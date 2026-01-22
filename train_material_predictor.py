#!/usr/bin/env python3
"""
Train an LSTM encoder + FFN to predict viscoelastic material parameters from stress-strain curves.

The model takes variable-length sequences of [S11, F11, F22, F33] and predicts
10 material parameters: kM, kF, m3, m5, q, m1, m2, m4, theta, alphaM.

Supports all loading scenarios:
- Scenario 0: Constant rate ramp (single curve)
- Scenario 1: Ramp + hold / stress relaxation (single curve)
- Scenario 2: Multi-step piecewise ramp (single curve)
- Scenario 3: Multi-rate repetition (MULTIPLE curves per sample)

For Scenario 3, each curve is encoded separately using the same LSTM encoder (weight sharing),
then embeddings are aggregated (mean pooling) before the MLP decoder.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


def init_weights_xavier(module: nn.Module):
    """Xavier (Glorot) initialization for both LSTM and MLP layers.
    Best suited for tanh and sigmoid activations.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)


def init_weights_kaiming(module: nn.Module):
    """Kaiming (He) initialization for ReLU-based networks.
    Uses fan_in mode which is standard for ReLU activations.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LSTM):
        # LSTM uses tanh and sigmoid internally, so Xavier is more appropriate
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)  # Helps with gradient flow in RNNs
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
                # Set forget gate bias to 1.0 for better gradient flow
                # Forget gate is typically the second quarter of the bias vector
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)


def get_init_function_for_activation(activation: str):
    """
    Returns the appropriate weight initialization function based on activation type.
    
    - ReLU: Kaiming/He initialization (designed for ReLU's non-symmetric nature)
    - Sigmoid/Tanh: Xavier/Glorot initialization (designed for symmetric activations)
    - None (linear): Xavier initialization
    
    Args:
        activation: One of 'relu', 'sigmoid', 'tanh', 'none'
    
    Returns:
        Initialization function to apply with model.apply()
    """
    activation = activation.lower()
    if activation == 'relu':
        return init_weights_kaiming
    elif activation in ['sigmoid', 'tanh', 'none']:
        return init_weights_xavier
    else:
        # Default to Xavier for unknown activations
        return init_weights_xavier


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss for material parameter prediction.
    
    Applies higher weights to parameters that are harder to predict (based on physics-based
    identifiability analysis). This helps the model focus more on difficult parameters
    like kF, alphaM, m5, q, m1 while not neglecting the easier ones.
    
    Default weights based on parameter prediction difficulty:
    - Easy params (m3, m2, theta): weight = 0.5
    - Medium params (kM, m4): weight = 1.0  
    - Hard params (kF, m5, q, m1): weight = 3.0
    - Very hard params (alphaM): weight = 5.0
    
    Parameter order: [kM, kF, m3, m5, q, m1, m2, m4, theta, alphaM]
    """
    
    # Default weights based on parameter difficulty
    # Order: [kM, kF, m3, m5, q, m1, m2, m4, theta, alphaM]
    DEFAULT_WEIGHTS = [1.0, 3.0, 0.5, 3.0, 3.0, 3.0, 0.5, 1.0, 0.5, 5.0]
    
    def __init__(self, weights: List[float] = None, param_names: List[str] = None):
        """
        Args:
            weights: List of weights for each parameter. If None, uses default weights.
            param_names: List of parameter names to predict. Used to select appropriate
                        default weights when predicting a subset of parameters.
        """
        super().__init__()
        
        # Full parameter name to default weight mapping
        self.param_weight_map = {
            'kM': 1.0,      # Medium - decent prediction
            'kF': 3.0,      # Hard - fiber viscosity hard to identify
            'm3': 0.5,      # Easy - direct effect on fiber stress
            'm5': 3.0,      # Hard - coupling parameter
            'q': 3.0,       # Hard - exponential coupling
            'm1': 3.0,      # Hard - volumetric penalty
            'm2': 0.5,      # Easy - direct effect on matrix stress
            'm4': 1.0,      # Medium - nonlinearity exponent
            'theta': 0.5,   # Easy - geometric effect
            'alphaM': 5.0,  # Very hard - coupled with kM, may be unidentifiable
        }
        
        if weights is not None:
            # Use provided weights
            self.weights = torch.tensor(weights, dtype=torch.float32)
        elif param_names is not None:
            # Build weights based on parameter names
            weights_list = [self.param_weight_map.get(name, 1.0) for name in param_names]
            self.weights = torch.tensor(weights_list, dtype=torch.float32)
        else:
            # Use default weights for all 10 parameters
            self.weights = torch.tensor(self.DEFAULT_WEIGHTS, dtype=torch.float32)
        
        # Register as buffer so it moves to correct device with model
        self.register_buffer('weight_buffer', self.weights)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            predictions: (batch_size, n_params) predicted parameters
            targets: (batch_size, n_params) ground truth parameters
            
        Returns:
            Weighted MSE loss (scalar)
        """
        # Ensure weights are on the same device as predictions
        weights = self.weight_buffer.to(predictions.device)
        
        # Compute squared errors
        squared_errors = (predictions - targets) ** 2
        
        # Apply weights and compute mean
        weighted_errors = weights * squared_errors
        
        return weighted_errors.mean()
    
    def get_weights_dict(self, param_names: List[str]) -> Dict[str, float]:
        """Return a dictionary of parameter names to weights for logging."""
        weights_np = self.weight_buffer.cpu().numpy()
        return {name: float(weights_np[i]) for i, name in enumerate(param_names)}


def safe_rate_calculation(F11: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calculate rate dF11/dt with protection against zero/tiny time differences.
    This is important for hold phases where time points can be identical.
    """
    if len(F11) < 2:
        return np.zeros_like(F11)
    
    dt = np.diff(t)
    dF11 = np.diff(F11)
    
    # Protect against division by zero or near-zero
    safe_dt = np.where(np.abs(dt) < 1e-9, np.sign(dt + 1e-12) * 1e-9, dt)
    rate_diff = dF11 / safe_dt
    
    # Prepend first value to match original length
    rate = np.concatenate([[rate_diff[0]], rate_diff])
    return rate


def extract_curve_features(
    S11: np.ndarray,
    F11: np.ndarray,
    F22: np.ndarray,
    F33: np.ndarray,
    t: np.ndarray,
    use_time: bool,
    use_rate: bool,
    max_seq_length: int = None,
    use_log_inputs: bool = False,
    s11_epsilon: float = None,
    normalize_s11_max: bool = False,
    normalize_inputs: bool = False,
    input_mean: np.ndarray = None,
    input_std: np.ndarray = None,
    use_scenario: bool = False,
    scenario: int = 0,
    use_physics_features: bool = False
) -> np.ndarray:
    """
    Extract and preprocess features from a single stress-strain curve.
    
    Args:
        S11: Stress values
        F11: Stretch in loading direction
        F22: Lateral stretch
        F33: Lateral stretch
        t: Time values
        use_time: Include time as feature
        use_rate: Include loading rate as feature
        max_seq_length: Maximum sequence length for downsampling
        use_log_inputs: Apply log10 to S11
        s11_epsilon: Epsilon for log transform
        normalize_s11_max: Normalize S11 by max value
        normalize_inputs: Apply z-score normalization
        input_mean: Mean for z-score normalization
        input_std: Std for z-score normalization
        use_scenario: Include one-hot encoded scenario as features
        scenario: Scenario type (0, 1, 2, or 3)
        
    Returns:
        sequence: (seq_len, n_features) array of preprocessed features
    """
    S11 = S11.copy()
    F11 = F11.copy()
    F22 = F22.copy()
    F33 = F33.copy()
    if t is not None:
        t = t.copy()
    
    seq_len = len(S11)
    
    # Uniform downsampling if sequence is too long
    if max_seq_length is not None and seq_len > max_seq_length:
        indices = np.round(np.linspace(0, seq_len - 1, max_seq_length)).astype(int)
        S11 = S11[indices]
        F11 = F11[indices]
        F22 = F22[indices]
        F33 = F33[indices]
        if t is not None:
            t = t[indices]
        seq_len = len(S11)
    
    phys_features = None
    if use_physics_features:
        if t is None:
            raise ValueError("Cannot calculate physics features without time 't'")
        phys_features = calculate_physics_features(S11, F11, F22, F33, t)
    
    # Apply log transformation to S11 if enabled (AFTER physics features calculation)
    if use_log_inputs and s11_epsilon is not None:
        S11 = np.log10(np.abs(S11) + s11_epsilon)
    
    # Apply max normalization to S11 if enabled
    if normalize_s11_max:
        max_s11 = np.max(np.abs(S11))
        if max_s11 > 1e-8:
            S11 = S11 / max_s11
    
    features = [S11, F11, F22, F33]
    if use_time:
        if t is None:
            raise ValueError("Cannot use time feature: 't' not provided")
        features.append(t)
    
    if use_rate:
        if t is None:
            raise ValueError("Cannot calculate rate without time 't'")
        rate = safe_rate_calculation(F11, t)
        features.append(rate)
    
    # Add physics features (already calculated on original S11)
    if use_physics_features and phys_features is not None:
        features.extend(phys_features)
    
    sequence = np.stack(features, axis=1)
    
    # Apply input normalization if enabled (before adding scenario features)
    if normalize_inputs and input_mean is not None and input_std is not None:
        sequence = (sequence - input_mean) / input_std
    
    # Add one-hot encoded scenario features AFTER normalization
    # (scenario features are already 0/1, no normalization needed)
    if use_scenario:
        # Create one-hot encoding for scenarios 0, 1, 2, 3
        # Each scenario feature is constant across the sequence
        is_scenario_0 = np.ones((seq_len, 1)) if scenario == 0 else np.zeros((seq_len, 1))
        is_scenario_1 = np.ones((seq_len, 1)) if scenario == 1 else np.zeros((seq_len, 1))
        is_scenario_2 = np.ones((seq_len, 1)) if scenario == 2 else np.zeros((seq_len, 1))
        is_scenario_3 = np.ones((seq_len, 1)) if scenario == 3 else np.zeros((seq_len, 1))
        scenario_features = np.hstack([is_scenario_0, is_scenario_1, is_scenario_2, is_scenario_3])
        sequence = np.hstack([sequence, scenario_features])
    
    return sequence


def calculate_physics_features(
    S11: np.ndarray,
    F11: np.ndarray,
    F22: np.ndarray,
    F33: np.ndarray,
    t: np.ndarray
) -> List[np.ndarray]:
    """
    Calculate physics-informed features: derivatives, ratios, normalized shapes,
    and volumetric invariants.
    
    Features:
    1. dS11_dt: Stress rate (Relaxation)
    2. dF22_dt: Lateral stretch rate (Lateral creep)
    3. dF11_dt: Loading rate
    4. dS11_dF11: Tangent stiffness
    5. d2S11_dF112: Curvature
    6. F22_F11_Ratio: Poisson effect
    7. F22_F33_Ratio: Anisotropy
    8. S_norm: Normalized stress shape
    9. J: Volume ratio (det(F) = F11 * F22 * F33 for diagonal F)
    10. log_J: log(|J| + epsilon) - volumetric strain measure
    11. I1: First invariant of b (= F11^2 + F22^2 + F33^2 for diagonal F)
    
    Returns:
        List of feature arrays
    """
    # Check minimum array size for gradient calculation
    if len(t) < 2:
        # Return zeros for all features if array is too small
        n = len(t) if len(t) > 0 else 1
        zeros = np.zeros(n)
        return [zeros] * 11  # 11 physics features
    
    # 1. Time derivatives (Rates)
    # Use safe gradient calculation
    dt = np.gradient(t)
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)  # Avoid division by zero
    
    dS11_dt = np.gradient(S11, t)
    dF22_dt = np.gradient(F22, t)
    dF11_dt = np.gradient(F11, t)
    
    # 2. Strain derivatives (Stiffness/Curvature)
    # dS11/dF11 = (dS11/dt) / (dF11/dt)
    # Add epsilon to denominator to avoid division by zero during holds
    dF11_dt_safe = np.where(np.abs(dF11_dt) < 1e-9, 1e-9 * np.sign(dF11_dt + 1e-12), dF11_dt)
    dS11_dF11 = dS11_dt / dF11_dt_safe
    
    # d2S11/dF11^2 = d(dS11_dF11)/dt / (dF11/dt)
    d_stiffness_dt = np.gradient(dS11_dF11, t)
    d2S11_dF112 = d_stiffness_dt / dF11_dt_safe
    
    # 3. Ratios
    # Avoid division by zero for F11 (though F11 usually >= 1)
    F11_safe = np.where(np.abs(F11) < 1e-9, 1e-9, F11)
    F33_safe = np.where(np.abs(F33) < 1e-9, 1e-9, F33)
    
    F22_F11_Ratio = F22 / F11_safe
    F22_F33_Ratio = F22 / F33_safe
    
    # 4. Normalized Shape
    max_s11 = np.max(np.abs(S11))
    if max_s11 < 1e-9:
        max_s11 = 1.0
    S_norm = S11 / max_s11
    
    # 5. Volumetric invariants (target m5, alphaM)
    # J = det(F) = F11 * F22 * F33 for diagonal deformation gradient
    J = F11 * F22 * F33
    
    # log(|J| + epsilon) - logarithmic volumetric strain
    log_J = np.log(np.abs(J) + 1e-9)
    
    # I1 = tr(b) = F11^2 + F22^2 + F33^2 for diagonal left Cauchy-Green tensor b = F @ F.T
    I1 = F11**2 + F22**2 + F33**2
    
    return [
        dS11_dt,
        dF22_dt,
        dF11_dt,
        dS11_dF11,
        d2S11_dF112,
        F22_F11_Ratio,
        F22_F33_Ratio,
        S_norm,
        J,
        log_J,
        I1
    ]


class MaterialDataset(Dataset):
    """
    Dataset for loading viscoelastic material curves from .npz files.
    
    Supports all scenarios:
    - Scenarios 0, 1, 2: Single curve per sample (standard format)
    - Scenario 3: Multiple curves per sample (multi-rate format)
    
    For Scenario 3, returns a list of curve tensors instead of a single tensor.
    """

    # All available parameters (excluding mu0, rate, lambda_max)
    ALL_PARAM_INDICES = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]  # kM, kF, m3, m5, q, m1, m2, m4, theta, alphaM
    ALL_PARAM_NAMES = ['kM', 'kF', 'm3', 'm5', 'q', 'm1', 'm2', 'm4', 'theta', 'alphaM']

    # Parameters that should use log transformation (based on large dynamic range)
    LOG_PARAM_NAMES = ['kM', 'kF', 'm3', 'm5', 'q', 'm1', 'm2', 'alphaM']
    # Parameters that should NOT use log (narrow range or bounded)
    NO_LOG_PARAM_NAMES = ['m4', 'theta']

    def __init__(
        self,
        npz_files: List[Path],
        normalize: bool = True,
        use_time: bool = False,
        use_rate: bool = False,
        target_params: List[str] = None,
        max_seq_length: int = None,
        use_log_targets: bool = False,
        normalize_inputs: bool = False,
        use_log_inputs: bool = False,
        normalize_s11_max: bool = False,
        use_scenario: bool = False,
        use_physics_features: bool = False,
        # Precomputed normalization parameters (to avoid redundant computation)
        precomputed_param_norm: dict = None,
        precomputed_input_norm: dict = None,
    ):
        """
        Args:
            npz_files: List of paths to .npz files
            normalize: Whether to normalize parameters (targets)
            use_time: Whether to include time as a feature (default: False)
            use_rate: Whether to include loading rate as a feature (default: False)
            target_params: List of parameter names to predict (default: all 10)
            max_seq_length: Maximum sequence length for uniform downsampling (default: None)
            use_log_targets: Whether to apply log transform to targets with large dynamic range
            normalize_inputs: Whether to normalize input sequences with z-score
            use_log_inputs: Whether to apply log10 to S11 before normalization
            normalize_s11_max: Whether to normalize S11 by its per-curve max value
            use_scenario: Whether to include one-hot encoded scenario as input features
            precomputed_param_norm: Dict with 'mean' and 'std' arrays (skip computation if provided)
            precomputed_input_norm: Dict with 'mean', 'std', and optionally 's11_epsilon'
        """
        if len(npz_files) == 0:
            raise ValueError("Cannot create dataset with no files")
        
        self.npz_files = npz_files
        self.normalize = normalize
        self.use_time = use_time
        self.use_rate = use_rate
        self.max_seq_length = max_seq_length
        self.use_log_targets = use_log_targets
        self.normalize_inputs = normalize_inputs
        self.use_log_inputs = use_log_inputs
        self.normalize_s11_max = normalize_s11_max
        self.use_scenario = use_scenario
        self.use_physics_features = use_physics_features

        # Determine which parameters to predict
        if target_params is None:
            self.PARAM_NAMES = self.ALL_PARAM_NAMES.copy()
            self.PARAM_INDICES = self.ALL_PARAM_INDICES.copy()
        else:
            self.PARAM_NAMES = []
            self.PARAM_INDICES = []
            for param in target_params:
                if param not in self.ALL_PARAM_NAMES:
                    raise ValueError(
                        f"Invalid parameter '{param}'. "
                        f"Available: {', '.join(self.ALL_PARAM_NAMES)}"
                    )
                idx = self.ALL_PARAM_NAMES.index(param)
                self.PARAM_NAMES.append(param)
                self.PARAM_INDICES.append(self.ALL_PARAM_INDICES[idx])

        # Determine which of the selected parameters should use log
        self.param_use_log = np.array([
            name in self.LOG_PARAM_NAMES for name in self.PARAM_NAMES
        ])

        # Initialize s11_epsilon
        self.s11_epsilon = None

        # Warn about conflicting S11 transformations
        if use_log_inputs and normalize_s11_max:
            print("WARNING: Both use_log_inputs and normalize_s11_max enabled. "
                  "Max normalization will be applied to log-transformed values.")

        # Validate time data availability if needed
        if use_time or use_rate or use_physics_features:
            self._validate_time_data()

        # Use precomputed or compute normalization parameters
        if normalize:
            if precomputed_param_norm is not None:
                self.param_mean = precomputed_param_norm['mean']
                self.param_std = precomputed_param_norm['std']
            else:
                self._compute_normalization_params()

        if normalize_inputs:
            if precomputed_input_norm is not None:
                self.input_mean = precomputed_input_norm['mean']
                self.input_std = precomputed_input_norm['std']
                if use_log_inputs:
                    self.s11_epsilon = precomputed_input_norm.get('s11_epsilon')
            else:
                self._compute_input_normalization()
    
    def _has_field(self, obj, key):
        """Helper to check if a key/attribute exists in a dict, object, or numpy structured array."""
        if isinstance(obj, dict):
            return key in obj
        # Handle numpy structured array (has dtype.names)
        if hasattr(obj, 'dtype') and obj.dtype.names is not None:
            return key in obj.dtype.names
        return hasattr(obj, key)

    def _get_field(self, obj, key, default=None):
        """Helper to get a value from a dict, object, or numpy structured array."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        # Handle numpy structured array (has dtype.names)
        if hasattr(obj, 'dtype') and obj.dtype.names is not None:
            if key in obj.dtype.names:
                return obj[key]
            return default
        return getattr(obj, key, default)
    
    def _flatten_array(self, arr):
        """
        Flatten array from (N, 1) or similar 2D shapes to 1D (N,).
        MATLAB-generated NPZ files often store arrays as 2D column vectors.
        Also handles nested (1,1) object arrays from MATLAB structured arrays.
        """
        if arr is None:
            return None
        arr = np.asarray(arr)
        # Handle (1,1) object arrays containing the actual data (from MATLAB structured arrays)
        while arr.shape == (1, 1) and arr.dtype == object:
            arr = arr[0, 0]
            arr = np.asarray(arr)
        if arr.ndim > 1:
            return arr.flatten()
        return arr
    
    def _validate_time_data(self):
        """Check that time data is available in files if use_time or use_rate is enabled."""
        # Check first file as representative sample
        with np.load(self.npz_files[0], allow_pickle=True) as data:
            scenario_val = data.get('scenario', 0)
            scenario = int(scenario_val.item()) if hasattr(scenario_val, 'item') else int(scenario_val)
            
            if scenario == 3:
                # Scenario 3: check multi_curves structure
                multi_curves = data.get('multi_curves', None)
                if multi_curves is None:
                    raise ValueError(
                        f"Scenario 3 file {self.npz_files[0]} missing 'multi_curves' array"
                    )
                # Flatten multi_curves - MATLAB exports as (1, N) 2D array
                multi_curves = multi_curves.flatten()
                # Check first curve in the multi_curves array
                first_curve = multi_curves[0] if len(multi_curves) > 0 else {}
                if not self._has_field(first_curve, 't'):
                    if self.use_time:
                        raise ValueError(
                            f"use_time=True but 't' not found in multi_curves of {self.npz_files[0]}. "
                            "Time data is required when use_time is enabled."
                        )
                    if self.use_rate:
                        raise ValueError(
                            f"use_rate=True but 't' not found in multi_curves of {self.npz_files[0]}. "
                            "Time data is required to calculate rate."
                        )
            else:
                # Scenarios 0, 1, 2: standard single-curve format
                if 't' not in data:
                    if self.use_time:
                        raise ValueError(
                            f"use_time=True but 't' not found in {self.npz_files[0]}. "
                            "Time data is required when use_time is enabled."
                        )
                    if self.use_rate:
                        raise ValueError(
                            f"use_rate=True but 't' not found in {self.npz_files[0]}. "
                            "Time data is required to calculate rate."
                        )

    def _compute_normalization_params(self):
        """Compute mean and std for parameter normalization (after log transform if enabled)."""
        all_params = []
        for npz_file in self.npz_files:
            with np.load(npz_file, allow_pickle=True) as data:
                # Flatten par from (N, 1) to (N,) - MATLAB exports as 2D column vectors
                par = self._flatten_array(data['par'])
                selected_params = par[self.PARAM_INDICES].copy()

                if self.use_log_targets:
                    log_values = selected_params[self.param_use_log]
                    if np.any(log_values <= 0):
                        raise ValueError(
                            f"Cannot apply log to non-positive values in {npz_file}. "
                            f"Values: {log_values[log_values <= 0]}"
                        )
                    selected_params[self.param_use_log] = np.log(log_values)

                all_params.append(selected_params)

        all_params = np.array(all_params)
        self.param_mean = all_params.mean(axis=0)
        self.param_std = all_params.std(axis=0)

        # Prevent division by zero
        self.param_std = np.where(self.param_std < 1e-8, 1.0, self.param_std)

        # Normalization parameters computed (not printing details to reduce verbosity)

    def _compute_input_normalization(self):
        """
        Compute mean and std for input sequence normalization using Welford's 
        online algorithm (memory-efficient streaming statistics).
        
        Handles both single-curve (scenarios 0-2) and multi-curve (scenario 3) data.
        """
        feature_names = ['S11', 'F11', 'F22', 'F33']
        if self.use_time:
            feature_names.append('t')
        if self.use_rate:
            feature_names.append('rate')
        if self.use_physics_features:
            feature_names.extend([
                'dS11_dt', 'dF22_dt', 'dF11_dt', 
                'dS11_dF11', 'd2S11_dF112', 
                'F22_F11_Ratio', 'F22_F33_Ratio', 
                'S_norm', 'J', 'log_J', 'I1'
            ])

        n_features = len(feature_names)
        
        # Welford's online algorithm for streaming mean and variance
        count = np.zeros(n_features)
        mean = np.zeros(n_features)
        M2 = np.zeros(n_features)  # Sum of squared differences from mean
        
        # First pass: find max S11 for epsilon calculation if using log
        if self.use_log_inputs:
            max_s11 = 0.0
            for npz_file in self.npz_files:
                with np.load(npz_file, allow_pickle=True) as data:
                    scenario_val = data.get('scenario', 0)
                    scenario = int(scenario_val.item()) if hasattr(scenario_val, 'item') else int(scenario_val)
                    if scenario == 3:
                        # Scenario 3: check all curves in multi_curves
                        multi_curves = data['multi_curves'].flatten()  # MATLAB exports as (1, N)
                        for curve in multi_curves:
                            s11_val = self._flatten_array(self._get_field(curve, 'S11'))
                            max_s11 = max(max_s11, np.max(np.abs(s11_val)))
                    else:
                        s11_val = self._flatten_array(data['S11'])
                        max_s11 = max(max_s11, np.max(np.abs(s11_val)))
            self.s11_epsilon = 1e-6 * max_s11
            print(f"  S11 log transform: ε = {self.s11_epsilon:.6e}")

        # Helper function to update stats for a single curve
        def update_stats_for_curve(S11, F11, F22, F33, t_arr):
            nonlocal count, mean, M2
            
            phys_features = None
            if self.use_physics_features:
                phys_features = calculate_physics_features(S11, F11, F22, F33, t_arr)
            
            # Apply log transform to S11 if enabled (AFTER physics features calculation)
            if self.use_log_inputs:
                S11 = np.log10(np.abs(S11) + self.s11_epsilon)

            features = [S11, F11, F22, F33]
            
            if self.use_time:
                features.append(t_arr)
            
            if self.use_rate:
                rate = safe_rate_calculation(F11, t_arr)
                features.append(rate)

            # Add physics features (already calculated on original S11)
            if self.use_physics_features and phys_features is not None:
                features.extend(phys_features)

            # Update statistics for each feature using Welford's algorithm
            for i, feat_values in enumerate(features):
                for x in feat_values:
                    count[i] += 1
                    delta = x - mean[i]
                    mean[i] += delta / count[i]
                    delta2 = x - mean[i]
                    M2[i] += delta * delta2

        # Second pass: compute streaming statistics
        for npz_file in self.npz_files:
            with np.load(npz_file, allow_pickle=True) as data:
                scenario_val = data.get('scenario', 0)
                scenario = int(scenario_val.item()) if hasattr(scenario_val, 'item') else int(scenario_val)
                
                if scenario == 3:
                    # Scenario 3: process all curves in multi_curves
                    multi_curves = data['multi_curves'].flatten()  # MATLAB exports as (1, N)
                    for curve in multi_curves:
                        t_val = self._flatten_array(self._get_field(curve, 't'))
                        # Validate time data when physics features are enabled
                        if self.use_physics_features and t_val is None:
                            raise ValueError(
                                f"Cannot compute physics features: 't' not found in multi_curves of {npz_file}. "
                                "Time data is required when use_physics_features is enabled."
                            )
                        update_stats_for_curve(
                            self._flatten_array(self._get_field(curve, 'S11')), 
                            self._flatten_array(self._get_field(curve, 'F11')), 
                            self._flatten_array(self._get_field(curve, 'F22')), 
                            self._flatten_array(self._get_field(curve, 'F33')), 
                            t_val
                        )
                else:
                    # Scenarios 0, 1, 2: single curve
                    # Flatten arrays from (N, 1) to (N,) - MATLAB exports as 2D column vectors
                    S11 = self._flatten_array(data['S11'])
                    F11 = self._flatten_array(data['F11'])
                    F22 = self._flatten_array(data['F22'])
                    F33 = self._flatten_array(data['F33'])
                    t_arr = self._flatten_array(data['t']) if 't' in data else None
                    
                    # Validate time data when physics features are enabled
                    if self.use_physics_features and t_arr is None:
                        raise ValueError(
                            f"Cannot compute physics features: 't' not found in {npz_file}. "
                            "Time data is required when use_physics_features is enabled."
                        )
                    
                    update_stats_for_curve(S11, F11, F22, F33, t_arr)

        # Compute final variance and std
        variance = M2 / count
        std = np.sqrt(variance)
        
        self.input_mean = mean
        self.input_std = np.where(std < 1e-8, 1.0, std)

    def __len__(self) -> int:
        return len(self.npz_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            For scenarios 0-2 (single curve):
                sequences: (seq_len, input_dim) tensor of [S11, F11, F22, F33, (t), (rate)]
                parameters: (n_params,) tensor of selected parameters
                seq_len: actual sequence length (after downsampling if applicable)
            
            For scenario 3 (multi-curve):
                sequences: list of (seq_len_i, input_dim) tensors, one per curve
                parameters: (n_params,) tensor of selected parameters (same for all curves)
                n_curves: number of curves in this sample
        """
        with np.load(self.npz_files[idx], allow_pickle=True) as data:
            scenario_val = data.get('scenario', 0)
            scenario = int(scenario_val.item()) if hasattr(scenario_val, 'item') else int(scenario_val)
            # Flatten par from (N, 1) to (N,) - MATLAB exports as 2D column vectors
            par = self._flatten_array(data['par'])
            
            if scenario == 3:
                # Scenario 3: Multiple curves per sample
                multi_curves = data['multi_curves'].flatten()  # MATLAB exports as (1, N)
                curve_sequences = []
                
                for curve in multi_curves:
                    # Flatten arrays from (N, 1) to (N,) - MATLAB exports as 2D column vectors
                    S11 = self._flatten_array(self._get_field(curve, 'S11')).copy()
                    F11 = self._flatten_array(self._get_field(curve, 'F11')).copy()
                    F22 = self._flatten_array(self._get_field(curve, 'F22')).copy()
                    F33 = self._flatten_array(self._get_field(curve, 'F33')).copy()
                    t_val = self._flatten_array(self._get_field(curve, 't'))
                    t = t_val.copy() if t_val is not None else None
                    
                    # Use the helper function to extract features
                    sequence = extract_curve_features(
                        S11, F11, F22, F33, t,
                        use_time=self.use_time,
                        use_rate=self.use_rate,
                        max_seq_length=self.max_seq_length,
                        use_log_inputs=self.use_log_inputs,
                        s11_epsilon=self.s11_epsilon,
                        normalize_s11_max=self.normalize_s11_max,
                        normalize_inputs=self.normalize_inputs,
                        input_mean=self.input_mean if self.normalize_inputs else None,
                        input_std=self.input_std if self.normalize_inputs else None,
                        use_scenario=self.use_scenario,
                        scenario=scenario,
                        use_physics_features=self.use_physics_features
                    )
                    curve_sequences.append(torch.tensor(sequence, dtype=torch.float32))
                
                # Process parameters (same for all curves)
                selected_params = par[self.PARAM_INDICES].copy()
                
                if self.use_log_targets:
                    log_values = selected_params[self.param_use_log]
                    if np.any(log_values <= 0):
                        raise ValueError(
                            f"Cannot apply log to non-positive values. File index: {idx}"
                        )
                    selected_params[self.param_use_log] = np.log(log_values)
                
                if self.normalize:
                    selected_params = (selected_params - self.param_mean) / self.param_std
                
                return (
                    curve_sequences,  # List of tensors for multi-curve
                    torch.tensor(selected_params, dtype=torch.float32),
                    len(curve_sequences)  # Number of curves
                )
            
            else:
                # Scenarios 0, 1, 2: Single curve
                # Flatten arrays from (N, 1) to (N,) - MATLAB exports as 2D column vectors
                S11 = self._flatten_array(data['S11']).copy()
                F11 = self._flatten_array(data['F11']).copy()
                F22 = self._flatten_array(data['F22']).copy()
                F33 = self._flatten_array(data['F33']).copy()
                t = self._flatten_array(data['t']).copy() if 't' in data else None

                seq_len = len(S11)

                # Uniform downsampling if sequence is too long
                if self.max_seq_length is not None and seq_len > self.max_seq_length:
                    indices = np.round(np.linspace(0, seq_len - 1, self.max_seq_length)).astype(int)
                    S11 = S11[indices]
                    F11 = F11[indices]
                    F22 = F22[indices]
                    F33 = F33[indices]
                    if t is not None:
                        t = t[indices]
                    seq_len = len(S11)

                phys_features = None
                if self.use_physics_features:
                    if t is None:
                        raise ValueError("Cannot calculate physics features without time 't' in data")
                    phys_features = calculate_physics_features(S11, F11, F22, F33, t)

                # Apply log transformation to S11 if enabled (AFTER physics features calculation)
                if self.use_log_inputs:
                    S11 = np.log10(np.abs(S11) + self.s11_epsilon)
                
                # Apply max normalization to S11 if enabled
                if self.normalize_s11_max:
                    max_s11 = np.max(np.abs(S11))
                    if max_s11 > 1e-8:
                        S11 = S11 / max_s11

                features = [S11, F11, F22, F33]
                if self.use_time:
                    if t is None:
                        raise ValueError("Cannot use time feature: 't' not in data")
                    features.append(t)
                
                if self.use_rate:
                    if t is None:
                        raise ValueError("Cannot calculate rate without time 't' in data")
                    rate = safe_rate_calculation(F11, t)
                    features.append(rate)

                # Add physics features (already calculated on original S11)
                if self.use_physics_features and phys_features is not None:
                    features.extend(phys_features)

                sequences = np.stack(features, axis=1)

                if self.normalize_inputs:
                    sequences = (sequences - self.input_mean) / self.input_std

                # Add one-hot encoded scenario features AFTER normalization
                if self.use_scenario:
                    is_scenario_0 = np.ones((seq_len, 1)) if scenario == 0 else np.zeros((seq_len, 1))
                    is_scenario_1 = np.ones((seq_len, 1)) if scenario == 1 else np.zeros((seq_len, 1))
                    is_scenario_2 = np.ones((seq_len, 1)) if scenario == 2 else np.zeros((seq_len, 1))
                    is_scenario_3 = np.ones((seq_len, 1)) if scenario == 3 else np.zeros((seq_len, 1))
                    scenario_features = np.hstack([is_scenario_0, is_scenario_1, is_scenario_2, is_scenario_3])
                    sequences = np.hstack([sequences, scenario_features])

                selected_params = par[self.PARAM_INDICES].copy()

                if self.use_log_targets:
                    log_values = selected_params[self.param_use_log]
                    if np.any(log_values <= 0):
                        raise ValueError(
                            f"Cannot apply log to non-positive values. File index: {idx}"
                        )
                    selected_params[self.param_use_log] = np.log(log_values)

                if self.normalize:
                    selected_params = (selected_params - self.param_mean) / self.param_std

                return (
                    torch.tensor(sequences, dtype=torch.float32),
                    torch.tensor(selected_params, dtype=torch.float32),
                    len(sequences)
                )


def collate_fn(batch: List[Tuple]):
    """
    Collate function to pad sequences to the same length in a batch.
    
    Handles both:
    - Single-curve samples (scenarios 0-2): sequences is a tensor
    - Multi-curve samples (scenario 3): sequences is a list of tensors
    
    Returns:
        For uniform batches (all single-curve or all multi-curve):
            padded_sequences: (batch_size, max_seq_len, input_dim) or 
                             (total_curves, max_seq_len, input_dim) for multi-curve
            parameters: (batch_size, n_params)
            lengths: (batch_size,) or (total_curves,) for multi-curve
            curve_counts: (batch_size,) number of curves per sample (1 for single-curve)
    """
    sequences_list, parameters, counts = zip(*batch)
    
    # Handle mixed batches: treat everything as multi-curve (list of tensors)
    # If an item is a single tensor (Scenario 0-2), wrap it in a list
    all_curves = []
    all_lengths = []
    curve_counts = []
    
    for item in sequences_list:
        if isinstance(item, list):
            # Multi-curve (Scenario 3)
            curves = item
        else:
            # Single-curve (Scenario 0-2)
            curves = [item]
            
        curve_counts.append(len(curves))
        for curve in curves:
            all_curves.append(curve)
            all_lengths.append(len(curve))
    
    padded_sequences = nn.utils.rnn.pad_sequence(all_curves, batch_first=True)
    parameters = torch.stack(parameters)
    lengths = torch.tensor(all_lengths, dtype=torch.long)
    curve_counts = torch.tensor(curve_counts, dtype=torch.long)
    
    return padded_sequences, parameters, lengths, curve_counts


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder + FFN for material parameter prediction.
    
    Supports both single-curve and multi-curve inputs:
    - Single curve: Standard LSTM encoding → MLP prediction
    - Multi-curve: Encode each curve → Aggregate embeddings → MLP prediction
    
    Aggregation method for multi-curve:
    - 'mean': Simple average of embeddings
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_dim: int = 10,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_cell_state: bool = False,
        mlp_layers: List[int] = None,
        activation: str = 'relu',
        aggregation: str = 'mean'
    ):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.use_cell_state = use_cell_state
        self.activation_name = activation.lower()
        aggregation = aggregation.lower()
        if aggregation != 'mean':
            raise ValueError("Only 'mean' aggregation is supported.")
        self.aggregation = aggregation

        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        state_multiplier = 2 if use_cell_state else 1
        self.embedding_dim = hidden_size * self.num_directions * state_multiplier
        ffn_input_dim = self.embedding_dim

        if mlp_layers is None:
            mlp_layers = [256, 128, 64]

        if self.activation_name == 'relu':
            activation_cls = nn.ReLU
        elif self.activation_name == 'sigmoid':
            activation_cls = nn.Sigmoid
        elif self.activation_name == 'none':
            activation_cls = None
        else:
            raise ValueError("activation must be one of ['relu', 'sigmoid', 'none']")

        layers = []
        prev_dim = ffn_input_dim

        for hidden_dim in mlp_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation_cls is not None:
                layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.ffn = nn.Sequential(*layers)
    
    def encode_sequences(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode sequences using LSTM and return the embedding.
        
        Args:
            x: (batch_size, seq_len, input_dim) padded sequences
            lengths: (batch_size,) actual sequence lengths
        Returns:
            embeddings: (batch_size, embedding_dim) encoded representations
        """
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, (h_n, c_n) = self.lstm(packed)

        if self.num_directions == 2:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h_last = torch.cat([h_forward, h_backward], dim=1)

            if self.use_cell_state:
                c_forward = c_n[-2, :, :]
                c_backward = c_n[-1, :, :]
                c_last = torch.cat([c_forward, c_backward], dim=1)
                state = torch.cat([h_last, c_last], dim=1)
            else:
                state = h_last
        else:
            h_last = h_n[-1, :, :]
            if self.use_cell_state:
                c_last = c_n[-1, :, :]
                state = torch.cat([h_last, c_last], dim=1)
            else:
                state = h_last
        
        return state
    
    def aggregate_embeddings(
        self, 
        embeddings: torch.Tensor, 
        curve_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate embeddings from multiple curves for each sample.
        
        Args:
            embeddings: (total_curves, embedding_dim) embeddings for all curves
            curve_counts: (batch_size,) number of curves per sample
        Returns:
            aggregated: (batch_size, embedding_dim) aggregated embeddings
        """
        batch_size = len(curve_counts)
        device = embeddings.device
        
        aggregated = []
        idx = 0
        for count in curve_counts:
            count = count.item()
            sample_embeddings = embeddings[idx:idx + count]  # (n_curves, embedding_dim)
            aggregated.append(sample_embeddings.mean(dim=0))
            idx += count
        return torch.stack(aggregated)
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor,
        curve_counts: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass supporting both single-curve and multi-curve inputs.
        
        Args:
            x: (batch_size, seq_len, input_dim) for single-curve, or
               (total_curves, seq_len, input_dim) for multi-curve
            lengths: (batch_size,) or (total_curves,) actual sequence lengths
            curve_counts: (batch_size,) number of curves per sample.
                         If None or all ones, treated as single-curve mode.
        Returns:
            predictions: (batch_size, output_dim) predicted parameters
        """
        # Encode all sequences
        embeddings = self.encode_sequences(x, lengths)
        
        # Check if multi-curve aggregation is needed
        if curve_counts is not None and (curve_counts > 1).any():
            # Multi-curve mode: aggregate embeddings
            embeddings = self.aggregate_embeddings(embeddings, curve_counts)
        
        # Pass through MLP
        predictions = self.ffn(embeddings)
        return predictions


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    use_amp: bool = False,
    max_grad_norm: float = 1.0
) -> float:
    """Train for one epoch with gradient clipping. Supports multi-curve data."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for sequences, parameters, lengths, curve_counts in dataloader:
        batch_size = parameters.size(0)  # Use parameters size since sequences may be flattened
        sequences = sequences.to(device)
        parameters = parameters.to(device)
        curve_counts = curve_counts.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(device_type='cuda'):
                predictions = model(sequences, lengths, curve_counts)
                loss = criterion(predictions, parameters)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(sequences, lengths, curve_counts)
            loss = criterion(predictions, parameters)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False
) -> float:
    """Validate the model. Supports multi-curve data."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for sequences, parameters, lengths, curve_counts in dataloader:
            batch_size = parameters.size(0)
            sequences = sequences.to(device)
            parameters = parameters.to(device)
            curve_counts = curve_counts.to(device)

            if use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    predictions = model(sequences, lengths, curve_counts)
                    loss = criterion(predictions, parameters)
            else:
                predictions = model(sequences, lengths, curve_counts)
                loss = criterion(predictions, parameters)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def denormalize_parameters(
    normalized_params: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """Denormalize parameters back to original scale."""
    return normalized_params * std + mean


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    dataset: MaterialDataset,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model and return predictions and ground truth in original scale.
    Supports multi-curve data.
    
    Returns:
        predictions: (n_samples, n_params) array of predicted parameters
        ground_truth: (n_samples, n_params) array of true parameters
    """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for sequences, parameters, lengths, curve_counts in dataloader:
            sequences = sequences.to(device)
            curve_counts = curve_counts.to(device)
            predictions = model(sequences, lengths, curve_counts)
            all_preds.append(predictions.cpu().numpy())
            all_true.append(parameters.numpy())

    predictions = np.vstack(all_preds)
    ground_truth = np.vstack(all_true)

    if dataset.normalize:
        predictions = denormalize_parameters(predictions, dataset.param_mean, dataset.param_std)
        ground_truth = denormalize_parameters(ground_truth, dataset.param_mean, dataset.param_std)

    if dataset.use_log_targets:
        predictions[:, dataset.param_use_log] = np.exp(predictions[:, dataset.param_use_log])
        ground_truth[:, dataset.param_use_log] = np.exp(ground_truth[:, dataset.param_use_log])

    return predictions, ground_truth


def save_figure_with_svg(fig_or_none, save_path: Path, dpi: int = 150):
    """
    Save figure to PNG and also create an SVG copy in the svg subdirectory.
    
    Args:
        fig_or_none: matplotlib figure object, or None to use current figure
        save_path: Path for the PNG file
        dpi: DPI for PNG output
    """
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PNG
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Create SVG directory and save SVG copy
    svg_dir = save_path.parent / 'svg'
    svg_dir.mkdir(parents=True, exist_ok=True)
    svg_path = svg_dir / save_path.with_suffix('.svg').name
    plt.savefig(svg_path, format='svg', bbox_inches='tight')


def plot_training_curves(train_losses: List[float], val_losses: List[float], save_path: Path):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    save_figure_with_svg(None, save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    param_names: List[str],
    save_path: Path,
    log_param_names: List[str] = None
):
    """Plot predictions vs ground truth for all parameters."""
    if log_param_names is None:
        log_param_names = []

    n_params = len(param_names)
    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes[:n_params], param_names)):
        pred = predictions[:, i]
        true = ground_truth[:, i]
        use_log = name in log_param_names
        
        ax.scatter(true, pred, alpha=0.5, s=10)
        
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        if use_log:
            true_log = np.log10(np.maximum(true, 1e-10))
            pred_log = np.log10(np.maximum(pred, 1e-10))
            
            ss_res = np.sum((true_log - pred_log) ** 2)
            ss_tot = np.sum((true_log - true_log.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((true_log - pred_log) ** 2))
            metric_str = f'Log R² = {r2:.4f}, Log RMSE = {rmse:.4f}'
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ss_res = np.sum((true - pred) ** 2)
            ss_tot = np.sum((true - true.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((true - pred) ** 2))
            metric_str = f'R² = {r2:.4f}, RMSE = {rmse:.4f}'
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}\n{metric_str}')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
    
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_figure_with_svg(None, save_path)
    plt.close()
    print(f"Prediction plots saved to {save_path}")


def plot_predictions_with_worst(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    param_names: List[str],
    save_path: Path,
    worst_indices_path: Path = None,
    n_worst: int = 10,
    log_param_names: List[str] = None
) -> dict:
    """
    Plot predictions vs ground truth with worst predictions highlighted in red.
    
    Args:
        predictions: (n_samples, n_params) predicted values
        ground_truth: (n_samples, n_params) true values
        param_names: List of parameter names
        save_path: Path to save the plot
        worst_indices_path: Path to save worst indices as CSV
        n_worst: Number of worst predictions to highlight per parameter
        log_param_names: Parameters that should use log scale
    
    Returns:
        Dictionary containing:
        - 'worst_indices': dict mapping param_name -> array of worst sample indices
        - 'worst_errors': dict mapping param_name -> array of error values for worst samples
        - 'overall_worst_idx': Index of sample with largest overall normalized error
        - 'overall_worst_error': The error value for that sample
    """
    if log_param_names is None:
        log_param_names = []

    n_params = len(param_names)
    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    
    worst_indices = {}
    worst_errors = {}
    
    # Compute normalized errors for overall worst (using relative error for each param)
    all_relative_errors = np.zeros(predictions.shape[0])
    
    for i, (ax, name) in enumerate(zip(axes[:n_params], param_names)):
        pred = predictions[:, i]
        true = ground_truth[:, i]
        use_log = name in log_param_names
        
        # Compute errors (relative for log-scale params, absolute for others)
        if use_log:
            # For log-scale params, use log-space error
            true_safe = np.maximum(np.abs(true), 1e-10)
            pred_safe = np.maximum(np.abs(pred), 1e-10)
            errors = np.abs(np.log10(pred_safe) - np.log10(true_safe))
        else:
            # For linear params, use relative error (normalized by range)
            param_range = true.max() - true.min()
            if param_range > 0:
                errors = np.abs(pred - true) / param_range
            else:
                errors = np.abs(pred - true)
        
        # Add to overall error (normalized contribution)
        all_relative_errors += errors / (errors.max() + 1e-10)
        
        # Find worst predictions
        worst_idx = np.argsort(errors)[-n_worst:][::-1]
        worst_indices[name] = worst_idx
        worst_errors[name] = errors[worst_idx]
        
        # Plot all points in blue
        ax.scatter(true, pred, alpha=0.5, s=10, c='tab:blue', label='Predictions')
        
        # Highlight worst in red
        ax.scatter(true[worst_idx], pred[worst_idx], alpha=0.8, s=30, c='red', 
                   marker='o', edgecolors='darkred', linewidths=0.5, label=f'Worst {n_worst}')
        
        # Perfect prediction line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'g--', lw=2, label='Perfect')
        
        if use_log:
            true_log = np.log10(np.maximum(true, 1e-10))
            pred_log = np.log10(np.maximum(pred, 1e-10))
            ss_res = np.sum((true_log - pred_log) ** 2)
            ss_tot = np.sum((true_log - true_log.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            metric_str = f'Log R² = {r2:.4f}'
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ss_res = np.sum((true - pred) ** 2)
            ss_tot = np.sum((true - true.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            metric_str = f'R² = {r2:.4f}'
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}\n{metric_str}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
    
    # Find overall worst sample
    overall_worst_idx = np.argmax(all_relative_errors)
    overall_worst_error = all_relative_errors[overall_worst_idx]
    
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Test Predictions with Worst {n_worst} Highlighted (Sample {overall_worst_idx} has largest overall error)', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure_with_svg(None, save_path)
    plt.close()
    print(f"Prediction plots with worst highlighted saved to {save_path}")
    
    # Save worst indices to CSV if path provided
    if worst_indices_path is not None:
        import pandas as pd
        # Create DataFrame with worst indices for each parameter
        max_len = max(len(v) for v in worst_indices.values())
        data = {}
        for name in param_names:
            idx_arr = worst_indices[name]
            err_arr = worst_errors[name]
            data[f'{name}_idx'] = list(idx_arr) + [np.nan] * (max_len - len(idx_arr))
            data[f'{name}_error'] = list(err_arr) + [np.nan] * (max_len - len(err_arr))
        
        # Add overall worst
        data['overall_worst_idx'] = [overall_worst_idx] + [np.nan] * (max_len - 1)
        data['overall_worst_error'] = [overall_worst_error] + [np.nan] * (max_len - 1)
        
        df = pd.DataFrame(data)
        df.to_csv(worst_indices_path, index=False)
        print(f"Worst prediction indices saved to {worst_indices_path}")
    
    return {
        'worst_indices': worst_indices,
        'worst_errors': worst_errors,
        'overall_worst_idx': overall_worst_idx,
        'overall_worst_error': overall_worst_error
    }


def plot_dual_predictions_with_worst(
    train_preds: np.ndarray,
    train_true: np.ndarray,
    val_preds: np.ndarray,
    val_true: np.ndarray,
    param_names: List[str],
    save_path: Path,
    worst_indices_path: Path = None,
    n_worst: int = 10,
    log_param_names: List[str] = None
) -> dict:
    """
    Plot Train vs Validation predictions with worst predictions highlighted in red.
    
    Returns:
        Dictionary containing:
        - 'train_worst_indices': dict mapping param_name -> array of worst train sample indices
        - 'val_worst_indices': dict mapping param_name -> array of worst val sample indices
        - 'overall_worst_train_idx': Index of train sample with largest overall error
        - 'overall_worst_val_idx': Index of val sample with largest overall error
    """
    if log_param_names is None:
        log_param_names = []

    n_params = len(param_names)
    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    
    train_worst_indices = {}
    train_worst_errors = {}
    val_worst_indices = {}
    val_worst_errors = {}
    
    # Compute overall errors
    train_overall_errors = np.zeros(train_preds.shape[0])
    val_overall_errors = np.zeros(val_preds.shape[0])
    
    for i, (ax, name) in enumerate(zip(axes[:n_params], param_names)):
        t_pred = train_preds[:, i]
        t_true = train_true[:, i]
        v_pred = val_preds[:, i]
        v_true = val_true[:, i]
        
        use_log = name in log_param_names
        
        # Compute errors
        if use_log:
            t_errors = np.abs(np.log10(np.maximum(np.abs(t_pred), 1e-10)) - 
                             np.log10(np.maximum(np.abs(t_true), 1e-10)))
            v_errors = np.abs(np.log10(np.maximum(np.abs(v_pred), 1e-10)) - 
                             np.log10(np.maximum(np.abs(v_true), 1e-10)))
        else:
            all_true = np.concatenate([t_true, v_true])
            param_range = all_true.max() - all_true.min()
            if param_range > 0:
                t_errors = np.abs(t_pred - t_true) / param_range
                v_errors = np.abs(v_pred - v_true) / param_range
            else:
                t_errors = np.abs(t_pred - t_true)
                v_errors = np.abs(v_pred - v_true)
        
        # Add to overall errors
        train_overall_errors += t_errors / (t_errors.max() + 1e-10) if t_errors.max() > 0 else t_errors
        val_overall_errors += v_errors / (v_errors.max() + 1e-10) if v_errors.max() > 0 else v_errors
        
        # Find worst predictions
        t_worst_idx = np.argsort(t_errors)[-n_worst:][::-1]
        v_worst_idx = np.argsort(v_errors)[-n_worst:][::-1]
        
        train_worst_indices[name] = t_worst_idx
        train_worst_errors[name] = t_errors[t_worst_idx]
        val_worst_indices[name] = v_worst_idx
        val_worst_errors[name] = v_errors[v_worst_idx]
        
        # Plot all train points in blue
        ax.scatter(t_true, t_pred, alpha=0.3, s=10, label='Train', c='tab:blue')
        # Plot all val points in orange
        ax.scatter(v_true, v_pred, alpha=0.5, s=10, label='Val', c='tab:orange')
        
        # Highlight worst train in red
        ax.scatter(t_true[t_worst_idx], t_pred[t_worst_idx], alpha=0.8, s=25, c='red', 
                   marker='o', edgecolors='darkred', linewidths=0.5, label=f'Worst Train')
        # Highlight worst val in dark red with different marker
        ax.scatter(v_true[v_worst_idx], v_pred[v_worst_idx], alpha=0.8, s=25, c='darkred', 
                   marker='s', edgecolors='black', linewidths=0.5, label=f'Worst Val')
        
        all_true = np.concatenate([t_true, v_true])
        all_pred = np.concatenate([t_pred, v_pred])
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'g--', lw=2, label='Perfect')
        
        if use_log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}')
        ax.legend(fontsize=6, loc='upper left')
        ax.grid(True, alpha=0.3, which='both')
    
    # Find overall worst samples
    overall_worst_train_idx = np.argmax(train_overall_errors)
    overall_worst_val_idx = np.argmax(val_overall_errors)
    
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Train/Val Predictions with Worst {n_worst} Highlighted\n'
                 f'(Worst Train: #{overall_worst_train_idx}, Worst Val: #{overall_worst_val_idx})', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_figure_with_svg(None, save_path)
    plt.close()
    print(f"Dual prediction plots with worst highlighted saved to {save_path}")
    
    # Save worst indices to CSV if path provided
    if worst_indices_path is not None:
        import pandas as pd
        max_len = max(
            max(len(v) for v in train_worst_indices.values()),
            max(len(v) for v in val_worst_indices.values())
        )
        data = {}
        for name in param_names:
            t_idx = train_worst_indices[name]
            t_err = train_worst_errors[name]
            v_idx = val_worst_indices[name]
            v_err = val_worst_errors[name]
            
            data[f'{name}_train_idx'] = list(t_idx) + [np.nan] * (max_len - len(t_idx))
            data[f'{name}_train_error'] = list(t_err) + [np.nan] * (max_len - len(t_err))
            data[f'{name}_val_idx'] = list(v_idx) + [np.nan] * (max_len - len(v_idx))
            data[f'{name}_val_error'] = list(v_err) + [np.nan] * (max_len - len(v_err))
        
        data['overall_worst_train_idx'] = [overall_worst_train_idx] + [np.nan] * (max_len - 1)
        data['overall_worst_val_idx'] = [overall_worst_val_idx] + [np.nan] * (max_len - 1)
        
        df = pd.DataFrame(data)
        df.to_csv(worst_indices_path, index=False)
        print(f"Worst prediction indices saved to {worst_indices_path}")
    
    return {
        'train_worst_indices': train_worst_indices,
        'train_worst_errors': train_worst_errors,
        'val_worst_indices': val_worst_indices,
        'val_worst_errors': val_worst_errors,
        'overall_worst_train_idx': overall_worst_train_idx,
        'overall_worst_val_idx': overall_worst_val_idx
    }

def plot_dual_predictions(
    train_preds: np.ndarray,
    train_true: np.ndarray,
    val_preds: np.ndarray,
    val_true: np.ndarray,
    param_names: List[str],
    save_path: Path,
    log_param_names: List[str] = None
):
    """Plot Train vs Validation predictions for all parameters."""
    if log_param_names is None:
        log_param_names = []

    n_params = len(param_names)
    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes[:n_params], param_names)):
        t_pred = train_preds[:, i]
        t_true = train_true[:, i]
        v_pred = val_preds[:, i]
        v_true = val_true[:, i]
        
        use_log = name in log_param_names
        
        ax.scatter(t_true, t_pred, alpha=0.3, s=10, label='Train', c='tab:blue')
        ax.scatter(v_true, v_pred, alpha=0.5, s=10, label='Val', c='tab:orange')
        
        all_true = np.concatenate([t_true, v_true])
        all_pred = np.concatenate([t_pred, v_pred])
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        if use_log:
            t_true_log = np.log10(np.maximum(t_true, 1e-10))
            t_pred_log = np.log10(np.maximum(t_pred, 1e-10))
            v_true_log = np.log10(np.maximum(v_true, 1e-10))
            v_pred_log = np.log10(np.maximum(v_pred, 1e-10))
            
            ss_res_t = np.sum((t_true_log - t_pred_log) ** 2)
            ss_tot_t = np.sum((t_true_log - t_true_log.mean()) ** 2)
            r2_t = 1 - (ss_res_t / ss_tot_t) if ss_tot_t > 0 else 0
            
            ss_res_v = np.sum((v_true_log - v_pred_log) ** 2)
            ss_tot_v = np.sum((v_true_log - v_true_log.mean()) ** 2)
            r2_v = 1 - (ss_res_v / ss_tot_v) if ss_tot_v > 0 else 0
            
            metric_str = f'Log R²: Train={r2_t:.3f}, Val={r2_v:.3f}'
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ss_res_t = np.sum((t_true - t_pred) ** 2)
            ss_tot_t = np.sum((t_true - t_true.mean()) ** 2)
            r2_t = 1 - (ss_res_t / ss_tot_t) if ss_tot_t > 0 else 0
            
            ss_res_v = np.sum((v_true - v_pred) ** 2)
            ss_tot_v = np.sum((v_true - v_true.mean()) ** 2)
            r2_v = 1 - (ss_res_v / ss_tot_v) if ss_tot_v > 0 else 0
            
            metric_str = f'R²: Train={r2_t:.3f}, Val={r2_v:.3f}'
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}\n{metric_str}')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
    
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_figure_with_svg(None, save_path)
    plt.close()
    print(f"Dual prediction plots saved to {save_path}")


def save_unified_worst_indices(
    train_preds: np.ndarray, train_true: np.ndarray, train_files: List[Path],
    val_preds: np.ndarray, val_true: np.ndarray, val_files: List[Path],
    test_preds: np.ndarray, test_true: np.ndarray, test_files: List[Path],
    save_path: Path,
    n_worst: int = 10,
    param_names: List[str] = None,
    # New parameters to match training metric exactly
    use_log_targets: bool = False,
    param_use_log: np.ndarray = None,
    param_mean: np.ndarray = None,
    param_std: np.ndarray = None,
    use_weighted_loss: bool = False,
    loss_weights: List[float] = None
) -> dict:
    """
    Save unified CSV with best, worst, and random indices per split, and a comprehensive sample mapping.
    
    The error metric used for ranking is EXACTLY the same as the training loss:
    - Predictions and ground truth are transformed back to the normalized, log-transformed space
      used during training.
    - MSE (or Weighted MSE if use_weighted_loss=True) is computed per sample.
    
    Args:
        *_preds: Predictions for each split (in original, denormalized scale)
        *_true: Ground truth for each split (in original, denormalized scale)
        *_files: File paths for each split
        save_path: Path to save the unified worst/best/random CSV
        n_worst: Number of samples to select for each category (best/worst/random)
        param_names: Names of parameters for column labels
        use_log_targets: Whether log transform was applied to targets during training
        param_use_log: Boolean array indicating which parameters use log transform
        param_mean: Mean used for z-score normalization during training
        param_std: Std used for z-score normalization during training
        use_weighted_loss: Whether weighted MSE loss was used during training
        loss_weights: Weights for each parameter (only used if use_weighted_loss=True)
        
    Returns:
        Dictionary with indices info for each split
    """
    if param_names is None:
        param_names = []
    if param_use_log is None:
        param_use_log = np.zeros(len(param_names), dtype=bool)
    
    # Default weights (all 1.0) if not using weighted loss
    n_params = len(param_names) if param_names else 10
    if use_weighted_loss and loss_weights is not None:
        weights = np.array(loss_weights)
    elif use_weighted_loss:
        # Default weights from WeightedMSELoss
        weights = np.array([1.0, 3.0, 0.5, 3.0, 3.0, 3.0, 0.5, 1.0, 0.5, 5.0])[:n_params]
    else:
        weights = np.ones(n_params)
    
    def compute_training_loss(preds, true_vals):
        """
        Compute per-sample loss using the EXACT same metric as training.
        
        Steps:
        1. Apply log transform to relevant parameters (reverse of exp during evaluation)
        2. Apply z-score normalization (reverse of denormalization during evaluation)
        3. Compute MSE (or Weighted MSE) per sample
        """
        n_samples = preds.shape[0]
        
        # Copy to avoid modifying original
        preds_transformed = preds.copy()
        true_transformed = true_vals.copy()
        
        # Step 1: Apply log transform to relevant parameters
        if use_log_targets and param_use_log is not None:
            preds_transformed[:, param_use_log] = np.log(np.maximum(preds_transformed[:, param_use_log], 1e-10))
            true_transformed[:, param_use_log] = np.log(np.maximum(true_transformed[:, param_use_log], 1e-10))
        
        # Step 2: Apply z-score normalization
        if param_mean is not None and param_std is not None:
            preds_transformed = (preds_transformed - param_mean) / param_std
            true_transformed = (true_transformed - param_mean) / param_std
        
        # Step 3: Compute (Weighted) MSE per sample
        squared_errors = (preds_transformed - true_transformed) ** 2  # (n_samples, n_params)
        weighted_squared_errors = squared_errors * weights  # Apply weights
        per_sample_loss = np.mean(weighted_squared_errors, axis=1)  # (n_samples,)
        
        return per_sample_loss
    
    results = {'train': [], 'val': [], 'test': []}
    all_sample_mapping = []

    # RNG for random samples
    rng = np.random.RandomState(42)
    
    for split_name, preds, true_vals, files in [
        ('train', train_preds, train_true, train_files),
        ('val', val_preds, val_true, val_files),
        ('test', test_preds, test_true, test_files)
    ]:
        if preds is None or len(preds) == 0:
            continue
            
        per_sample_loss = compute_training_loss(preds, true_vals)
        num_samples = len(per_sample_loss)
        
        # 1. Worst Samples (Highest Loss)
        worst_indices = np.argsort(per_sample_loss)[-n_worst:][::-1]
        
        # 2. Best Samples (Lowest Loss)
        best_indices = np.argsort(per_sample_loss)[:n_worst]
        
        # 3. Random Samples
        n_random = min(n_worst, num_samples)
        random_indices = rng.choice(num_samples, size=n_random, replace=False)
        
        # Helper to process a set of indices
        def process_indices(indices, category_type):
            for rank, idx in enumerate(indices, 1):
                file_path = str(files[idx]) if idx < len(files) else "unknown"
                loss = per_sample_loss[idx]
                results[split_name].append({
                    'type': category_type,
                    'rank': rank,
                    'index': int(idx),
                    'file_path': file_path,
                    'loss': float(loss)
                })

        process_indices(worst_indices, 'worst')
        process_indices(best_indices, 'best')
        process_indices(random_indices, 'random')

        # Collect data for sample mapping
        for idx in range(num_samples):
            file_path = str(files[idx]) if idx < len(files) else "unknown"
            all_sample_mapping.append({
                'split': split_name,
                'index': int(idx),
                'file_path': file_path,
                'loss': float(per_sample_loss[idx])
            })
    
    # Write Unified Indices CSV
    # Columns: split, type, rank, index, file_path, loss
    with open(save_path, 'w') as f:
        f.write("split,type,rank,index,file_path,loss\n")
        for split_name in ['train', 'val', 'test']:
            for row in results[split_name]:
                f.write(f"{split_name},{row['type']},{row['rank']},{row['index']},{row['file_path']},{row['loss']:.6f}\n")
    
    print(f"Unified worst/best/random indices saved to {save_path}")

    # Write Sample Mapping CSV
    mapping_path = save_path.parent / "sample_mapping.csv"
    with open(mapping_path, 'w') as f:
        f.write("split,index,file_path,loss\n")
        for row in all_sample_mapping:
            f.write(f"{row['split']},{row['index']},{row['file_path']},{row['loss']:.6f}\n")

    print(f"Full sample mapping saved to {mapping_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train LSTM encoder for material parameter prediction')
    parser.add_argument('--data-dir', type=Path, default=Path('raw_npz'),
                        help='Directory containing .npz files')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--use-time', action='store_true',
                        help='Include time as an input feature')
    parser.add_argument('--output-dir', type=Path, default=Path('output'),
                        help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision training')
    parser.add_argument('--target-params', type=str, nargs='+', default=None,
                        help='Specific parameters to predict (default: all 10)')
    parser.add_argument('--max-seq-length', type=int, default=None,
                        help='Maximum sequence length for uniform downsampling')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of .npz files to use')
    parser.add_argument('--use-cell-state', action='store_true',
                        help='Use both hidden state and cell state from LSTM')
    parser.add_argument('--mlp-layers', type=int, nargs='+', default=[256, 128, 64],
                        help='Hidden layer dimensions for the MLP')
    parser.add_argument('--mlp-activation', type=str, default='relu', 
                        choices=['relu', 'sigmoid', 'none'],
                        help='Activation function for MLP layers')
    parser.add_argument('--weight-init', type=str, default='auto', 
                        choices=['default', 'xavier', 'kaiming', 'auto'],
                        help='Weight initialization strategy. "auto" selects based on activation (Kaiming for ReLU, Xavier for others)')
    parser.add_argument('--use-log-targets', action='store_true',
                        help='Apply log transformation to target parameters with large dynamic ranges')
    parser.add_argument('--normalize-inputs', action='store_true',
                        help='Apply z-score normalization to input sequences')
    parser.add_argument('--use-log-inputs', action='store_true',
                        help='Apply log10 transformation to S11 before normalization')
    parser.add_argument('--normalize-s11-max', action='store_true',
                        help='Normalize S11 by its per-curve maximum value')
    parser.add_argument('--early-stop-patience', type=int, default=20,
                        help='Early stopping patience (epochs without improvement). Set to 0 to disable.')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--resume', type=Path, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--use-scenario', action='store_true',
                        help='Include one-hot encoded scenario type as input features')
    parser.add_argument('--use-weighted-loss', action='store_true',
                        help='Use weighted MSE loss to emphasize hard-to-predict parameters (kF, alphaM, m5, q, m1)')
    parser.add_argument('--loss-weights', type=float, nargs='+', default=None,
                        help='Custom weights for each parameter (order: kM, kF, m3, m5, q, m1, m2, m4, theta, alphaM). '
                             'Only used with --use-weighted-loss. If not provided, uses default weights based on parameter difficulty.')
    parser.add_argument('--use-physics-features', action='store_true',
                        help='Include physics-informed features (derivatives, ratios) to improve parameter identification')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='Learning rate scheduler type: plateau (ReduceLROnPlateau) or cosine (CosineAnnealingWarmRestarts)')
    parser.add_argument('--scheduler-t0', type=int, default=10,
                        help='T_0 for CosineAnnealingWarmRestarts: number of epochs before first restart (default: 10)')
    parser.add_argument('--scheduler-tmult', type=int, default=2,
                        help='T_mult for CosineAnnealingWarmRestarts: factor to increase T_i after restart (default: 2)')

    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    use_amp = args.use_amp and torch.cuda.is_available()
    if use_amp:
        print("Mixed precision training: ENABLED")
    elif args.use_amp:
        print("Warning: --use-amp requested but CUDA not available, using FP32")

    # Load data files
    npz_files = sorted(args.data_dir.glob('*_case_*.npz'))
    print(f"Found {len(npz_files)} data files")

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir}")

    if args.max_files is not None and args.max_files < len(npz_files):
        rng = np.random.RandomState(args.seed)
        selected_indices = rng.choice(len(npz_files), size=args.max_files, replace=False)
        npz_files = [npz_files[i] for i in sorted(selected_indices)]
        print(f"Randomly selected {len(npz_files)} files")

    # Split data
    train_files, test_files = train_test_split(npz_files, test_size=0.15, random_state=args.seed)
    train_files, val_files = train_test_split(train_files, test_size=0.15, random_state=args.seed)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Create training dataset (computes normalization statistics)
    use_rate = False
    normalize = True
    print(f"\nPredicting {len(MaterialDataset.ALL_PARAM_NAMES)} parameters")
    if args.use_log_targets:
        print(f"Log transformation enabled for: {', '.join(MaterialDataset.LOG_PARAM_NAMES)}")
    
    train_dataset = MaterialDataset(
        train_files, normalize=normalize, use_time=args.use_time, use_rate=use_rate,
        target_params=args.target_params, max_seq_length=args.max_seq_length,
        use_log_targets=args.use_log_targets, normalize_inputs=args.normalize_inputs,
        use_log_inputs=args.use_log_inputs, normalize_s11_max=args.normalize_s11_max,
        use_scenario=args.use_scenario, use_physics_features=args.use_physics_features
    )

    # Prepare precomputed normalization for val/test (avoids redundant computation)
    param_norm = {'mean': train_dataset.param_mean, 'std': train_dataset.param_std}
    
    input_norm = None
    if args.normalize_inputs:
        input_norm = {
            'mean': train_dataset.input_mean,
            'std': train_dataset.input_std,
            's11_epsilon': train_dataset.s11_epsilon
        }

    # Create val/test datasets with precomputed normalization
    val_dataset = MaterialDataset(
        val_files, normalize=normalize, use_time=args.use_time, use_rate=use_rate,
        target_params=args.target_params, max_seq_length=args.max_seq_length,
        use_log_targets=args.use_log_targets, normalize_inputs=args.normalize_inputs,
        use_log_inputs=args.use_log_inputs, normalize_s11_max=args.normalize_s11_max,
        use_scenario=args.use_scenario, use_physics_features=args.use_physics_features,
        precomputed_param_norm=param_norm, precomputed_input_norm=input_norm
    )

    test_dataset = MaterialDataset(
        test_files, normalize=normalize, use_time=args.use_time, use_rate=use_rate,
        target_params=args.target_params, max_seq_length=args.max_seq_length,
        use_log_targets=args.use_log_targets, normalize_inputs=args.normalize_inputs,
        use_log_inputs=args.use_log_inputs, normalize_s11_max=args.normalize_s11_max,
        use_scenario=args.use_scenario, use_physics_features=args.use_physics_features,
        precomputed_param_norm=param_norm, precomputed_input_norm=input_norm
    )

    # Create dataloaders
    num_workers = 4 if torch.cuda.is_available() else 0
    
    # Use aggregate mode: separate LSTM per curve
    active_collate_fn = collate_fn
    print("Multi-curve mode: AGGREGATE (method=mean)")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=active_collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=active_collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             collate_fn=active_collate_fn, num_workers=num_workers, pin_memory=True)
    
    # Create model
    # Base features: S11, F11, F22, F33 (4)
    # Optional: time (+1), physics (+11), scenario one-hot (+4)
    base_input_dim = 4 + int(args.use_time) + (11 if args.use_physics_features else 0) + (4 if args.use_scenario else 0)
    input_dim = base_input_dim
    
    output_dim = len(train_dataset.PARAM_NAMES)
    model = LSTMEncoder(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_dim=output_dim,
        dropout=args.dropout,
        bidirectional=True,
        use_cell_state=args.use_cell_state,
        mlp_layers=args.mlp_layers,
        activation=args.mlp_activation,
        aggregation='mean'
    ).to(device)


    print("Multi-curve aggregation method: mean")
    if args.use_scenario:
        print("Scenario one-hot encoding: ENABLED (adds 4 features: is_scenario_0/1/2/3)")
    if args.use_physics_features:
        print("Physics-informed features: ENABLED (adds 11 features: derivatives, ratios, volumetric invariants)")
    
    if args.weight_init == 'xavier':
        print("Applying Xavier (Glorot) initialization")
        model.apply(init_weights_xavier)
    elif args.weight_init == 'kaiming':
        print("Applying Kaiming (He) initialization (best for ReLU)")
        model.apply(init_weights_kaiming)
    elif args.weight_init == 'auto':
        init_fn = get_init_function_for_activation(args.mlp_activation)
        init_name = 'Kaiming (He)' if args.mlp_activation == 'relu' else 'Xavier (Glorot)'
        print(f"Applying {init_name} initialization (auto-selected for {args.mlp_activation} activation)")
        model.apply(init_fn)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function (weighted or standard MSE)
    if args.use_weighted_loss:
        if args.loss_weights is not None:
            # Use custom weights provided by user
            if len(args.loss_weights) != len(train_dataset.PARAM_NAMES):
                raise ValueError(
                    f"Number of loss weights ({len(args.loss_weights)}) must match "
                    f"number of parameters ({len(train_dataset.PARAM_NAMES)})"
                )
            criterion = WeightedMSELoss(weights=args.loss_weights)
            print(f"\nUsing WEIGHTED MSE Loss with custom weights:")
        else:
            # Use default weights based on parameter names
            criterion = WeightedMSELoss(param_names=train_dataset.PARAM_NAMES)
            print(f"\nUsing WEIGHTED MSE Loss with default difficulty-based weights:")
        
        # Print the weights being used
        weights_dict = criterion.get_weights_dict(train_dataset.PARAM_NAMES)
        for name, weight in weights_dict.items():
            difficulty = "easy" if weight <= 0.5 else "medium" if weight <= 1.0 else "hard" if weight <= 3.0 else "very hard"
            print(f"  {name:8s}: weight = {weight:.1f} ({difficulty})")
    else:
        criterion = nn.MSELoss()
        print("\nUsing standard MSE Loss")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=args.scheduler_t0,
            T_mult=args.scheduler_tmult,
            eta_min=1e-7
        )
        print(f"\nLearning rate scheduler: CosineAnnealingWarmRestarts (T_0={args.scheduler_t0}, T_mult={args.scheduler_tmult})")
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        print("\nLearning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")
        
    scaler = GradScaler('cuda') if use_amp else None

    # Training state
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if args.resume is not None:
        if args.resume.exists():
            print(f"\nResuming from checkpoint: {args.resume}")
            resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(resume_checkpoint['model_state_dict'])
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            start_epoch = resume_checkpoint.get('epoch', 0) + 1
            best_val_loss = resume_checkpoint.get('val_loss', float('inf'))
            train_losses = resume_checkpoint.get('train_losses', [])
            val_losses = resume_checkpoint.get('val_losses', [])
            print(f"  Resumed from epoch {start_epoch}, best val_loss: {best_val_loss:.6f}")
        else:
            print(f"Warning: Resume checkpoint not found: {args.resume}, starting from scratch")

    print("\nStarting training...")
    training_start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                                  scaler, use_amp, args.max_grad_norm)
        val_loss = validate(model, val_loader, criterion, device, use_amp)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Step the scheduler (different behavior for different schedulers)
        if args.scheduler == 'cosine':
            scheduler.step()  # CosineAnnealingWarmRestarts steps per epoch
        else:
            scheduler.step(val_loss)  # ReduceLROnPlateau needs the metric
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_loss': val_loss,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'bidirectional': True,
                'normalization': {
                    'mean': train_dataset.param_mean,
                    'std': train_dataset.param_std
                },
                'input_normalization': {
                    'mean': train_dataset.input_mean.tolist() if args.normalize_inputs else None,
                    'std': train_dataset.input_std.tolist() if args.normalize_inputs else None
                },
                'use_time': args.use_time,
                'use_rate': use_rate,
                'target_params': train_dataset.PARAM_NAMES,
                'param_indices': train_dataset.PARAM_INDICES,
                'max_seq_length': args.max_seq_length,
                'use_cell_state': args.use_cell_state,
                'mlp_layers': args.mlp_layers,
                'mlp_activation': args.mlp_activation,
                'weight_init': args.weight_init,
                'use_log_targets': args.use_log_targets,
                'param_use_log': train_dataset.param_use_log.tolist(),
                'normalize_inputs': args.normalize_inputs,
                'use_log_inputs': args.use_log_inputs,
                's11_epsilon': train_dataset.s11_epsilon if args.use_log_inputs else None,
                'normalize_s11_max': args.normalize_s11_max,
                'aggregation': 'mean',
                'use_scenario': args.use_scenario,
                'use_physics_features': args.use_physics_features,
                'use_weighted_loss': args.use_weighted_loss,
                'loss_weights': args.loss_weights if args.use_weighted_loss else None
            }, args.output_dir / 'best_model.pth')

            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        else:
            early_stop_counter += 1
            if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.early_stop_patience} epochs)")
                break
        
    
    # Calculate and print training duration
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    seconds = training_duration % 60
    if hours > 0:
        print(f"Training completed in {hours}h {minutes}m {seconds:.1f}s")
    elif minutes > 0:
        print(f"Training completed in {minutes}m {seconds:.1f}s")
    else:
        print(f"Training completed in {seconds:.1f}s")
    
    plot_training_curves(train_losses, val_losses, args.output_dir / 'training_curves.png')
    
    # Load best model for evaluation
    checkpoint = torch.load(args.output_dir / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating on test set...")
    test_start_time = time.time()
    predictions, ground_truth = evaluate_model(model, test_loader, test_dataset, device)
    test_end_time = time.time()
    
    # Calculate average prediction time per sample
    test_duration = test_end_time - test_start_time
    n_test_samples = len(predictions)
    avg_time_per_sample_ms = (test_duration / n_test_samples) * 1000
    
    mse = np.mean((predictions - ground_truth) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - ground_truth), axis=0)
    
    print("\nTest Set Results:")
    for i, name in enumerate(test_dataset.PARAM_NAMES):
        print(f"  {name:8s}: RMSE = {rmse[i]:10.6f}, MAE = {mae[i]:10.6f}")
    print(f"\nAverage prediction time: {avg_time_per_sample_ms:.3f} ms/sample ({n_test_samples} samples)")


    plot_predictions(
        predictions, ground_truth, test_dataset.PARAM_NAMES,
        args.output_dir / 'predictions.png',
        log_param_names=MaterialDataset.LOG_PARAM_NAMES
    )

    plot_predictions(
        predictions, ground_truth, test_dataset.PARAM_NAMES,
        args.output_dir / 'predictions_linear.png',
        log_param_names=[]
    )

    # Plot with worst predictions highlighted (for test data)
    print("\nGenerating test predictions with worst highlighted...")
    worst_test_info = plot_predictions_with_worst(
        predictions, ground_truth, test_dataset.PARAM_NAMES,
        args.output_dir / 'predictions_worst_highlighted.png',
        worst_indices_path=args.output_dir / 'test_worst_indices.csv',
        n_worst=10,
        log_param_names=MaterialDataset.LOG_PARAM_NAMES
    )
    print(f"  Overall worst test sample: index {worst_test_info['overall_worst_idx']} "
          f"(total error: {worst_test_info['overall_worst_error']:.4f})")

    # Save predictions
    header = ','.join([f'true_{name}' for name in test_dataset.PARAM_NAMES] +
                      [f'pred_{name}' for name in test_dataset.PARAM_NAMES])
    results = np.hstack([ground_truth, predictions])
    np.savetxt(args.output_dir / 'test_predictions.csv', results, delimiter=',', header=header, comments='')
    print(f"\nTest predictions saved to {args.output_dir / 'test_predictions.csv'}")

    # Train vs Validation Analysis
    print("\nGenerating Train vs Validation plots...")
    
    # Evaluate on Train set (use a subset if too large)
    train_subset_indices = np.random.choice(len(train_dataset), min(2000, len(train_dataset)), replace=False)
    train_subset = torch.utils.data.Subset(train_dataset, train_subset_indices)
    train_subset_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, 
                                      collate_fn=active_collate_fn, num_workers=num_workers, pin_memory=True)
    
    print("  Evaluating on Training subset...")
    train_preds, train_true = evaluate_model(model, train_subset_loader, train_dataset, device)
    
    print("  Evaluating on Validation set...")
    val_preds, val_true = evaluate_model(model, val_loader, val_dataset, device)
    
    # Log-space comparison
    plot_dual_predictions(
        train_preds, train_true,
        val_preds, val_true,
        train_dataset.PARAM_NAMES,
        args.output_dir / 'train_val_scatter.png',
        log_param_names=MaterialDataset.LOG_PARAM_NAMES
    )
    
    # Linear-space comparison
    plot_dual_predictions(
        train_preds, train_true,
        val_preds, val_true,
        train_dataset.PARAM_NAMES,
        args.output_dir / 'train_val_scatter_linear.png',
        log_param_names=[]
    )
    
    # Train/Val with worst predictions highlighted
    print("\nGenerating Train/Val plots with worst highlighted...")
    worst_trainval_info = plot_dual_predictions_with_worst(
        train_preds, train_true,
        val_preds, val_true,
        train_dataset.PARAM_NAMES,
        args.output_dir / 'train_val_worst_highlighted.png',
        worst_indices_path=args.output_dir / 'train_val_worst_indices.csv',
        n_worst=10,
        log_param_names=MaterialDataset.LOG_PARAM_NAMES
    )
    print(f"  Worst train sample: index {worst_trainval_info['overall_worst_train_idx']}")
    print(f"  Worst val sample: index {worst_trainval_info['overall_worst_val_idx']}")
    
    # Save unified worst indices CSV with file paths
    print("\nSaving unified worst indices CSV...")
    # Map train subset indices back to original file indices
    train_files_subset = [train_files[i] for i in train_subset_indices]
    
    save_unified_worst_indices(
        train_preds=train_preds, train_true=train_true, train_files=train_files_subset,
        val_preds=val_preds, val_true=val_true, val_files=val_files,
        test_preds=predictions, test_true=ground_truth, test_files=test_files,
        save_path=args.output_dir / 'worst_indices_unified.csv',
        n_worst=10,
        param_names=train_dataset.PARAM_NAMES,
        use_log_targets=args.use_log_targets,
        param_use_log=train_dataset.param_use_log,
        param_mean=train_dataset.param_mean,
        param_std=train_dataset.param_std,
        use_weighted_loss=args.use_weighted_loss,
        loss_weights=args.loss_weights
    )
    
if __name__ == '__main__':
    main()
