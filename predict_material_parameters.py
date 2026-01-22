#!/usr/bin/env python3
"""
Inference script to predict material parameters from stress-strain curves using a trained model.

Supports all scenarios:
- Scenarios 0, 1, 2: Single curve prediction
- Scenario 3: Multi-curve prediction (multiple curves from the same material)

Usage:
    python predict_material_parameters.py --model path/to/best_model.pth --input path/to/curve.npz
    python predict_material_parameters.py --model path/to/best_model.pth --input path/to/curve.npz --output out.npz
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from train_material_predictor import LSTMEncoder, MaterialDataset, extract_curve_features


def flatten_matlab_array(arr):
    """
    Flatten array from MATLAB exports.
    Handles: (N, 1) column vectors, (1,1) object arrays containing data.
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


def get_field(obj, key, default=None):
    """
    Helper function to get a field from either a dict or numpy structured array.
    Automatically handles nested MATLAB (1,1) object arrays.
    
    Args:
        obj: Dictionary, numpy structured array, or object with attributes
        key: Field name to retrieve
        default: Default value if field not found
        
    Returns:
        Field value or default (flattened if array)
    """
    value = None
    if hasattr(obj, 'dtype') and hasattr(obj.dtype, 'names') and obj.dtype.names:
        # Numpy structured array
        value = obj[key] if key in obj.dtype.names else default
    elif isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    
    # Flatten MATLAB-style arrays
    if value is not None and hasattr(value, 'ndim'):
        return flatten_matlab_array(value)
    return value


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine input dimension
    use_time = checkpoint.get('use_time', False)
    use_rate = checkpoint.get('use_rate', False)
    use_scenario = checkpoint.get('use_scenario', False)
    use_physics_features = checkpoint.get('use_physics_features', False)
    
    # Base features (4) + optional time (+1) + optional rate (+1) + optional physics (+11) + optional scenario one-hot (+4)
    input_dim = 4 + int(use_time) + int(use_rate) + (11 if use_physics_features else 0) + (4 if use_scenario else 0)

    # Get target parameters
    target_params = checkpoint.get('target_params', MaterialDataset.ALL_PARAM_NAMES)
    param_indices = checkpoint.get('param_indices', MaterialDataset.ALL_PARAM_INDICES)
    output_dim = len(target_params)

    # Get model architecture hyperparameters
    hidden_size = checkpoint.get('hidden_size', 128)
    num_layers = checkpoint.get('num_layers', 3)
    dropout = checkpoint.get('dropout', 0.2)
    bidirectional = checkpoint.get('bidirectional', True)
    use_cell_state = checkpoint.get('use_cell_state', False)
    mlp_layers = checkpoint.get('mlp_layers', [256, 128, 64])
    mlp_activation = checkpoint.get('mlp_activation', 'relu')
    max_seq_length = checkpoint.get('max_seq_length', None)
    aggregation = checkpoint.get('aggregation', 'mean')

    # Get log transformation configuration
    use_log_targets = checkpoint.get('use_log_targets', False)
    param_use_log = np.array(checkpoint.get('param_use_log', [False] * len(target_params)))

    # Get input normalization configuration
    normalize_inputs = checkpoint.get('normalize_inputs', False)
    input_norm_params = checkpoint.get('input_normalization', {})

    # Get input log transformation configuration
    use_log_inputs = checkpoint.get('use_log_inputs', False)
    s11_epsilon = checkpoint.get('s11_epsilon', None)

    # Get S11 max normalization configuration
    normalize_s11_max = checkpoint.get('normalize_s11_max', False)

    # Create model with same architecture as training
    model = LSTMEncoder(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout,
        bidirectional=bidirectional,
        use_cell_state=use_cell_state,
        mlp_layers=mlp_layers,
        activation=mlp_activation,
        aggregation=aggregation
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    norm_params = checkpoint.get('normalization', {})

    return (model, norm_params, use_time, use_rate, target_params, param_indices, max_seq_length,
            use_cell_state, mlp_layers, mlp_activation, use_log_targets, param_use_log,
            normalize_inputs, input_norm_params, use_log_inputs, s11_epsilon, normalize_s11_max,
            aggregation, use_scenario, use_physics_features)


def predict_parameters(
    model: torch.nn.Module,
    S11: np.ndarray,
    F11: np.ndarray,
    F22: np.ndarray,
    F33: np.ndarray,
    t: np.ndarray,
    norm_params: dict,
    use_time: bool,
    use_rate: bool,
    device: torch.device,
    max_seq_length: int = None,
    use_log_targets: bool = False,
    param_use_log: np.ndarray = None,
    normalize_inputs: bool = False,
    input_norm_params: dict = None,
    use_log_inputs: bool = False,
    s11_epsilon: float = None,
    normalize_s11_max: bool = False,
    use_scenario: bool = False,
    scenario: int = None,
    use_physics_features: bool = False
) -> np.ndarray:
    """
    Predict parameters for a single curve (Scenarios 0, 1, 2).

    Args:
        model: Trained LSTM model
        S11: (seq_len,) array of stress values
        F11: (seq_len,) array of stretch in loading direction
        F22: (seq_len,) array of lateral stretch
        F33: (seq_len,) array of lateral stretch
        t: (seq_len,) array of time values
        norm_params: Normalization parameters from training
        use_time: Whether to include time as a feature
        use_rate: Whether to include loading rate as a feature
        device: Torch device
        max_seq_length: Maximum sequence length for downsampling
        use_log_targets: Whether log transformation was used during training
        param_use_log: Boolean array indicating which parameters use log
        normalize_inputs: Whether input normalization was used during training
        input_norm_params: Input normalization parameters (mean, std)
        use_log_inputs: Whether log transformation was applied to S11 during training
        s11_epsilon: Epsilon value for S11 log transformation
        normalize_s11_max: Whether to normalize S11 by its per-curve max value
        use_scenario: Whether to include one-hot encoded scenario as features
        scenario: Scenario type (0, 1, 2, or 3)
        use_physics_features: Whether to include physics-informed features

    Returns:
        parameters: (n_params,) array of predicted parameters in original scale
    """
    # Get input normalization arrays
    input_mean = np.array(input_norm_params['mean']) if normalize_inputs and input_norm_params else None
    input_std = np.array(input_norm_params['std']) if normalize_inputs and input_norm_params else None
    
    # Use the helper function to extract and preprocess features
    sequence = extract_curve_features(
        S11, F11, F22, F33, t,
        use_time=use_time,
        use_rate=use_rate,
        max_seq_length=max_seq_length,
        use_log_inputs=use_log_inputs,
        s11_epsilon=s11_epsilon,
        normalize_s11_max=normalize_s11_max,
        normalize_inputs=normalize_inputs,
        input_mean=input_mean,
        input_std=input_std,
        use_scenario=use_scenario,
        scenario=scenario,
        use_physics_features=use_physics_features
    )

    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    length_tensor = torch.tensor([len(sequence)], dtype=torch.long)
    curve_counts = torch.ones(1, dtype=torch.long).to(device)  # Single curve

    # Predict
    with torch.no_grad():
        prediction = model(sequence_tensor, length_tensor, curve_counts)

    params = prediction.cpu().numpy().squeeze()

    # Denormalize if normalization was used
    if norm_params.get('mean') is not None and norm_params.get('std') is not None:
        mean = norm_params['mean']
        std = norm_params['std']
        params = params * std + mean

    # Apply inverse log transformation if needed
    if use_log_targets and param_use_log is not None:
        params[param_use_log] = np.exp(params[param_use_log])

    return params


def predict_parameters_multi_curve(
    model: torch.nn.Module,
    curves: list,
    norm_params: dict,
    use_time: bool,
    use_rate: bool,
    device: torch.device,
    max_seq_length: int = None,
    use_log_targets: bool = False,
    param_use_log: np.ndarray = None,
    normalize_inputs: bool = False,
    input_norm_params: dict = None,
    use_log_inputs: bool = False,
    s11_epsilon: float = None,
    normalize_s11_max: bool = False,
    use_scenario: bool = False,
    scenario: int = None,
    use_physics_features: bool = False
) -> np.ndarray:
    """
    Predict parameters from multiple curves (Scenario 3).

    Args:
        model: Trained LSTM model
        curves: List of curve dictionaries, each containing S11, F11, F22, F33, t arrays
        norm_params: Normalization parameters from training
        use_time: Whether to include time as a feature
        use_rate: Whether to include loading rate as a feature
        device: Torch device
        max_seq_length: Maximum sequence length for downsampling
        use_log_targets: Whether log transformation was used during training
        param_use_log: Boolean array indicating which parameters use log
        normalize_inputs: Whether input normalization was used during training
        input_norm_params: Input normalization parameters (mean, std)
        use_log_inputs: Whether log transformation was applied to S11 during training
        s11_epsilon: Epsilon value for S11 log transformation
        normalize_s11_max: Whether to normalize S11 by its per-curve max value
        use_scenario: Whether to include one-hot encoded scenario as features
        scenario: Scenario type (0, 1, 2, or 3)
        use_physics_features: Whether to include physics-informed features

    Returns:
        parameters: (n_params,) array of predicted parameters in original scale
    """
    # Get input normalization arrays
    input_mean = np.array(input_norm_params['mean']) if normalize_inputs and input_norm_params else None
    input_std = np.array(input_norm_params['std']) if normalize_inputs and input_norm_params else None
    
    # Process each curve
    curve_tensors = []
    for curve in curves:
        sequence = extract_curve_features(
            curve['S11'], curve['F11'], curve['F22'], curve['F33'], curve['t'],
            use_time=use_time,
            use_rate=use_rate,
            max_seq_length=max_seq_length,
            use_log_inputs=use_log_inputs,
            s11_epsilon=s11_epsilon,
            normalize_s11_max=normalize_s11_max,
            normalize_inputs=normalize_inputs,
            input_mean=input_mean,
            input_std=input_std,
            use_scenario=use_scenario,
            scenario=scenario,
            use_physics_features=use_physics_features
        )
        curve_tensors.append(torch.tensor(sequence, dtype=torch.float32))
    
    # Pad sequences
    lengths = torch.tensor([len(c) for c in curve_tensors], dtype=torch.long)
    padded_sequences = nn.utils.rnn.pad_sequence(curve_tensors, batch_first=True).to(device)
    curve_counts = torch.tensor([len(curves)], dtype=torch.long).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(padded_sequences, lengths, curve_counts)
    
    params = prediction.cpu().numpy().squeeze()
    
    # Denormalize if normalization was used
    if norm_params.get('mean') is not None and norm_params.get('std') is not None:
        mean = norm_params['mean']
        std = norm_params['std']
        params = params * std + mean
    
    # Apply inverse log transformation if needed
    if use_log_targets and param_use_log is not None:
        params[param_use_log] = np.exp(params[param_use_log])
    
    return params


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description='Predict material parameters from stress-strain curves')
    parser.add_argument('--model', type=Path, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--input', type=Path, required=True,
                        help='Path to input .npz file')
    parser.add_argument('--output', type=Path, default=None,
                        help='Path to save predictions (optional)')
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.model}...")
    (model, norm_params, use_time, use_rate, target_params, param_indices, max_seq_length,
     use_cell_state, mlp_layers, mlp_activation, use_log_targets, param_use_log,
     normalize_inputs, input_norm_params, use_log_inputs, s11_epsilon, normalize_s11_max,
     aggregation, use_scenario, use_physics_features) = load_model(args.model, device)
    
    print("Model loaded successfully!")
    print(f"Model predicts {len(target_params)} parameters: {', '.join(target_params)}")
    print(f"MLP architecture: {mlp_layers}, activation: {mlp_activation}")
    print(f"Multi-curve aggregation: {aggregation}")
    
    if max_seq_length is not None:
        print(f"Maximum sequence length: {max_seq_length}")
    if use_cell_state:
        print("Using both hidden and cell states for prediction")
    if use_log_targets:
        log_params = [name for name, use_log in zip(target_params, param_use_log) if use_log]
        print(f"Log transformation used for: {', '.join(log_params)}")
    if normalize_inputs:
        print("Input normalization: ENABLED")
    if use_log_inputs:
        print(f"Input log transformation: ENABLED (epsilon = {s11_epsilon:.6e})")
    if normalize_s11_max:
        print("S11 max normalization: ENABLED")
    if use_scenario:
        print("Scenario one-hot encoding: ENABLED")
    if use_physics_features:
        print("Physics-informed features: ENABLED (11 additional features)")

    # Load input data and detect scenario
    print(f"\nLoading data from {args.input}...")
    with np.load(args.input, allow_pickle=True) as data:
        scenario_val = data.get('scenario', 0)
        scenario = int(scenario_val.item()) if hasattr(scenario_val, 'item') else int(scenario_val)
        true_par = flatten_matlab_array(data.get('par', None))
        
        print(f"Detected Scenario: {scenario}")
        
        if scenario == 3:
            # Multi-curve scenario - flatten (1,N) to (N,)
            multi_curves = data['multi_curves'].flatten()
            n_curves = len(multi_curves)
            print(f"Multi-curve data: {n_curves} curves")
            
            curves = []
            for i, curve in enumerate(multi_curves):
                curve_dict = {
                    'S11': get_field(curve, 'S11'),
                    'F11': get_field(curve, 'F11'),
                    'F22': get_field(curve, 'F22'),
                    'F33': get_field(curve, 'F33'),
                    't': get_field(curve, 't', None)
                }
                curves.append(curve_dict)
                rate_val = get_field(curve, 'rate', 'N/A')
                s11_arr = get_field(curve, 'S11')
                print(f"  Curve {i+1}: {len(s11_arr)} timesteps, rate={rate_val}")
            
            # Predict using multi-curve function
            print("\nPredicting parameters from multiple curves...")
            pred_params = predict_parameters_multi_curve(
                model, curves, norm_params, use_time, use_rate, device,
                max_seq_length, use_log_targets, param_use_log,
                normalize_inputs, input_norm_params, use_log_inputs, s11_epsilon, normalize_s11_max,
                use_scenario=use_scenario, scenario=scenario,
                use_physics_features=use_physics_features
            )
        else:
            # Single-curve scenarios (0, 1, 2)
            S11 = flatten_matlab_array(data['S11'])
            F11 = flatten_matlab_array(data['F11'])
            F22 = flatten_matlab_array(data['F22'])
            F33 = flatten_matlab_array(data['F33'])
            t = flatten_matlab_array(data['t']) if 't' in data else None
            
            print(f"Single curve: {len(S11)} timesteps")
            if max_seq_length is not None and len(S11) > max_seq_length:
                print(f"  → Will downsample to {max_seq_length} timesteps")
            
            # Predict using single-curve function
            print("\nPredicting parameters...")
            pred_params = predict_parameters(
                model, S11, F11, F22, F33, t, norm_params, use_time, use_rate, device,
                max_seq_length, use_log_targets, param_use_log,
                normalize_inputs, input_norm_params, use_log_inputs, s11_epsilon, normalize_s11_max,
                use_scenario=use_scenario, scenario=scenario,
                use_physics_features=use_physics_features
            )

    # Display results
    print("\nPredicted parameters:")
    for i, name in enumerate(target_params):
        print(f"  {name:8s}: {pred_params[i]:12.6f}")

    if true_par is not None:
        true_params = true_par[param_indices]

        print("\nTrue parameters:")
        for i, name in enumerate(target_params):
            print(f"  {name:8s}: {true_params[i]:12.6f}")

        print("\nAbsolute errors:")
        errors = np.abs(pred_params - true_params)
        for i, name in enumerate(target_params):
            print(f"  {name:8s}: {errors[i]:12.6f}")

        print(f"\nMean absolute error: {errors.mean():.6f}")

        rel_errors = errors / (np.abs(true_params) + 1e-8) * 100
        print(f"Mean relative error: {rel_errors.mean():.2f}%")

    # Save predictions if requested
    if args.output:
        output_data = {
            'predicted_parameters': pred_params,
            'parameter_names': target_params,
            'scenario': scenario,
        }
        if true_par is not None:
            true_params = true_par[param_indices]
            output_data['true_parameters'] = true_params
            output_data['absolute_errors'] = np.abs(pred_params - true_params)
            output_data['relative_errors'] = np.abs(pred_params - true_params) / (np.abs(true_params) + 1e-8) * 100

        np.savez(args.output, **output_data)
        print(f"\nPredictions saved to {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
