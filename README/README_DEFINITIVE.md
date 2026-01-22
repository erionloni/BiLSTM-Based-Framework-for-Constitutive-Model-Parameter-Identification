# Complete Technical Documentation & User Guide

## About This Project

This repository accompanies the semester project:

**BiLSTM-Based Framework for Constitutive Model Parameter Identification**

*Erion Isljami*  
Semester Project, January 2026

Institute of Mechanical Systems & Signal and Information Processing Laboratory (ISI)  
Swiss Federal Institute of Technology (ETH) Zurich  
In collaboration with Empa – Swiss Federal Laboratories for Materials Science and Technology

**Advisors:** Prof. Hans-Andrea Loeliger, Dr. Ehsan Hosseini, Haotian Xu

---



## Abstract

Identifying parameters of history-dependent constitutive laws from uniaxial tension measurements is a central inverse problem in computational mechanics. Conventional workflows embed a forward simulator within iterative optimization, which becomes expensive when many evaluations are required. This thesis develops a data-driven surrogate that estimates Rubin–Bodner (RB) viscoelastic model parameters directly from stress–stretch histories using a bidirectional Long Short-Term Memory (BiLSTM) network. We evaluate multiple loading scenarios, including relaxation holds and multi-rate ramps, to study how protocol design affects identifiability.

The results show that several parameters exhibit strong predicted–true agreement (high R² / log-R²) from uniaxial histories, while a subset of volumetric parameters remains weakly identifiable under traction-free uniaxial loading. Constraining these weakly identifiable parameters enables forward reconstructions with a mean trajectory NRMSE of 12.81% on holdout data. After training, inference requires only a few milliseconds per specimen, enabling rapid parameter estimation and practical forward prediction.

## Reproducing the Experiments

To reconstruct the experiments, use the configuration files in the `\txt_files` folder and run the forward solver for data generation.

---



## 1. Project Overview

### 1.1 The Forward Problem (MATLAB Simulation)

The MATLAB simulation (`SA_matlab/`) solves the forward problem:

```
Material Parameters + Loading Protocol → Stress-Strain Curves
```

A compressible Rubin-Bodner viscoelastic material model with fiber reinforcement is subjected to uniaxial tensile loading. The simulation computes the stress response and lateral deformations over time using a predictor-corrector integration scheme.

### 1.2 The Inverse Problem (Machine Learning)

This project solves the inverse problem using a neural network:

```
Stress-Strain Curves → Material Parameters
```

Traditional methods (Finite Element Model Updating, curve fitting) require expensive iterative optimization for every experiment. This neural network approximates the inverse mapping **instantly** once trained.

### 1.3 Key Scripts and Folders

| Script | Purpose |
|--------|---------|
| `train_material_predictor.py` | Train/evaluate BiLSTM inverse model (single-curve + multi-curve) |
| `predict_material_parameters.py` | Inference script for new curves |
| `curve_reconstruction.py` | Compare true vs predicted curves using MATLAB forward model |
| `ambiguity_analysis.py` | Ambiguity diagnostics  |
| `SA_matlab` | Matlab forward solver and plotting scripts |

---

## 2. The Physics: Compressible Rubin-Bodner Model

### 2.1 Stress Decomposition

The Cauchy stress tensor **σ** is decomposed into three contributions:

$$\boldsymbol{\sigma} = \boldsymbol{\sigma}_{\text{matrix}} + \boldsymbol{\sigma}_{\text{fiber}} + \boldsymbol{\sigma}_{\text{volumetric}}$$

### 2.2 Loading Configuration

- **Uniaxial tension**: Stretch applied in the 1-direction (F₁₁ = λ₁)
- **Stress-free lateral boundaries**: σ₂₂ = σ₃₃ = 0 (solved iteratively)
- **Result**: F₂₂ and F₃₃ are computed to satisfy equilibrium

### 2.3 Fiber Architecture

- **16 fiber directions** distributed in 3D space
- Fiber angle **θ** (theta) controls out-of-plane pitch
- Fibers only resist **tension** (compression → zero contribution)
- Each fiber direction evolves independently via viscous flow

### 2.4 Time Integration

The constitutive equations are integrated using a backward-Euler predictor-corrector scheme (`pc_scheme.m`) with:
- `lsqnonlin` solver for equilibrium
- Tolerances: FunctionTolerance = 1e-12, StepTolerance = 1e-12

---

## 3. Material Parameters

### 3.1 The 10 Predicted Parameters

| # | Symbol | MATLAB Index | Physical Meaning | Range | Log Scale? |
|---|--------|--------------|------------------|-------|------------|
| 1 | **kM** | par(1) | Matrix dissipation prefactor | [10⁻⁴, 10¹] | ✅ Yes |
| 2 | **kF** | par(2) | Fiber dissipation prefactor | [10⁻⁸, 10⁻³] | ✅ Yes |
| 3 | **m3** | par(4) | Fiber energy scale | [0.1, 300] | ✅ Yes |
| 4 | **m5** | par(5) | Matrix compressibility exponent | [0.1, 10] | ✅ Yes |
| 5 | **q** | par(6) | Exponential coupling | [10⁻⁴, 10¹] | ✅ Yes |
| 6 | **m1** | par(7) | Volumetric penalty | [0.1, 300] | ✅ Yes |
| 7 | **m2** | par(8) | Neo-Hookean shear scale | [10⁻⁵, 10⁻¹] | ✅ Yes |
| 8 | **m4** | par(9) | Fiber nonlinearity exponent | [1.1, 1.5] | ❌ No |
| 9 | **θ** (theta) | par(10) | Fiber angle (radians) | [0.02, 0.35] | ❌ No |
| 10 | **αM** (alphaM) | par(11) | Matrix dissipation exponent | [0.1, 10] | ✅ Yes |

### 3.2 Parameters NOT Predicted (Fixed/Input)

| Symbol | MATLAB Index | Description |
|--------|--------------|-------------|
| **μ₀** (mu0) | par(3) | Reference shear modulus (always = 1) |
| **rate** | par(12) | Loading rate (input) |
| **λ₁_max** | par(13) | Maximum stretch (input) |



---

## 4. Loading Scenarios

The model supports 4 loading scenarios, each revealing different material behavior:

### 4.1 Scenario 0: Constant Rate Ramp

Simple monotonic loading at constant strain rate.

```
Stretch λ₁
    │
λmax├─────────────●
    │           ╱
    │        ╱
    │     ╱
  1 ├──●╱
    └────────────────► Time (t_ramp)
```

**Reveals**: Rate-dependent stiffness, elastic response

### 4.2 Scenario 1: Ramp + Hold (Stress Relaxation)

Ramp to max stretch, then hold constant while stress relaxes.

```
Stretch λ₁                  Stress S₁₁
    │                           │
λmax├────●━━━━━━━━●       Smax├────●
    │  ╱  (HOLD)              │  ╱ ╲
    │╱                        │╱   ╲___●
  1 ├                       0 ├
    └────────────────►          └────────────►
```

**Reveals**: Viscous relaxation time constants, equilibrium stress


### 4.3 Scenario 2: Multi-Step Piecewise Ramp

Loading with up to 3 segments, each at a different strain rate.

```
Stretch λ₁
    │
λmax├───────────────●
    │            ╱ (rate 3)
λ₂  ├─────────●╱
    │       ╱ (rate 2)
λ₁  ├─────●╱
    │   ╱ (rate 1)
  1 ├──●
    └────────────────► Time
```

**Reveals**: Rate sensitivity, transition behavior at rate changes

### 4.4 Scenario 3: Multi-Rate Repetition

Multiple **independent** ramp (or ramp+hold) tests at different rates on the same material. Controlled by the `hold_time` parameter:

- `hold_time = 0` → **Ramp-only mode** (original behavior)
- `hold_time > 0` → **Ramp+Hold mode** (same hold duration for all rates)

**Ramp-Only Mode** (`hold_time = 0`):
```
Test 1 (slow)      Test 2 (medium)    Test 3 (fast)
λmax├──────●       λmax├────●         λmax├──●
    │    ╱             │  ╱               │╱
  1 ├──●             1 ├──●             1 ├●
```

**Ramp+Hold Mode** (`hold_time > 0`):
```
Test 1 (slow)      Test 2 (medium)    Test 3 (fast)
λmax├──────●━━━●   λmax├────●━━━●     λmax├──●━━━●
    │    ╱             │  ╱               │╱
  1 ├──●             1 ├──●             1 ├●
        (hold)            (hold)           (hold)
```

**Data Structure**: Contains `multi_curves` array + top-level `hold_time` (shared across all rates).

**Reveals**: Direct rate comparison, viscous contribution at different timescales, and (with hold) stress relaxation behavior

---

## 5. Data Format

### 5.1 NPZ File Structure

#### Scenarios 0, 1, 2 (Single Curve)

```python
{
    'scenario': int,           # 0, 1, or 2
    'par': ndarray (13,),      # Material parameters
    'S11': ndarray (T,),       # 1st Piola-Kirchhoff stress
    'F11': ndarray (T,),       # Axial stretch
    'F22': ndarray (T,),       # Lateral stretch
    'F33': ndarray (T,),       # Lateral stretch
    't': ndarray (T,),         # Time vector
    'segment_rates': ndarray,  # Rates used
    'hold_time': float         # (Scenario 1, 2)
}
```

#### Scenario 3 (Multi-Curve)

```python
{
    'scenario': 3,
    'par': ndarray (13,),      # Shared parameters
    'multi_rates': ndarray (N,), # Array of N rates
    'hold_time': float,        # Hold duration (0 = ramp-only, >0 = ramp+hold per rate)
    'multi_curves': [          # Array of N curve objects
        {
            'S11': ndarray, 'F11': ndarray, 'F22': ndarray,
            'F33': ndarray, 't': ndarray, 'rate': float,
            'hold_time': float  # Same as top-level for convenience
        },
        ...
    ]
}
```


### 5.2 Parameter Vector (`par`) Indexing

```python
# Python 0-indexed
par[0]  = kM        # Matrix dissipation prefactor
par[1]  = kF        # Fiber dissipation prefactor
par[2]  = mu0       # Always 1.0
par[3]  = m3        # Fiber energy scale
par[4]  = m5        # Matrix compressibility exponent
par[5]  = q         # Exponential coupling
par[6]  = m1        # Volumetric penalty
par[7]  = m2        # Neo-Hookean shear scale
par[8]  = m4        # Fiber nonlinearity exponent
par[9]  = theta     # Fiber angle (radians)
par[10] = alphaM    # Matrix dissipation exponent
par[11] = rate      # Loading rate (protocol input, not predicted)
par[12] = lambda_max # Final stretch (input)

# Indices predicted by the model:
PREDICT_INDICES = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
```

---

## 6. Neural Network Architecture

### 6.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT SEQUENCE                           │
│  Variable-length: [S11, F11, F22, F33, (t), (scenario), …] │
│  Shape: (seq_len, input_dim) where input_dim = 4-20        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              BIDIRECTIONAL LSTM ENCODER                     │
│  - Default: 3 layers, 128 hidden units                     │
│  - Dropout between layers (default 0.2)                    │
│  - Uses pack_padded_sequence for variable lengths          │
│  - Output: Final hidden state (forward + backward concat)  │
│  - Embedding dim: 256 (128 × 2 for bidirectional)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼ (For Scenario 3: Multi-Curve Handling)
┌─────────────────────────────────────────────────────────────┐
│          MULTI-CURVE HANDLING (Scenario 3 Only)             │
│                                                             │
│  Each curve → Separate LSTM pass → Mean pool embeddings     │
│  (mean across curves, shared encoder weights)               │
└────────────────────┬────────────────────────────────────────┘

                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          FEED-FORWARD NETWORK (MLP Decoder)                 │
│  Default layers: [256, 128, 64]                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Linear(256 → 256) + ReLU + Dropout(0.2)           │   │
│  │            ↓                                        │   │
│  │  Linear(256 → 128) + ReLU + Dropout(0.2)           │   │
│  │            ↓                                        │   │
│  │  Linear(128 → 64) + ReLU + Dropout(0.2)            │   │
│  │            ↓                                        │   │
│  │  Linear(64 → 10)  (output layer, no activation)    │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT                                   │
│  10 normalized parameters (denormalized post-prediction)   │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 LSTMEncoder Class Details

```python
class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,          # Number of input features per timestep
        hidden_size: int = 128,      # LSTM hidden dimension
        num_layers: int = 3,         # Number of LSTM layers
        output_dim: int = 10,        # Number of output parameters
        dropout: float = 0.2,        # Dropout rate
        bidirectional: bool = True,  # Use bidirectional LSTM
        use_cell_state: bool = False,# Concatenate cell state with hidden
        mlp_layers: List[int] = [256, 128, 64],  # MLP layer sizes
        activation: str = 'relu',    # Options: 'relu', 'sigmoid', 'none'
        aggregation: str = 'mean'    # Mean pooling across curves (Scenario 3)
    )
```

### 6.3 Activation Functions

| Location | Activation | Purpose |
|----------|------------|---------|
| LSTM internal | Tanh, Sigmoid | Standard LSTM gates |
| MLP hidden layers | **ReLU** (default) | Non-linearity |
| MLP output layer | **None** | Linear output for regression |

### 6.4 Embedding Dimension Calculation

```python
embedding_dim = hidden_size × num_directions × state_multiplier
# Default: 128 × 2 × 1 = 256

# If use_cell_state=True:
# embedding_dim = 128 × 2 × 2 = 512
```

### 6.5 Total Parameters (Default Config)

```
LSTM:
  - Layer 1 (input): ~270K params
  - Layers 2-3: ~540K params each
  
MLP:
  - Linear 256→256: ~65K params
  - Linear 256→128: ~33K params
  - Linear 128→64:  ~8K params
  - Linear 64→10:   ~640 params

Total: ~1.45M parameters
```

---

## 7. Input Preprocessing & Feature Engineering

### 7.1 Base Input Features (4 features)

```python
[S11, F11, F22, F33]
```

### 7.2 Optional Features

| Flag | Feature Added | Total Dim | Description |
|------|---------------|-----------|-------------|
| `--use-time` | `t` | 5 | Time values (critical for rate-dependent behavior) |
| `--use-scenario` | 4 one-hot | +4 | Scenario type encoding |
| `--use-physics-features` | 11 features | +11 | Physics-informed derivatives, ratios, and volumetric invariants |

### 7.3 Physics-Informed Features

When `--use-physics-features` is enabled, 11 additional features are computed:

| # | Feature | Formula | Physical Meaning | Helps Identify |
|---|---------|---------|------------------|----------------|
| 1 | `dS11_dt` | ∂S11/∂t | Stress rate | kM, kF, alphaM |
| 2 | `dF22_dt` | ∂F22/∂t | Lateral creep rate | alphaM |
| 3 | `dF11_dt` | ∂F11/∂t | Loading rate | Rate-dependent params |
| 4 | `dS11_dF11` | (∂S11/∂t) / (∂F11/∂t) | Tangent stiffness | m2, m3 |
| 5 | `d2S11_dF112` | ∂(dS11/dF11)/∂t / (∂F11/∂t) | Curvature | m4, q |
| 6 | `F22_F11_Ratio` | F22 / F11 | Poisson effect | m5 |
| 7 | `F22_F33_Ratio` | F22 / F33 | Anisotropy indicator | theta |
| 8 | `S_norm` | S11 / max(\|S11\|) | Normalized shape | Rate independence |
| 9 | `J` | F11 · F22 · F33 | Volume ratio (det F) | m5, alphaM |
| 10 | `log_J` | log(\|J\| + ε) | Logarithmic volumetric strain | m5, alphaM |
| 11 | `I1` | F11² + F22² + F33² | First invariant of b | m5, alphaM |

**Implementation**: `calculate_physics_features()` in `train_material_predictor.py`

### 7.4 Input Normalization Options

| Flag | Effect | Use When |
|------|--------|----------|
| `--normalize-inputs` | Z-score normalization: (x - μ) / σ | Always recommended |
| `--use-log-inputs` | log₁₀(|S11| + ε) before normalization | Stress spans decades |
| `--normalize-s11-max` | S11 / max(|S11|) per curve | Rate-independent analysis |

### 7.5 Maximum Sequence Length

```python
--max-seq-length 100  # Downsample to 100 points if longer
```

Downsampling uses uniform index selection to preserve curve shape.

---

## 8. Training Details

### 8.1 Loss Function

**Standard MSE Loss**:
```python
criterion = nn.MSELoss()
```




### 8.2 Target Normalization

**Log Transformation** (for parameters with large dynamic range):
```python
LOG_PARAM_NAMES = ['kM', 'kF', 'm3', 'm5', 'q', 'm1', 'm2', 'alphaM']
NO_LOG_PARAM_NAMES = ['m4', 'theta']

# During training:
target[LOG_PARAM_NAMES] = log(target[LOG_PARAM_NAMES])
```

**Z-Score Normalization**:
```python
normalized_target = (target - mean) / std
```

### 8.3 Optimizer & Scheduler

**Default Scheduler (ReduceLROnPlateau)**:
```bash
--scheduler plateau
```
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,       # Reduce LR by half
    patience=10,      # Wait 10 epochs before reducing
    min_lr=1e-6
)
```

**Alternative (CosineAnnealingWarmRestarts)**:
```bash
--scheduler cosine --scheduler-t0 10 --scheduler-tmult 2
```
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,           # Epochs until first restart
    T_mult=2          # Multiply T_0 by this after each restart
)
```

| Scheduler | Best For | Flag |
|-----------|----------|------|
| `plateau` (default) | Stable convergence, auto-adjusting | `--scheduler plateau` |
| `cosine` | Faster exploration, warm restarts | `--scheduler cosine` |

### 8.4 Early Stopping

Early stopping is controlled by `--early-stop-patience`:

```bash
--early-stop-patience 20  # Stop if no improvement for 20 epochs (default)
--early-stop-patience 0   # Disable early stopping, train for full epochs
```

### 8.5 Weight Initialization Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| **Auto** (default) | `--weight-init auto` | Kaiming for ReLU, Xavier for Tanh/Sigmoid |
| Kaiming (He) | `--weight-init kaiming` | Best for ReLU networks |
| Xavier (Glorot) | `--weight-init xavier` | Best for Tanh/Sigmoid |
| Default | `--weight-init default` | PyTorch defaults |

The `auto` strategy automatically selects based on `--mlp-activation`:
- ReLU → Kaiming (He) uniform initialization
- Tanh/Sigmoid → Xavier (Glorot) uniform initialization

### 8.6 Automatic Mixed Precision (AMP)

```python
--use-amp  # Enables FP16 training on supported GPUs
```

Uses `torch.amp.autocast` and `GradScaler` for faster training with lower memory usage.

### 8.7 Gradient Clipping

```python
--max-grad-norm 1.0  # Default, gradients clipped to prevent explosions
```

### 8.8 Resume Training

```bash
--resume path/to/checkpoint.pth  # Resume training from a checkpoint
```

---

## 9. User Guide: Training

### 9.1 Basic Training Command

```bash
python train_material_predictor.py \
    --data-dir path/to/npz_data \
    --epochs 100 \
    --batch-size 32
```



### 9.2 All Command-Line Arguments

#### Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `raw_npz` | Directory containing .npz files |
| `--max-files` | None | Limit number of files (for testing) |
| `--max-seq-length` | None | Downsample sequences to this length |

#### Model Architecture Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-size` | 128 | LSTM hidden dimension |
| `--num-layers` | 3 | Number of LSTM layers |
| `--dropout` | 0.2 | Dropout rate |
| `--mlp-layers` | 256 128 64 | MLP layer sizes |
| `--use-cell-state` | False | Include LSTM cell state in embedding |
| `--mlp-activation` | relu | MLP activation: relu, sigmoid, none |

#### Feature Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-time` | False | Include time as input feature |
| `--use-scenario` | False | Include one-hot scenario encoding |
| `--use-physics-features` | False | Include 11 physics-derived features |

#### Preprocessing Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--normalize-inputs` | False | Z-score normalize inputs |
| `--use-log-inputs` | False | Apply log₁₀ to S11 |
| `--normalize-s11-max` | False | Normalize S11 by max value |
| `--use-log-targets` | False | Apply log to target parameters |

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--use-amp` | False | Use automatic mixed precision |
| `--weight-init` | auto | Initialization: auto, kaiming, xavier, default |
| `--scheduler` | plateau | LR scheduler: plateau, cosine |
| `--scheduler-t0` | 10 | CosineAnnealingWarmRestarts T_0 |
| `--scheduler-tmult` | 2 | CosineAnnealingWarmRestarts T_mult |
| `--early-stop-patience` | 20 | Epochs without improvement before stopping. **Set to 0 to disable.** |
| `--max-grad-norm` | 1.0 | Maximum gradient norm for clipping |
| `--seed` | 42 | Random seed |
| `--resume` | None | Path to checkpoint to resume from |

#### Loss Function Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-weighted-loss` | False | Use weighted MSE loss |
| `--loss-weights` | None | Custom weights for each parameter |

Multi-curve handling is fixed to mean pooling across curves for Scenario 3.

#### Output Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | output | Directory for saved models and plots |

### 9.3 Training Outputs

After training, the following files are saved in `--output-dir`:

| File | Description |
|------|-------------|
| `best_model.pth` | Best model checkpoint (lowest validation loss) |
| `training_curves.png` | Loss vs epoch plot |
| `predictions.png` | True vs predicted scatter plots (log scale) |
| `predictions_linear.png` | True vs predicted scatter plots (linear scale) |
| `predictions_worst_highlighted.png` | Scatter plots with worst 10 predictions highlighted in red |
| `train_val_scatter.png` | Train vs validation comparison (log scale) |
| `train_val_scatter_linear.png` | Train vs validation comparison (linear) |
| `train_val_worst_highlighted.png` | Train/val scatter with worst predictions highlighted |
| `test_predictions.csv` | CSV with `true_*` and `pred_*` columns for all test samples |
| `test_worst_indices.csv` | Worst test samples per-parameter (indices + error values) + overall worst |
| `train_val_worst_indices.csv` | Worst train/val samples per-parameter (indices + error values) + overall worst indices |
| `worst_indices_unified.csv` | Unified CSV of `worst`/`best`/`random` samples per split (default: 10 each; typically 90 rows total), with file paths + per-sample loss |
| `sample_mapping.csv` | Full per-sample mapping (`split,index,file_path,loss`) for all evaluated samples (train subset + full val/test) |

---

## 10. User Guide: Inference

### 10.1 Basic Inference

```bash
python predict_material_parameters.py \
    --model output/best_model.pth \
    --input path/to/curve.npz
```

To also save the predictions to disk:

```bash
python predict_material_parameters.py \
    --model output/best_model.pth \
    --input path/to/curve.npz \
    --output predicted_parameters.npz
```

The saved `predicted_parameters.npz` contains `predicted_parameters`, `parameter_names`, and `scenario` (and, if the input NPZ also contains `par`, it adds `true_parameters`, `absolute_errors`, and `relative_errors`).

### 10.2 Inference Output Example

```
Using device: cuda
Loading model from output/best_model.pth...
Model loaded successfully!
Model predicts 10 parameters: kM, kF, m3, m5, q, m1, m2, m4, theta, alphaM
MLP architecture: [256, 128, 64], activation: relu
Multi-curve aggregation: mean

Loading data from input_curve.npz...
Detected Scenario: 1
Single curve: 500 timesteps
  → Will downsample to 100 timesteps

Predicting parameters...

Predicted parameters:
  kM      :     1.234567
  kF      :     0.000005
  m3      :    45.123456
  m5      :     2.345678
  q       :     0.012345
  m1      :    12.345678
  m2      :     0.001234
  m4      :     1.234567
  theta   :     0.123456
  alphaM  :     3.456789

True parameters:
  kM      :     1.200000
  ...

Absolute errors:
  kM      :     0.034567
  ...

Mean absolute error: 0.123456
Mean relative error: 5.67%
```


---

## 11. Curve Reconstruction & Validation

The `curve_reconstruction.py` script provides a complete pipeline to **validate model predictions** by:
1. Predicting material parameters from input curves
2. Running MATLAB forward simulation with predicted parameters
3. Generating comparison plots of true vs predicted curves

This is the ultimate test of model accuracy: "Do the predicted parameters produce the same curves as the true parameters?"

### 11.1 Basic Usage

```bash
python curve_reconstruction.py \
    --model-dir path/to/model_folder \
    --npz-file path/to/input_curve.npz
```

### 11.2 Batch Processing

To process multiple random files from a directory:

```bash
python curve_reconstruction.py \
    --model-dir path/to/model_folder \
    --data-dir path/to/npz_data \
    --n-random 5
```

### 11.3 Reconstruct Worst/Best/Random Predictions

After training, a `worst_indices_unified.csv` file is saved containing `worst`/`best`/`random` samples per split (default: 10 each, ranked by per-sample training loss) including file paths. You can reconstruct curves for these cases:

```bash
# Reconstruct top 3 worst from test split
python curve_reconstruction.py \
    --model-dir path/to/model_folder \
    --worst-csv path/to/worst_indices_unified.csv \
    --split test \
    --top-n 3

# Reconstruct ALL rows from CSV (includes worst+best+random across splits)
python curve_reconstruction.py \
    --model-dir path/to/model_folder \
    --worst-csv path/to/worst_indices_unified.csv \
    --reconstruct-all-csv

# Faster for many files: run MATLAB once for the full list
python curve_reconstruction.py \
    --model-dir path/to/model_folder \
    --worst-csv path/to/worst_indices_unified.csv \
    --reconstruct-all-csv \
    --matlab-batch
```

### 11.4 Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-dir` | ✅ Yes | - | Directory containing `best_model.pth` |
| `--npz-file` | ⚠️ One of three | - | Single NPZ file to analyze |
| `--data-dir` | ⚠️ One of three | - | Directory with NPZ files (batch mode) |
| `--sample-mapping` | ❌ No | - | Enable dataset evaluation mode using `sample_mapping.csv` (requires `--data-dir`) |
| `--worst-csv` | ⚠️ One of three | - | Path to `worst_indices_unified.csv` from training |
| `--n-random` | ❌ No | 1 | Number of random files to process in batch mode |
| `--split` | ❌ No | `all` | Filter worst CSV by split: `train`, `val`, `test`, or `all` |
| `--top-n` | ❌ No | 10 | Reconstruct first N rows after applying `--split` (CSV is written worst-first per split) |
| `--reconstruct-all-csv` | ❌ No | False | Reconstruct all rows from `--worst-csv` (ignores `--top-n`) |
| `--splits` | ❌ No | `test` | Legacy/ignored (kept for backward compatibility) |
| `--output-dir` | ❌ No | `curve_reconstruction_results` | Output directory for results |
| `--job-id` | ❌ No | - | Job identifier to isolate MATLAB files for parallel runs |
| `--skip-matlab` | ❌ No | False | Skip MATLAB simulation (only predict params) |
| `--skip-plot` | ❌ No | False | Skip plotting |
| `--dpi` | ❌ No | 600 | DPI for plots (also used by dataset evaluation mode) |
| `--matlab-batch` | ❌ No | False | Run MATLAB once for all selected samples (faster for `--worst-csv` / `--data-dir`) |
| `--matlab-timeout` | ❌ No | Auto | MATLAB timeout in seconds (use `0` to disable) |
| `--csv-path-root` | ❌ No | - | Optional root directory to resolve CSV file paths (useful on clusters) |
| `--csv-offset` | ❌ No | 0 | Skip the first N rows of the selected CSV subset |
| `--csv-limit` | ❌ No | - | Only process the next N rows of the selected CSV subset |

### 11.5 Workflow Steps

The script performs the following steps:

1. **Load Model**: Loads the trained model from `best_model.pth`
2. **Predict Parameters**: Uses the model to predict 10 material parameters from input curve(s)
3. **Generate Parameters File**: Writes predicted parameters to `generated_parameters.txt` in MATLAB format
4. **Run MATLAB Simulation**: Executes `main_R_local_Empa_flex.m` to generate curves from predicted parameters
5. **Convert Output**: Converts MATLAB `.mat` output to `.npz` format
6. **Plot Results**: Generates comparison plots showing true vs predicted curves

```
Input NPZ → Model Prediction → generated_parameters.txt → MATLAB → Output MAT → NPZ → Plots
```

### 11.6 Output Files

For each processed file, the following outputs are generated in the output directory:

| File | Description |
|------|-------------|
| `predicted_curves.npz` | MATLAB simulation output in NPZ format |
| `predicted_curves_plot.png` | Optional: plot of `predicted_curves.npz` (run `python SA_matlab/plot_results.py predicted_curves.npz`) |
| `comparison_true_vs_predicted.png` | **Key output**: Side-by-side comparison of true vs predicted curves |
| `comparison_rate_N.png` | (Scenario 3 only) Per-rate comparison plots |

### 11.7 Understanding the Comparison Plot

The comparison plot shows:
- **Solid lines**: Curves generated from TRUE parameters
- **Dashed lines**: Curves generated from PREDICTED parameters
- **Right panel**: Parameter comparison table with relative errors

**For Scenario 3 (Multi-Rate)**:
- Each color represents a different strain rate
- All curves share the same material parameters
- A legend shows which color corresponds to which rate

### 11.8 MATLAB Requirements

- MATLAB must be installed and accessible via command line (`matlab -batch`)
- The script uses `main_R_local_Empa_flex.m` which requires the Parallel Computing Toolbox
- First run may take longer (~60-90s) due to parallel pool initialization
- Subsequent runs are faster (~30-60s per simulation)

---

## 12. Plotting Results

### 12.1 Using plot_results.py

The `SA_matlab/plot_results.py` script provides visualization for stress-strain curves from NPZ files.

**Basic usage:**

```bash
python SA_matlab/plot_results.py path/to/curve.npz
```

**Skip showing the window (save to file only):**

```bash
python SA_matlab/plot_results.py path/to/curve.npz --no-show
```

### 12.2 Plot Structure

The script generates a 2×3 grid of plots plus a parameter panel:

**Row 1 (vs F11):**
- S11 vs F11 (stress-strain curve)
- F22 vs F11 (lateral deformation)
- F33 vs F11 (lateral deformation)

**Row 2 (vs Time):**
- F11 vs Time (loading history)
- S11 vs Time (stress relaxation visualization)
- F22, F33 vs Time (lateral deformation history)

**Parameter Panel:**
- Material parameters displayed in formatted table
- For Scenario 1: Hold time and relaxation info
- For Scenario 3: Applied rates legend

### 12.3 Scenario-Specific Visualization

- **Scenario 0 (Constant Rate)**: Single curve with uniform color
- **Scenario 1 (Ramp+Hold)**: Blue lines for ramp, red lines for hold phase
- **Scenario 3 (Multi-Rate)**: Color-coded curves for each rate, viridis colormap

---

