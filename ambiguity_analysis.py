#!/usr/bin/env python3
"""
Ambiguity Diagnostic Tool (Shape-Only Analysis)

Detect many-to-one mappings where similar curve shapes correspond to different material parameters.
This is a physics-based identifiability diagnostic that does NOT require a trained ML model.

Usage:
    python ambiguity_analysis.py --data-dir <input_directory>

    python ambiguity_analysis.py \
        --data-dir ./npz_data \
        --output-dir ./results \
        --k-neighbors 20
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# ==============================================================================
# Constants
# ==============================================================================

PREDICT_INDICES = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
PARAM_NAMES = ['kM', 'kF', 'm3', 'm5', 'q', 'm1', 'm2', 'm4', 'theta', 'alphaM']
LOG_PARAM_NAMES = ['kM', 'kF', 'm3', 'm5', 'q', 'm1', 'm2', 'alphaM']
PARAM_USE_LOG = np.array([name in LOG_PARAM_NAMES for name in PARAM_NAMES])


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class CurveData:
    """Single curve data."""
    S11: np.ndarray
    F11: np.ndarray
    F22: np.ndarray
    F33: np.ndarray
    t: np.ndarray
    rate: float = 0.0
    hold_time: float = 0.0


@dataclass
class Sample:
    """A single dataset sample (one NPZ file)."""
    file_path: Path
    sample_id: int
    scenario: int
    par: np.ndarray
    curves: List[CurveData]
    hold_time: float = 0.0
    detected_hold: Optional[bool] = None
    
    @property
    def n_curves(self) -> int:
        return len(self.curves)
    
    @property
    def has_hold(self) -> bool:
        if self.detected_hold is None:
            self.detected_hold = any(
                detect_hold_start_index(curve.F11) is not None
                for curve in self.curves
            )
        return self.detected_hold


@dataclass
class SampleEmbedding:
    """Embedded representation of a sample."""
    sample_id: int
    file_path: Path
    scenario: int
    par: np.ndarray
    curve_embeddings: List[np.ndarray]
    n_curves: int = 1
    has_hold: bool = False


# ==============================================================================
# Data Loading
# ==============================================================================

def get_field(obj, key, default=None):
    """Helper to get a field from dict, structured array, or object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, 'dtype') and obj.dtype.names is not None:
        if key in obj.dtype.names:
            return obj[key]
        return default
    if hasattr(obj, key):
        return getattr(obj, key)
    return default


def flatten_matlab_array(arr):
    """Flatten array from MATLAB exports."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    while arr.shape == (1, 1) and arr.dtype == object:
        arr = arr[0, 0]
        arr = np.asarray(arr)
    if arr.ndim > 1:
        return arr.flatten()
    return arr


def load_sample(file_path: Path, sample_id: int) -> Optional[Sample]:
    """Load a single NPZ file into a Sample object."""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            scenario_val = data.get('scenario', 0)
            scenario = int(scenario_val.item()) if hasattr(scenario_val, 'item') else int(scenario_val)
            par = flatten_matlab_array(data['par'])
            hold_time_val = data.get('hold_time', 0)
            hold_time = float(hold_time_val.item()) if hasattr(hold_time_val, 'item') else float(hold_time_val)
            
            curves = []
            
            if scenario == 3:
                multi_curves = data.get('multi_curves', None)
                if multi_curves is None:
                    return None
                multi_curves = multi_curves.flatten()
                
                for mc in multi_curves:
                    S11 = flatten_matlab_array(get_field(mc, 'S11'))
                    F11 = flatten_matlab_array(get_field(mc, 'F11'))
                    F22 = flatten_matlab_array(get_field(mc, 'F22'))
                    F33 = flatten_matlab_array(get_field(mc, 'F33'))
                    t = flatten_matlab_array(get_field(mc, 't'))
                    rate_val = get_field(mc, 'rate', 0)
                    rate = float(flatten_matlab_array(rate_val).flat[0] if hasattr(rate_val, 'flat') else rate_val)
                    curve_hold_val = get_field(mc, 'hold_time', hold_time)
                    curve_hold = float(flatten_matlab_array(curve_hold_val).flat[0] if hasattr(curve_hold_val, 'flat') else curve_hold_val)
                    
                    if S11 is None or F11 is None:
                        continue
                    
                    curves.append(CurveData(
                        S11=S11,
                        F11=F11,
                        F22=F22 if F22 is not None else np.ones_like(S11),
                        F33=F33 if F33 is not None else np.ones_like(S11),
                        t=t if t is not None else np.arange(len(S11)),
                        rate=rate,
                        hold_time=curve_hold
                    ))
            else:
                S11 = flatten_matlab_array(data['S11'])
                F11 = flatten_matlab_array(data['F11'])
                F22 = flatten_matlab_array(data.get('F22', np.ones_like(S11)))
                F33 = flatten_matlab_array(data.get('F33', np.ones_like(S11)))
                t = flatten_matlab_array(data.get('t', np.arange(len(S11))))
                
                curves.append(CurveData(
                    S11=S11,
                    F11=F11,
                    F22=F22,
                    F33=F33,
                    t=t,
                    rate=float(par[11]) if len(par) > 11 else 0.0,
                    hold_time=hold_time
                ))
            
            if len(curves) == 0:
                return None
            
            return Sample(
                file_path=file_path,
                sample_id=sample_id,
                scenario=scenario,
                par=par,
                curves=curves,
                hold_time=hold_time
            )
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def load_and_group_samples(
    data_dir: Path,
    max_files: Optional[int] = None,
    seed: int = 42
) -> Dict[int, List[Sample]]:
    """Load all NPZ files and group by scenario."""
    npz_files = sorted(data_dir.glob("*.npz"))
    
    if max_files is not None and len(npz_files) > max_files:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(npz_files), size=max_files, replace=False)
        npz_files = [npz_files[i] for i in sorted(indices)]
    
    print(f"Loading {len(npz_files)} NPZ files from {data_dir}...")
    
    samples_by_scenario: Dict[int, List[Sample]] = {0: [], 1: [], 2: [], 3: []}
    
    for idx, file_path in enumerate(tqdm(npz_files, desc="Loading samples")):
        sample = load_sample(file_path, idx)
        if sample is not None:
            samples_by_scenario[sample.scenario].append(sample)
    
    for scenario, samples in samples_by_scenario.items():
        if len(samples) > 0:
            n_with_hold = sum(1 for s in samples if s.has_hold)
            print(f"  Scenario {scenario}: {len(samples)} samples ({n_with_hold} with hold)")
    
    return samples_by_scenario


# ==============================================================================
# Curve Embedding
# ==============================================================================

def interpolate_to_grid(values: np.ndarray, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate values onto a fixed grid."""
    if len(x) < 2:
        return np.full(len(grid), values[0] if len(values) > 0 else 0.0)
    
    _, unique_idx = np.unique(x, return_index=True)
    x_unique = x[unique_idx]
    values_unique = values[unique_idx]
    
    if len(x_unique) < 2:
        return np.full(len(grid), values_unique[0] if len(values_unique) > 0 else 0.0)
    
    interp_func = interp1d(x_unique, values_unique, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    return interp_func(grid)


def detect_hold_start_index(F11: np.ndarray, tol: float = 1e-4) -> Optional[int]:
    """Detect where the hold phase starts (F11 becomes constant)."""
    if len(F11) < 10:
        return None
    
    dF11 = np.abs(np.diff(F11))
    threshold = tol * (F11.max() - F11.min() + 1e-10)
    hold_mask = dF11 < threshold
    
    if not np.any(hold_mask):
        return None
    
    for i in range(len(hold_mask) - 10):
        if np.all(hold_mask[i:i+10]):
            return i
    
    return None


def embed_ramp_stretch_domain(
    curve: CurveData,
    grid_size: int = 200,
    hold_start_idx: Optional[int] = None
) -> np.ndarray:
    """Embed ramp portion of curve in stretch-domain."""
    if hold_start_idx is not None and hold_start_idx > 10:
        S11 = curve.S11[:hold_start_idx]
        F11 = curve.F11[:hold_start_idx]
        F22 = curve.F22[:hold_start_idx]
        F33 = curve.F33[:hold_start_idx]
    else:
        S11 = curve.S11
        F11 = curve.F11
        F22 = curve.F22
        F33 = curve.F33
    
    F11_min = 1.0
    F11_max = F11.max()
    
    if F11_max - F11_min < 1e-6:
        return np.zeros((grid_size, 3))
    
    xi = (F11 - F11_min) / (F11_max - F11_min)
    xi_grid = np.linspace(0, 1, grid_size)
    
    S11_interp = interpolate_to_grid(S11, xi, xi_grid)
    F22_interp = interpolate_to_grid(F22, xi, xi_grid)
    F33_interp = interpolate_to_grid(F33, xi, xi_grid)
    
    return np.column_stack([S11_interp, F22_interp, F33_interp])


def embed_hold_time_domain(
    curve: CurveData,
    hold_start_idx: int,
    grid_size: int = 100
) -> Optional[np.ndarray]:
    """Embed hold portion of curve in time-domain."""
    if hold_start_idx is None or hold_start_idx >= len(curve.S11) - 5:
        return None
    
    S11 = curve.S11[hold_start_idx:]
    F22 = curve.F22[hold_start_idx:]
    F33 = curve.F33[hold_start_idx:]
    t = curve.t[hold_start_idx:]
    
    if len(t) < 5:
        return None
    
    t_min = t[0]
    t_max = t[-1]
    
    if t_max - t_min < 1e-6:
        return None
    
    tau = (t - t_min) / (t_max - t_min)
    tau_grid = np.linspace(0, 1, grid_size)
    
    S11_interp = interpolate_to_grid(S11, tau, tau_grid)
    F22_interp = interpolate_to_grid(F22, tau, tau_grid)
    F33_interp = interpolate_to_grid(F33, tau, tau_grid)
    
    return np.column_stack([S11_interp, F22_interp, F33_interp])


def normalize_curve_shape(embedding: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-curve normalization to focus on shape, not magnitude."""
    normalized = np.zeros_like(embedding)
    
    # S11 (column 0)
    col = embedding[:, 0]
    col_shifted = col - col[0]
    range_val = col.max() - col.min()
    normalized[:, 0] = col_shifted / (range_val + eps)
    
    # F22 (column 1)
    col = embedding[:, 1]
    col_shifted = col - 1.0
    scale = np.max(np.abs(col_shifted)) + eps
    normalized[:, 1] = col_shifted / scale
    
    # F33 (column 2)
    col = embedding[:, 2]
    col_shifted = col - 1.0
    scale = np.max(np.abs(col_shifted)) + eps
    normalized[:, 2] = col_shifted / scale
    
    return normalized


def build_curve_embedding(
    curve: CurveData,
    ramp_grid_size: int = 200,
    hold_grid_size: int = 100,
    allow_hold: bool = True,
    force_hold_block: bool = False,
    hold_tol: float = 1e-4
) -> np.ndarray:
    """Build shape-only embedding for a single curve."""
    hold_start_idx = None
    if allow_hold:
        hold_start_idx = detect_hold_start_index(curve.F11, tol=hold_tol)
    
    ramp_embed = embed_ramp_stretch_domain(curve, ramp_grid_size, hold_start_idx)
    ramp_embed_norm = normalize_curve_shape(ramp_embed)
    
    if allow_hold and hold_start_idx is not None:
        hold_embed = embed_hold_time_domain(curve, hold_start_idx, hold_grid_size)
        if hold_embed is not None:
            hold_embed_norm = normalize_curve_shape(hold_embed)
            return np.concatenate([ramp_embed_norm.flatten(), hold_embed_norm.flatten()])

    if force_hold_block:
        hold_block = np.zeros((hold_grid_size, 3))
        return np.concatenate([ramp_embed_norm.flatten(), hold_block.flatten()])

    return ramp_embed_norm.flatten()


def embed_sample(
    sample: Sample,
    ramp_grid_size: int = 200,
    hold_grid_size: int = 100,
    allow_hold: bool = True,
    force_hold_block: bool = False
) -> SampleEmbedding:
    """Embed a sample using shape-only embedding."""
    curve_embeddings = [
        build_curve_embedding(
            curve, ramp_grid_size, hold_grid_size,
            allow_hold=allow_hold,
            force_hold_block=force_hold_block
        )
        for curve in sample.curves
    ]
    
    return SampleEmbedding(
        sample_id=sample.sample_id,
        file_path=sample.file_path,
        scenario=sample.scenario,
        par=sample.par,
        curve_embeddings=curve_embeddings,
        n_curves=sample.n_curves,
        has_hold=sample.has_hold
    )


# ==============================================================================
# Distance Computation
# ==============================================================================

def curve_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between curve embeddings."""
    return np.linalg.norm(a - b)


def chamfer_distance(set_a: List[np.ndarray], set_b: List[np.ndarray]) -> float:
    """Bidirectional nearest-neighbor (Chamfer) distance between two sets."""
    if len(set_a) == 0 or len(set_b) == 0:
        return float('inf')
    
    sum_a_to_b = sum(min(curve_distance(a, b) for b in set_b) for a in set_a)
    sum_b_to_a = sum(min(curve_distance(b, a) for a in set_a) for b in set_b)
    
    return sum_a_to_b / len(set_a) + sum_b_to_a / len(set_b)


# ==============================================================================
# k-Nearest Neighbors
# ==============================================================================

def compute_pooled_signature(embeddings: List[SampleEmbedding]) -> np.ndarray:
    """Compute pooled signatures for two-stage retrieval."""
    all_curve_embeds = []
    for sample in embeddings:
        for curve_embed in sample.curve_embeddings:
            all_curve_embeds.append(curve_embed)
    
    if len(all_curve_embeds) == 0:
        return np.zeros((len(embeddings), 1))
    
    all_curve_embeds = np.array(all_curve_embeds)
    embed_dim = all_curve_embeds.shape[1]
    
    signatures = []
    for sample in embeddings:
        if len(sample.curve_embeddings) == 0:
            signatures.append(np.zeros(embed_dim * 2))
            continue
        
        curve_embeds = np.array(sample.curve_embeddings)
        mean_embed = curve_embeds.mean(axis=0)
        std_embed = curve_embeds.std(axis=0) if len(curve_embeds) > 1 else np.zeros_like(mean_embed)
        signatures.append(np.concatenate([mean_embed, std_embed]))
    
    return np.array(signatures)


def find_neighbors(
    embeddings: List[SampleEmbedding],
    k: int,
    candidate_multiplier: int = 20
) -> Dict[int, List[Tuple[int, float]]]:
    """Two-stage kNN: signatures for candidates, Chamfer for exact distance."""
    n_samples = len(embeddings)
    if n_samples <= k:
        neighbors = {}
        for i, emb_i in enumerate(embeddings):
            dists = []
            for j, emb_j in enumerate(embeddings):
                if i != j:
                    d = chamfer_distance(emb_i.curve_embeddings, emb_j.curve_embeddings)
                    dists.append((emb_j.sample_id, d))
            dists.sort(key=lambda x: x[1])
            neighbors[emb_i.sample_id] = dists[:k]
        return neighbors
    
    print("  Stage 1: Computing pooled signatures...")
    signatures = compute_pooled_signature(embeddings)
    
    n_candidates = min(k * candidate_multiplier, n_samples - 1)
    nn_model = NearestNeighbors(n_neighbors=n_candidates + 1, metric='euclidean')
    nn_model.fit(signatures)
    
    _, candidate_indices = nn_model.kneighbors(signatures)
    
    print("  Stage 2: Computing exact Chamfer distances...")
    neighbors = {}
    
    for i, emb_i in enumerate(tqdm(embeddings, desc="  kNN")):
        candidates = candidate_indices[i]
        candidates = [c for c in candidates if c != i]
        
        dists = []
        for c in candidates:
            emb_c = embeddings[c]
            d = chamfer_distance(emb_i.curve_embeddings, emb_c.curve_embeddings)
            dists.append((emb_c.sample_id, d))
        
        dists.sort(key=lambda x: x[1])
        neighbors[emb_i.sample_id] = dists[:k]
    
    return neighbors


# ==============================================================================
# Parameter Distance
# ==============================================================================

def transform_parameters(par: np.ndarray) -> np.ndarray:
    """Transform parameters: log for log-scale params, linear for others."""
    selected = par[PREDICT_INDICES].copy()
    
    theta = np.zeros_like(selected)
    for i, (val, use_log) in enumerate(zip(selected, PARAM_USE_LOG)):
        if use_log:
            theta[i] = np.log(max(val, 1e-15))
        else:
            theta[i] = val
    
    return theta


def compute_param_statistics(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of transformed parameters across samples."""
    transformed = [transform_parameters(s.par) for s in samples]
    transformed = np.array(transformed)
    return transformed.mean(axis=0), transformed.std(axis=0) + 1e-10


def param_distance(
    par_i: np.ndarray,
    par_j: np.ndarray,
    param_mean: np.ndarray,
    param_std: np.ndarray
) -> float:
    """L2 distance between transformed and standardized parameters."""
    theta_i = (transform_parameters(par_i) - param_mean) / param_std
    theta_j = (transform_parameters(par_j) - param_mean) / param_std
    return float(np.linalg.norm(theta_i - theta_j))


def param_differences(par_i: np.ndarray, par_j: np.ndarray) -> Dict[str, float]:
    """Compute per-parameter absolute differences in transformed space."""
    theta_i = transform_parameters(par_i)
    theta_j = transform_parameters(par_j)
    return {name: float(diff) for name, diff in zip(PARAM_NAMES, np.abs(theta_i - theta_j))}


# ==============================================================================
# Ambiguity Metrics
# ==============================================================================

def compute_ambiguity_score(
    d_y_neighbors: List[float],
    d_p_neighbors: List[float],
    eps: float = 1e-8
) -> float:
    """Local ambiguity score: A(i) = median(d_p) / (median(d_y) + ε)"""
    if len(d_y_neighbors) == 0 or len(d_p_neighbors) == 0:
        return 0.0
    return np.median(d_p_neighbors) / (np.median(d_y_neighbors) + eps)


def compute_global_ambiguity_stats(all_pairs: pd.DataFrame) -> dict:
    """Compute global ambiguity statistics."""
    tau_curve = np.percentile(all_pairs['d_y'], 20)
    tau_param = np.percentile(all_pairs['d_p'], 80)
    
    ambiguous_mask = (all_pairs['d_y'] < tau_curve) & (all_pairs['d_p'] > tau_param)
    ambiguity_rate = ambiguous_mask.sum() / len(all_pairs) * 100
    
    close_pairs = all_pairs[all_pairs['d_y'] < tau_curve]
    per_param_spreads = {}
    for name in PARAM_NAMES:
        col = f'delta_{name}'
        if col in close_pairs.columns:
            per_param_spreads[name] = float(close_pairs[col].median())
    
    return {
        'tau_curve': float(tau_curve),
        'tau_param': float(tau_param),
        'ambiguity_rate_pct': float(ambiguity_rate),
        'n_ambiguous_pairs': int(ambiguous_mask.sum()),
        'per_param_median_spread': per_param_spreads
    }


# ==============================================================================
# Analysis and Output
# ==============================================================================

def run_ambiguity_analysis(
    samples: List[Sample],
    embeddings: List[SampleEmbedding],
    neighbors: Dict[int, List[Tuple[int, float]]],
    scenario: int
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run ambiguity analysis for a scenario."""
    sample_by_id = {s.sample_id: s for s in samples}
    param_mean, param_std = compute_param_statistics(samples)
    
    pairs_data = []
    for sample_id, neighbor_list in neighbors.items():
        sample_i = sample_by_id[sample_id]
        
        for neighbor_id, d_y in neighbor_list:
            sample_j = sample_by_id[neighbor_id]
            d_p = param_distance(sample_i.par, sample_j.par, param_mean, param_std)
            diffs = param_differences(sample_i.par, sample_j.par)
            
            row = {
                'i': sample_id,
                'j': neighbor_id,
                'd_y': d_y,
                'd_p': d_p,
                'file_i': str(sample_i.file_path.name),
                'file_j': str(sample_j.file_path.name),
                'n_curves_i': sample_i.n_curves,
                'n_curves_j': sample_j.n_curves,
                'has_hold_i': sample_i.has_hold,
                'has_hold_j': sample_j.has_hold,
            }
            for name, diff in diffs.items():
                row[f'delta_{name}'] = diff
            pairs_data.append(row)
    
    pairs_df = pd.DataFrame(pairs_data)
    
    scores_data = []
    for sample_id in neighbors.keys():
        sample = sample_by_id[sample_id]
        neighbor_list = neighbors[sample_id]
        
        d_y_list = [d for _, d in neighbor_list]
        d_p_list = [
            param_distance(sample.par, sample_by_id[nid].par, param_mean, param_std)
            for nid, _ in neighbor_list
        ]
        
        A_score = compute_ambiguity_score(d_y_list, d_p_list)
        
        scores_data.append({
            'sample_id': sample_id,
            'file_path': str(sample.file_path.name),
            'A_score': A_score,
            'median_d_y': np.median(d_y_list) if d_y_list else 0,
            'median_d_p': np.median(d_p_list) if d_p_list else 0,
            'n_curves': sample.n_curves,
            'has_hold': sample.has_hold
        })
    
    scores_df = pd.DataFrame(scores_data).sort_values('A_score', ascending=False)
    
    global_stats = compute_global_ambiguity_stats(pairs_df)
    
    summary = {
        'scenario': scenario,
        'n_samples': len(samples),
        'n_pairs_analyzed': len(pairs_df),
        'k_neighbors': len(neighbors.get(samples[0].sample_id, [])) if samples else 0,
        **global_stats,
        'top_ambiguous_samples': scores_df.head(10)[['sample_id', 'file_path', 'A_score']].to_dict('records')
    }
    
    return pairs_df, scores_df, summary


def save_outputs(
    pairs_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    summary: dict,
    scenario: int,
    output_dir: Path,
    group_label: Optional[str] = None
):
    """Save analysis outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{group_label}" if group_label else ""
    
    pairs_path = output_dir / f"pairs_s{scenario}{suffix}.csv"
    pairs_df.to_csv(pairs_path, index=False)
    print(f"  Saved {pairs_path}")
    
    scores_path = output_dir / f"ambiguity_scores_s{scenario}{suffix}.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"  Saved {scores_path}")
    
    summary_path = output_dir / f"summary_s{scenario}{suffix}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {summary_path}")


def create_visualizations(
    pairs_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    scenario: int,
    output_dir: Path,
    group_label: Optional[str] = None
):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Warning: matplotlib not available, skipping visualizations")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{group_label}" if group_label else ""
    
    dataset_map = {0: '0', 1: '1', 2: '2', 3: '3'}
    d_sub = dataset_map.get(scenario, str(scenario))
    plot_title = f"Ambiguity Analysis for Dataset $D_{d_sub}$"
    
    tau_curve = np.percentile(pairs_df['d_y'], 20)
    tau_param = np.percentile(pairs_df['d_p'], 80)
    
    ambiguous = (pairs_df['d_y'] < tau_curve) & (pairs_df['d_p'] > tau_param)
    n_total = len(pairs_df)
    n_ambiguous = ambiguous.sum()
    pct_ambiguous = n_ambiguous / n_total * 100
    
    stats_text_tl = (
        f"$\\tau_\\mathrm{{curve}}$: {tau_curve:.4f}\n"
        f"$\\tau_\\mathrm{{param}}$: {tau_param:.4f}"
    )
    stats_text_br = (
        f"Total pairs: {n_total:,}\n"
        f"Ambiguous pairs: {n_ambiguous:,} ({pct_ambiguous:.2f}%)"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pairs_df['d_y'], pairs_df['d_p'], 
               alpha=0.2, s=8, c='steelblue', label='Normal pairs', rasterized=True)
    
    if n_ambiguous > 0:
        ax.scatter(pairs_df.loc[ambiguous, 'd_y'], pairs_df.loc[ambiguous, 'd_p'],
                   alpha=0.8, s=25, c='red', marker='o', 
                   edgecolors='darkred', linewidths=0.5, label='Ambiguous pairs')
    
    ax.axvline(tau_curve, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhline(tau_param, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.text(0.02, 0.98, stats_text_tl, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left', bbox=props)
    ax.text(0.98, 0.02, stats_text_br, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    ax.set_xlabel(r"Observation Distance ($d_\mathrm{y}$)", fontsize=14)
    ax.set_ylabel(r"Parameter Distance ($d_\mathrm{p}$)", fontsize=14)
    ax.set_title(plot_title, fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    scatter_path = output_dir / f"scatter_dy_vs_dp_s{scenario}{suffix}.png"
    fig.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {scatter_path}")
    
    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores_df['A_score'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    
    pct_90 = scores_df['A_score'].quantile(0.9)
    ax.axvline(pct_90, color='red', linestyle='--', linewidth=2,
               label=f'90th percentile: {pct_90:.2f}')
    
    ax.set_xlabel('Ambiguity Score A(i)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(plot_title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    hist_path = output_dir / f"ambiguity_hist_s{scenario}{suffix}.png"
    fig.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {hist_path}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ambiguity Diagnostic Tool - Detect many-to-one mappings in material parameter identification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--output-dir', type=Path, default=Path('ambiguity_results'),
                        help='Output directory (default: ambiguity_results)')
    parser.add_argument('--k-neighbors', type=int, default=10,
                        help='Number of nearest neighbors (default: 10)')
    parser.add_argument('--candidate-multiplier', type=int, default=20,
                        help='Candidate pool multiplier for two-stage retrieval (default: 20)')
    parser.add_argument('--ramp-grid-size', type=int, default=200,
                        help='Grid size for ramp embedding (default: 200)')
    parser.add_argument('--hold-grid-size', type=int, default=100,
                        help='Grid size for hold embedding (default: 100)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to load (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Ambiguity Diagnostic Tool (Shape-Only)")
    print("=" * 60)
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"k-neighbors: {args.k_neighbors}")
    print()
    
    samples_by_scenario = load_and_group_samples(
        args.data_dir, args.max_files, args.seed
    )
    
    scenarios = [s for s in [0, 1, 2, 3] if len(samples_by_scenario[s]) > 0]
    
    print(f"\nAnalyzing scenarios: {scenarios}")
    
    for scenario in scenarios:
        samples = samples_by_scenario[scenario]
        
        if len(samples) < 2:
            print(f"\nSkipping scenario {scenario}: too few samples ({len(samples)})")
            continue
        
        print(f"\n{'='*60}")
        print(f"Scenario {scenario}: {len(samples)} samples")
        print('='*60)
        
        # Split by hold behavior
        hold_samples = [s for s in samples if s.has_hold]
        ramp_samples = [s for s in samples if not s.has_hold]
        groups = []
        if hold_samples:
            groups.append(('hold', hold_samples))
        if ramp_samples:
            groups.append(('ramp', ramp_samples))
        if not groups:
            groups = [(None, samples)]
        
        for group_label, group_samples in groups:
            if len(group_samples) < 2:
                print(f"\nSkipping subgroup {group_label}: too few samples ({len(group_samples)})")
                continue
            
            if group_label:
                print(f"\nSubgroup {group_label}: {len(group_samples)} samples")
            
            print("Embedding samples...")
            allow_hold = group_label == 'hold'
            force_hold_block = group_label == 'hold'
            
            embeddings = [
                embed_sample(
                    s, args.ramp_grid_size, args.hold_grid_size,
                    allow_hold=allow_hold,
                    force_hold_block=force_hold_block
                )
                for s in tqdm(group_samples, desc="  Embedding")
            ]
            
            print("Finding nearest neighbors...")
            neighbors = find_neighbors(
                embeddings, args.k_neighbors, args.candidate_multiplier
            )
            
            print("Computing ambiguity metrics...")
            pairs_df, scores_df, summary = run_ambiguity_analysis(
                group_samples, embeddings, neighbors, scenario
            )
            summary['group_label'] = group_label
            
            print("Saving outputs...")
            save_outputs(pairs_df, scores_df, summary, scenario, args.output_dir, group_label)
            
            print("Creating visualizations...")
            create_visualizations(pairs_df, scores_df, scenario, args.output_dir, group_label)
            
            label_suffix = f" ({group_label})" if group_label else ""
            print(f"\n  Summary for Scenario {scenario}{label_suffix}:")
            print(f"    Samples: {summary['n_samples']}")
            print(f"    Pairs analyzed: {summary['n_pairs_analyzed']}")
            print(f"    Ambiguity rate: {summary['ambiguity_rate_pct']:.2f}%")
            print(f"    Top 3 ambiguous samples:")
            for item in summary['top_ambiguous_samples'][:3]:
                print(f"      - {item['file_path']}: A={item['A_score']:.2f}")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print('='*60)


if __name__ == '__main__':
    main()
