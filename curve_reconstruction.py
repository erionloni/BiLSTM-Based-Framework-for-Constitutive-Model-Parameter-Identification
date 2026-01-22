#!/usr/bin/env python3
"""
Curve reconstruction workflow to predict material parameters, run the MATLAB forward model,
and compare true vs reconstructed curves.

Workflow:
1. Predict parameters from NPZ file(s)
2. Write `generated_parameters.txt` for MATLAB
3. Run `main_R_local_Empa_flex.m`
4. Convert MATLAB output to NPZ
5. Plot and summarize comparisons

Usage:
    python curve_reconstruction.py --model-dir path/to/model --npz-file path/to/sample.npz
    python curve_reconstruction.py --model-dir path/to/model --data-dir path/to/npz --n-random 5
    python curve_reconstruction.py --model-dir path/to/model --worst-csv path/to/worst_indices_unified.csv --split test --top-n 20
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from predict_material_parameters import (
    flatten_matlab_array,
    get_field,
    load_model,
    predict_parameters,
    predict_parameters_multi_curve,
)
from SA_matlab.plot_results import format_parameter_text, plot_results


def save_figure_with_svg(fig_or_none, save_path: Path, dpi: int = 800):
    """
    Save figure in both PNG and SVG formats.
    
    Args:
        fig_or_none: matplotlib Figure object or None (uses current figure)
        save_path: Path to save PNG file (SVG saved in 'svg' subdirectory)
        dpi: Resolution for PNG output (default 800 for publication quality)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    # Save SVG in svg subdirectory
    svg_dir = save_path.parent / 'svg'
    svg_dir.mkdir(parents=True, exist_ok=True)
    svg_path = svg_dir / save_path.with_suffix('.svg').name
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"  Saved: {save_path} and {svg_path}")

def compute_curve_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> Dict[str, float]:
    """
    Compute comparison metrics between true and predicted curve values.
    
    Args:
        true_vals: Ground truth array (1D)
        pred_vals: Predicted/reconstructed array (1D)
    
    Returns:
        Dictionary with metrics:
        - RMSE: Root Mean Square Error (absolute units)
        - NRMSE: Normalized RMSE (% of signal range)
        - MAE: Mean Absolute Error
        - MaxError: Maximum absolute deviation
        - R2: Coefficient of determination (1.0 = perfect fit)
    """
    # Handle length mismatch by interpolating to common length
    if len(true_vals) != len(pred_vals):
        common_len = min(len(true_vals), len(pred_vals))
        true_x = np.linspace(0, 1, len(true_vals))
        pred_x = np.linspace(0, 1, len(pred_vals))
        common_x = np.linspace(0, 1, common_len)
        true_vals = np.interp(common_x, true_x, true_vals)
        pred_vals = np.interp(common_x, pred_x, pred_vals)
    
    # Ensure 1D
    true_vals = np.array(true_vals).flatten()
    pred_vals = np.array(pred_vals).flatten()
    
    # RMSE
    mse = np.mean((true_vals - pred_vals) ** 2)
    rmse = np.sqrt(mse)
    
    # NRMSE (normalized by range)
    val_range = np.max(true_vals) - np.min(true_vals)
    nrmse = (rmse / val_range * 100) if val_range > 1e-12 else 0.0
    
    # MAE
    mae = np.mean(np.abs(true_vals - pred_vals))
    
    # Max Error
    max_error = np.max(np.abs(true_vals - pred_vals))
    
    # R² (coefficient of determination)
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
    
    return {
        'RMSE': float(rmse),
        'NRMSE': float(nrmse),
        'MAE': float(mae),
        'MaxError': float(max_error),
        'R2': float(r2)
    }


def compute_full_curve_metrics(true_data: dict, pred_data: dict, scenario: int) -> Dict[str, Any]:
    """
    Compute metrics for a full curve comparison (S11, F22, F33).
    
    For Scenario 3 (multi-curve), computes per-curve and aggregate metrics.
    
    Args:
        true_data: Dict with true curve arrays
        pred_data: Dict with predicted/reconstructed curve arrays
        scenario: Scenario number (0, 1, 2, or 3)
    
    Returns:
        Dictionary with:
        - metrics for S11, F22, F33
        - overall_NRMSE (mean of S11, F22, F33 NRMSEs)
        - For Scenario 3: per_curve metrics and aggregate
    """
    def extract_curves(mc):
        """Extract list of curve dicts from multi_curves array"""
        curves = []
        if hasattr(mc, 'ndim') and mc.ndim == 2:
            for i in range(mc.shape[1]):
                curve = mc[0, i]
                if hasattr(curve, 'dtype') and curve.dtype.names:
                    def get_val(c, name):
                        val = c[name]
                        while isinstance(val, np.ndarray):
                            if val.size == 1:
                                val = val.flat[0]
                            elif val.ndim > 1 and val.shape[0] == 1:
                                val = val[0]
                            else:
                                break
                        return val.flatten() if isinstance(val, np.ndarray) else val
                    curves.append({
                        'F11': get_val(curve, 'F11'), 'S11': get_val(curve, 'S11'),
                        'F22': get_val(curve, 'F22'), 'F33': get_val(curve, 'F33'),
                        't': get_val(curve, 't'),
                        'rate': get_val(curve, 'rate') if 'rate' in curve.dtype.names else np.nan
                    })
        elif hasattr(mc, 'ndim') and mc.ndim == 1:
            for curve in mc:
                if hasattr(curve, 'F11'):
                    curves.append({
                        'F11': np.array(curve.F11).flatten(), 'S11': np.array(curve.S11).flatten(),
                        'F22': np.array(curve.F22).flatten(), 'F33': np.array(curve.F33).flatten(),
                        't': np.array(curve.t).flatten(),
                        'rate': float(curve.rate) if hasattr(curve, 'rate') else np.nan
                    })
                elif hasattr(curve, 'dtype') and curve.dtype.names:
                    def get_val(c, name):
                        val = c[name]
                        while isinstance(val, np.ndarray):
                            if val.size == 1:
                                val = val.flat[0]
                            elif val.ndim > 1 and val.shape[0] == 1:
                                val = val[0]
                            else:
                                break
                        return val.flatten() if isinstance(val, np.ndarray) else val
                    curves.append({
                        'F11': get_val(curve, 'F11'), 'S11': get_val(curve, 'S11'),
                        'F22': get_val(curve, 'F22'), 'F33': get_val(curve, 'F33'),
                        't': get_val(curve, 't'),
                        'rate': get_val(curve, 'rate') if 'rate' in curve.dtype.names else np.nan
                    })
        return curves
    
    if scenario == 3:
        # Multi-curve scenario
        true_curves = extract_curves(true_data['multi_curves'])
        pred_curves = extract_curves(pred_data['multi_curves'])
        
        per_curve_metrics = []
        s11_nrmses, f22_nrmses, f33_nrmses = [], [], []
        
        for i, (tc, pc) in enumerate(zip(true_curves, pred_curves)):
            rate = tc.get('rate', i)
            if hasattr(rate, 'item'):
                rate = rate.item()
            
            s11_m = compute_curve_metrics(tc['S11'], pc['S11'])
            f22_m = compute_curve_metrics(tc['F22'], pc['F22'])
            f33_m = compute_curve_metrics(tc['F33'], pc['F33'])
            
            s11_nrmses.append(s11_m['NRMSE'])
            f22_nrmses.append(f22_m['NRMSE'])
            f33_nrmses.append(f33_m['NRMSE'])
            
            per_curve_metrics.append({
                'curve_index': i,
                'rate': float(rate) if not np.isnan(rate) else None,
                'S11': s11_m,
                'F22': f22_m,
                'F33': f33_m,
                'overall_NRMSE': float(np.mean([s11_m['NRMSE'], f22_m['NRMSE'], f33_m['NRMSE']]))
            })
        
        aggregate = {
            'mean_NRMSE_S11': float(np.mean(s11_nrmses)),
            'mean_NRMSE_F22': float(np.mean(f22_nrmses)),
            'mean_NRMSE_F33': float(np.mean(f33_nrmses)),
            'max_NRMSE_S11': float(np.max(s11_nrmses)),
            'std_NRMSE_S11': float(np.std(s11_nrmses)),
            'overall_NRMSE': float(np.mean(s11_nrmses + f22_nrmses + f33_nrmses)),
            'n_curves': len(true_curves)
        }
        
        return {
            'scenario': scenario,
            'aggregate': aggregate,
            'per_curve': per_curve_metrics
        }
    else:
        # Single curve scenarios (0, 1, 2)
        true_S11 = true_data['S11'].flatten()
        true_F22 = true_data['F22'].flatten()
        true_F33 = true_data['F33'].flatten()
        
        pred_S11 = pred_data['S11'].flatten()
        pred_F22 = pred_data['F22'].flatten()
        pred_F33 = pred_data['F33'].flatten()
        
        s11_m = compute_curve_metrics(true_S11, pred_S11)
        f22_m = compute_curve_metrics(true_F22, pred_F22)
        f33_m = compute_curve_metrics(true_F33, pred_F33)
        
        return {
            'scenario': scenario,
            'S11': s11_m,
            'F22': f22_m,
            'F33': f33_m,
            'overall_NRMSE': float(np.mean([s11_m['NRMSE'], f22_m['NRMSE'], f33_m['NRMSE']])),
            'aggregate': {
                'mean_NRMSE_S11': s11_m['NRMSE'],
                'mean_NRMSE_F22': f22_m['NRMSE'],
                'mean_NRMSE_F33': f33_m['NRMSE'],
                'overall_NRMSE': float(np.mean([s11_m['NRMSE'], f22_m['NRMSE'], f33_m['NRMSE']]))
            }
        }


def save_curve_metrics(metrics: Dict[str, Any], output_path: Path, file_name: str = None):
    """Save curve comparison metrics to JSON file."""
    if file_name:
        metrics['file'] = file_name
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def generate_folder_summaries(output_dir: Path):
    """
    Generate summary statistics for each split/type folder and cross-split comparison.
    
    Creates:
    - output_dir/<split>/<type>/summary.json for each folder with metrics
    - output_dir/cross_split_comparison.csv comparing all splits/types
    """
    import csv
    
    summary_rows = []
    
    # Find all split/type folders
    for split_dir in output_dir.iterdir():
        if not split_dir.is_dir() or split_dir.name.startswith('.'):
            continue
        
        for type_dir in split_dir.iterdir():
            if not type_dir.is_dir():
                continue
            
            # Collect all curve_metrics.json files in this folder
            metrics_files = list(type_dir.glob('*/curve_metrics.json'))
            if not metrics_files:
                continue
            
            all_metrics = []
            for mf in metrics_files:
                try:
                    with open(mf, 'r') as f:
                        m = json.load(f)
                        m['sample_dir'] = mf.parent.name
                        all_metrics.append(m)
                except Exception as e:
                    print(f"  Warning: Could not load {mf}: {e}")
            
            if not all_metrics:
                continue
            
            # Compute folder-level statistics
            nrmses_s11 = [m['aggregate']['mean_NRMSE_S11'] if 'aggregate' in m else m.get('S11', {}).get('NRMSE', 0) for m in all_metrics]
            nrmses_f22 = [m['aggregate']['mean_NRMSE_F22'] if 'aggregate' in m else m.get('F22', {}).get('NRMSE', 0) for m in all_metrics]
            nrmses_f33 = [m['aggregate']['mean_NRMSE_F33'] if 'aggregate' in m else m.get('F33', {}).get('NRMSE', 0) for m in all_metrics]
            overall_nrmses = [m['aggregate']['overall_NRMSE'] if 'aggregate' in m else m.get('overall_NRMSE', 0) for m in all_metrics]
            
            folder_summary = {
                'folder': f"{split_dir.name}/{type_dir.name}",
                'split': split_dir.name,
                'type': type_dir.name,
                'n_samples': len(all_metrics),
                'mean_NRMSE_S11': float(np.mean(nrmses_s11)),
                'std_NRMSE_S11': float(np.std(nrmses_s11)),
                'min_NRMSE_S11': float(np.min(nrmses_s11)),
                'max_NRMSE_S11': float(np.max(nrmses_s11)),
                'mean_NRMSE_F22': float(np.mean(nrmses_f22)),
                'mean_NRMSE_F33': float(np.mean(nrmses_f33)),
                'mean_overall_NRMSE': float(np.mean(overall_nrmses)),
                'std_overall_NRMSE': float(np.std(overall_nrmses)),
                'samples': [
                    {
                        'name': m['sample_dir'],
                        'file': m.get('file', ''),
                        'NRMSE_S11': m['aggregate']['mean_NRMSE_S11'] if 'aggregate' in m else m.get('S11', {}).get('NRMSE', 0),
                        'NRMSE_F22': m['aggregate']['mean_NRMSE_F22'] if 'aggregate' in m else m.get('F22', {}).get('NRMSE', 0),
                        'NRMSE_F33': m['aggregate']['mean_NRMSE_F33'] if 'aggregate' in m else m.get('F33', {}).get('NRMSE', 0),
                        'overall_NRMSE': m['aggregate']['overall_NRMSE'] if 'aggregate' in m else m.get('overall_NRMSE', 0)
                    }
                    for m in all_metrics
                ]
            }
            
            # Save folder summary
            summary_path = type_dir / 'summary.json'
            with open(summary_path, 'w') as f:
                json.dump(folder_summary, f, indent=2)
            print(f"  Saved folder summary: {summary_path}")
            
            # Add to cross-split comparison
            summary_rows.append({
                'split': split_dir.name,
                'type': type_dir.name,
                'n_samples': len(all_metrics),
                'mean_NRMSE_S11': float(np.mean(nrmses_s11)),
                'std_NRMSE_S11': float(np.std(nrmses_s11)),
                'mean_NRMSE_F22': float(np.mean(nrmses_f22)),
                'mean_NRMSE_F33': float(np.mean(nrmses_f33)),
                'mean_overall_NRMSE': float(np.mean(overall_nrmses)),
                'std_overall_NRMSE': float(np.std(overall_nrmses))
            })
    
    # Write cross-split comparison CSV
    if summary_rows:
        csv_path = output_dir / 'cross_split_comparison.csv'
        fieldnames = ['split', 'type', 'n_samples', 'mean_NRMSE_S11', 'std_NRMSE_S11', 
                      'mean_NRMSE_F22', 'mean_NRMSE_F33', 'mean_overall_NRMSE', 'std_overall_NRMSE']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Saved cross-split comparison: {csv_path}")


def resolve_npz_path(raw_path: str, repo_root: Path, csv_path_root: Optional[Path] = None) -> Path:
    """
    Resolve a NPZ file path coming from a CSV.

    The `worst_indices_unified.csv` often contains absolute Windows paths, which don't exist on Linux
    clusters. This tries a few fallbacks so the same CSV can be re-used on the cluster.
    """
    path = Path(raw_path)
    if path.exists():
        return path

    normalized = raw_path.replace('\\', '/')
    normalized_lower = normalized.lower()

    # If the CSV path contains ".../SA_matlab/...", treat everything from SA_matlab as repo-relative.
    token = 'sa_matlab/'
    idx = normalized_lower.find(token)
    if idx != -1:
        rel = normalized[idx:]  # keep original casing

        if csv_path_root is not None:
            candidate = csv_path_root / rel
            if candidate.exists():
                return candidate

        candidate = repo_root / rel
        if candidate.exists():
            return candidate

    # If it's already relative, optionally resolve it relative to csv_path_root.
    if csv_path_root is not None:
        looks_like_windows_abs = len(normalized) >= 3 and normalized[1:3] == ':/'
        looks_like_posix_abs = normalized.startswith('/')
        if not looks_like_windows_abs and not looks_like_posix_abs:
            candidate = csv_path_root / normalized
            if candidate.exists():
                return candidate

    return path


def build_generated_parameters_row(pred_par, scenario, hold_time, multi_rates):
    """
    Build one row in the format expected by main_R_local_Empa_flex.m
    
    Column order (from main_R_local_Empa_flex.m):
    1: lambda1_max, 2: theta, 3: q, 4: m1, 5: m2, 6: m3, 7: m4, 8: m5, 
    9: kM, 10: alphaM, 11: kF, 12: lambda1_dot, 13: scenario,
    14: n_segments, 15-20: seg params, 21: hold_time, 22: multi_rate_count, 23+: rates
    
    pred_par indices: 0=kM, 1=kF, 2=mu0, 3=m3, 4=m5, 5=q, 6=m1, 7=m2, 8=m4, 9=theta, 10=alphaM, 11=rate, 12=lambda_max
    """
    row = []
    row.append(pred_par[12])  # col 1: lambda1_max
    row.append(pred_par[9])   # col 2: theta
    row.append(pred_par[5])   # col 3: q
    row.append(pred_par[6])   # col 4: m1
    row.append(pred_par[7])   # col 5: m2
    row.append(pred_par[3])   # col 6: m3
    row.append(pred_par[8])   # col 7: m4
    row.append(pred_par[4])   # col 8: m5
    row.append(pred_par[0])   # col 9: kM
    row.append(pred_par[10])  # col 10: alphaM
    row.append(pred_par[1])   # col 11: kF
    row.append(pred_par[11])  # col 12: lambda1_dot
    row.append(scenario)      # col 13: scenario_type
    row.append(0)             # col 14: n_segments
    row.extend([0] * 6)       # cols 15-20: seg params
    row.append(hold_time)     # col 21: hold_time
    
    if multi_rates is not None and len(multi_rates) > 0:
        row.append(len(multi_rates))  # col 22: multi_rate_count
        row.extend(multi_rates)       # col 23+: rates
    else:
        row.append(0)

    return row


def write_generated_parameters_txt(pred_par, scenario, hold_time, multi_rates, output_path):
    """Write a single parameter row to disk for main_R_local_Empa_flex.m."""
    rows = [build_generated_parameters_row(pred_par, scenario, hold_time, multi_rates)]
    write_generated_parameters_txt_rows(rows, output_path)


def write_generated_parameters_txt_rows(rows, output_path):
    """Write one or more parameter rows (padding to a consistent column count)."""
    if not rows:
        raise ValueError("rows must be non-empty")

    max_len = max(len(r) for r in rows)
    with open(output_path, 'w') as f:
        for r in rows:
            if len(r) < max_len:
                r = list(r) + [0.0] * (max_len - len(r))
            f.write(','.join(f'{float(x):.12e}' for x in r) + '\n')


def run_matlab_main(matlab_dir, start_idx=1, end_idx=1, timeout_s=600, param_file='generated_parameters.txt'):
    """
    Run main_R_local_Empa_flex.m which reads generated_parameters.txt
    and outputs to matlab_data/data_matlab_augmented_case_<idx>.mat.
    """
    print("  Running MATLAB main_R_local_Empa_flex.m...")
    
    # Set environment variables for batch processing
    cmd = ['matlab', '-singleCompThread', '-batch',
           f"cd('{str(matlab_dir).replace(chr(92), '/')}'); "
           f"setenv('PARAM_FILE', '{param_file}'); "
           f"main_R_local_Empa_flex"]
    
    print(f"  Debug: MATLAB dir = {matlab_dir}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(matlab_dir))
        if result.returncode != 0:
            print(f"  MATLAB error (code {result.returncode}):")
            print(f"    {result.stderr if result.stderr else result.stdout}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  MATLAB timed out after {timeout_s}s")
        return False
    except Exception as e:
        print(f"  MATLAB error: {e}")
        return False


def convert_mat_to_npz(mat_file, npz_file):
    """Convert MATLAB output MAT file to NPZ for plot_results.py"""
    import scipy.io as sio
    
    try:
        mat_data = sio.loadmat(str(mat_file))
    except Exception as e:
        print(f"  Error loading MAT file: {e}")
        return False
    
    # Copy all data fields
    npz_data = {}
    for key in mat_data:
        if not key.startswith('_'):
            npz_data[key] = mat_data[key]
    
    np.savez(npz_file, **npz_data)
    return True


def resolve_matlab_timeout_s(arg_value, n_cases):
    """
    Resolve the MATLAB timeout:
      - None: choose default (600 single-case, max(600, 60*n_cases) batch)
      - <= 0: disable timeout
      - else: use provided seconds
    """
    if arg_value is None:
        if n_cases <= 1:
            return 600
        return max(600, 60 * n_cases)
    if arg_value <= 0:
        return None
    return arg_value


def plot_comparison(true_npz_file, pred_npz_file, output_file, true_par, pred_par, predicted_names=None):
    """
    Generates comparison plot with A4-compact styling (MPa, Black Table, L2 vs T).
    Material Box: Symbol and Err% only.
    """
    try:
        true_data = np.load(true_npz_file, allow_pickle=True)
        pred_data = np.load(pred_npz_file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading NPZ files: {e}")
        return {}
        
    def extract_curves_local(data, multi_curves):
        curves = []
        if multi_curves is not None:
            mc = np.atleast_1d(multi_curves)
            if mc.ndim == 2: mc = mc.flatten()
            elif mc.ndim > 2: mc = mc.ravel()
            for i in range(len(mc)):
                c = mc[i]
                cd = {}
                for k in ['F11', 'S11', 'F22', 'F33', 't', 'rate']:
                    try:
                        val = get_field(c, k)
                        if isinstance(val, np.ndarray): val = val.flatten()
                        cd[k] = val
                    except: pass
                curves.append(cd)
        else:
            cd = {}
            for k in ['F11', 'S11', 'F22', 'F33', 't']:
                 val = data.get(k, None)
                 if val is not None: cd[k] = np.array(val).flatten()
            curves.append(cd)
        return curves

    scenario = true_data.get('scenario', 0)
    if hasattr(scenario, 'item'): scenario = scenario.item()
    
    true_multi = true_data.get('multi_curves', None)
    pred_multi = pred_data.get('multi_curves', None)
    true_curves = extract_curves_local(true_data, true_multi)
    pred_curves = extract_curves_local(pred_data, pred_multi)
    
    curve_metrics = compute_full_curve_metrics(true_data, pred_data, scenario)
    
    # Plot Setup
    fig = plt.figure(figsize=(13, 8)) 
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.35, hspace=0.35)
    
    axs = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    ])
    ax_params = fig.add_subplot(gs[:, 3])
    ax_params.axis('off')
    
    curve_colors = ['blue', 'red', 'green', 'magenta']
    rate_labels = []
    
    if scenario == 3:
        rates = []
        for tc in true_curves:
            r = tc.get('rate', np.nan)
            if hasattr(r, 'item'): r = r.item()
            rates.append(r)
        unique_rates = sorted(list(set([r for r in rates if not np.isnan(r)])))
        rate_to_color = {r: curve_colors[i % len(curve_colors)] for i, r in enumerate(unique_rates)}
        
        plot_items = []
        n_curves = min(len(true_curves), len(pred_curves))
        for i in range(n_curves):
            tc = true_curves[i]
            pc = pred_curves[i]
            rate = tc.get('rate', np.nan)
            if hasattr(rate, 'item'): rate = rate.item()
            color = rate_to_color.get(rate, 'black') if not np.isnan(rate) else 'black'
            rate_labels.append((color, rate, i))
            plot_items.append((tc, pc, color))
    else:
        if len(true_curves)>0 and len(pred_curves)>0:
            plot_items = [(true_curves[0], pred_curves[0], 'blue')]
        else: plot_items = []
    
    if scenario == 3:
        for tc, pc, color in plot_items:
            # Plots (Pa units for S11/P11)
            axs[0, 0].plot(tc['t'], tc['F11'], color=color, ls='-', lw=1.5, alpha=0.8)
            axs[0, 0].plot(pc['t'], pc['F11'], color=color, ls='--', lw=1.5, alpha=0.8)
            axs[0, 1].plot(tc['F11'], tc['S11'], color=color, ls='-', lw=1.5, alpha=0.8)
            axs[0, 1].plot(pc['F11'], pc['S11'], color=color, ls='--', lw=1.5, alpha=0.8)
            axs[0, 2].plot(tc['F11'], tc['F22'], color=color, ls='-', lw=1.5, alpha=0.8)
            axs[0, 2].plot(pc['F11'], pc['F22'], color=color, ls='--', lw=1.5, alpha=0.8)
            axs[1, 0].plot(tc['F11'], tc['F33'], color=color, ls='-', lw=1.5, alpha=0.8)
            axs[1, 0].plot(pc['F11'], pc['F33'], color=color, ls='--', lw=1.5, alpha=0.8)
            axs[1, 1].plot(tc['t'], tc['S11'], color=color, ls='-', lw=1.5, alpha=0.8)
            axs[1, 1].plot(pc['t'], pc['S11'], color=color, ls='--', lw=1.5, alpha=0.8)
            axs[1, 2].plot(tc['t'], tc['F22'], color=color, ls='-', lw=1.5, alpha=0.8)
            axs[1, 2].plot(pc['t'], pc['F22'], color=color, ls='--', lw=1.5, alpha=0.8)
    else:
        # Scenario 0/1/2 - Specific colors (Black/Blue/Red/Green)
        for tc, pc, _ in plot_items:
             # 1. L1 vs Time (Black)
            axs[0, 0].plot(tc['t'], tc['F11'], color='black', ls='-', lw=1.5, alpha=0.8)
            axs[0, 0].plot(pc['t'], pc['F11'], color='black', ls='--', lw=1.5, alpha=0.8)
            
            # 2. L1 vs P11 (Blue)
            axs[0, 1].plot(tc['F11'], tc['S11'], color='blue', ls='-', lw=1.5, alpha=0.8)
            axs[0, 1].plot(pc['F11'], pc['S11'], color='blue', ls='--', lw=1.5, alpha=0.8)
            
            # 3. L1 vs L2 (Red)
            axs[0, 2].plot(tc['F11'], tc['F22'], color='red', ls='-', lw=1.5, alpha=0.8)
            axs[0, 2].plot(pc['F11'], pc['F22'], color='red', ls='--', lw=1.5, alpha=0.8)
            
            # 4. L1 vs L3 (Green)
            axs[1, 0].plot(tc['F11'], tc['F33'], color='green', ls='-', lw=1.5, alpha=0.8)
            axs[1, 0].plot(pc['F11'], pc['F33'], color='green', ls='--', lw=1.5, alpha=0.8)
            
            # 5. P11 vs Time (Blue)
            axs[1, 1].plot(tc['t'], tc['S11'], color='blue', ls='-', lw=1.5, alpha=0.8)
            axs[1, 1].plot(pc['t'], pc['S11'], color='blue', ls='--', lw=1.5, alpha=0.8)
            
            # 6. L2 & L3 vs Time (Red & Green)
            axs[1, 2].plot(tc['t'], tc['F22'], color='red', ls='-', lw=1.5, alpha=0.8, label=r'$\lambda_2$')
            axs[1, 2].plot(pc['t'], pc['F22'], color='red', ls='--', lw=1.5, alpha=0.8)
            axs[1, 2].plot(tc['t'], tc['F33'], color='green', ls='-', lw=1.5, alpha=0.8, label=r'$\lambda_3$')
            axs[1, 2].plot(pc['t'], pc['F33'], color='green', ls='--', lw=1.5, alpha=0.8)
            axs[1, 2].legend(loc='best', fontsize=8, frameon=False)
            
            # Scenario 1: Vertical line for hold start
            if scenario == 1:
                hv = true_data.get('hold_time', 0)
                if hasattr(hv, 'item'): hv = hv.item()
                # Use True curve time to determine start
                t_plot = tc['t']
                if hv > 0 and len(t_plot) > 0:
                    t_hold_start = t_plot[-1] - hv
                    # 1. L1 vs Time
                    axs[0, 0].axvline(x=t_hold_start, color='lightgrey', linestyle='--', zorder=0)
                    # 5. P11 vs Time
                    axs[1, 1].axvline(x=t_hold_start, color='lightgrey', linestyle='--', zorder=0)
                    # 6. L2/L3 vs Time
                    axs[1, 2].axvline(x=t_hold_start, color='lightgrey', linestyle='--', zorder=0)
            
            # Scenario 2: Vertical lines for segment transitions
            if scenario == 2:
                seg_rates = true_data.get('segment_rates', None)
                if seg_rates is not None:
                    seg_rates_arr = np.atleast_1d(seg_rates).flatten()
                    # Count all non-zero rates (including artifact at index 0)
                    all_seg_rates = [r for r in seg_rates_arr if r > 0]
                    total_seg = len(all_seg_rates)
                    # Displayed segments (excluding first/artifact)
                    n_displayed = len([r for r in seg_rates_arr[1:] if r > 0])
                    F11_true = tc['F11']
                    t_true = tc['t']
                    if n_displayed > 1 and len(t_true) > 0 and len(F11_true) > 0:
                        # Divide by total segments (including artifact)
                        lambda_start = F11_true[0]
                        lambda_end = F11_true[-1]
                        delta_lambda = (lambda_end - lambda_start) / total_seg
                        # Draw lines starting after segment 2 (index 1), which is the first displayed
                        # Line 1: between displayed seg 1 and 2 (= after total seg 2)
                        for seg_idx in range(2, total_seg):
                            target_lambda = lambda_start + seg_idx * delta_lambda
                            idx = np.searchsorted(F11_true, target_lambda)
                            if 0 < idx < len(t_true):
                                t_transition = t_true[idx]
                                axs[0, 0].axvline(x=t_transition, color='lightgrey', linestyle='--', zorder=0)
                                axs[1, 1].axvline(x=t_transition, color='lightgrey', linestyle='--', zorder=0)
                                axs[1, 2].axvline(x=t_transition, color='lightgrey', linestyle='--', zorder=0)

    axs[0, 0].set_xlabel('Time [s]'); axs[0, 0].set_ylabel(r'In-plane stretch $\lambda_1$ [-]'); axs[0, 0].set_title(r'$\lambda_1$ vs Time')
    axs[0, 1].set_xlabel(r'In-plane stretch $\lambda_1$ [-]'); axs[0, 1].set_ylabel(r'1st PK stress $P_{11}$ [MPa]'); axs[0, 1].set_title(r'$\lambda_1$ vs $P_{11}$')
    axs[0, 2].set_xlabel(r'In-plane stretch $\lambda_1$ [-]'); axs[0, 2].set_ylabel(r'Lateral stretch $\lambda_2$ [-]'); axs[0, 2].set_title(r'$\lambda_1$ vs $\lambda_2$')
    axs[1, 0].set_xlabel(r'In-plane stretch $\lambda_1$ [-]'); axs[1, 0].set_ylabel(r'Thickness stretch $\lambda_3$ [-]'); axs[1, 0].set_title(r'$\lambda_1$ vs $\lambda_3$')
    axs[1, 1].set_xlabel('Time [s]'); axs[1, 1].set_ylabel(r'1st PK stress $P_{11}$ [MPa]'); axs[1, 1].set_title(r'$P_{11}$ vs Time')
    axs[1, 2].set_xlabel('Time [s]'); axs[1, 2].set_ylabel(r'Lateral stretch $\lambda_2$ [-]'); axs[1, 2].set_title(r'$\lambda_2$ vs Time')
    for ax in axs.flatten(): 
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=False, prune=None, nbins=5))
    fig.suptitle(f'Scenario {scenario} - Curve Reconstruction', fontsize=14, fontweight='bold', y=0.98)

    # --- PARAMETER PANEL (Simplified) ---
    param_info = [
        (0,  r'$\bar{k}_M$',    r'$\mathrm{s}^{-1}$', 'kM'),
        (10, r'$\alpha_M$',     '', 'alphaM'),
        (1,  r'$\bar{k}_F$',    r'$\mathrm{s}^{-1}$', 'kF'),
        (6,  r'$m_1$',          '', 'm1'),
        (7,  r'$m_2$',          '', 'm2'),
        (4,  r'$m_5$',          '', 'm5'),
        (3,  r'$\bar{m}_3$',    '', 'm3'),
        (8,  r'$m_4$',          '', 'm4'),
        (5,  r'$q$',            '', 'q'),
        (9,  r'$\vartheta$',    'rad', 'theta'),
    ]
    if predicted_names is None:
        predicted_names = ['kM', 'kF', 'm3', 'm5', 'q', 'm1', 'm2', 'm4', 'theta', 'alphaM']
    n_material = len(param_info)

    protocol_params = []
    if len(true_par) >= 13:
        lambda_max = true_par[12]
        protocol_params.append((r'$\lambda_{\max}$', f'{lambda_max:.3f}', ''))
    
    unique_rates_list = []
    if scenario == 3:
        unique_rates_list = sorted(list(set([r for c, r, i in rate_labels if not np.isnan(r)])))
        protocol_params.append((r'$N_{\mathrm{rate}}$', str(len(unique_rates_list)), ''))
        hold_val = true_data.get('hold_time', None)
        if hold_val is not None:
             if hasattr(hold_val, 'item'): hold_val = hold_val.item()
             if hold_val > 0: protocol_params.append((r'$t_{\mathrm{hold}}$', f'{hold_val:.1f}', 's'))
    elif scenario == 2:
        # Scenario 2: Multi-segment piecewise ramp
        seg_rates = true_data.get('segment_rates', None)
        if seg_rates is not None:
            seg_rates = np.atleast_1d(seg_rates).flatten()
            # Skip first value (numerical artifact) and filter zeros
            seg_rates = [r for r in seg_rates[1:] if r > 0]
            n_seg = len(seg_rates)
            if n_seg > 0:
                rates_str = '{' + ', '.join([f'{r:.3f}' for r in seg_rates]) + '}'
                protocol_params.append((r'$\{\dot{\lambda}_{i}\}_{i=1}^{N_{\mathrm{seg}}}$', rates_str, r'$\mathrm{s}^{-1}$'))
                protocol_params.append((r'$N_{\mathrm{seg}}$', str(n_seg), ''))
        else:
            # Fallback: use single rate
            if len(true_par) >= 12:
                rate = true_par[11]
                rate_str = f"{rate:.4f}" if abs(rate) >= 0.001 and abs(rate) <= 1000 else f"{rate:.2e}"
                protocol_params.append((r'$\dot{\lambda}_1$', rate_str, r'$\mathrm{s}^{-1}$'))
        # NO hold time for Scenario 2
    else:
        # Scenario 0/1: single rate
        if len(true_par) >= 12:
            rate = true_par[11]
            rate_str = f"{rate:.4f}" if abs(rate) >= 0.001 and abs(rate) <= 1000 else f"{rate:.2e}"
            protocol_params.append((r'$\dot{\lambda}_1$', rate_str, r'$\mathrm{s}^{-1}$'))
        # Hold time only for Scenario 1
        if scenario == 1:
            hold_val = true_data.get('hold_time', None)
            if hold_val is not None:
                 if hasattr(hold_val, 'item'): hold_val = hold_val.item()
                 if hold_val > 0: protocol_params.append((r'$t_{\mathrm{hold}}$', f'{hold_val:.1f}', 's'))

    n_protocol = len(protocol_params)
    n_rates = len(unique_rates_list) if scenario == 3 else 0

    row_height = 0.035
    header_height = 0.05
    title_height = 0.04
    gap = 0.02
    rates_section_height = (0.04 + n_rates * 0.03) if n_rates > 0 else 0
    material_box_height = title_height + header_height + n_material * row_height + 0.02
    protocol_box_height = title_height + header_height + n_protocol * row_height + rates_section_height + 0.02
    metrics_box_height = 0.08
    
    material_top = 0.98
    material_bottom = material_top - material_box_height
    protocol_top = material_bottom - gap
    protocol_bottom = protocol_top - protocol_box_height
    metrics_top = protocol_bottom - gap
    metrics_bottom = metrics_top - metrics_box_height
    
    # 1. Material (Simplified: Symbol & Error only)
    ax_params.add_patch(plt.Rectangle((0.0, material_bottom), 1.0, material_box_height, transform=ax_params.transAxes, facecolor='#E8E8E8', edgecolor='gray', alpha=0.95, linewidth=1, zorder=1))
    
    y = material_top - 0.025
    ax_params.text(0.5, y, 'Material Parameters', transform=ax_params.transAxes, fontsize=10, fontweight='bold', ha='center', va='center', zorder=2)
    y -= 0.04
    
    # Simpler headers: Symbol (0.3), Err% (0.7)
    ax_params.text(0.3, y, 'Symbol', transform=ax_params.transAxes, fontsize=9, fontweight='bold', ha='center', zorder=2)
    ax_params.text(0.7, y, 'Err%', transform=ax_params.transAxes, fontsize=9, fontweight='bold', ha='center', zorder=2)
    ax_params.plot([0.1, 0.9], [y-0.015, y-0.015], color='gray', linewidth=0.5, transform=ax_params.transAxes, zorder=2)
    
    y_start = y - 0.035
    for i, (idx, sym, unit, key) in enumerate(param_info):
        y_pos = y_start - i*row_height
        tv = true_par[idx] if idx < len(true_par) else 0
        pv = pred_par[idx] if idx < len(pred_par) else 0
        
        err_str = "—"
        if key in predicted_names and abs(tv)>1e-12:
            err = abs(pv-tv)/abs(tv)*100
            err_str = f"{err:.1f}%"
            
        ax_params.text(0.3, y_pos, sym, transform=ax_params.transAxes, fontsize=9, ha='center', va='center', zorder=2)
        ax_params.text(0.7, y_pos, err_str, transform=ax_params.transAxes, fontsize=9, ha='center', va='center', color='black', zorder=2)

    # 2. Protocol (Same as before)
    ax_params.add_patch(plt.Rectangle((0.0, protocol_bottom), 1.0, protocol_box_height, transform=ax_params.transAxes, facecolor='#E8E8E8', edgecolor='gray', alpha=0.95, linewidth=1, zorder=1))
    y = protocol_top - 0.025
    ax_params.text(0.5, y, 'Protocol Parameters', transform=ax_params.transAxes, fontsize=10, fontweight='bold', ha='center', va='center', zorder=2)
    y -= 0.04
    ax_params.text(0.20, y, 'Symbol', transform=ax_params.transAxes, fontsize=9, fontweight='bold', ha='center', zorder=2)
    ax_params.text(0.55, y, 'Value', transform=ax_params.transAxes, fontsize=9, fontweight='bold', ha='center', zorder=2)
    ax_params.text(0.88, y, 'Unit', transform=ax_params.transAxes, fontsize=9, fontweight='bold', ha='center', zorder=2)
    ax_params.plot([0.05, 0.95], [y-0.015, y-0.015], color='gray', linewidth=0.5, transform=ax_params.transAxes, zorder=2)
    
    y_start = y - 0.035
    for i, (sym, val, unit) in enumerate(protocol_params):
        y_pos = y_start - i*row_height
        ax_params.text(0.20, y_pos, sym, transform=ax_params.transAxes, fontsize=9, ha='center', va='center', zorder=2)
        ax_params.text(0.55, y_pos, val, transform=ax_params.transAxes, fontsize=9, ha='center', va='center', zorder=2)
        ax_params.text(0.88, y_pos, unit, transform=ax_params.transAxes, fontsize=9, ha='center', va='center', zorder=2)
        
    if n_rates > 0:
        y_rates = y_start - n_protocol * row_height - 0.02
        ax_params.text(0.5, y_rates, 'Applied Rates:', transform=ax_params.transAxes, fontsize=9, fontweight='bold', ha='center', va='top', zorder=2)
        for i, r in enumerate(unique_rates_list):
            color = rate_to_color.get(r, 'black')
            ax_params.text(0.5, y_rates - 0.03 - i*0.03, f'{r:.2e} /s', transform=ax_params.transAxes, fontsize=8, ha='center', va='top', color=color, fontweight='bold', zorder=2)

    # 3. Metrics (Same as before)
    ax_params.add_patch(plt.Rectangle((0.0, metrics_bottom), 1.0, metrics_box_height, transform=ax_params.transAxes, facecolor='#E8E8E8', edgecolor='gray', alpha=0.95, linewidth=1, zorder=1))
    agg = curve_metrics.get('aggregate', curve_metrics)
    overall_nrmse = agg.get('overall_NRMSE', curve_metrics.get('overall_NRMSE', 0))
    ax_params.text(0.5, metrics_top - 0.04, f"Overall NRMSE: {overall_nrmse:.2f}%", transform=ax_params.transAxes, fontsize=11, fontweight='bold', ha='center', va='center', zorder=2)

    save_figure_with_svg(fig, output_file, dpi=600)
    plt.close(fig)
    return curve_metrics


def plot_per_rate_comparisons(true_npz_file, pred_npz_file, base_output_file, true_par, pred_par, predicted_names=None):
    """
    Generate individual comparison plots for each strain rate in Scenario 3.
    Each plot shows one rate's true vs predicted curves.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.lines import Line2D
    
    # Load both NPZ files
    true_data = np.load(true_npz_file, allow_pickle=True)
    pred_data = np.load(pred_npz_file, allow_pickle=True)
    
    # Use formatted text logic similar to plot_comparison
    name_to_symbol = {
        'kM': 'kM', 'kF': 'kF', 'm3': 'm3', 'm5': 'm5', 'q': 'q', 
        'm1': 'm1', 'm2': 'm2', 'm4': 'm4', 'theta': 'θ', 'alphaM': 'αM'
    }
    all_param_indices = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    
    if predicted_names is None:
        predicted_names = ['kM', 'kF', 'm3', 'm5', 'q', 'm1', 'm2', 'm4', 'theta', 'alphaM']
    
    lines = []
    lines.append("═" * 32)
    lines.append("   Parameter Comparison")
    lines.append("   (P)=Predicted, (T)=True")
    lines.append("═" * 32)
    lines.append(f"{'Param':<8} {'True':>9} {'Pred':>9} {'Err%':>5}")
    lines.append("─" * 36)
    
    for name_key, idx in zip(['kM', 'kF', 'm3', 'm5', 'q', 'm1', 'm2', 'm4', 'theta', 'alphaM'], all_param_indices):
        symbol = name_to_symbol.get(name_key, name_key)
        is_predicted = name_key in predicted_names
        marker = "(P)" if is_predicted else "(T)"
        label = f"{symbol} {marker}"
        
        true_val = true_par[idx]
        pred_val = pred_par[idx]
        
        if is_predicted and abs(true_val) > 1e-12:
            err = abs(pred_val - true_val) / abs(true_val) * 100
        else:
            err = 0.0
            
        err_str = f"{err:>5.1f}%" if is_predicted else " --- "
        lines.append(f"{label:<8} {true_val:>9.2e} {pred_val:>9.2e} {err_str}")
    
    lines.append("═" * 32)
    param_text = "\n".join(lines)
    
    output_dir = Path(base_output_file).parent
    
    # Extract curves using the same logic as plot_comparison
    def extract_curves(mc):
        curves = []
        if hasattr(mc, 'ndim') and mc.ndim == 2:
            for i in range(mc.shape[1]):
                curve = mc[0, i]
                if hasattr(curve, 'dtype') and curve.dtype.names:
                    def get_val(c, name):
                        val = c[name]
                        while isinstance(val, np.ndarray):
                            if val.size == 1:
                                val = val.flat[0]
                            elif val.ndim > 1 and val.shape[0] == 1:
                                val = val[0]
                            else:
                                break
                        return val.flatten() if isinstance(val, np.ndarray) else val
                    curves.append({
                        'F11': get_val(curve, 'F11'), 'S11': get_val(curve, 'S11'),
                        'F22': get_val(curve, 'F22'), 'F33': get_val(curve, 'F33'),
                        't': get_val(curve, 't'),
                        'rate': get_val(curve, 'rate') if 'rate' in curve.dtype.names else np.nan
                    })
        elif hasattr(mc, 'ndim') and mc.ndim == 1:
            for curve in mc:
                if hasattr(curve, 'F11'):
                    curves.append({
                        'F11': np.array(curve.F11).flatten(), 'S11': np.array(curve.S11).flatten(),
                        'F22': np.array(curve.F22).flatten(), 'F33': np.array(curve.F33).flatten(),
                        't': np.array(curve.t).flatten(),
                        'rate': float(curve.rate) if hasattr(curve, 'rate') else np.nan
                    })
        return curves
    
    true_curves = extract_curves(true_data['multi_curves'])
    pred_curves = extract_curves(pred_data['multi_curves'])
    
    output_dir = Path(base_output_file).parent
    
    # Parameter comparison text (same for all plots)
    param_names = ['kM', 'kF', 'm3', 'm5', 'q', 'm1', 'm2', 'm4', 'θ', 'αM']
    param_indices = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    
    text_lines = []
    text_lines.append("═" * 26)
    text_lines.append("   Parameter Comparison")
    text_lines.append("═" * 26)
    text_lines.append(f"{'Param':<6} {'True':>10} {'Pred':>10} {'Err%':>7}")
    text_lines.append("─" * 36)
    
    for name, idx in zip(param_names, param_indices):
        true_val = true_par[idx]
        pred_val = pred_par[idx]
        err = abs(pred_val - true_val) / abs(true_val) * 100 if abs(true_val) > 1e-12 else 0
        text_lines.append(f"{name:<6} {true_val:>10.3e} {pred_val:>10.3e} {err:>6.1f}%")
    
    text_lines.append("═" * 26)
    param_text = '\n'.join(text_lines)
    
    # Generate one plot for each rate
    for i, (tc, pc) in enumerate(zip(true_curves, pred_curves)):
        rate = tc.get('rate', i)
        if hasattr(rate, 'item'):
            rate = rate.item()
        
        # Create figure with parameter panel
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.35], wspace=0.3, hspace=0.3)
        
        axs = np.array([
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
        ])
        
        ax_params = fig.add_subplot(gs[:, 3])
        ax_params.axis('off')
        
        # Plot this rate's curves (blue=true, red=predicted)
        axs[0, 0].plot(tc['F11'], tc['S11'], 'b-', linewidth=2, label='True')
        axs[0, 0].plot(pc['F11'], pc['S11'], 'r--', linewidth=2, label='Predicted')
        
        axs[0, 1].plot(tc['F11'], tc['F22'], 'b-', linewidth=2)
        axs[0, 1].plot(pc['F11'], pc['F22'], 'r--', linewidth=2)
        
        axs[0, 2].plot(tc['F11'], tc['F33'], 'b-', linewidth=2)
        axs[0, 2].plot(pc['F11'], pc['F33'], 'r--', linewidth=2)
        
        axs[1, 0].plot(tc['t'], tc['F11'], 'b-', linewidth=2)
        axs[1, 0].plot(pc['t'], pc['F11'], 'r--', linewidth=2)
        
        axs[1, 1].plot(tc['t'], tc['S11'], 'b-', linewidth=2)
        axs[1, 1].plot(pc['t'], pc['S11'], 'r--', linewidth=2)
        
        axs[1, 2].plot(tc['t'], tc['F22'], 'b-', linewidth=2)
        axs[1, 2].plot(pc['t'], pc['F22'], 'r--', linewidth=2)
        
        # Labels
        axs[0, 0].set_xlabel(r'$F_{11}$'); axs[0, 0].set_ylabel(r'$S_{11}$ [MPa]'); axs[0, 0].set_title(r'$S_{11}$ vs $F_{11}$')
        axs[0, 1].set_xlabel(r'$F_{11}$'); axs[0, 1].set_ylabel(r'$F_{22}$'); axs[0, 1].set_title(r'$F_{22}$ vs $F_{11}$')
        axs[0, 2].set_xlabel(r'$F_{11}$'); axs[0, 2].set_ylabel(r'$F_{33}$'); axs[0, 2].set_title(r'$F_{33}$ vs $F_{11}$')
        axs[1, 0].set_xlabel('Time [s]'); axs[1, 0].set_ylabel(r'$F_{11}$'); axs[1, 0].set_title(r'$F_{11}$ vs Time')
        axs[1, 1].set_xlabel('Time [s]'); axs[1, 1].set_ylabel(r'$S_{11}$ [MPa]'); axs[1, 1].set_title(r'$S_{11}$ vs Time')
        axs[1, 2].set_xlabel('Time [s]'); axs[1, 2].set_ylabel(r'$F_{22}$'); axs[1, 2].set_title(r'$F_{22}$ vs Time')
        
        for ax in axs.flat:
            ax.grid(True, alpha=0.3)
        
        # Legend
        axs[0, 0].legend(loc='upper left', fontsize=10)
        
        # Parameter panel
        ax_params.text(0.05, 0.95, param_text, transform=ax_params.transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                                edgecolor='gray', alpha=0.9))
        
        # Rate info
        rate_str = f'{rate:.4f} /s' if not np.isnan(rate) else f'Curve {i+1}'
        rate_info = f"═══════════════════════\n  Rate {i+1}: {rate_str}\n═══════════════════════"
        ax_params.text(0.05, 0.35, rate_info, transform=ax_params.transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', 
                                edgecolor='teal', alpha=0.9))
        
        # Title
        fig.suptitle(f'True vs Predicted Curves - Rate {i+1} ({rate_str})', fontsize=14, fontweight='bold')
        
        # Save
        output_file = output_dir / f'comparison_rate_{i+1}.png'
        save_figure_with_svg(None, output_file)
        plt.close()
    
    print(f"  Per-rate plots saved: {len(true_curves)} files (comparison_rate_1.png, ...)")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Curve reconstruction workflow")
    parser.add_argument('--model-dir', required=True, help='Directory with best_model.pth')
    parser.add_argument('--npz-file', help='Single NPZ file to analyze')
    parser.add_argument('--data-dir', help='Directory with NPZ files (batch mode)')
    parser.add_argument('--n-random', type=int, default=1, help='Number of random files (batch mode)')
    parser.add_argument('--splits', nargs='+', default=['test'], help='Splits (ignored)')
    parser.add_argument('--output-dir', default='curve_reconstruction_results', help='Output directory')
    parser.add_argument('--skip-matlab', action='store_true', help='Skip MATLAB simulation')
    parser.add_argument('--skip-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--matlab-batch', action='store_true',
                        help='Run MATLAB once for all selected samples (recommended for --worst-csv / --data-dir)')
    parser.add_argument('--sample-mapping', type=str, help='Path to sample_mapping.csv for dataset evaluation')
    parser.add_argument('--matlab-timeout', type=int, default=None,
                        help="MATLAB timeout in seconds (default: 600 single-case, max(600, 60*#cases) batch). Use 0 to disable.")
    parser.add_argument('--csv-path-root', default=None,
                        help="Optional root directory to resolve CSV file paths (useful on clusters).")
    parser.add_argument('--csv-offset', type=int, default=0,
                        help="Skip the first N rows of the selected CSV subset (after --split / --top-n / --reconstruct-all-csv).")
    parser.add_argument('--csv-limit', type=int, default=None,
                        help="Only process the next N rows of the selected CSV subset (after --csv-offset).")
    parser.add_argument('--worst-csv', help='Path to worst_indices_unified.csv from training')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all',
                        help='Which split to reconstruct from CSV (default: all)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Reconstruct top N worst from CSV (default: 10)')
    parser.add_argument('--reconstruct-all-csv', action='store_true',
                        help='Reconstruct ALL samples from the worst-csv file (ignores --top-n)')
    parser.add_argument('--job-id', type=str, default=None,
                        help='Unique job ID for parallel execution (prevents parameter file conflicts)')
    parser.add_argument('--dpi', type=int, default=600,
                        help='DPI for plots (passed recursively by dataset evaluation mode)')
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.sample_mapping:
        if not args.data_dir:
            print("Error: --data-dir is required when using --sample-mapping")
            return 1
        
        
        model_dir = Path(args.model_dir) if args.model_dir else Path('.')
        output_dir = Path(args.output_dir) if args.output_dir else Path('results')
        
        evaluate_dataset_mode(model_dir, args.sample_mapping, args.data_dir, output_dir, device)
        return 0

    
    # Validate arguments
    options_count = sum([
        args.npz_file is not None,
        args.data_dir is not None,
        args.worst_csv is not None
    ])
    if options_count == 0:
        parser.error("One of --npz-file, --data-dir, or --worst-csv is required")
    

    
    model_dir = Path(args.model_dir)
    matlab_dir = Path(__file__).parent / 'SA_matlab'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path_root = Path(args.csv_path_root) if args.csv_path_root else None
    
    # Job isolation for parallel execution
    # When --job-id is specified, use unique parameter files and output directories
    # to prevent race conditions between parallel jobs
    job_id = args.job_id
    if job_id is None:
        # Auto-detect from environment (SLURM array task)
        slurm_array_task = os.environ.get('SLURM_ARRAY_TASK_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_array_task:
            job_id = f"{slurm_job_id}_{slurm_array_task}"
        elif slurm_job_id:
            job_id = slurm_job_id
    
    if job_id:
        print(f"Job isolation enabled: job_id={job_id}")
        # Create job-specific subdirectory for MATLAB files
        job_matlab_dir = matlab_dir / f'job_{job_id}'
        job_matlab_dir.mkdir(parents=True, exist_ok=True)
        (job_matlab_dir / 'matlab_data').mkdir(parents=True, exist_ok=True)
        
        # Symlink all .m files to the job directory (MATLAB needs them)
        for m_file in matlab_dir.glob('*.m'):
            target = job_matlab_dir / m_file.name
            if not target.exists():
                try:
                    target.symlink_to(m_file)
                except OSError:
                    # Fallback: copy if symlink not supported (Windows)
                    import shutil
                    shutil.copy2(m_file, target)
        
        matlab_dir = job_matlab_dir
        param_file_name = 'generated_parameters.txt'
    else:
        param_file_name = 'generated_parameters.txt'
    
    # Get files to process based on input mode
    # Store tuples of (path, split, type)
    files_info = []
    
    if args.worst_csv:
        # Load worst indices CSV and extract file paths
        import csv

        rows = []
        with open(args.worst_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if args.split != 'all' and row.get('split') != args.split:
                    continue
                rows.append(row)

        if not args.reconstruct_all_csv:
            rows = rows[:args.top_n]

        if args.csv_offset:
            rows = rows[args.csv_offset:]

        if args.csv_limit is not None:
            rows = rows[:args.csv_limit]

        for row in rows:
            file_path = resolve_npz_path(row['file_path'], repo_root=Path(__file__).parent, csv_path_root=csv_path_root)
            if file_path.exists():
                sample_type = row.get('type', 'unknown')
                split = row.get('split', 'unknown')
                files_info.append((file_path, split, sample_type))
            else:
                print(f"  Warning: File not found: {file_path}")
        
        print(f"Selected {len(files_info)} file(s) from {args.worst_csv}")
        if args.reconstruct_all_csv:
            print("  Reconstructing ALL samples from CSV.")
        else:
            print(f"  Split filter: {args.split}, Top-N: {args.top_n}")
        if args.csv_offset or args.csv_limit is not None:
            print(f"  CSV slice: offset={args.csv_offset}, limit={args.csv_limit}")
            
    elif args.npz_file:
        files_info = [(Path(args.npz_file), None, None)]
    else:
        data_dir = Path(args.data_dir)
        all_files = sorted(data_dir.glob('*.npz'))
        selected_files = random.sample(all_files, min(args.n_random, len(all_files)))
        files_info = [(f, None, None) for f in selected_files]
        print(f"Selected {len(files_info)} file(s) from {data_dir}")
    
    # Load model once
    print("Loading model...")
    result = load_model(model_dir / 'best_model.pth', device)
    (model, norm_params, use_time, use_rate, target_params, param_indices, max_seq_length,
     use_cell_state, mlp_layers, mlp_activation, use_log_targets, param_use_log,
     normalize_inputs, input_norm_params, use_log_inputs, s11_epsilon, normalize_s11_max,
     aggregation, use_scenario, use_physics_features) = result

    use_matlab_batch = (
        args.matlab_batch
        and (not args.skip_matlab)
        and len(files_info) > 1
    )
    queued_jobs = []
    
    for npz_file, split, sample_type in files_info:
        print(f"\n{'='*60}")
        print(f"Processing: {npz_file.name}")
        if split and sample_type:
             print(f"Type: {sample_type} ({split})")
        print(f"{'='*60}")
        
        # Determine output folder structure
        if split and sample_type:
             # e.g. output_dir/train/best/case_name
             sample_output_dir = output_dir / split / sample_type / npz_file.stem
        else:
             # e.g. output_dir/case_name
             sample_output_dir = output_dir / npz_file.stem
             
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load NPZ
        with np.load(npz_file, allow_pickle=True) as data:
            true_data = {k: data[k] for k in data.files}
        
        scenario_val = true_data.get('scenario', 0)
        scenario = int(scenario_val.item()) if hasattr(scenario_val, 'item') else int(scenario_val)
        true_par = flatten_matlab_array(true_data['par'])
        
        # Get hold_time - may need to calculate for Scenario 1 or 3
        hold_time_val = true_data.get('hold_time', 0)
        hold_time = float(hold_time_val.item()) if hasattr(hold_time_val, 'item') else float(hold_time_val)
        
        if hold_time == 0 and scenario == 1 and 't' in true_data and 'F11' in true_data:
            # Calculate hold time from data for Scenario 1 (ramp+hold)
            t = flatten_matlab_array(true_data['t'])
            F11 = flatten_matlab_array(true_data['F11'])
            F11_max = F11.max()
            # Find where F11 first reaches ~99.9% of max (hold phase start)
            hold_start_idx = np.argmax(F11 > F11_max * 0.999)
            if hold_start_idx > 0 and hold_start_idx < len(t) - 1:
                hold_time = t.max() - t[hold_start_idx]
                print(f"  Calculated hold_time from Scenario 1 data: {hold_time:.1f} s")
        
        elif hold_time == 0 and scenario == 3 and 'multi_curves' in true_data:
            # Calculate hold time from Scenario 3 multi_curves (detect plateau in F11)
            curves = true_data['multi_curves'].flatten()  # MATLAB exports as (1, N)
            hold_times_inferred = []
            for c in curves:
                try:
                    F11_c = get_field(c, 'F11')  # get_field now auto-flattens
                    t_c = get_field(c, 't')
                    F11_max_c = F11_c.max()
                    # Find hold start: first point at ~max where there are multiple such points
                    hold_mask = np.abs(F11_c - F11_max_c) < 1e-6
                    if np.sum(hold_mask) > 1:
                        hold_start_idx = np.argmax(hold_mask)
                        hold_time_c = t_c[-1] - t_c[hold_start_idx]
                        if hold_time_c > 0.5:  # minimum 0.5s to count as real hold
                            hold_times_inferred.append(hold_time_c)
                except Exception:
                    pass
            if hold_times_inferred:
                hold_time = float(np.median(hold_times_inferred))
                if np.max(hold_times_inferred) - np.min(hold_times_inferred) > 2.0:
                    print(f"  Warning: Inconsistent hold times across curves (range: {np.min(hold_times_inferred):.1f}-{np.max(hold_times_inferred):.1f}s)")
                print(f"  Calculated hold_time from Scenario 3 data: {hold_time:.1f} s")
        
        multi_rates = true_data.get('multi_rates', None)
        if multi_rates is not None:
            multi_rates = flatten_matlab_array(multi_rates).tolist()
        
        print(f"Scenario: {scenario}")
        print("\nTRUE PARAMETERS:")
        print(format_parameter_text(true_par))
        
        # Predict
        if scenario == 3:
            curves = true_data.get('multi_curves', np.array([])).flatten()  # MATLAB exports as (1, N)
            curve_list = [{
                'S11': get_field(c, 'S11'),  # get_field now auto-flattens
                'F11': get_field(c, 'F11'),
                'F22': get_field(c, 'F22'),
                'F33': get_field(c, 'F33'),
                't': get_field(c, 't')
            } for c in curves]
            pred_params = predict_parameters_multi_curve(
                model=model, curves=curve_list, norm_params=norm_params,
                use_time=use_time, use_rate=use_rate, device=device,
                max_seq_length=max_seq_length, use_log_targets=use_log_targets,
                param_use_log=param_use_log, normalize_inputs=normalize_inputs,
                input_norm_params=input_norm_params if isinstance(input_norm_params, dict) else None,
                use_log_inputs=use_log_inputs, s11_epsilon=s11_epsilon,
                normalize_s11_max=normalize_s11_max, use_scenario=use_scenario,
                scenario=scenario, use_physics_features=use_physics_features
            )
        else:
            pred_params = predict_parameters(
                model=model,
                S11=flatten_matlab_array(true_data['S11']), F11=flatten_matlab_array(true_data['F11']),
                F22=flatten_matlab_array(true_data['F22']), F33=flatten_matlab_array(true_data['F33']),
                t=flatten_matlab_array(true_data['t']) if 't' in true_data else None,
                norm_params=norm_params, use_time=use_time, use_rate=use_rate, device=device,
                max_seq_length=max_seq_length, use_log_targets=use_log_targets,
                param_use_log=param_use_log, normalize_inputs=normalize_inputs,
                input_norm_params=input_norm_params if isinstance(input_norm_params, dict) else None,
                use_log_inputs=use_log_inputs, s11_epsilon=s11_epsilon,
                normalize_s11_max=normalize_s11_max, use_scenario=use_scenario,
                scenario=scenario, use_physics_features=use_physics_features
            )
        
        # Build predicted par
        pred_par = true_par.copy()
        for i, idx in enumerate(param_indices):
            if i < len(pred_params) and idx < len(pred_par):
                pred_par[idx] = pred_params[i]
        
        print("\nPREDICTED PARAMETERS:")
        print(format_parameter_text(pred_par))
        
        print("\nRELATIVE ERRORS (Predicted Parameters only):")
        for name, idx in zip(target_params, param_indices):
            if abs(true_par[idx]) > 1e-12:
                err = abs(pred_par[idx] - true_par[idx]) / abs(true_par[idx]) * 100
                print(f"  {name:8s}: {err:8.1f}%")
        
        if args.skip_matlab:
            continue

        if use_matlab_batch:
            queued_jobs.append({
                'npz_file': npz_file,
                'sample_output_dir': sample_output_dir,
                'scenario': scenario,
                'hold_time': hold_time,
                'multi_rates': multi_rates,
                'true_par': true_par,
                'pred_par': pred_par,
            })
            print(f"\n  Queued for batch MATLAB reconstruction (index {len(queued_jobs)})")
            continue
        
        # Write predicted params to generated_parameters.txt
        param_file = matlab_dir / 'generated_parameters.txt'
        write_generated_parameters_txt(pred_par, scenario, hold_time, multi_rates, param_file)
        print(f"\n  Wrote: {param_file}")
        
        # Run main_R_local_Empa_flex.m
        matlab_timeout_s = resolve_matlab_timeout_s(args.matlab_timeout, 1)
        if not run_matlab_main(matlab_dir, start_idx=1, end_idx=1, timeout_s=matlab_timeout_s):
            print("  MATLAB failed!")
            # Cleanup param file even on failure
            if param_file.exists():
                try:
                     param_file.unlink()
                except:
                     pass
            continue
        
        # Cleanup generated_parameters.txt immediately after MATLAB run
        if param_file.exists():
            try:
                param_file.unlink()
            except Exception as e:
                print(f"  Warning: could not delete {param_file}: {e}")
        
        # Find the MAT output
        mat_file = matlab_dir / 'matlab_data' / 'data_matlab_augmented_case_1.mat'
        if not mat_file.exists():
            print(f"  MATLAB output not found: {mat_file}")
            continue
        
        # Convert MAT to NPZ
        pred_npz = sample_output_dir / 'predicted_curves.npz'
        if convert_mat_to_npz(mat_file, pred_npz):
            print(f"  Saved: {pred_npz}")
            try:
                mat_file.unlink()
            except Exception as e:
                print(f"  Warning: could not delete {mat_file}: {e}")
        else:
            # Cleanup even if conversion failed
            if mat_file.exists():
                try:
                    mat_file.unlink()
                except:
                    pass
            continue
        
        # Plot
        if not args.skip_plot:
            print("\n  Plotting predicted curves...")
            plot_results(str(pred_npz), show_plot=False)
            
            # Create comparison plot and compute metrics
            print("\n  Creating comparison plot (True vs Predicted)...")
            comparison_file = sample_output_dir / 'comparison_true_vs_predicted.png'
            curve_metrics = plot_comparison(str(npz_file), str(pred_npz), str(comparison_file), true_par, pred_par, target_params)
            
            # Save curve metrics to JSON
            metrics_file = sample_output_dir / 'curve_metrics.json'
            save_curve_metrics(curve_metrics, metrics_file, npz_file.name)
            print(f"  Saved curve metrics: {metrics_file}")
        
        print(f"\nDone: {npz_file.name}")

    if use_matlab_batch and queued_jobs:
        print(f"\n{'='*60}")
        print(f"Running batch MATLAB reconstruction for {len(queued_jobs)} case(s)...")
        print(f"{'='*60}")

        param_file = matlab_dir / 'generated_parameters.txt'
        param_rows = [
            build_generated_parameters_row(j['pred_par'], j['scenario'], j['hold_time'], j['multi_rates'])
            for j in queued_jobs
        ]
        write_generated_parameters_txt_rows(param_rows, param_file)
        print(f"\n  Wrote: {param_file}")

        # Remove any stale output files for these indices (prevents mixing old results on partial failures)
        for i in range(1, len(queued_jobs) + 1):
            stale = matlab_dir / 'matlab_data' / f'data_matlab_augmented_case_{i}.mat'
            if stale.exists():
                try:
                    stale.unlink()
                except Exception:
                    pass

        matlab_timeout_s = resolve_matlab_timeout_s(args.matlab_timeout, len(queued_jobs))
        matlab_ok = run_matlab_main(matlab_dir, start_idx=1, end_idx=len(queued_jobs), timeout_s=matlab_timeout_s)
        if not matlab_ok:
            # MATLAB may have completed but hung on exit (zombie), so continue to check for output files
            print("  MATLAB returned error/timeout - checking for partial results...")

        for i, job in enumerate(queued_jobs, start=1):
            npz_file = job['npz_file']
            sample_output_dir = job['sample_output_dir']
            true_par = job['true_par']
            pred_par = job['pred_par']

            mat_file = matlab_dir / 'matlab_data' / f'data_matlab_augmented_case_{i}.mat'
            if not mat_file.exists():
                print(f"  MATLAB output not found for {npz_file.name}: {mat_file}")
                continue

            pred_npz = sample_output_dir / 'predicted_curves.npz'
            if convert_mat_to_npz(mat_file, pred_npz):
                print(f"  Saved: {pred_npz}")
            else:
                continue

            if not args.skip_plot:
                print(f"\n  Plotting predicted curves for {npz_file.name}...")
                plot_results(str(pred_npz), show_plot=False)

                print(f"\n  Creating comparison plot for {npz_file.name}...")
                comparison_file = sample_output_dir / 'comparison_true_vs_predicted.png'
                curve_metrics = plot_comparison(str(npz_file), str(pred_npz), str(comparison_file), true_par, pred_par, target_params)
                
                # Save curve metrics to JSON
                metrics_file = sample_output_dir / 'curve_metrics.json'
                save_curve_metrics(curve_metrics, metrics_file, npz_file.name)
                print(f"  Saved curve metrics: {metrics_file}")
    
    # Generate folder summaries if we processed multiple files with CSV input
    if args.worst_csv and not args.skip_matlab and not args.skip_plot:
        print("\n" + "="*60)
        print("Generating folder summaries...")
        print("="*60)
        generate_folder_summaries(output_dir)
    
    print("\n" + "="*60)
    print("All files processed!")
    print("="*60)
    
    # Cleanup job-specific directory if used
    if job_id and matlab_dir.name.startswith('job_'):
        import shutil
        try:
            shutil.rmtree(matlab_dir)
            print(f"Cleaned up job directory: {matlab_dir}")
        except Exception as e:
            print(f"Warning: Could not cleanup job directory {matlab_dir}: {e}")

    return 0


def evaluate_dataset_mode(model_dir, mapping_file, data_dir, output_dir, device):
    """
    Evaluates model on Train/Val/Test sets defined in sample_mapping.csv.
    Computes Mean Parameter NRMSE and Mean Relative Error per parameter.
    Also runs full curve reconstruction for 15 random test samples using subprocess.
    """
    import pandas as pd
    from tqdm import tqdm
    import subprocess
    import random
    import torch
    
    print(f"\n=== Dataset Evaluation Mode ===")
    print(f"Mapping: {mapping_file}")
    
    # Load Model to get normalization params and configuration
    print("Loading model for metrics calculation...")
    result = load_model(model_dir / 'best_model.pth', device)
    (model, norm_params, use_time, use_rate, target_params, param_indices, max_seq_length,
     use_cell_state, mlp_layers, mlp_activation, use_log_targets, param_use_log,
     normalize_inputs, input_norm_params, use_log_inputs, s11_epsilon, normalize_s11_max,
     aggregation, use_scenario, use_physics_features) = result

    df = pd.read_csv(mapping_file)
    splits = ['train', 'val', 'test']
    
    metrics_report = ["=== Dataset Evaluation Metrics ==="]
    test_recon_files = []
    
    data_dir_path = Path(data_dir)
    
    for split in splits:
        subset = df[df['split'] == split]
        if len(subset) == 0: continue
        
        print(f"\nProcessing {split.upper()} set ({len(subset)} samples)...")
        
        Y_true_list = []
        Y_pred_list = []
        
        valid_count = 0
        for idx, row in tqdm(subset.iterrows(), total=len(subset)):
            orig_path = row['file_path']
            fname = Path(orig_path).name
            local_path = data_dir_path / fname
            
            if not local_path.exists():
                continue
                
            try:
                # Load NPZ
                with np.load(local_path, allow_pickle=True) as data:
                    true_data = {k: data[k] for k in data.files}
                
                # Extract True Params
                true_par = flatten_matlab_array(true_data['par'])
                scenario_val = true_data.get('scenario', 0)
                scenario = int(scenario_val.item()) if hasattr(scenario_val, 'item') else int(scenario_val)
                
                # Predict (Reuse prediction logic)
                if scenario == 3:
                     curves = true_data.get('multi_curves', np.array([])).flatten()
                     curve_list = [{
                        'S11': get_field(c, 'S11'),
                        'F11': get_field(c, 'F11'),
                        'F22': get_field(c, 'F22'),
                        'F33': get_field(c, 'F33'),
                        't': get_field(c, 't')
                     } for c in curves]
                     pred_params_raw = predict_parameters_multi_curve(
                        model=model, curves=curve_list, norm_params=norm_params,
                        use_time=use_time, use_rate=use_rate, device=device,
                        max_seq_length=max_seq_length, use_log_targets=use_log_targets,
                        param_use_log=param_use_log, normalize_inputs=normalize_inputs,
                        input_norm_params=input_norm_params if isinstance(input_norm_params, dict) else None,
                        use_log_inputs=use_log_inputs, s11_epsilon=s11_epsilon,
                        normalize_s11_max=normalize_s11_max, use_scenario=use_scenario,
                        scenario=scenario, use_physics_features=use_physics_features
                     )
                else:
                     pred_params_raw = predict_parameters(
                        model=model,
                        S11=flatten_matlab_array(true_data['S11']), F11=flatten_matlab_array(true_data['F11']),
                        F22=flatten_matlab_array(true_data['F22']), F33=flatten_matlab_array(true_data['F33']),
                        t=flatten_matlab_array(true_data['t']) if 't' in true_data else None,
                        norm_params=norm_params, use_time=use_time, use_rate=use_rate, device=device,
                        max_seq_length=max_seq_length, use_log_targets=use_log_targets,
                        param_use_log=param_use_log, normalize_inputs=normalize_inputs,
                        input_norm_params=input_norm_params if isinstance(input_norm_params, dict) else None,
                        use_log_inputs=use_log_inputs, s11_epsilon=s11_epsilon,
                        normalize_s11_max=normalize_s11_max, use_scenario=use_scenario,
                        scenario=scenario, use_physics_features=use_physics_features
                     )

                # Construct pred vector
                pred_par = true_par.copy()
                for i, p_idx in enumerate(param_indices):
                    if i < len(pred_params_raw) and p_idx < len(pred_par):
                        pred_par[p_idx] = pred_params_raw[i]
                
                Y_true_list.append(true_par)
                Y_pred_list.append(pred_par)
                
                if split == 'test':
                    test_recon_files.append(str(local_path))
                
                valid_count += 1
                
            except Exception:
                pass
        
        if valid_count > 0:
            Y_true = np.array(Y_true_list)
            Y_pred = np.array(Y_pred_list)
            
            # Subselect target params
            Y_true_sub = Y_true[:, param_indices]
            Y_pred_sub = Y_pred[:, param_indices]
            
            # 1. Parameter NRMSE (Normalized by Range)
            ranges = Y_true_sub.max(axis=0) - Y_true_sub.min(axis=0)
            ranges[ranges < 1e-6] = 1.0 # Avoid div by zero
            
            norm_diff = (Y_pred_sub - Y_true_sub) / ranges
            nrmse_per_sample = np.sqrt(np.mean(norm_diff**2, axis=1))
            
            mean_nrmse = np.mean(nrmse_per_sample)
            std_nrmse = np.std(nrmse_per_sample)
            
            metrics_report.append(f"\n[{split.upper()}]")
            metrics_report.append(f"  Mean Parameter NRMSE: {mean_nrmse:.6f} +/- {std_nrmse:.6f}")
            
            # 2. Relative Errors
            eps = 1e-8
            rel_errors = np.abs(Y_pred_sub - Y_true_sub) / (np.abs(Y_true_sub) + eps) * 100
            mean_rel_errors = np.mean(rel_errors, axis=0)
            
            metrics_report.append("  Mean Relative Errors:")
            for name, err in zip(target_params, mean_rel_errors):
                 metrics_report.append(f"    {name:<8}: {err:.4f}%")

    print("\n" + "="*40)
    print("\n".join(metrics_report))
    print("="*40)
    
    # Save Report
    with open(output_dir / "dataset_eval_metrics.txt", "w") as f:
        f.write("\n".join(metrics_report))
        
    # 3. 15 Random Reconstructions
    if test_recon_files:
        print("\n" + "="*60)
        print("Running reconstruction for 500 random Test files...")
        print("="*60)
        subset = random.sample(test_recon_files, min(500, len(test_recon_files)))
        
        # Use subprocess to call this script in single-file mode
        # This ensures exact same plotting logic is used
        script_path = sys.argv[0]
        
        for idx, npz_path in enumerate(subset):
            print(f"Reconstructing [{idx+1}/{len(subset)}]: {Path(npz_path).name}")
            try:
                # We save output to output_dir/test_reconstructions/case_name
                case_name = Path(npz_path).stem
                out_path = output_dir / "test_reconstructions" / case_name
                out_path.mkdir(parents=True, exist_ok=True)
                
                cmd = [
                    sys.executable, script_path,
                    "--model-dir", str(model_dir),
                    "--npz-file", str(npz_path),
                    "--output-dir", str(out_path.parent), # Logic in main appends case name
                    "--matlab-timeout", "600",
                    "--dpi", "300" # Reduced to 300 to save space/time, was 800
                ]

                
                # Execute
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                
            except subprocess.CalledProcessError as e:
                print(f"  Failed reconstruction for {npz_path}")
                print(f"  STDOUT: {e.stdout}")
                print(f"  STDERR: {e.stderr}")
        
        # Aggregate curve NRMSE from reconstructions
        curve_nrmse_values = []
        for npz_path in subset:
            case_name = Path(npz_path).stem
            metrics_file = output_dir / "test_reconstructions" / case_name / "curve_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        cm = json.load(f)
                    # overall_NRMSE may be in root or under 'aggregate'
                    if 'aggregate' in cm:
                        nrmse = cm['aggregate'].get('overall_NRMSE', cm.get('overall_NRMSE'))
                    else:
                        nrmse = cm.get('overall_NRMSE')
                    if nrmse is not None:
                        curve_nrmse_values.append(float(nrmse))
                except Exception:
                    pass
        
        if curve_nrmse_values:
            mean_curve_nrmse = np.mean(curve_nrmse_values)
            std_curve_nrmse = np.std(curve_nrmse_values)
            curve_report = f"\n[TEST CURVE METRICS (from {len(curve_nrmse_values)} reconstructions)]\n"
            curve_report += f"  Mean Curve NRMSE: {mean_curve_nrmse:.4f}% +/- {std_curve_nrmse:.4f}%"
            print(curve_report)
            
            # Append to saved report
            with open(output_dir / "dataset_eval_metrics.txt", "a") as f:
                f.write("\n" + curve_report)


if __name__ == '__main__':
    raise SystemExit(main())
