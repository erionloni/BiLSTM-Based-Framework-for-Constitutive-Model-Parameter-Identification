#!/usr/bin/env python3
"""
Plot simulation results from a MATLAB-exported NPZ file.

Usage:
    python SA_matlab/plot_results.py path/to/case.npz
    python SA_matlab/plot_results.py path/to/case.npz --no-show
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def _flatten_matlab_value(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        while isinstance(value, np.ndarray):
            if value.size == 1:
                value = value.flat[0]
            elif value.ndim > 1 and value.shape[0] == 1:
                value = value[0]
            else:
                break
        if isinstance(value, np.ndarray):
            return value.flatten()
    return value


def _get_matlab_field(obj, key: str):
    try:
        val = obj[key]
    except Exception:
        return None
    return _flatten_matlab_value(val)


def format_parameter_text(par: Sequence[float]) -> List[Tuple[str, str, str]]:
    """Format parameter vector into (symbol, value, unit) rows."""
    par = np.array(par).flatten()
    param_info = [
        (0, r'$\bar{k}_M$', r'$\mathrm{s}^{-1}$'),
        (10, r'$\alpha_M$', ''),
        (1, r'$\bar{k}_F$', r'$\mathrm{s}^{-1}$'),
        (6, r'$m_1$', ''),
        (7, r'$m_2$', ''),
        (4, r'$m_5$', ''),
        (3, r'$\bar{m}_3$', ''),
        (8, r'$m_4$', ''),
        (5, r'$q$', ''),
        (9, r'$\vartheta$', 'rad'),
    ]
    result = []
    for idx, symbol, unit in param_info:
        if idx < len(par):
            val = par[idx]
            if val == 0:
                val_str = "0"
            elif abs(val) < 0.001 or abs(val) > 1000:
                val_str = f"{val:.2e}"
            elif abs(val) < 1:
                val_str = f"{val:.4f}"
            else:
                val_str = f"{val:.3f}"
            result.append((symbol, val_str, unit))
    return result


def _extract_curves(data: dict, scenario: int) -> List[Dict[str, np.ndarray]]:
    curves_data = []
    if scenario == 3:
        multi_curves = data.get('multi_curves', None)
        if multi_curves is not None:
            mc = np.atleast_1d(multi_curves)
            if mc.ndim == 2:
                mc = mc.flatten()
            elif mc.ndim > 2:
                mc = mc.ravel()

            for curve in mc:
                cd: Dict[str, np.ndarray] = {}
                try:
                    for key in ['F11', 'S11', 'F22', 'F33', 't', 'rate']:
                        val = _get_matlab_field(curve, key)
                        if val is None and hasattr(curve, key):
                            val = getattr(curve, key)
                        if val is not None:
                            cd[key] = val
                except Exception:
                    pass

                if 'F11' in cd:
                    curves_data.append(cd)
    else:
        cd = {}
        for key in ['F11', 'S11', 'F22', 'F33', 't']:
            val = data.get(key, None)
            if val is not None:
                cd[key] = np.array(val).flatten()
        if 'F11' in cd:
            curves_data.append(cd)

    return curves_data


def plot_results(npz_file: str, show_plot: bool = True) -> None:
    """Plot results in A4 Compact style (13x8, 2x3 grid, MPa)."""
    npz_path = Path(npz_file)
    if not npz_path.exists():
        print(f"Error: File not found: {npz_file}")
        return

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        print(f"Error loading {npz_file}: {exc}")
        return

    scenario = data.get('scenario', 0)
    if hasattr(scenario, 'item'): scenario = scenario.item()

    par = data.get('par', None)
    if par is not None:
        if isinstance(par, np.ndarray) and par.ndim == 0: par = par.item()
        par = np.array(par).flatten()
    has_params = (par is not None)

    print(f"Loaded {npz_file}")
    print(f"Scenario: {scenario}")
    
    # Setup Figure (A4 Compact)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.35, hspace=0.35)
    
    axs = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    ])
    
    ax_params = fig.add_subplot(gs[:, 3])
    ax_params.axis('off')

    curves_data = _extract_curves(data, scenario)

    # Colors
    curve_colors = ['blue', 'red', 'green', 'magenta']
    rates = []
    for c in curves_data:
        r = c.get('rate', np.nan)
        if hasattr(r, 'item'):
            r = r.item()
        rates.append(float(r))
    unique_rates = sorted(list(set([r for r in rates if not np.isnan(r)])))
    rate_to_color = {r: curve_colors[i % len(curve_colors)] for i, r in enumerate(unique_rates)}

    # Plot
    for i, c_data in enumerate(curves_data):
        rate = c_data.get('rate', np.nan)
        if hasattr(rate, 'item'):
            rate = rate.item()
        color = rate_to_color.get(rate, 'blue') if not np.isnan(rate) else 'blue'
        
        t = c_data.get('t')
        F11 = c_data.get('F11')
        S11 = c_data.get('S11') # Pa
        F22 = c_data.get('F22')
        F33 = c_data.get('F33')
        
        if t is None or F11 is None:
            continue

        if scenario == 3:
            # Multi-rate color logic (per rate)
            # 1. L1 vs Time
            axs[0, 0].plot(t, F11, color=color, ls='-', lw=1.5, alpha=0.8)
            # 2. L1 vs P11 (Pa)
            if S11 is not None:
                axs[0, 1].plot(F11, S11, color=color, ls='-', lw=1.5, alpha=0.8)
            # 3. L1 vs L2
            if F22 is not None:
                axs[0, 2].plot(F11, F22, color=color, ls='-', lw=1.5, alpha=0.8)
            # 4. L1 vs L3
            if F33 is not None:
                axs[1, 0].plot(F11, F33, color=color, ls='-', lw=1.5, alpha=0.8)
            # 5. P11 vs Time (Pa)
            if S11 is not None:
                axs[1, 1].plot(t, S11, color=color, ls='-', lw=1.5, alpha=0.8)
            # 6. L2 vs Time
            if F22 is not None:
                axs[1, 2].plot(t, F22, color=color, ls='-', lw=1.5, alpha=0.8)
        else:
            # Scenario 0/1/2 (Single curve logic) - Specific Colors requested by User
            # 1. L1 vs Time (Black)
            axs[0, 0].plot(t, F11, color='black', ls='-', lw=1.5, alpha=0.8)
            # 2. L1 vs P11 (Blue)
            if S11 is not None:
                axs[0, 1].plot(F11, S11, color='blue', ls='-', lw=1.5, alpha=0.8)
            # 3. L1 vs L2 (Red)
            if F22 is not None:
                axs[0, 2].plot(F11, F22, color='red', ls='-', lw=1.5, alpha=0.8)
            # 4. L1 vs L3 (Green)
            if F33 is not None:
                axs[1, 0].plot(F11, F33, color='green', ls='-', lw=1.5, alpha=0.8)
            # 5. P11 vs Time (Blue)
            if S11 is not None:
                axs[1, 1].plot(t, S11, color='blue', ls='-', lw=1.5, alpha=0.8)
            # 6. L2 & L3 vs Time (Red & Green)
            if F22 is not None:
                axs[1, 2].plot(t, F22, color='red', ls='-', lw=1.5, alpha=0.8, label=r'$\lambda_2$')
            if F33 is not None:
                axs[1, 2].plot(t, F33, color='green', ls='-', lw=1.5, alpha=0.8, label=r'$\lambda_3$')
            # Add legend only once
            if i == 0:
                axs[1, 2].legend(loc='best', fontsize=8, frameon=False)
            
            # Scenario 1: Vertical line for hold start
            if scenario == 1:
                hv = data.get('hold_time', 0)
                if hasattr(hv, 'item'):
                    hv = hv.item()
                if hv > 0 and len(t) > 0:
                    t_hold_start = t[-1] - hv
                    # 1. L1 vs Time
                    axs[0, 0].axvline(x=t_hold_start, color='lightgrey', linestyle='--', zorder=0)
                    # 5. P11 vs Time
                    axs[1, 1].axvline(x=t_hold_start, color='lightgrey', linestyle='--', zorder=0)
                    # 6. L2/L3 vs Time
                    axs[1, 2].axvline(x=t_hold_start, color='lightgrey', linestyle='--', zorder=0)
            
            # Scenario 2: Vertical lines for segment transitions
            if scenario == 2:
                seg_rates = data.get('segment_rates', None)
                if seg_rates is not None:
                    seg_rates = np.atleast_1d(seg_rates).flatten()
                    # Valid rates generally start from index 1 (skipping artifact)
                    valid_rates = seg_rates[1:] if len(seg_rates) > 1 else seg_rates
                    valid_rates = [r for r in valid_rates if r > 0]
                    
                    if len(valid_rates) > 1 and len(t) > 10:
                        # Detect rate changes by finding kinks in F11 vs time
                        dt = np.diff(t)
                        dF11 = np.diff(F11)
                        mask = dt > 1e-9
                        rate_calc = np.zeros_like(dF11)
                        rate_calc[mask] = dF11[mask] / dt[mask]
                        
                        # Find significant changes in the rate derivative
                        dRate = np.diff(rate_calc)
                        change_mag = np.abs(dRate)
                        
                        if len(change_mag) > 0:
                            # Threshold: 20% of max change typically isolates the kinks
                            threshold = np.max(change_mag) * 0.2
                            peaks = np.where(change_mag > threshold)[0]
                            
                            if len(peaks) > 0:
                                # Cluster consecutive peaks 
                                splits = np.where(np.diff(peaks) > 5)[0] + 1
                                clusters = np.split(peaks, splits)
                                transition_times = []
                                for clust in clusters:
                                    if len(clust) == 0: continue
                                    idx_max = clust[np.argmax(change_mag[clust])]
                                    t_trans = t[idx_max + 1]
                                    transition_times.append(t_trans)
                                
                                for t_trans in transition_times:
                                    axs[0, 0].axvline(x=t_trans, color='lightgrey', linestyle='--', zorder=0)
                                    axs[1, 1].axvline(x=t_trans, color='lightgrey', linestyle='--', zorder=0)
                                    axs[1, 2].axvline(x=t_trans, color='lightgrey', linestyle='--', zorder=0)

    # Labels
    axs[0, 0].set_xlabel('Time [s]')
    axs[0, 0].set_ylabel(r'In-plane stretch $\lambda_1$ [-]')
    axs[0, 0].set_title(r'$\lambda_1$ vs Time')
    axs[0, 1].set_xlabel(r'In-plane stretch $\lambda_1$ [-]')
    axs[0, 1].set_ylabel(r'1st Piola-Kirchhoff stress $P_{11}$ [MPa]')
    axs[0, 1].set_title(r'$\lambda_1$ vs $P_{11}$')
    axs[0, 2].set_xlabel(r'In-plane stretch $\lambda_1$ [-]')
    axs[0, 2].set_ylabel(r'Lateral stretch $\lambda_2$ [-]')
    axs[0, 2].set_title(r'$\lambda_1$ vs $\lambda_2$')
    axs[1, 0].set_xlabel(r'In-plane stretch $\lambda_1$ [-]')
    axs[1, 0].set_ylabel(r'Thickness stretch $\lambda_3$ [-]')
    axs[1, 0].set_title(r'$\lambda_1$ vs $\lambda_3$')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel(r'1st Piola-Kirchhoff stress $P_{11}$ [MPa]')
    axs[1, 1].set_title(r'$P_{11}$ vs Time')
    axs[1, 2].set_xlabel('Time [s]')
    axs[1, 2].set_ylabel(r'Lateral stretch $\lambda_2$ [-]')
    axs[1, 2].set_title(r'$\lambda_2$ vs Time')
    for ax in axs.flatten():
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=False, prune=None, nbins=5))
    fig.suptitle(f'Scenario {scenario} - Simulation Results', fontsize=14, fontweight='bold', y=0.98)

    # --- PARAMETER PANEL ---
    if has_params:
        param_data = format_parameter_text(par)
        n_material = len(param_data)
        
        protocol_params = []
        if len(par) >= 13:
            lambda_max = par[12]
            protocol_params.append((r'$\lambda_{\max}$', f'{lambda_max:.3f}', ''))
        
        if scenario == 3:
            protocol_params.append((r'$N_{\mathrm{rate}}$', str(len(unique_rates)), ''))
        elif scenario == 2:
            # Scenario 2: Multi-segment piecewise ramp
            # Extract segment rates from data
            seg_rates = data.get('segment_rates', None)
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
                if len(par) >= 12:
                    rate = par[11]
                    r_str = f"{rate:.4f}" if 0.001 <= abs(rate) <= 1000 else f"{rate:.2e}"
                    protocol_params.append((r'$\dot{\lambda}_1$', r_str, r'$\mathrm{s}^{-1}$'))
        else:
            # Scenario 0/1: single rate
            if len(par) >= 12:
                rate = par[11]
                r_str = f"{rate:.4f}" if 0.001 <= abs(rate) <= 1000 else f"{rate:.2e}"
                protocol_params.append((r'$\dot{\lambda}_1$', r_str, r'$\mathrm{s}^{-1}$'))
        
        # Hold time (Scenario 1 only, not Scenario 2)
        if scenario == 1:
            try:
                hv = data.get('hold_time', None)
                if hv is not None:
                    if hasattr(hv, 'item'):
                        hv = hv.item()
                    if hv > 0:
                        protocol_params.append((r'$t_{\mathrm{hold}}$', f'{hv:.1f}', 's'))
            except Exception:
                pass
        
        n_protocol = len(protocol_params)
        n_rates = len(unique_rates) if scenario == 3 else 0
        
        row_height = 0.035
        header_height = 0.05
        title_height = 0.04
        gap = 0.02
        rates_section_height = (0.04 + n_rates * 0.03) if n_rates > 0 else 0
        material_box_height = title_height + header_height + n_material * row_height + 0.02
        protocol_box_height = title_height + header_height + n_protocol * row_height + rates_section_height + 0.02
        
        material_top = 0.98
        material_bottom = material_top - material_box_height
        protocol_top = material_bottom - gap
        protocol_bottom = protocol_top - protocol_box_height
        
        # 1. Material
        ax_params.add_patch(
            plt.Rectangle(
                (0.0, material_bottom),
                1.0,
                material_box_height,
                transform=ax_params.transAxes,
                facecolor='#E8E8E8',
                edgecolor='gray',
                alpha=0.95,
                linewidth=1,
                zorder=1,
            )
        )
        y = material_top - 0.025
        ax_params.text(0.5, y, 'Material Parameters', transform=ax_params.transAxes, fontsize=10, fontweight='bold', ha='center', va='center', zorder=2)
        y -= 0.04
        ax_params.text(0.20, y, 'Symbol', transform=ax_params.transAxes, fontsize=9, fontweight='bold', ha='center', zorder=2)
        ax_params.text(0.55, y, 'Value', transform=ax_params.transAxes, fontsize=9, fontweight='bold', ha='center', zorder=2)
        ax_params.text(0.88, y, 'Unit', transform=ax_params.transAxes, fontsize=9, fontweight='bold', ha='center', zorder=2)
        ax_params.plot([0.05, 0.95], [y-0.015, y-0.015], color='gray', linewidth=0.5, transform=ax_params.transAxes, zorder=2)
        
        y_start = y - 0.035
        for i, (sym, val, unit) in enumerate(param_data):
            y_pos = y_start - i*row_height
            ax_params.text(0.20, y_pos, sym, transform=ax_params.transAxes, fontsize=9, ha='center', va='center', zorder=2)
            ax_params.text(0.55, y_pos, val, transform=ax_params.transAxes, fontsize=9, ha='center', va='center', zorder=2)
            ax_params.text(0.88, y_pos, unit, transform=ax_params.transAxes, fontsize=9, ha='center', va='center', zorder=2)

        # 2. Protocol
        ax_params.add_patch(
            plt.Rectangle(
                (0.0, protocol_bottom),
                1.0,
                protocol_box_height,
                transform=ax_params.transAxes,
                facecolor='#E8E8E8',
                edgecolor='gray',
                alpha=0.95,
                linewidth=1,
                zorder=1,
            )
        )
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
            for i, r in enumerate(unique_rates):
                color = rate_to_color.get(r, 'black')
                ax_params.text(0.5, y_rates - 0.03 - i*0.03, f'{r:.2e} /s', transform=ax_params.transAxes, fontsize=8, ha='center', va='top', color=color, fontweight='bold', zorder=2)

    # Save
    out_path = npz_path.with_suffix('.png')
    plt.savefig(out_path, dpi=800, bbox_inches='tight')
    svg_dir = out_path.parent / 'svg'
    svg_dir.mkdir(exist_ok=True)
    plt.savefig(svg_dir / out_path.with_suffix('.svg').name, format='svg', bbox_inches='tight')
    print(f"Saved plot to {out_path} (DPI=800) and svg/")
    if show_plot:
        plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description='Plot simulation results (A4 Compact)')
    parser.add_argument('npz_file', help='Path to NPZ file')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    plot_results(args.npz_file, show_plot=not args.no_show)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
