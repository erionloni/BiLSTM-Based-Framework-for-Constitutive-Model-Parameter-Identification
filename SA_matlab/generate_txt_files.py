#!/usr/bin/env python3
"""
Generate parameter-set TXT files for the MATLAB Rubin-Bodner solver.

This script produces comma-separated rows matching the expected input format
for `SA_matlab/main_R_local_Empa_flex.m`.

Scenario mapping:
    1 = single-rate ramp
    2 = ramp + hold
    3 = multi-step piecewise ramp
    4 = multi-rate repetition

Defaults are defined in the configuration section and can be overridden with
CLI flags for common settings.

Usage:
    python SA_matlab/generate_txt_files.py
    python SA_matlab/generate_txt_files.py --output D.txt --total-files 5000
    python SA_matlab/generate_txt_files.py --scenario-dist 0 0 0 1
    python SA_matlab/generate_txt_files.py --seed 42
"""

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# =============================================================================
# USER CONFIGURATION
# =============================================================================

TOTAL_FILES = 10000

SCENARIO_DISTRIBUTION = {
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 1,
}

OUTPUT_FILENAME = 'D.txt'

PARAM_RANGES_LINEAR = {
    'lambda1_max': [1.8, 3],
    'theta': [0.02, 0.35],
    'm4': [1.1, 1.5],
}

PARAM_RANGES_LOG = {
    'lambda1_dot': [0.0009, 0.11],
    'q': [1e-4, 1e1],
    'm1': [1e-1, 300.0],
    'm2': [1e-5, 1e-1],
    'm3': [1e-1, 300.0],
    'kM': [1e-4, 1e1],
    'kF': [1e-8, 1e-3],
    'm5': [0.1, 10.0],
    'alphaM': [0.1, 10.0],
}

HOLD_TIME_RANGE = [400.0, 600.0]

N_SEGMENTS_OPTIONS = [2]

N_RATES_OPTIONS = [4]

MULTI_RATE_SAMPLING = "uniform_logspace"
MAX_MULTI_RATES = max(N_RATES_OPTIONS)

MULTIRATE_HOLD_MODE = "uniform"
MULTIRATE_HOLD_TIME_RANGE = [400.0, 600.0]
MULTIRATE_HOLD_TIME_FIXED = 600.0

# =============================================================================
# END USER CONFIGURATION
# =============================================================================


@dataclass
class GenerationConfig:
    """Configuration for parameter-set generation."""

    total_files: int = TOTAL_FILES
    scenario_distribution: Dict[str, float] = field(default_factory=lambda: dict(SCENARIO_DISTRIBUTION))
    output_filename: str = OUTPUT_FILENAME
    param_ranges_linear: Dict[str, Sequence[float]] = field(default_factory=lambda: dict(PARAM_RANGES_LINEAR))
    param_ranges_log: Dict[str, Sequence[float]] = field(default_factory=lambda: dict(PARAM_RANGES_LOG))
    hold_time_range: Sequence[float] = tuple(HOLD_TIME_RANGE)
    n_segments_options: List[int] = field(default_factory=lambda: list(N_SEGMENTS_OPTIONS))
    n_rates_options: List[int] = field(default_factory=lambda: list(N_RATES_OPTIONS))
    multi_rate_sampling: str = MULTI_RATE_SAMPLING
    max_multi_rates: int = MAX_MULTI_RATES
    multi_rate_hold_mode: str = MULTIRATE_HOLD_MODE
    multi_rate_hold_time_range: Sequence[float] = tuple(MULTIRATE_HOLD_TIME_RANGE)
    multi_rate_hold_time_fixed: float = MULTIRATE_HOLD_TIME_FIXED


def sample_linear(low: float, high: float) -> float:
    """Sample a value uniformly from [low, high] in linear space."""
    return random.uniform(low, high)


def sample_log(low: float, high: float) -> float:
    """Sample a value uniformly in log10 space between low and high."""
    log_low = math.log10(low)
    log_high = math.log10(high)
    return 10 ** random.uniform(log_low, log_high)


def linspace(low: float, high: float, count: int) -> List[float]:
    """Generate a list of linearly spaced values."""
    if count <= 0:
        return []
    if count == 1:
        return [(low + high) / 2.0]
    step = (high - low) / (count - 1)
    return [low + i * step for i in range(count)]


def logspace(low: float, high: float, count: int) -> List[float]:
    """Generate a list of log-spaced values."""
    if count <= 0:
        return []
    if count == 1:
        return [10 ** ((math.log10(low) + math.log10(high)) / 2.0)]
    log_low = math.log10(low)
    log_high = math.log10(high)
    step = (log_high - log_low) / (count - 1)
    return [10 ** (log_low + i * step) for i in range(count)]


def generate_data(config: Optional[GenerationConfig] = None) -> List[List[float]]:
    """
    Generate parameter rows for all scenarios based on the provided configuration.
    """
    if config is None:
        config = GenerationConfig()

    data = []

    counts = {
        '1': int(config.total_files * config.scenario_distribution['1']),
        '2': int(config.total_files * config.scenario_distribution['2']),
        '3': int(config.total_files * config.scenario_distribution['3']),
        '4': int(config.total_files * config.scenario_distribution['4']),
    }

    current_total = sum(counts.values())
    if current_total < config.total_files:
        counts['4'] += (config.total_files - current_total)

    for scenario_key, count in counts.items():
        if count == 0:
            continue

        for _ in range(count):
            row = {}

            for key, (low, high) in config.param_ranges_linear.items():
                row[key] = sample_linear(low, high)

            for key, (low, high) in config.param_ranges_log.items():
                row[key] = sample_log(low, high)

            scenario_type = 0
            n_segments = 0
            seg1_lambda = 0.0
            seg1_rate = 0.0
            seg2_lambda = 0.0
            seg2_rate = 0.0
            seg3_lambda = 0.0
            seg3_rate = 0.0
            hold_time = 0.0
            n_rates_multi = 0
            rate_values = [0.0] * config.max_multi_rates

            if scenario_key == '1':
                scenario_type = 0
                hold_time = 0.0
            elif scenario_key == '2':
                scenario_type = 1
                hold_time = sample_linear(config.hold_time_range[0], config.hold_time_range[1])
            elif scenario_key == '3':
                scenario_type = 2
                n_segments = random.choice(config.n_segments_options)

                target_lambda = row['lambda1_max']

                intermediate_lambdas = []
                for _ in range(n_segments - 1):
                    intermediate_lambdas.append(random.uniform(1.0, target_lambda))
                intermediate_lambdas.sort()

                rates = []
                for _ in range(n_segments):
                    rates.append(sample_log(
                        config.param_ranges_log['lambda1_dot'][0],
                        config.param_ranges_log['lambda1_dot'][1],
                    ))

                if n_segments >= 1:
                    seg1_lambda = intermediate_lambdas[0] if n_segments > 1 else target_lambda
                    seg1_rate = rates[0]

                if n_segments >= 2:
                    seg2_lambda = target_lambda if n_segments == 2 else intermediate_lambdas[1]
                    seg2_rate = rates[1]

                if n_segments >= 3:
                    seg3_lambda = target_lambda
                    seg3_rate = rates[2]
            elif scenario_key == '4':
                scenario_type = 3
                n_rates_multi = random.choice(config.n_rates_options)
                rate_min, rate_max = config.param_ranges_log['lambda1_dot']

                if config.multi_rate_sampling == "independent_log":
                    sampled_rates = [sample_log(rate_min, rate_max) for _ in range(n_rates_multi)]
                elif config.multi_rate_sampling == "uniform_logspace":
                    sampled_rates = logspace(rate_min, rate_max, n_rates_multi)
                elif config.multi_rate_sampling == "uniform_linear":
                    sampled_rates = linspace(rate_min, rate_max, n_rates_multi)
                else:
                    raise ValueError(f"Unknown MULTI_RATE_SAMPLING: {config.multi_rate_sampling}")

                for idx, rate_val in enumerate(sampled_rates):
                    if idx < config.max_multi_rates:
                        rate_values[idx] = rate_val

                if config.multi_rate_hold_mode == "none":
                    hold_time = 0.0
                elif config.multi_rate_hold_mode == "fixed":
                    hold_time = config.multi_rate_hold_time_fixed
                elif config.multi_rate_hold_mode == "uniform":
                    hold_time = sample_linear(
                        config.multi_rate_hold_time_range[0],
                        config.multi_rate_hold_time_range[1],
                    )
                else:
                    raise ValueError(f"Unknown MULTIRATE_HOLD_MODE: {config.multi_rate_hold_mode}")

            out_row = [
                row['lambda1_max'],
                row['theta'],
                row['q'],
                row['m1'],
                row['m2'],
                row['m3'],
                row['m4'],
                row['m5'],
                row['kM'],
                row['alphaM'],
                row['kF'],
                row['lambda1_dot'],
                scenario_type,
                n_segments,
                seg1_lambda,
                seg1_rate,
                seg2_lambda,
                seg2_rate,
                seg3_lambda,
                seg3_rate,
                hold_time,
                n_rates_multi,
            ]

            out_row.extend(rate_values)
            data.append(out_row)

    return data


def save_to_txt(data: List[Sequence[float]], filename: Union[str, Path]) -> None:
    """Write rows to disk as comma-separated values with fixed precision."""
    try:
        with open(filename, 'w') as file_handle:
            for row in data:
                line = ",".join([f"{x:.16f}" for x in row])
                file_handle.write(line + "\n")
        print(f"Successfully generated {len(data)} parameter sets in '{filename}'.")
    except Exception as exc:
        print(f"Error writing file: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the generator."""
    parser = argparse.ArgumentParser(
        description="Generate parameter sets for the Rubin-Bodner MATLAB solver."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output filename for the generated parameters.",
    )
    parser.add_argument(
        "--total-files",
        type=int,
        default=None,
        help="Total number of parameter sets to generate.",
    )
    parser.add_argument(
        "--scenario-dist",
        type=float,
        nargs=4,
        default=None,
        metavar=("S1", "S2", "S3", "S4"),
        help="Scenario distribution for scenarios 1-4 (must sum to 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = GenerationConfig()
    if args.total_files is not None:
        config.total_files = args.total_files
    if args.output is not None:
        config.output_filename = str(args.output)
    if args.scenario_dist is not None:
        config.scenario_distribution = {
            '1': args.scenario_dist[0],
            '2': args.scenario_dist[1],
            '3': args.scenario_dist[2],
            '4': args.scenario_dist[3],
        }
    if args.seed is not None:
        random.seed(args.seed)

    print("Generating parameter file...")
    data = generate_data(config)
    random.shuffle(data)
    save_to_txt(data, config.output_filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
