"""Command-line experiments and transparent, machine-readable outputs."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
from typing import Any

from . import __version__
from .core import Observer, Settings, bayesian, staircase


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Compare adaptive threshold methods on synthetic observers."
    )
    result.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing paths are refused.")
    result.add_argument("--sensitivities", type=float, nargs="+", default=[10, 20, 30])
    result.add_argument("--lapses", type=float, nargs="+", default=[0.0, 0.1])
    result.add_argument("--false-positives", type=float, nargs="+", default=[0.0, 0.05])
    result.add_argument("--slope", type=float, default=2.0,
                        help="Synthetic observer logistic scale, in dB.")
    result.add_argument("--trials", type=int, default=100,
                        help="Independent observer runs per parameter combination.")
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--min-presentations", type=int, default=8)
    result.add_argument("--max-presentations", type=int, default=40)
    result.add_argument("--fixed-budget", action="store_true",
                        help="Disable early stopping for both procedures.")
    result.add_argument("--stop-sd", type=float, default=1.5,
                        help="Bayesian posterior SD stopping threshold, in dB.")
    result.add_argument("--reversals", type=int, default=6,
                        help="Staircase reversal stopping target.")
    result.add_argument("--reversal-average", type=int, default=4,
                        help="Number of latest reversal intensities to average.")
    result.add_argument("--start", type=float, default=25.0)
    result.add_argument("--coarse-step", type=float, default=4.0)
    result.add_argument("--fine-step", type=float, default=2.0)
    result.add_argument("--lower", type=float, default=0.0)
    result.add_argument("--upper", type=float, default=40.0)
    result.add_argument("--grid-step", type=float, default=0.5)
    result.add_argument("--prior-mean", type=float, default=25.0)
    result.add_argument("--prior-sd", type=float, default=12.0)
    result.add_argument("--assumed-slope", type=float, default=None,
                        help="Bayesian model scale; defaults to true observer scale.")
    result.add_argument("--assumed-lapse", type=float, default=None,
                        help="Bayesian lapse; omitted means the true rate in each regime.")
    result.add_argument("--assumed-false-positive", type=float, default=None,
                        help="Bayesian false-positive rate; omitted means the true rate.")
    result.add_argument("--save-presentations", action="store_true",
                        help="Also save every stimulus, response, and posterior summary.")
    return result


def validate(args: argparse.Namespace, cli: argparse.ArgumentParser) -> None:
    scalar_names = (
        "slope", "stop_sd", "start", "coarse_step", "fine_step", "lower", "upper",
        "grid_step", "prior_mean", "prior_sd", "assumed_slope", "assumed_lapse",
        "assumed_false_positive",
    )
    numbers = [getattr(args, name) for name in scalar_names]
    numbers += args.sensitivities + args.lapses + args.false_positives
    if any(value is not None and not math.isfinite(value) for value in numbers):
        cli.error("All numeric parameters must be finite.")
    positive_names = ("slope", "stop_sd", "coarse_step", "fine_step", "grid_step", "prior_sd")
    if any(getattr(args, name) <= 0 for name in positive_names):
        cli.error("Slope, stopping SD, steps, and prior SD must be positive.")
    if args.assumed_slope is not None and args.assumed_slope <= 0:
        cli.error("--assumed-slope must be positive.")
    if args.trials < 1 or args.reversals < 1 or args.reversal_average < 1:
        cli.error("Trials and reversal counts must be positive integers.")
    if not 1 <= args.min_presentations <= args.max_presentations:
        cli.error("Require 1 <= min-presentations <= max-presentations.")
    if args.lower >= args.upper:
        cli.error("--lower must be smaller than --upper.")
    if not args.lower <= args.start <= args.upper:
        cli.error("--start must fall inside the stimulus range.")
    if any(not args.lower <= value <= args.upper for value in args.sensitivities):
        cli.error("Sensitivities must fall inside the threshold grid.")
    grid_intervals = (args.upper - args.lower) / args.grid_step
    if not 1 <= grid_intervals <= 10000 or not math.isclose(
        grid_intervals, round(grid_intervals), rel_tol=0.0, abs_tol=1e-8
    ):
        cli.error("The range must contain an integer number of grid steps, from 1 to 10000.")
    for values in (args.sensitivities, args.lapses, args.false_positives):
        if len(set(values)) != len(values):
            cli.error("Parameter lists must not contain duplicate values.")
    for lapse, false_positive in itertools.product(args.lapses, args.false_positives):
        rates = [lapse, false_positive]
        model_lapse = lapse if args.assumed_lapse is None else args.assumed_lapse
        model_fp = false_positive if args.assumed_false_positive is None else args.assumed_false_positive
        rates.extend([model_lapse, model_fp])
        if any(rate < 0 or rate >= 1 for rate in rates):
            cli.error("Lapse and false-positive rates must be in [0, 1).")
        if lapse + false_positive >= 1 or model_lapse + model_fp >= 1:
            cli.error("Lapse + false-positive must be less than 1 for observer and model.")
    if args.output.exists():
        cli.error("Output path already exists; choose a new directory.")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in runs:
        key = tuple(row[name] for name in ("sensitivity_db", "lapse", "false_positive", "method"))
        groups.setdefault(key, []).append(row)
    summaries = []
    for (sensitivity, lapse, false_positive, method), group in groups.items():
        errors = [row["error_db"] for row in group]
        counts = [row["presentations"] for row in group]
        coverage = [row["interval_covers_truth"] for row in group if row["interval_covers_truth"] is not None]
        summaries.append({
            "sensitivity_db": sensitivity,
            "lapse": lapse,
            "false_positive": false_positive,
            "method": method,
            "runs": len(group),
            "bias_db": statistics.mean(errors),
            "mae_db": statistics.mean(abs(error) for error in errors),
            "rmse_db": math.sqrt(statistics.mean(error * error for error in errors)),
            "mean_presentations": statistics.mean(counts),
            "median_presentations": statistics.median(counts),
            "min_presentations": min(counts),
            "max_presentations": max(counts),
            "fraction_stopped_by_criterion": statistics.mean(
                row["stop_reason"] != "budget" for row in group
            ),
            "posterior_interval_coverage": statistics.mean(coverage) if coverage else None,
        })
    return summaries


def source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {
        "visual_field/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.glob("*.py"))
    }


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    validate(args, cli)
    settings = Settings(
        lower=args.lower, upper=args.upper, grid_step=args.grid_step,
        prior_mean=args.prior_mean, prior_sd=args.prior_sd, start=args.start,
        coarse_step=args.coarse_step, fine_step=args.fine_step,
        reversal_target=args.reversals, reversal_average=args.reversal_average,
        min_presentations=args.min_presentations, max_presentations=args.max_presentations,
        stop_sd=args.stop_sd, fixed_budget=args.fixed_budget,
    )
    model_slope = args.slope if args.assumed_slope is None else args.assumed_slope
    seed_generator = random.Random(args.seed)
    runs: list[dict[str, Any]] = []
    presentations: list[dict[str, Any]] = []
    regimes = []
    started_at = datetime.now(timezone.utc).isoformat()

    for regime_id, (sensitivity, lapse, false_positive) in enumerate(
        itertools.product(args.sensitivities, args.lapses, args.false_positives), start=1
    ):
        observer = Observer(sensitivity, args.slope, lapse, false_positive)
        model_lapse = lapse if args.assumed_lapse is None else args.assumed_lapse
        model_fp = false_positive if args.assumed_false_positive is None else args.assumed_false_positive
        regimes.append({
            "regime_id": regime_id, "observer": asdict(observer),
            "bayesian_assumptions": {"slope": model_slope, "lapse": model_lapse, "false_positive": model_fp},
        })
        for trial in range(1, args.trials + 1):
            pair_seed = seed_generator.getrandbits(64)
            estimates = {
                "staircase": staircase(observer, settings, pair_seed),
                "bayesian": bayesian(observer, settings, pair_seed, model_slope, model_lapse, model_fp),
            }
            for method, estimate in estimates.items():
                covered = None
                if estimate.interval_low is not None and estimate.interval_high is not None:
                    covered = estimate.interval_low <= sensitivity <= estimate.interval_high
                runs.append({
                    "regime_id": regime_id, "trial": trial, "pair_seed": pair_seed,
                    "method": method, "sensitivity_db": sensitivity, "slope_db": args.slope,
                    "lapse": lapse, "false_positive": false_positive,
                    "estimate_db": estimate.threshold, "error_db": estimate.threshold - sensitivity,
                    "presentations": len(estimate.presentations), "stop_reason": estimate.stop_reason,
                    "reversals": estimate.reversals, "posterior_sd_db": estimate.posterior_sd,
                    "interval_low_db": estimate.interval_low, "interval_high_db": estimate.interval_high,
                    "interval_covers_truth": covered,
                })
                if args.save_presentations:
                    for event in estimate.presentations:
                        presentations.append({
                            "regime_id": regime_id, "trial": trial, "method": method,
                            "presentation": event.number, "stimulus_db": event.stimulus,
                            "seen": event.seen, "posterior_mean_db": event.posterior_mean,
                            "posterior_sd_db": event.posterior_sd,
                        })

    summaries = aggregate(runs)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "runs.csv", runs)
    write_csv(args.output / "summary.csv", summaries)
    if presentations:
        write_csv(args.output / "presentations.csv", presentations)
    provenance = {
        "project": "visual-field-adaptive-testing",
        "version": __version__,
        "data_origin": "synthetic logistic observer; no human measurements",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "command_arguments": sys.argv[1:],
        "master_seed": args.seed,
        "trials_per_regime": args.trials,
        "settings": asdict(settings),
        "regimes": regimes,
        "source_sha256": source_hashes(),
        "pairing": "Each method starts a fresh PRNG with the same pair seed; stimuli differ.",
        "stopping": "Both use the full budget." if args.fixed_budget else "Method-specific early stopping with common minimum and maximum counts.",
        "model_assumptions": "Bayesian response parameters match each observer unless explicitly overridden.",
        "interval": "Equal-tail 95% grid-posterior credible interval, conditional on model and prior.",
        "scope": "Exploratory offline single-location simulation; no clinical validation.",
    }
    (args.output / "settings.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = [
        "# Synthetic-observer experiment", "",
        "Generated from an offline simulation. Results are conditional on the observer model and selected settings.", "",
        "| Sensitivity (dB) | Lapse | False positive | Method | Runs | Bias (dB) | MAE (dB) | RMSE (dB) | Mean presentations |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        report.append(
            f"| {row['sensitivity_db']:g} | {row['lapse']:g} | {row['false_positive']:g} | "
            f"{row['method']} | {row['runs']} | {row['bias_db']:.3f} | {row['mae_db']:.3f} | "
            f"{row['rmse_db']:.3f} | {row['mean_presentations']:.2f} |"
        )
    report.extend(["", "See settings.json for assumptions and provenance, runs.csv for individual estimates, and summary.csv for additional measures.", ""])
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved {len(runs)} synthetic runs across {len(regimes)} regimes to {args.output}")


if __name__ == "__main__":
    main()
