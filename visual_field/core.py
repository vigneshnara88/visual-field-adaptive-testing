"""Synthetic observer and two single-location threshold procedures.

All intensities are abstract attenuation values in dB. Increasing attenuation
makes a stimulus dimmer; increasing observer sensitivity improves detection.
No display, perimeter, patient records, or device calibration is involved.
"""

from dataclasses import dataclass
import math
import random
from typing import Optional


@dataclass(frozen=True)
class Observer:
    sensitivity: float
    slope: float
    lapse: float
    false_positive: float


@dataclass(frozen=True)
class Settings:
    lower: float = 0.0
    upper: float = 40.0
    grid_step: float = 0.5
    prior_mean: float = 25.0
    prior_sd: float = 12.0
    start: float = 25.0
    coarse_step: float = 4.0
    fine_step: float = 2.0
    reversal_target: int = 6
    reversal_average: int = 4
    min_presentations: int = 8
    max_presentations: int = 40
    stop_sd: float = 1.5
    fixed_budget: bool = False


@dataclass(frozen=True)
class Presentation:
    number: int
    stimulus: float
    seen: bool
    posterior_mean: Optional[float] = None
    posterior_sd: Optional[float] = None


@dataclass(frozen=True)
class Estimate:
    threshold: float
    stop_reason: str
    presentations: tuple[Presentation, ...]
    reversals: Optional[int] = None
    posterior_sd: Optional[float] = None
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None


def probability_seen(
    stimulus: float,
    sensitivity: float,
    slope: float,
    lapse: float,
    false_positive: float,
) -> float:
    """Logistic psychometric function with a stable exponential evaluation.

    P(seen) = f + (1 - f - l) / (1 + exp((stimulus - sensitivity) / slope)).
    The threshold is the midpoint between response asymptotes, which need not
    equal an absolute 50% detection probability when f and l differ.
    """
    z = (stimulus - sensitivity) / slope
    if z >= 0.0:
        tail = math.exp(-z)
        logistic = tail / (1.0 + tail)
    else:
        logistic = 1.0 / (1.0 + math.exp(z))
    return false_positive + (1.0 - false_positive - lapse) * logistic


def respond(observer: Observer, stimulus: float, rng: random.Random) -> bool:
    return rng.random() < probability_seen(
        stimulus, observer.sensitivity, observer.slope,
        observer.lapse, observer.false_positive,
    )


def clip(value: float, settings: Settings) -> float:
    return min(settings.upper, max(settings.lower, value))


def staircase(observer: Observer, settings: Settings, seed: int) -> Estimate:
    """One-up/one-down staircase; average the latest reversal intensities.

    A reversal means a change in the direction requested by the responses,
    including at a clipped boundary. This is a deliberately simple baseline.
    """
    rng = random.Random(seed)
    stimulus = clip(settings.start, settings)
    previous_direction = None
    reversals: list[float] = []
    events: list[Presentation] = []
    reason = "budget"

    for number in range(1, settings.max_presentations + 1):
        seen = respond(observer, stimulus, rng)
        events.append(Presentation(number, stimulus, seen))
        direction = 1 if seen else -1
        if previous_direction is not None and direction != previous_direction:
            reversals.append(stimulus)

        if (
            not settings.fixed_budget
            and number >= settings.min_presentations
            and len(reversals) >= settings.reversal_target
        ):
            reason = "reversal_target"
            break

        step = settings.fine_step if reversals else settings.coarse_step
        stimulus = clip(stimulus + direction * step, settings)
        previous_direction = direction

    recent = reversals[-settings.reversal_average:]
    threshold = sum(recent) / len(recent) if recent else events[-1].stimulus
    return Estimate(threshold, reason, tuple(events), reversals=len(reversals))


def threshold_grid(settings: Settings) -> tuple[float, ...]:
    steps = round((settings.upper - settings.lower) / settings.grid_step)
    return tuple(settings.lower + i * settings.grid_step for i in range(steps + 1))


def normalize_logs(log_weights: list[float]) -> list[float]:
    peak = max(log_weights)
    weights = [math.exp(value - peak) for value in log_weights]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def moments(grid: tuple[float, ...], weights: list[float]) -> tuple[float, float]:
    mean = math.fsum(theta * weight for theta, weight in zip(grid, weights))
    variance = math.fsum(
        weight * (theta - mean) ** 2 for theta, weight in zip(grid, weights)
    )
    return mean, math.sqrt(max(0.0, variance))


def weighted_quantile(
    grid: tuple[float, ...], weights: list[float], probability: float
) -> float:
    cumulative = 0.0
    for theta, weight in zip(grid, weights):
        cumulative += weight
        if cumulative >= probability:
            return theta
    return grid[-1]


def bayesian(
    observer: Observer,
    settings: Settings,
    seed: int,
    assumed_slope: float,
    assumed_lapse: float,
    assumed_false_positive: float,
) -> Estimate:
    """Discrete posterior update, with stimuli placed at the posterior mean.

    This is a generic sequential Bayesian estimator, not SITA or an exact
    reproduction of a published ZEST implementation.
    """
    rng = random.Random(seed)
    grid = threshold_grid(settings)
    log_weights = [
        -0.5 * ((theta - settings.prior_mean) / settings.prior_sd) ** 2
        for theta in grid
    ]
    weights = normalize_logs(log_weights)
    events: list[Presentation] = []
    reason = "budget"

    for number in range(1, settings.max_presentations + 1):
        mean, _ = moments(grid, weights)
        stimulus = clip(mean, settings)
        seen = respond(observer, stimulus, rng)
        for index, theta in enumerate(grid):
            p_seen = probability_seen(
                stimulus, theta, assumed_slope, assumed_lapse,
                assumed_false_positive,
            )
            likelihood = p_seen if seen else 1.0 - p_seen
            # Keep numerical tails representable; this is not an extra lapse.
            log_weights[index] += math.log(max(likelihood, 1e-300))
        weights = normalize_logs(log_weights)
        mean, sd = moments(grid, weights)
        events.append(Presentation(number, stimulus, seen, mean, sd))
        if (
            not settings.fixed_budget
            and number >= settings.min_presentations
            and sd <= settings.stop_sd
        ):
            reason = "posterior_sd"
            break

    mean, sd = moments(grid, weights)
    return Estimate(
        mean, reason, tuple(events), posterior_sd=sd,
        interval_low=weighted_quantile(grid, weights, 0.025),
        interval_high=weighted_quantile(grid, weights, 0.975),
    )
