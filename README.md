# Adaptive visual-field testing

An exploratory Python framework for studying the trade-off between threshold-estimation error and presentation count. It compares a simple staircase with a discrete Bayesian procedure using synthetic observers, configurable response errors, and explicit stopping rules.

The research question is: **how do response reliability, prior assumptions, and stopping criteria change the amount of information obtained from an adaptive visual-sensitivity test?**

This repository contains an offline, single-location simulation. It does not present stimuli to people, operate a perimeter, or implement SITA. Presentation counts are a measurement-burden proxy; they are not measured test duration or patient comfort. No benchmark results or clinical performance claims are included.

## Run an experiment

Python 3.10 or newer is required. The simulator uses only the standard library and can run directly from this directory without installation.

```powershell
python -m visual_field --output outputs/baseline --sensitivities 10 20 30 --lapses 0 0.1 --false-positives 0 0.05 --trials 100 --seed 2026
```

Each combination of sensitivity, lapse rate, and false-positive rate receives the requested number of independent trials. Both methods run for each trial. Choose a new output directory for each experiment; existing paths are refused.

To compare estimation error at an identical presentation budget:

```powershell
python -m visual_field --output outputs/fixed-budget --fixed-budget --max-presentations 30 --trials 200 --seed 2026
```

To examine a deliberately misspecified Bayesian response model:

```powershell
python -m visual_field --output outputs/model-mismatch --lapses 0.15 --false-positives 0.1 --assumed-lapse 0 --assumed-false-positive 0 --assumed-slope 1.5 --save-presentations
```

To change the prior, presentation range, and stopping criteria:

```powershell
python -m visual_field --output outputs/stopping --prior-mean 20 --prior-sd 10 --stop-sd 1 --min-presentations 10 --max-presentations 60 --reversals 8 --reversal-average 6
```

Run `python -m visual_field --help` for the complete argument list.

## Observer model

The stimulus value `d` represents attenuation in dB. **A larger stimulus dB value is dimmer. A larger sensitivity `theta` means that the observer can detect dimmer stimuli.** The values are abstract model quantities and are not calibrated to a particular device.

The probability of responding “seen” is:

```text
P(seen | d, theta) = f + (1 - f - l) / (1 + exp((d - theta) / s))
```

Here `s > 0` is the logistic scale in dB, `f` is the false-positive floor, and `l` is the lapse ceiling deficit. The constraints are `0 <= f, l < 1` and `f + l < 1`. At very bright stimuli the detection probability approaches `1 - l`; at very dim stimuli it approaches `f`.

`theta` is the midpoint of the psychometric function's response range. At `d = theta`, detection probability is `(1 + f - l) / 2`; it is exactly 0.5 only when the two response-error rates are equal. This distinction matters when interpreting staircase bias under asymmetric response errors.

Responses are independent conditional on the fixed observer parameters. The model has no fatigue, learning, fixation losses, spatial dependence, disease progression, or response-time component.

## Procedures

| Procedure | Next presentation | Estimate | Optional early stop |
| --- | --- | --- | --- |
| Staircase | After “seen,” increase attenuation; after “not seen,” decrease it. Use a coarse step until the first reversal and a fine step thereafter. | Mean of the latest reversal intensities; final presented intensity if no reversal occurs. | Requested reversal count, after a common minimum number of presentations. |
| Bayesian | Present at the current posterior mean on a bounded, continuous stimulus range. | Mean of a posterior over a discrete sensitivity grid. | Posterior SD at or below the selected target, after the common minimum. |

The staircase is one-up/one-down and approximately targets an absolute 50% response probability. When lapse and false-positive rates differ, that point differs from `theta`; finite steps and range boundaries add further effects. A reversal is a change in requested response direction, including at a clipped boundary. This simple convention is recorded explicitly so boundary behavior can be interpreted.

The Bayesian procedure begins with a Gaussian prior normalized over the bounded threshold grid. Each observation multiplies the prior mass by the corresponding Bernoulli likelihood. Updates use log weights for numerical stability. The default stimulus range is 0–40 dB, with a 0.5 dB threshold grid, a prior mean of 25 dB, and prior SD of 12 dB. Posterior-mean stimulus placement is a simple policy; it does not optimize expected information gain.

The Bayesian method belongs to the broad family of sequential Bayesian threshold estimators. It is not an implementation of a published ZEST protocol, ARREST, qVFM, or any proprietary test strategy.

## Comparison design

Both methods receive the same true sensitivity, response-error rates, logistic scale, and minimum/maximum presentation limits. Within a trial they start fresh pseudorandom generators with the same pair seed. This creates a common uniform response sequence indexed by presentation number; their responses can still differ because their chosen stimuli differ. The pair seed is saved in each run record. Changing the regime list changes the master seed's allocation to trials.

By default, the Bayesian procedure is given the true response-model parameters. This is a favorable assumption for that procedure, not a claim that such parameters would be known in practice. The staircase does not use those parameters. The `--assumed-*` options expose likelihood misspecification; `--prior-mean` and `--prior-sd` expose prior sensitivity.

Under default stopping, a reversal target and a posterior SD target do not imply equal precision. Compare error and presentation count together. Use `--fixed-budget` to disable both early-stop rules for an equal-count comparison. A `budget` stop means the procedure completed the maximum count without taking its early-stop branch; an early-stop criterion reached on the final presentation is recorded as that criterion.

Thresholds and true sensitivities are bounded by the configured range. A truncated grid can produce strong boundary effects and optimistic posterior uncertainty. A sensitivity can lie between grid points, so grid approximation is another source of error.

## Output files

Files are created only when an experiment is run.

| File | Contents |
| --- | --- |
| `runs.csv` | One record per method and synthetic trial: true sensitivity, estimate, signed error, presentation count, stop reason, pair seed, and method-specific diagnostics. |
| `summary.csv` | Per-regime bias, mean absolute error, RMSE, presentation-count summaries, stopping fraction, and Bayesian posterior-interval coverage. |
| `settings.json` | Observer and inference parameters, stopping settings, master seed, Python/package versions, UTC timestamps, command arguments, and SHA-256 hashes of the Python source. |
| `report.md` | Human-readable aggregate error and presentation-count table. |
| `presentations.csv` | Optional complete stimulus/response history and Bayesian posterior summaries, enabled by `--save-presentations`. |

Signed error is `estimate - true sensitivity`; positive bias means overestimated sensitivity. RMSE is `sqrt(mean(error**2))`. The Bayesian interval is a 95% equal-tail credible interval over the grid. Its simulated coverage is the fraction of runs containing the true sensitivity, conditional on the chosen regimes. It is not guaranteed to have 95% frequentist coverage. Staircase interval and posterior fields are blank.

The CLI stores all run summaries in memory, and stores presentation histories when requested. Use smaller batches for very large parameter sweeps. Generated output directories are ignored by Git by default.

## Research context

The bibliography in [SOURCES.md](SOURCES.md) connects this exercise to adaptive perimetry, spatial testing strategies, open research software, and patient experience. Those publications motivate questions; their findings are not results of this implementation.

Potential next studies include equal-budget comparisons, deliberately incorrect priors, changing response reliability during a run, and spatially correlated sensitivities. The current model cannot establish diagnostic accuracy, real patient burden, progression detection, hardware agreement, or safety for clinical use.

## Repository layout

```text
visual_field/
  core.py          Observer, psychometric function, staircase, Bayesian updates
  __main__.py      CLI, experiment loops, aggregate measures, output provenance
  __init__.py      Package version
SOURCES.md         Primary literature and methodological boundaries
pyproject.toml     Package metadata; no runtime dependencies
```
