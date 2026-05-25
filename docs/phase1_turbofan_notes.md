# Phase 1 Turbofan Domain Notes

These notes are the Phase 1 learning checkpoint for the C-MAPSS RUL track. They are intentionally short and practical: the goal is to connect the code artifacts to the physical maintenance problem.

## What C-MAPSS Represents

C-MAPSS is a simulated turbofan run-to-failure benchmark. Each engine unit starts from a healthy state, runs for many cycles, and eventually reaches an end-of-life condition. The model observes per-cycle operating settings and sensor readings, then predicts remaining useful life (RUL) for held-out engine units.

The FD001-FD004 subsets are not interchangeable:

| Subset | Operating conditions | Fault modes | Practical implication |
|---|---:|---:|---|
| FD001 | 1 | 1 | Cleanest baseline split; good first sanity check. |
| FD002 | 6 | 1 | Requires handling operating regimes before comparing sensor values. |
| FD003 | 1 | 2 | Same operating regime, but degradation can come from more than one fault path. |
| FD004 | 6 | 2 | Hardest classical subset because regime and fault effects are entangled. |

## Sensor Interpretation

The raw files expose 21 anonymized sensor columns plus 3 operating settings. The public documentation describes them as turbofan measurements such as temperatures, pressures, shaft speeds, fuel flow, and cooling flows around the fan, compressors, combustor, and turbines. The project code keeps the neutral `sensor_01` to `sensor_21` names because the text files themselves do not embed labels.

The important modelling lesson is that not every sensor is useful in every subset. Some channels are nearly flat, some mostly track operating condition, and some show degradation drift. That is why the EDA workflow records flat-sensor flags, correlations, operating-setting summaries, and per-sensor drift indicators before fitting any model.

## RUL Target Convention

C-MAPSS models usually use a piecewise-linear RUL target. Early in life, RUL is capped around 120-130 cycles because distinguishing "very healthy" from "extremely healthy" is less operationally useful than ranking engines near failure. This repo defaults to a cap of 125 cycles and keeps it configurable through the CLI.

## Evaluation Implications

RMSE is useful but incomplete for aerospace maintenance. NASA's asymmetric score penalizes late predictions more heavily than early predictions because overestimating RUL can delay maintenance past a failure point. The baseline workflow reports both metrics for every subset.

## Current Phase 1 Baseline Framing

The first baseline is a scikit-learn histogram gradient-boosted regressor over engineered per-unit features. It is not meant to be state of the art. It exists to prove that ingestion, target generation, leakage-resistant standardization, reproducible scoring, and artifact generation all work before moving to sequence models.

The Phase 2 models should improve on this by using sliding windows directly: 1D-CNN, LSTM/BiLSTM, TCN, and Transformer variants. Those models should still report against this Phase 1 table so progress is visible and honest.
