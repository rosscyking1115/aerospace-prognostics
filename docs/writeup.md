# Distribution-Free Uncertainty for Turbofan Remaining-Useful-Life Prediction

**Cheng-Yuan King** · aerospace-prognostics · assembled 2026-08-09

This document assembles the project's phase records into one account: what was
built, what was measured, and what the measurements are and are not allowed to
support. It is written for a reader who will check it. Every conformal figure
below is read from a git-tracked artifact under `artifacts/conformal/`; every
earlier figure is read from a git-tracked phase record. Section 9 maps each
number to its source and says plainly which of the two it is, because those are
different grades of evidence.

---

## 1. Summary

The project is a prognostics and health management (PHM) pipeline built around
NASA's C-MAPSS turbofan degradation benchmark, with a second track on spacecraft
telemetry anomaly detection. Its point predictors are ordinary: a
histogram-gradient-boosting policy and a set of sequence models. What the
project is actually about is the discipline around them — held-out protocol,
selection hygiene stated rather than implied, a claims ledger with an enforcing
test, and negative results reported at the same volume as positive ones.

The contribution recorded here is uncertainty quantification with a guarantee
attached. Prior to this round the repository produced RUL intervals by taking a
quantile of validation residuals. That is calibration, and calibration is not a
guarantee: nothing in it bounds the probability that the interval contains the
truth. This round adds split conformal prediction, whose coverage guarantee is
finite-sample and distribution-free — and which is worth exactly nothing if its
exchangeability assumption is violated, which on run-to-failure fleet data it
very easily is.

Three findings are worth stating up front, two of them negative.

1. **On FD001 at 90% nominal, unit-grouped conformal intervals attain coverage
   on average across ten independent unit splits (mean `0.948`, range `0.86` to
   `0.99`), at a mean width of `56.33` cycles against a mean label-only
   reference width of `120.80` over the same ten splits.** Eight of ten splits
   reach nominal individually; the
   guarantee is an average over splits and is reported as one.
2. **FD001 cannot support a distribution-free 99% guarantee at the unit level.**
   This is arithmetic, not a limitation of the model: 99% requires 99 exchangeable
   calibration units, and FD001 has 100 training engines in total. The honest
   interval at that confidence is infinite, and the pipeline returns an infinite
   one rather than a plausible number.
3. **The naive row-pooled design manufactures a finite 99% interval on the same
   data.** Treating each of 3,780 calibration cycles as an independent draw makes
   the rank attainable and yields a radius of `44.99` cycles with `0.99`
   empirical coverage. It looks like a success. It is an artefact of counting one
   engine's correlated trajectory as thousands of engines, and it is the clearest
   evidence in this document of what the exchangeability assumption is for.

---

## 2. Scope, and what this is not

C-MAPSS is simulation output. A result on it is evidence about modelling, not
about engines. No number here is evidence about real fleets, real airframes,
real spacecraft operations, or the safety of any maintenance decision, and
nothing here is a certification, airworthiness, or compliance claim. These
constraints are enforced in the repository by `claims.md`, whose rules forbid
operational readings of benchmark numbers and require that any figure selected
on a test set say so in its own row.

---

## 3. Data and protocol

**C-MAPSS** (NASA turbofan degradation simulation) supplies four subsets. Each
provides run-to-failure training trajectories and a test set whose trajectories
are truncated at an unknown point before failure, with one true remaining useful
life per test engine. FD001 has 100 training and 100 test engines under a single
operating condition; FD002 has 260 training and 259 test engines across six
operating regimes. Targets are capped at 125 cycles, the standard convention:
remaining life is not meaningfully estimable far from failure, and an uncapped
target makes the early-life plateau dominate the loss.

Two structural features of the benchmark matter for everything in section 6.
First, test engines are scored **once**, at their final observed window — not
along a trajectory. Second, C-MAPSS numbers training and test engines
independently, so `train_FD001` unit 10 and `test_FD001` unit 10 are unrelated
machines. The second point is not pedantry: the disjointness guard added in this
round rejected its first legitimate split because bare unit numbers collided,
and the fix was to namespace identifiers by source rather than to weaken the
guard.

**Spacecraft telemetry.** A second track covers SMAP/MSL and the ESA Anomaly
Detection Benchmark (ESA-ADB), scored event-wise on chronological splits. It is
summarised in section 7 and is not the subject of this document.

---

## 4. Point prediction, and how the numbers were selected

The deployable model is a validation-selected histogram-gradient-boosting
policy over engineered rolling-window features, with an operating-regime
transformer fitted on training rows only. On FD001 it reaches official-test RMSE
`13.012889` and NASA score `253.465322`, against a constant-predictor floor of
RMSE `49.819876` and NASA score `166570.542613` — a factor of 3.8 on RMSE and
657 on the NASA score. The NASA gap is the larger one because that score
penalises late predictions exponentially and a constant is late on roughly half
the fleet. Across subsets: FD002 `27.568929` / `8697.396161`, FD003 `14.394433`
/ `358.507217`, FD004 `29.215870` / `7106.739881`.

A deep track (CNN, residual CNN, LSTM/BiLSTM, TCN, Transformer) runs end to end
with NASA-aware asymmetric losses and inference-safe calibration. Its best FD001
row is a calibrated Transformer at RMSE `14.246672` / NASA `271.486206` — behind
the classical policy. That is reported as a negative result on purpose.

**Selection hygiene.** The feature policy, the gradient-boosting parameters, and
the sensor filter were chosen on a unit-held-out validation split that truncates
held-out histories to a 30-cycle horizon, mirroring the official test task. The
per-subset rolling window was **not**: it is the argmax of a sweep scored on the
official test set, and no validation-side window sweep exists in the codebase.
The official test set has also been consulted repeatedly across four successive
test-scored comparisons. The consequence is stated rather than buried: the
headline FD001 figures should be read as mildly optimistic, in the way any
repeatedly-consulted benchmark leaderboard is, and they are not a clean
single-shot held-out estimate. This is recorded in the claims ledger as
disclosed items D1, D2 and D4.

This inheritance matters for section 6. The conformal machinery calibrates only
on engines the model never saw, so the interval is not test-leaked — but the
point predictor it wraps carries a test-selected hyperparameter, and a conformal
interval around a mildly optimistic predictor is a conformal interval around a
mildly optimistic predictor.

---

## 5. Monotonicity: what it is, stated at its real strength

Degradation does not reverse, so an estimate of remaining life should not rise
as an engine ages. The repository uses this expectation in two distinct ways,
and they are frequently conflated in the literature, so they are separated here.

**A diagnostic.** After prediction, violations are counted: cases where the same
engine's predicted RUL increases at a later cycle. This is a post-hoc audit of
model behaviour against a physical expectation. It changes nothing about the
model; it tells you whether the model is behaving sensibly.

**An optional soft penalty.** Some deep training runs add a term to the loss
that penalises same-unit predicted RUL increases at later end cycles, at weight
`0.1`. It is a regularisation term motivated by physics.

Neither is a physics-informed model in the sense that term carries in the
literature. No governing equation, degradation law, or physical residual enters
the objective; the penalty is a monotonicity prior on the output ordering, not a
statement about the system's dynamics. Nor is it a hard constraint — the
architecture cannot guarantee monotone outputs and does not attempt to. The
penalty variants are also not the deployed policy: the Phase 3 recommendation
declines to adopt constrained training losses, on the ground that the available
official-test final rows show zero raw and calibrated monotonicity violations,
so the evidence points at tail calibration rather than at the loss.

Anything stronger than "a physically-motivated diagnostic, plus an optional soft
penalty that is not deployed" would overstate what the code does.

---

## 6. Conformal prediction with unit-grouped calibration

### 6.1 What the guarantee is, and what it rests on

Split conformal prediction turns a point predictor into an interval predictor.
Hold out a calibration set the model never saw; compute a nonconformity score
per calibration point (here, absolute error); take the interval radius to be the
`ceil((n + 1)(1 - alpha))`-th smallest score among `n`. The resulting interval
covers a fresh test point with probability at least `1 - alpha`. The bound is
finite-sample and assumes nothing about the model or the noise distribution.

The `n + 1` is not decoration. It accounts for the test point, which is
exchangeable with the calibration points and could have landed anywhere among
them; a plain empirical quantile omits it and undercovers at every sample size.

The guarantee holds if and only if the calibration scores and the test score are
**exchangeable** — their joint distribution unchanged by permutation. This is
the whole proof. When it fails, the coverage figure is still a number, but it
estimates nothing.

### 6.2 Two ways C-MAPSS breaks exchangeability

**Within-unit dependence.** Consecutive cycles of one engine share a degradation
trajectory, a unit-specific manufacturing offset, and nearly the same noise
realisation. Splitting rows at random puts cycles of the same engine on both
sides, so the "test" residual is partly predicted by calibration residuals it is
not independent of. The implementation makes this unreachable by accident:
calibration takes one score per unit, and `require_disjoint_units` raises if a
unit appears on both sides of a split.

**Covariate shift between calibration and test regimes.** Grouping by unit is
necessary and not sufficient. C-MAPSS test engines are scored once, at a final
window. Calibration scores pooled over whole trajectories are drawn from a
different population — every RUL level, including the long early-life plateau no
test engine is scored on. Two populations that differ in distribution are not
exchangeable with each other however carefully the split is grouped. This is a
covariate-shift problem, not a dependence problem, and it needs its own fix: the
calibration set is built one cycle per unit, drawn uniformly at random from
cycles that could plausibly have been a truncation point.

Two eligibility rules apply to that draw, both stated because both are
assumptions. Cycles whose true RUL exceeds the training cap are excluded,
because the model's target is capped there and it cannot express a larger value;
this is a property of the predictor, not of the test labels. Cycles before cycle
20 are excluded so a calibration engine has comparable observed history to a
test engine. The truncation distribution is uniform by choice — NASA's is
unknown — and any difference between the two is residual covariate shift that
the reported calibration and test RUL distributions leave visible.

Other violations the implementation cannot detect: different fault modes between
calibration and deployment fleets, a model retrained between calibration and
use, operating-condition drift, and any selection of calibration units that
depends on their outcomes.

### 6.3 Design of the study

The same trained model is run through three calibration designs differing by
exactly one shortcut each, so the cost of each is a number rather than an
argument:

| Design | Scores | What it isolates |
|---|---|---|
| `matched_unit_grouped` | one per unit, at a drawn truncation point | the honest design; the only coverage figure that should be quoted |
| `pooled_within_cap` | every cycle within the RUL cap, each as its own draw | difference from the above isolates within-unit dependence |
| `pooled_full_trajectory` | every cycle | difference from the above isolates the calibration-to-test covariate shift |

A constant-median predictor is carried through the identical pipeline as a
control.

**Both directions are measured, always.** Coverage alone is satisfiable by
absence: a predictor emitting infinite intervals covers everything and says
nothing. Width alone is satisfiable by a confidently wrong model. Coverage and
width are therefore returned as one object, and there is deliberately no
function in the module that returns a coverage number on its own.
Informativeness is checked against the width a label-only predictor achieves at
the same confidence — a reference that adapts to the label distribution rather
than a threshold someone invented.

### 6.4 FD001 at 90% nominal

Population: 100 training engines split into 70 for fitting (14,235 cycles) and
30 held out for calibration; evaluation on all 100 official test engines, one
final-window row each. Model: regime-engineered HGB, official-test RMSE
`13.923633` on this 70-engine fit. Eleven test engines have true RUL above the
125-cycle training cap.

| Design | Scores | Rank | Radius | Width | Coverage | Uncovered | Informative |
|---|---:|---:|---:|---:|---:|---:|---|
| `matched_unit_grouped` | 30 | 28 | `37.939764` | `75.879527` | `0.990000` | 1 | yes |
| `pooled_within_cap` | 3,780 | 3,403 | `27.006416` | `54.012832` | `0.950000` | 5 | yes |
| `pooled_full_trajectory` | 6,396 | 5,758 | `78.721341` | `157.442682` | `1.000000` | 0 | **no** |
| control: constant median | 30 | 28 | `59.000000` | `118.000000` | `0.750000` | 25 | **no** |

Label-only reference width: `118.000000`.

Read this carefully, because the naive design does not look bad here. It is
*narrower* than the honest one and still clears nominal. That is exactly the
trap: a leaky calibration design cannot be detected from a single coverage
number, because on any one split it may land anywhere. What it has lost is the
guarantee, not necessarily the number.

The full-trajectory design is the both-directions case made concrete. It covers
100% of test engines — and its interval is `157.44` cycles wide against a
label-only reference of `118.00`, so it is *wider than knowing nothing about the
engine at all*. Reported as coverage alone it is a triumph. Reported as a pair it
is marked uninformative, which is what it is.

The control behaves as a control should: the constant-median predictor is marked
uninformative (its width equals the reference exactly, by construction) and, on
this split, fails coverage at `0.750000`. A framework in which a predictor that
ignores its inputs scores as a success is measuring nothing.

**Repeated splits.** Coverage is an average over exchangeable draws, so a single
split is not evidence either way. Across ten independent unit splits, each
retrained from scratch:

| Design | Mean coverage | Min | Max | Mean width | Splits at or above nominal |
|---|---:|---:|---:|---:|---:|
| `matched_unit_grouped` | `0.948000` | `0.860000` | `0.990000` | `56.329239` | 8 / 10 |
| `pooled_within_cap` | `0.909000` | `0.850000` | `0.970000` | `48.577518` | 5 / 10 |
| `pooled_full_trajectory` | `1.000000` | `1.000000` | `1.000000` | `157.455343` | 10 / 10 |
| control: constant median | `0.818000` | `0.690000` | `0.930000` | `120.800000` | 3 / 10 |

The control's mean width, `120.800000`, is also the mean label-only reference
width across these ten splits, because the constant-median predictor and the
reference are the same predictor by construction. The honest design's mean width
of `56.329239` is therefore less than half of what the label distribution alone
would give.

The honest design attains nominal on average and over-covers, as small-`n`
conformal prediction generally does. Two of ten splits fall below nominal
individually; that is the finite-sample variability the guarantee allows, not a
failure, and it is reported rather than smoothed. The row-pooled design sits on
the nominal line with half its splits below it — no better than a coin toss at
the level it claims.

**By subgroup.** Splitting the primary measurement by whether the truth is
inside the model's expressible range: within the cap, `0.988764` over 89
engines; above the cap, `1.000000` over 11 engines.

### 6.5 The attainability limit, derived

The rank is bounded by the calibration size — there is no `(n + 1)`-th order
statistic among `n` scores — so a rank above `n` leaves the infinite interval as
the only honest answer. Requiring `ceil((n + 1)(1 - alpha)) <= n` gives
`n >= 1/alpha - 1`:

| Nominal | Minimum calibration units | FD001 (100 train) leaves | FD002 (260 train) leaves |
|---:|---:|---:|---:|
| 0.80 | 4 | 96 | 256 |
| 0.90 | 9 | 91 | 251 |
| 0.95 | 19 | 81 | 241 |
| 0.98 | 49 | 51 | 211 |
| 0.99 | 99 | **1** | 161 |

This is a property of the dataset and the confidence level, not of the model.
A 99% distribution-free interval on FD001 needs 99 calibration engines and there
are 100 training engines in total, leaving one engine to fit on. FD001 cannot
support a 99% unit-level guarantee. Neither can FD003, which also has 100
training engines.

The pipeline behaves accordingly. At `alpha = 0.01` with 30 calibration units it
returns rank 31 against 30 scores, an infinite radius, coverage `1.000000`, and
an explicit uninformative flag. It refuses rather than inventing a number.

**And this is where the leak becomes visible.** On the identical data and the
identical model, the row-pooled design counts 3,780 calibration cycles as 3,780
draws, makes rank 3,744 attainable, and returns a finite radius of `44.994157`
with `0.990000` empirical coverage. A reader shown only that row would conclude
that 99% coverage on FD001 is achievable and costs about 90 cycles of width. It
is not achievable, and the number is an artefact of treating one engine's
correlated trajectory as thousands of independent engines. The honest design's
refusal and the naive design's confident answer are the same experiment.

### 6.6 FD002 at 99% nominal: attainable, and it misses

FD002 has 260 training engines, so the arithmetic bar is clear. This is the
control in the other direction: a subset where 99% is attainable, run to see
whether attainability is sufficient. It is not.

Population: 161 engines for fitting (33,797 cycles), 99 held out for
calibration, 259 official test engines, official-test RMSE `27.087121`,
57 test engines above the training cap. The rank is 99 against 99 scores — the
maximum calibration score exactly, the boundary case.

| Design | Scores | Width | Coverage | Nominal | Informative |
|---|---:|---:|---:|---:|---|
| `matched_unit_grouped` | 99 | `100.179520` | `0.907336` | `0.99` | yes |
| `pooled_within_cap` | 12,474 | `108.514180` | `0.930502` | `0.99` | yes |
| `pooled_full_trajectory` | 19,962 | `354.386401` | `1.000000` | `0.99` | **no** |
| control: constant median | 99 | `142.000000` | `0.830116` | `0.99` | **no** |

Empirical coverage of `0.907336` against a nominal `0.99` is a miss, and across
ten splits it is systematic: mean `0.916988`, range `0.891892` to `0.965251`,
**zero of ten splits at or above nominal**. The finite interval is attainable and
the coverage claim is not met.

The subgroup split locates the cause precisely. Within the training cap,
coverage is `0.985149` over 202 engines — three uncovered, essentially nominal.
Above the cap, coverage collapses to `0.631579` over 57 engines, 21 of them
uncovered. FD002 puts 22% of its test fleet outside the range the model was
trained to express, and a symmetric interval centred on a systematically capped
prediction cannot reach a truth that sits beyond the cap however wide it is.

This is a finding, not a knob. It could be made to pass by widening the interval,
by excluding above-cap engines from the report, or by raising the cap — the first
buys coverage with meaninglessness, the second is choosing the population after
seeing the result, and the third is a change to the point predictor that would
have to be re-evaluated on its own terms. None was done. What the failure
actually says is that distribution-free calibration cannot repair a systematically
biased centre, which is a limit of conformal prediction worth knowing.

---

## 7. Spacecraft anomaly track, in brief

The second track establishes protocol-shaped event-wise detection evidence.
On ESA-ADB Mission 1 (channels 41–46) a robust z-score baseline at fixed
threshold 5 gives precision `1.000`, recall `0.415`, F0.5 `0.780`; on Mission 2
(channels 18–28) a validation-selected threshold of 20 gives `0.999` / `0.986` /
`0.997`. Mission 2's figures are lenient rather than strong: its events are
mostly long "Rare Event" subsequences and event-wise detection counts an event
caught if any sample in its interval fires. Only the detection tier of the
official metric hierarchy is computed, and the reported precision uses a
run-based definition that diverges from ESA-ADB's TNR-corrected `EW_precision`.
On SMAP/MSL the comparison-ready robust policy scores mean point-wise F1
`0.160525` at false-alarm rate `0.134247`; point-adjusted F1 `0.541768` is
tracked because the literature uses it, and is not the primary readout because
point adjustment flatters weak detectors on long labelled intervals.

One correction is recorded because it *raised* a published number: an early
Mission 1 recall of `0.236842` counted training-window events, which have no
test-window predictions, as missed. Restricting the event set to events
overlapping the test window gives the honest `0.415385`.

---

## 8. What this work does not establish

Stated at the same length as what it does, because a reader who stops here
should not be misled.

**It does not establish anything about real engines.** C-MAPSS is simulation
output. Every coverage figure above is a statement about a simulator's
trajectories under a modelling protocol. No result here supports a claim about
real fleets, airframes, spacecraft operations, or the safety of any maintenance
decision, and none is certification evidence.

**It does not establish that the intervals are conditionally valid.** Conformal
coverage is *marginal*: averaged over the population. It does not promise that
coverage holds for a specific engine, a specific operating regime, or a specific
region of RUL. The FD002 subgroup result is a direct demonstration — marginal
coverage of `0.907` decomposes into `0.985` below the cap and `0.632` above it.
Marginal validity is compatible with badly unequal treatment of subgroups, and
subgroup reporting is a partial mitigation, not conditional validity.

**It does not establish that the exchangeability assumption holds.** It shows
that two specific violations were removed by construction and that a third —
the truncation-distribution mismatch between the uniform calibration draw and
NASA's unknown test truncation — remains. Fault-mode drift between calibration
and deployment fleets, operating-condition drift, and outcome-dependent unit
selection would all break the guarantee and are not detectable by the code.

**It does not establish a clean held-out estimate of point-predictor quality.**
The rolling window is a test-selected hyperparameter (D1), the official test set
has been consulted repeatedly (D4), and the deep comparison ranks candidates on
test (D2). The conformal calibration itself never sees test rows, but it wraps a
predictor whose configuration partly does. The reported coverage is a property
of the interval procedure; the reported width is a property of a mildly
test-selected model.

**It does not establish that 99% coverage is out of reach for turbofan RUL.** It
establishes that FD001 and FD003 cannot support it at the unit level with
100 training engines each, and that FD002 does not attain it with this predictor.
A larger fleet, or a predictor without a hard target cap, might.

**It does not establish that the seed sweep bounds the variability.** Ten splits
give a rough picture of spread and no more; nothing here supports a confidence
interval on the coverage itself, and two-of-ten below nominal on FD001 should not
be read as a 20% failure rate.

**It does not establish anything about the deep models' uncertainty.** The
conformal study runs on the CPU gradient-boosting policy. The deep track's
prediction CSVs carry only 20 validation-selection units, which supports 95% at
the boundary and cannot support 99% at all; extending the study to those models
would require re-running inference with a larger held-out fleet.

**It does not establish that the monotonicity work constrains the model.** It is
a diagnostic plus an optional, undeployed soft penalty. No physics enters the
objective.

---

## 9. Provenance

Two grades of evidence appear above, and they are not equivalent.

**Grade A — traces to a git-tracked artifact.** Every conformal figure. The
files are under `artifacts/conformal/`, which is tracked by an explicit
`.gitignore` negation, and `tests/test_conformal_artifacts_tracked.py` fails if
that tracking is removed — so the claim is enforced rather than attested.

| Figures | Artifact |
|---|---|
| §6.4 FD001 90% table, subgroups, seed sweep | `artifacts/conformal/cmapss_fd001_conformal.json`, `_variants.csv`, `_seed_sweep.csv`, `.md` |
| §6.5 attainability table, FD001 99% refusal and the pooled 99% artefact | `artifacts/conformal/cmapss_fd001_conformal_attainability.csv`, `cmapss_fd001_conformal_alpha001.json`, `_variants.csv` |
| §6.6 FD002 99% table, subgroups, seed sweep | `artifacts/conformal/cmapss_fd002_conformal_alpha001.json`, `_variants.csv`, `_seed_sweep.csv` |

**Grade B — traces to a git-tracked phase record, not to a committed artifact.**
Every figure in sections 4 and 7. The repository's run outputs are local by
default; these numbers were recorded into the phase documents when produced and
the artifacts that produced them are regenerable but not committed. That is an
attestation with a reproduction command behind it, and it is a weaker guarantee
than Grade A. Sources: `docs/public_results.md`, `docs/phase1_cmapss_baseline_results.md`,
`docs/phase2_cmapss_deep_baselines.md`, `docs/phase3_esa_adb_intake.md`,
`docs/phase3_cmapss_recommendation.md`, and the ledger in `claims.md`.

### Reproduction

```bash
uv run aerospace-prognostics cmapss-conformal --data-dir data/raw/cmapss --subset FD001 --alpha 0.10 --calibration-units 30 --seed-sweep
```

```bash
uv run aerospace-prognostics cmapss-conformal --data-dir data/raw/cmapss --subset FD002 --alpha 0.01 --calibration-units 99 --seed-sweep --stem cmapss_fd002_conformal_alpha001
```

C-MAPSS is not redistributed; `docs/datasets.md` records how to obtain it. Both
commands are CPU-only and complete in under three minutes on a laptop. The
conformal module is `src/aerospace_prognostics/uncertainty/conformal.py`, whose
module docstring states the exchangeability assumption and what violates it; the
study is `src/aerospace_prognostics/experiments/cmapss_conformal.py`.
