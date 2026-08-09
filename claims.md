# Claims ledger

Every number this repository publishes has a row here: what produced it, what it is allowed
to support, and whether anything about it is disclosed-but-unresolved. If a number appears in
the README or in `docs/` and has no row here, it is not a claim this project makes.

This ledger was created on 2026-07-27 during the first claims-versus-evidence audit
(`docs/release-check-2026-07-27.md`). It is deliberately scoped to what this repository can
evidence from its own committed code and recorded runs.

**This file is the worked template for the other repositories in the portfolio.** The audit
found that discipline is currently inverted across it: the pre-Gate-0 scaffold repositories
all ship a `claims.md` and an explicit "Nothing here is a result yet", while the mature
repositories — the ones with real results and real readers — carry neither. The scaffolding
template solved this before there was anything to overclaim; the repositories that predate
the template never received it, and those are precisely the ones whose numbers someone might
rely on. Copy the shape of this file, not its rows: the four columns, the disclosed-but-
unresolved section, and `tests/test_claims_ledger.py`, which enforces the ledger rather than
trusting it. See §6 and §7 of the audit report.

## Rules

- A row states the **scope** of its number: dataset, subset, protocol, and what was held out.
- **Benchmark data supports claims about method, never about operations.** No row may be read
  as evidence about real fleets, real airframes, real spacecraft, or the safety of any
  maintenance decision. C-MAPSS is simulation output; a result on it is evidence about
  modelling, not about engines.
- No certification, airworthiness, or compliance claim. "Production-grade" is banned.
- A number selected on the official test set must say so in its own row. Selection hygiene is
  part of the claim, not a footnote to it.
- Corrections that *raise* a reported number are recorded explicitly, never quietly adopted.
- Audited at each release; rows are added when a number is published, not afterwards.
- **A measurement taken during a change must be re-taken after that change lands**, or stamped
  with the commit it was measured at. The first audit shipped two figures measured mid-flight
  — the test count and the repo-wide mypy error count — both already stale by the time the
  branch was reviewed.
- A "Produced by" cell must name a symbol that exists. This is enforced by
  `tests/test_claims_ledger.py`, not by review: the first audit cited a function name that had
  never existed, and review is exactly where that kind of error survives.

## C-MAPSS turbofan RUL

| # | Claim (with scope) | Produced by | Allowed to claim | Status |
|---|---|---|---|---|
| C1 | FD001 official-test RMSE `13.012889`, NASA score `253.465322`, validation-selected HGB policy | `run_all_cmapss_validation_selected_hgb_policy_default_windows`; recorded in `docs/phase1_cmapss_baseline_results.md` | That this pipeline reproduces a competitive classical RUL baseline on a simulated benchmark | **Disclosed caveat — see D1** |
| C2 | FD002 `27.568929` / `8697.396161`, FD003 `14.394433` / `358.507217`, FD004 `29.215870` / `7106.739881` | as C1 | as C1, across the multi-regime subsets | **Disclosed caveat — see D1** |
| C3 | Best FD001 deep row: calibrated Transformer, asymmetric late loss + monotonic penalty, RMSE `14.246672` / NASA `271.486206` | `src/aerospace_prognostics/experiments/cmapss_deep_baseline.py` + `src/aerospace_prognostics/reports/cmapss_prediction_calibration.py`; `docs/phase2_cmapss_deep_baselines.md` | That the deep track runs end to end and is **behind** the classical policy — reported as a negative result on purpose | **Disclosed caveat — see D2** |
| C4 | Naive floor, FD001: `train_median` RMSE `49.819876` / NASA `166570.542613`; `rul_cap` RMSE `64.615323` / NASA `1502475.412851` | `run_cmapss_naive_baseline`, command `cmapss-naive-baseline`, measured 2026-07-27 on `data/raw/cmapss` | That C1 beats a constant predictor 3.8x on RMSE and 657x on NASA score, so the headline reflects learned signal rather than the label distribution | Clean |
| C5 | Calibration is fit on validation predictions only and never touches official test rows | `fit_cmapss_predicted_rul_bin_nasa_shift_calibrations`; verified by source inspection during the audit | That the reported calibration improvement is not test-leaked | Clean |
| C6 | The validation split holds out whole units and truncates their histories to a 30-cycle horizon | `make_cmapss_temporal_validation_split` | That model selection used a grouped, temporally realistic split — no engine appears on both sides, and the validation task mirrors the official test task | Clean |

## C-MAPSS uncertainty

Coverage and width are stated together in every row here. A coverage figure quoted without
its width is not a claim this project makes: an infinite interval covers everything.

Two different kinds of interval exist in this repository and they are not interchangeable.
U1–U7 describe unit-grouped split conformal intervals, which carry a distribution-free
coverage guarantee. U8 describes the band the **serving artifact** actually returns, which
does not. A reader who conflates them will over-read the deployed system, and U9 is the
test that stops the served field from inviting exactly that.

| # | Claim (with scope) | Produced by | Allowed to claim | Status |
|---|---|---|---|---|
| U1 | FD001, nominal `0.90`, unit-grouped split conformal across ten unit splits: mean coverage `0.948000`, mean width `56.329239` cycles, 8 of 10 splits at or above nominal. Single split (seed 42): coverage `0.990000` at width `75.879527` on all 100 official test engines, 30 calibration engines held out of training | `run_cmapss_conformal_seed_sweep`, command `cmapss-conformal`; `artifacts/conformal/cmapss_fd001_conformal_seed_sweep.csv` | That a distribution-free interval procedure attains its nominal **marginal** coverage on this benchmark when calibration is grouped by engine. Not conditional coverage, and not a statement about any single engine | **Disclosed caveat — see D1** |
| U2 | FD001 cannot support a distribution-free 99% guarantee at the unit level: the rank `ceil((n+1)(1-alpha))` is attainable only for `n >= 1/alpha - 1`, so 99% needs 99 calibration engines and FD001 has 100 training engines in total | `minimum_calibration_size`, `build_attainability_table`; `artifacts/conformal/cmapss_fd001_conformal_attainability.csv` | That the limit is arithmetic — a property of the fleet size and the confidence level, not of the model. FD003 has the same fleet size and the same limit | Clean |
| U3 | Row-pooled calibration manufactures a finite 99% FD001 interval — radius `44.994157`, coverage `0.990000` — where the unit-grouped design on identical data returns an infinite radius and refuses | `run_cmapss_conformal_study`; `artifacts/conformal/cmapss_fd001_conformal_alpha001_variants.csv` | That the naive design's apparent success is an artefact of counting 3,780 correlated cycles as 3,780 independent draws. Published as evidence **against** the naive design, not as a result | Clean |
| U4 | FD002, nominal `0.99`, unit-grouped: coverage `0.907336` at width `100.179520`, and zero of ten splits reach nominal (mean `0.916988`) | as U1; `artifacts/conformal/cmapss_fd002_conformal_alpha001_seed_sweep.csv` | A **negative result**: an attainable rank is not sufficient for coverage. Reported rather than tuned | **Disclosed caveat — see D6** |
| U5 | Calibration units are held out of training and may not reappear in evaluation; a unit on both sides raises | `require_disjoint_units`, `fit_split_conformal_interval` | That the reported coverage is not inflated by within-unit leakage, and that the guard is enforced in code rather than observed by convention | Clean |
| U6 | Both-direction controls: a constant-median predictor is marked uninformative (FD001 width `118.000000`, equal to the label-only reference), and a known-good predictor is not rejected | `label_only_reference_width`, `evaluate_conformal_intervals`; `tests/test_conformal.py` | That the framework can tell an informative interval from an uninformative one. Without this, "100% coverage" from an infinite interval would score as success | Clean |
| U7 | Every conformal number above traces to a git-tracked artifact under `artifacts/conformal/`, and the tracking is enforced | `tests/test_conformal_artifacts_tracked.py` | That this section is checkable rather than attested. The rest of this ledger traces to git-tracked phase records, which is a weaker guarantee, and `docs/writeup.md` §9 separates the two grades | Clean |
| U8 | The **served** interval is `train_residual_absolute_quantile` at `interval_quantile_level` `0.90`: the 0.90 absolute quantile of residuals on the model's own training rows | `_fit_rul_interval_calibration` in `src/aerospace_prognostics/deployment/artifacts.py`; exposed by `POST /predict` and the console | That a width is attached to each served prediction and its method is disclosed. **Not** that the band covers the truth 90% of the time: the quantile is in-sample, so it is optimistic by construction, and no coverage guarantee attaches to it. The conformal work in U1–U7 is not wired into serving | **Disclosed caveat — see D7** |
| U9 | No served surface names the quantile level a confidence. The field was `interval_confidence` until 2026-08-09; the value is unchanged | `tests/test_interval_naming.py`, checking the prediction contract, the API response schema, the app database schema, and the fitted calibration payload | That the correction is enforced rather than disclosed. A field name travels with the data into every consumer, so six documents carrying a caveat could not undo a name that asserted a guarantee | Clean |

## ESA-ADB spacecraft anomaly detection

| # | Claim (with scope) | Produced by | Allowed to claim | Status |
|---|---|---|---|---|
| E1 | Mission 1 (ch 41–46), fixed τ=5: precision `1.000`, recall `0.415`, F0.5 `0.780` | `run_mission_lightweight`; `docs/phase3_esa_adb_intake.md` | Event-wise **detection-tier** evidence under a stated protocol deviation | **Scope-bounded — see D3** |
| E2 | Mission 2 (ch 18–28), validation-selected τ=20: precision `0.999`, recall `0.986`, F0.5 `0.997` | as E1 | as E1. Explicitly **lenient, not SOTA** — Mission 2's events are mostly long "Rare Event" subsequences and event-wise detection counts an event caught if any sample in its interval fires | **Scope-bounded — see D3** |
| E3 | The robust z-score baseline is fit on nominal **training** points only; the split is chronological | `robust_train_fit_zscores`, `chronological_split` | That no test row informed the fit, standardisation, or threshold | Clean |
| E4 | Recall correction: first Mission 1 run reported `0.236842`; the honest figure is `0.415385` | `filter_events_to_window`; recorded in `docs/public_results.md` | That the scorer counted training-window events — which have no test-window predictions — as missed. The correction **raises** the number, so it is recorded rather than quietly adopted | Clean |

## SMAP/MSL spacecraft anomaly

| # | Claim (with scope) | Produced by | Allowed to claim | Status |
|---|---|---|---|---|
| S1 | Comparison-ready robust threshold policy: mean point-wise F1 `0.160525`, mean false-alarm rate `0.134247`. The default robust z-score baseline scores F1 `0.165343` and false-alarm rate `0.187988` | `docs/phase2_spacecraft_anomaly_baselines.md` | That the policy trades a *slightly worse* F1 for a materially lower false-alarm rate. Both figures are stated because the tradeoff, not a win, is the claim. A baseline and alert-policy layer only — **not** a reproduced Telemanom result | Clean |
| S2 | Point-adjusted F1 `0.541768` | as S1 | Tracked because the literature uses it; point-wise metrics remain the primary readout because point adjustment flatters weak detectors on long labelled intervals | Clean |

## Engineering envelope

| # | Claim (with scope) | Produced by | Allowed to claim | Status |
|---|---|---|---|---|
| T1 | `541 passed`, full suite, measured 2026-08-09 after the interval rename (was `534 passed` earlier that day on the conformal round, and `462 passed` on 2026-07-28) | `uv run pytest` | That the suite is green, across Python 3.11/3.12/3.13 in CI. The count is **not** pinned in a badge, because a hard-coded count rots silently | **Disclosed caveat — see D5** |
| T2 | The suite contains a genuine skill regression, not only plumbing tests | `test_learned_baseline_beats_the_naive_floor` on `write_discriminating_cmapss_subset` | That an estimator which lost its predictive power turns CI red. Demonstrated by sabotage: a `DummyRegressor` that ignores the sensors scores `33.72` against the floor's `34.63` and fails the test | Clean |
| T3 | Type checking is scoped, not repo-wide | `[tool.mypy]` in `pyproject.toml` | That eight numeric/evaluation modules type-check clean and are gated in CI — the conformal module was added to the gate in the same round that introduced it. This repo does **not** run mypy strict: `mypy src` emits 335 errors across 22 of 86 files, measured 2026-08-09 (was 331 across 21 of 81 on 2026-07-27) | Clean |

## Disclosed but unresolved

Open items. Each is stated in the public docs rather than fixed silently; none is a defect
being hidden until it is convenient.

| # | Item | Effect on published numbers | Why it is open, not fixed |
|---|---|---|---|
| **D1** | **The per-subset rolling window was selected on the official test set.** `CMAPSS_ENGINEERED_DEFAULT_WINDOWS` (FD001:10, FD002:3, FD003:5, FD004:3) is exactly the argmax of the rolling-window sweep in `docs/phase1_cmapss_baseline_results.md`, and that sweep is scored on official test. No validation-side window sweep exists in the codebase. The feature policy, HGB parameters and sensor filter *are* validation-selected; the window is not. | C1 and C2 should be read as **mildly optimistic**, in the way any repeatedly-consulted benchmark leaderboard is. They are honest reproducible checkpoints, not a clean single-shot held-out estimate. | Resolving it means sweeping the window on validation and re-reporting — a change to published numbers. Folding a number movement into an audit branch would hide it inside a cleanup. It gets its own round. |
| **D2** | The FD001 deep comparison ranks five candidates by official-test RMSE and NASA score. | C3's "best deep row" is selected on test, so the margin between the top deep rows is not a held-out estimate. The load-bearing claim — that the deep track is *behind* the classical policy — is unaffected. | Same as D1: re-ranking on validation moves published numbers. |
| **D3** | Only the detection tier of the official ESA-ADB metric hierarchy is computed, and the official zero-order-hold resampling is not applied. The lightweight channels for these missions share a native grid, so the baseline scores on that grid; the loader rejects any mission whose channels do not share one. Separately, the reported **precision** uses `telemeval`'s run-based event-wise definition, which diverges from ESA-ADB's TNR-corrected `EW_*`; recall matches ESA-ADB exactly. | E1 and E2 are **protocol-shaped detection evidence, not a leaderboard claim**, and every artifact is stamped as such. Precision figures should be read as telemeval event-wise precision, not as an ESA-ADB `EW_precision` value. | **The stated reason for this row has changed and is recorded rather than rewritten.** It previously read "applying the full hierarchy is upstream work in `telemeval`". That is no longer true: `telemeval` now ships ADTQC detection timing and channel/subsystem-aware F-beta. What remains is local — this pipeline does not yet call them, and the resampling is not implemented here. Neither is a correction to a wrong number; both widen what could be reported. |
| **D4** | The official C-MAPSS test set has been consulted repeatedly — `docs/phase1_cmapss_baseline_results.md` records four successive test-scored comparisons. | Compounds D1. | Recorded so a reader can weigh it; it cannot be undone retrospectively. |
| **D5** | **One unreproduced test failure. The cause is not known.** `test_seed_quickstart_workspace_persists_model_and_evidence` failed once on 2026-07-27, in a full-suite run whose preceding commit changed only markdown. It has passed on every attempt since — 20 consecutive runs across two branches, 10 of them targeting that file alone. Ruled out by checking rather than assuming: test-order dependence (no `pytest-randomly`/`xdist`/`pytest-order` is installed, so collection order is deterministic), shared state (per-test `tmp_path`), time dependence in the idempotency key (the conflict key `{type}:{artifact_id}:{sha256}` is content-addressed), and parallelism (single-process). **Which assertion failed was never established** — the traceback was not captured, which is why 20 runs settled so little. | T1's "the suite is green" is true of every observed run, and is **not** a claim that the suite is deterministic. One failure in 21 observed runs supports **no** failure-rate estimate; do not read a frequency into it. | The cause is unknown and guessing at a fix would be worse than disclosing it. What exists is narrower than a diagnosis: `test_a_missing_evidence_file_is_silently_skipped_not_reported` demonstrates **a** failure mode *consistent with* the symptom — a missing evidence file produces a silent short count with no error or warning — but nothing establishes that this is what actually happened. The flaky test now prints its evidence inputs on failure, so the next occurrence can be diagnosed from a CI log instead of another 20 runs. Deliberately not a retry or an `xfail`: both would suppress the signal rather than resolve it. |
| **D6** | **FD002's `0.907336` is marginal coverage that describes no engine in the fleet.** The conformal guarantee is marginal — averaged over the population — and promises nothing conditional on any subgroup. Splitting the same measurement by whether the truth lies inside the range the model can express: within the 125-cycle training cap, `0.985149` over 202 engines, essentially nominal; above the cap, `0.631579` over 57 engines, 21 uncovered. The cap carves out a subpopulation where calibration and test are not exchangeable — calibration cycles are drawn only from at or below the cap — so a band centred on a capped prediction cannot reach a truth beyond it at any width. The headline averages the two and lands on a figure neither subgroup experiences. | U4's figure stands exactly as reported. It is not evidence against conformal prediction, which never claimed conditional coverage; it is an empirical demonstration of the standard critique, and of the fact that distribution-free calibration cannot repair a systematically biased centre. A marginal coverage figure quoted without a subgroup breakdown is not safe to act on. | Fixing it means changing the point predictor — raising or removing the RUL cap — which moves every published FD002 number and needs its own evaluation. The two cheap alternatives were both rejected: widening the interval buys coverage with meaninglessness, and dropping above-cap engines from the report is choosing the population after seeing the result. |

| **D7** | **The served band is fitted in-sample.** Its width is a quantile of the model's absolute residuals on its own training rows, so it is optimistic by construction, and no coverage guarantee attaches to it. This is the remaining half of the original D7; the other half — the field being *named* `interval_confidence` — closed on 2026-08-09 and is now enforced by U9. | U8 stands as written. The number is honest about what it is and is not a coverage claim. | Closing it means either wiring unit-grouped conformal calibration into the packaged artifact or recomputing the band out-of-sample. Both change the deployed uncertainty policy and every served width, and the first carries a real design question this ledger will not settle in a footnote: what the API returns when the honest conformal answer is an infinite interval — see U2 and `docs/writeup.md` §6.6. It comes back as its own round, with a design. |

## Not claimed

Stated explicitly so absence is not mistaken for modesty:

- No claim about real fleets, real airframes, real spacecraft operations, or the safety of any
  maintenance decision.
- No airworthiness, certification, or compliance claim.
- No claim of a clean single-shot held-out C-MAPSS estimate (see D1, D2, D4).
- No full ESA-ADB leaderboard claim (see D3).
- No claim of repo-wide or strict type checking (see T3).
- No claim of **conditional** conformal coverage. U1 is marginal coverage, averaged over the
  population; D6 shows the same interval treating two subgroups very differently.
- No claim that the exchangeability assumption holds — only that two named violations were
  removed by construction and a third, the truncation-distribution mismatch, remains.
- No claim about datasets this repository does not use, or about any sibling repository.
