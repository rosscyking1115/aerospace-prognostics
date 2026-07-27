# Claims ledger

Every number this repository publishes has a row here: what produced it, what it is allowed
to support, and whether anything about it is disclosed-but-unresolved. If a number appears in
the README or in `docs/` and has no row here, it is not a claim this project makes.

This ledger was created on 2026-07-27 during the first claims-versus-evidence audit
(`docs/release-check-2026-07-27.md`). It is deliberately scoped to what this repository can
evidence from its own committed code and recorded runs.

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

## C-MAPSS turbofan RUL

| # | Claim (with scope) | Produced by | Allowed to claim | Status |
|---|---|---|---|---|
| C1 | FD001 official-test RMSE `13.012889`, NASA score `253.465322`, validation-selected HGB policy | `run_cmapss_validation_selected_hgb_policy_default_windows`; recorded in `docs/phase1_cmapss_baseline_results.md` | That this pipeline reproduces a competitive classical RUL baseline on a simulated benchmark | **Disclosed caveat — see D1** |
| C2 | FD002 `27.568929` / `8697.396161`, FD003 `14.394433` / `358.507217`, FD004 `29.215870` / `7106.739881` | as C1 | as C1, across the multi-regime subsets | **Disclosed caveat — see D1** |
| C3 | Best FD001 deep row: calibrated Transformer, asymmetric late loss + monotonic penalty, RMSE `14.246672` / NASA `271.486206` | `experiments/cmapss_deep_baseline.py` + `reports/cmapss_prediction_calibration.py`; `docs/phase2_cmapss_deep_baselines.md` | That the deep track runs end to end and is **behind** the classical policy — reported as a negative result on purpose | **Disclosed caveat — see D2** |
| C4 | Naive floor, FD001: `train_median` RMSE `49.819876` / NASA `166570.542613`; `rul_cap` RMSE `64.615323` / NASA `1502475.412851` | `run_cmapss_naive_baseline`, command `cmapss-naive-baseline`, measured 2026-07-27 on `data/raw/cmapss` | That C1 beats a constant predictor 3.8x on RMSE and 658x on NASA score, so the headline reflects learned signal rather than the label distribution | Clean |
| C5 | Calibration is fit on validation predictions only and never touches official test rows | `fit_cmapss_predicted_rul_bin_nasa_shift_calibrations`; verified by source inspection during the audit | That the reported calibration improvement is not test-leaked | Clean |
| C6 | The validation split holds out whole units and truncates their histories to a 30-cycle horizon | `make_cmapss_temporal_validation_split` | That model selection used a grouped, temporally realistic split — no engine appears on both sides, and the validation task mirrors the official test task | Clean |

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
| S1 | Comparison-ready robust threshold policy: mean point-wise F1 `0.160525`, mean false-alarm rate `0.134247`, versus the default baseline's `0.187988` | `docs/phase2_spacecraft_anomaly_baselines.md` | A baseline and alert-policy layer only — **not** a reproduced Telemanom result | Clean |
| S2 | Point-adjusted F1 `0.541768` | as S1 | Tracked because the literature uses it; point-wise metrics remain the primary readout because point adjustment flatters weak detectors on long labelled intervals | Clean |

## Engineering envelope

| # | Claim (with scope) | Produced by | Allowed to claim | Status |
|---|---|---|---|---|
| T1 | `456 passed`, full suite, measured 2026-07-27 | `uv run pytest` | That the suite is green, across Python 3.11/3.12/3.13 in CI. The count is **not** pinned in a badge, because a hard-coded count rots silently | Clean |
| T2 | The suite contains a genuine skill regression, not only plumbing tests | `test_learned_baseline_beats_the_naive_floor` on `write_discriminating_cmapss_subset` | That an estimator which lost its predictive power turns CI red. Demonstrated by sabotage: a `DummyRegressor` that ignores the sensors scores `33.72` against the floor's `34.63` and fails the test | Clean |
| T3 | Type checking is scoped, not repo-wide | `[tool.mypy]` in `pyproject.toml` | That seven numeric/evaluation modules type-check clean and are gated in CI. This repo does **not** run mypy strict: repo-wide checking emits 332 errors across 22 of 81 files | Clean |

## Disclosed but unresolved

Open items. Each is stated in the public docs rather than fixed silently; none is a defect
being hidden until it is convenient.

| # | Item | Effect on published numbers | Why it is open, not fixed |
|---|---|---|---|
| **D1** | **The per-subset rolling window was selected on the official test set.** `CMAPSS_ENGINEERED_DEFAULT_WINDOWS` (FD001:10, FD002:3, FD003:5, FD004:3) is exactly the argmax of the rolling-window sweep in `docs/phase1_cmapss_baseline_results.md`, and that sweep is scored on official test. No validation-side window sweep exists in the codebase. The feature policy, HGB parameters and sensor filter *are* validation-selected; the window is not. | C1 and C2 should be read as **mildly optimistic**, in the way any repeatedly-consulted benchmark leaderboard is. They are honest reproducible checkpoints, not a clean single-shot held-out estimate. | Resolving it means sweeping the window on validation and re-reporting — a change to published numbers. Folding a number movement into an audit branch would hide it inside a cleanup. It gets its own round. |
| **D2** | The FD001 deep comparison ranks five candidates by official-test RMSE and NASA score. | C3's "best deep row" is selected on test, so the margin between the top deep rows is not a held-out estimate. The load-bearing claim — that the deep track is *behind* the classical policy — is unaffected. | Same as D1: re-ranking on validation moves published numbers. |
| **D3** | Only the detection tier of the official ESA-ADB metric hierarchy is computed, and the official zero-order-hold resampling is not applied. The lightweight channels for these missions share a native grid, so the baseline scores on that grid; the loader rejects any mission whose channels do not share one. | E1 and E2 are **protocol-shaped detection evidence, not a leaderboard claim**, and every artifact is stamped as such. | Applying the full hierarchy is upstream work in `telemeval`, not a correction to a wrong number here. |
| **D4** | The official C-MAPSS test set has been consulted repeatedly — `docs/phase1_cmapss_baseline_results.md` records four successive test-scored comparisons. | Compounds D1. | Recorded so a reader can weigh it; it cannot be undone retrospectively. |

## Not claimed

Stated explicitly so absence is not mistaken for modesty:

- No claim about real fleets, real airframes, real spacecraft operations, or the safety of any
  maintenance decision.
- No airworthiness, certification, or compliance claim.
- No claim of a clean single-shot held-out C-MAPSS estimate (see D1, D2, D4).
- No full ESA-ADB leaderboard claim (see D3).
- No claim of repo-wide or strict type checking (see T3).
- No claim about datasets this repository does not use, or about any sibling repository.
