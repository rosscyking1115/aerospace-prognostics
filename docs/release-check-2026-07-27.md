# Claims-vs-Evidence Audit — 2026-07-27

First independent audit of this repository. Baseline at commit `4530928`
(148 tracked Python files, 294 commits, MIT). Every claim below was measured,
not inferred; the exact commands are recorded so the verdicts can be re-run.

**Headline:** the repository's public claims are, with a few exceptions,
accurate. The test badge was true. The evaluation is real. Three defects
mattered: (1) the README carried no benchmark-vs-operational boundary; (2) one
hyperparameter in the headline C-MAPSS policy was selected on the official test
set while the surrounding prose implied otherwise; and (3) the test fixtures
were degenerate, so the suite proved the plumbing worked but could not have
caught a model that lost its predictive power. All three are fixed, and the
third is verified by sabotage rather than by assertion.

## Verdicts

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Test-count badge | **Pass, then hardened** | `451 passed` measured; badge was a static shields.io URL |
| 2 | CI badge | Pass | Live GitHub Actions endpoint, cannot rot |
| 3 | Licence claim | Pass | `LICENSE` is MIT; `pyproject.toml` agrees |
| 4 | Dataset identity, licence, provenance | **Fixed** | C-MAPSS/SMAP-MSL terms were unrecorded |
| 5 | Benchmark-vs-operational boundary | **Fixed (highest priority)** | Present in `docs/`, absent from README |
| 6 | Held-out split genuine and temporal | Pass | Unit-wise + horizon-truncated; chronological |
| 7 | Baseline that could beat the model | **Fixed** | No trivial floor existed anywhere |
| 7b | A test that catches a broken model | **Fixed** | Fixtures were degenerate; 455 tests proved only plumbing |
| 8 | Selection hygiene / test reuse | **Fixed** | Rolling window is an official-test argmax |
| 9 | Calibration leakage | Pass | Validation-fitted, verified in source |
| 10 | Typing and tooling | **Fixed** | No type checker existed at all |
| 11 | Secrets and env hygiene | Pass | `.env` ignored, only `.env.example` tracked |
| 12 | Internal path / string leakage | Pass | No hits for internal paths or markers |
| 13 | Link integrity | **Fixed** | 15/15 markdown links resolved; 3 code paths wrong |

## 1. The 451-test badge

```bash
uv run pytest
```

Printed `451 passed, 1 warning in 209.66s`, exit code 0. **The badge was
accurate** — worth stating plainly, since a sibling repo's equivalent claim was
off by 94.

It was nonetheless a hard-coded `shields.io` URL
(`tests-451%20passed-brightgreen`), a static string with no link to the suite.
It was true by maintenance, not by construction, and would have gone stale on
the next test added — which is exactly what happened during this audit (the
count is now 455).

**Fix.** The badge no longer carries a number. It reads
`tests-pytest green in CI`, which the CI badge beside it already substantiates
via a live endpoint. The count moved into the evidence table as a stamped,
reproducible measurement with the command to regenerate it. A number that
cannot be checked from the badge should not be in the badge.

## 2. Data, and what may be claimed

Three datasets, all public benchmarks:

- **NASA C-MAPSS** turbofan degradation. Critically, this is **simulation
  output** — the readme shipped with the data calls it a "Degradation
  Simulation Data Set". It is not measured engines.
- **SMAP / MSL** spacecraft telemetry (NASA/JPL). The recorded ingestion is the
  Telemanom S3 archive, not the Kaggle mirror — the mirror is a documented
  fallback. An earlier draft of this report and of the README asserted the
  mirror as the source; the local download metadata contradicts that, and both
  are corrected.
- **ESA-ADB** spacecraft telemetry (European Space Agency).

**What was recorded before the audit.** Only ESA-ADB's terms (`CC BY 3.0 IGO`),
and only in two deep documents. C-MAPSS and SMAP/MSL had source URLs and a
citation string in `data/raw/cmapss/cmapss_download_metadata.json`, but no
licence position anywhere. The README said nothing about data at all.

**On safety framing — the item flagged as highest priority.** The audit found
the repository already handles this well *in `docs/`*: `architecture.md`,
`public_results.md`, `public_proof_assets.md`, `phase3_*.md` and
`CONTRIBUTING.md` all carry explicit "not certification evidence" language, and
`docs/deployment.md` calls RUL intervals "operational triage aids, not
certification guarantees". No document overclaims operational relevance.

The defect was one of **placement, not content**. The README — the only
document most readers open — described the project as "an operations system"
with "fleet triage", an "operator console" and "deployable ML engineering",
and nowhere stated that all of this runs on benchmark data and carries no
operational standing. A reader who stopped at the README could reasonably have
formed the wrong impression.

**Fix.** A new "Data, and what this repo may claim" section near the top of the
README: a provenance and terms table for all three datasets, and an explicit
statement that benchmark data supports claims about *method* and not about real
fleets, airframes, spacecraft, or maintenance decisions — plus a note that
words like "fleet" and "operator console" describe the shape of the software,
not a deployment.

## 3. Is the evaluation real?

Yes. This is the strongest part of the repository, and it survived hard checks.

**The split is genuinely held out and genuinely temporal.**
`make_cmapss_temporal_validation_split`
([cmapss_baseline.py:117](../src/aerospace_prognostics/experiments/cmapss_baseline.py#L117))
holds out whole **units** and then truncates each held-out engine's history to a
30-cycle horizon. Both halves matter: unit-wise grouping prevents the same
engine appearing on both sides, and the truncation makes the validation task
mirror the official test task (predict RUL from a partial trajectory). The
`random_state` shuffle selects *which engines* are held out — it is grouping,
not row-level shuffling. There is no random row split in the RUL path.

For ESA-ADB, `run_mission_lightweight` splits chronologically (first half
train, second half test), fits the robust z-score on **nominal training points
only**, and selects thresholds on the last three months of the training half.
No test row touches any fit.

**Leakage guards hold.** Feature standardisation, the operating-regime
clusterer, sensor filtering, and the NASA-shift calibration are all fit on
train/validation rows only — verified in source, not taken on trust. The
README's "without ever touching the official test rows" claim is true.

**The recorded leakage correction is real and was self-reported.** The ESA-ADB
recall correction from `0.24` to `0.42` is documented, and the correction
*raises* the number — recording it anyway is the right instinct.

### Finding: no baseline floor existed

The repository compared HGB against deep models and honestly reported that the
classical model wins. That is a real comparator. But there was **no trivial
baseline anywhere** — no constant predictor, no `DummyRegressor`, nothing. A
reader had no way to tell whether RMSE `13.01` reflected learned degradation
signal or the shape of the label distribution.

**Fix.** Added `run_cmapss_naive_baseline` with three constant strategies, a
`cmapss-naive-baseline` CLI command, and four tests. Measured on the real data:

| Subset | Naive median RMSE | Naive median NASA | HGB RMSE | HGB NASA |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 49.82 | 166,570 | 13.01 | 253 |

The model beats the floor 3.8x on RMSE and **657x** on the asymmetric NASA
score. The evaluation is meaningful, and now demonstrably so.

### Finding: no test could catch a model predicting nonsense — now fixed

The most serious finding of the audit, and the last to be closed.

The tiny fixture is degenerate for skill purposes. Its training RUL values are
`[2,1,0,2,1,0]`, giving a median of `1.0`; both its test units have a true RUL
of exactly `1`. A constant predictor therefore scores a **perfect RMSE of 0.0**,
and no model can do better. The learned baseline scored `0.0` too — not because
it worked, but because with six rows and sklearn's default `min_samples_leaf=20`
it cannot split at all, so it emits the training mean, which happened to be the
right answer.

The consequence: **455 tests proved the plumbing and not one proved the
modelling.** Every test verified that commands run, artifacts are written and
shapes match. None would have gone red if the estimator lost all predictive
power.

**Fix.** Added `write_discriminating_cmapss_subset`, a generated fixture (about
700 rows, no real NASA data, so the no-redistribution posture is unchanged)
that removes both halves of the coincidence:

- five test units truncated at different remaining lifetimes (`5, 20, 45, 70,
  90`), so no single constant fits them all and the floor is forced above zero;
- a monotone degradation signal in the sensor readings with a deterministic
  wobble derived from unit and cycle numbers rather than an RNG, so a real
  estimator can recover RUL and the fixture cannot flake.

`test_learned_baseline_beats_the_naive_floor` asserts the learned model clears
the floor by more than 2x. Measured on the fixture:

| Predictor | RMSE | NASA score |
| --- | ---: | ---: |
| Naive `train_median` (the floor) | 34.6266 | 128.04 |
| Naive `rul_cap` | 84.9412 | 202324.06 |
| HGB baseline | **1.9446** | **0.42** |

**The test was then proven non-vacuous by sabotage.** Replacing the estimator
with a `DummyRegressor(strategy="mean")` — a model that ignores the sensors
entirely — scores RMSE `33.7158` against the floor's `34.6266`, and the test
goes red. On the old degenerate fixture the same sabotage is invisible: healthy
and broken both score `0.0`. That contrast is the whole point of the fix.

| Fixture | Healthy estimator | Sabotaged estimator |
| --- | --- | --- |
| Degenerate (`write_tiny_cmapss_subset`) | test fails (floor is 0.0) | test fails — indistinguishable |
| Discriminating (new) | **test passes** | **test fails — caught** |

The tiny fixture is retained for the plumbing tests it was always right for; its
docstring now points at the skill test rather than claiming the comparison
cannot be made without real data. It can be, and now is.

### Finding: one hyperparameter was selected on the official test set

`docs/public_results.md` claimed model selection "is kept on train-side
validation where possible, then the official test table is used once".

The first half is mostly true — the feature policy, HGB parameters, and sensor
filter are all scored on the validation split. The second half is not. The
per-subset rolling window `CMAPSS_ENGINEERED_DEFAULT_WINDOWS`
(FD001:10, FD002:3, FD003:5, FD004:3) matches, exactly, the "Best by NASA
score" row of the rolling-window sweep in
[phase1_cmapss_baseline_results.md](phase1_cmapss_baseline_results.md) — and
that sweep is scored on **official test**. There is no validation-side window
sweep in the codebase (confirmed by search). The headline "validation-selected"
policy therefore inherits a test-selected hyperparameter. Separately, the
official test set has been consulted at least four times in Phase 1 alone, and
the FD001 deep table ranks five candidates by official-test score.

This is mild as leakage goes — it is one hyperparameter over a three-value grid,
not a leaked feature — but the prose asserted a cleanliness the code does not
have.

**Fix.** Replaced the claim with a "Selection hygiene" section in
`public_results.md` that itemises exactly what was chosen on validation and what
was chosen on test, and states the consequence: treat the headline FD001 numbers
as **mildly optimistic**, in the way any repeatedly-consulted leaderboard is.
The README now carries a one-line pointer to it. Sweeping the window on
validation and re-reporting is named as open work rather than quietly dropped.

## 4. Typing and tooling

**Before this audit the repository ran no type checker at all** — no mypy, no
pyright, no annotation lint rules. `ruff` covered `E`, `F`, `I`, `UP`, `B`,
`SIM` repo-wide, and CI ran ruff + pytest + pip-audit + SBOM + a genuinely
thorough set of container smoke tests. The README made no typing claim, so
there was no overclaim to correct — but the gap against the sibling
`cashflow-risk` repo (mypy strict) was real.

Measured cost of closing it:

```bash
uv run mypy src/aerospace_prognostics --ignore-missing-imports
```

**331 errors across 21 of 81 files.** Inspection showed most are not latent
bugs but a single idiom: long `argparse` dispatch functions that rebind one
`result` variable across many branches, so mypy narrows to the first branch's
type. `cli_workflows.py:413` looks alarming (`Phase1WorkflowResult` has no
attribute `ok`) but is safe at runtime, since each branch returns before the
next binding.

Adding mypy strict repo-wide was therefore not an audit fix — it is a separate
project, and a permissive repo-wide config that passes vacuously would be worse
than nothing.

**Fix.** A *scoped, ratcheting* gate. The numeric and evaluation core — the
seven modules where a type error would corrupt a reported metric — now
type-checks clean and is gated in CI. Closing it surfaced one genuine implicit
invariant in `esa_adb_mission.py`, where `global_test_score` was narrowed on
`grid is None` rather than on itself; safe today, fragile under edit, now
explicit. `[tool.mypy]` in `pyproject.toml` documents that the file list only
grows, and the README states the posture plainly rather than implying more
coverage than exists.

## 5. Hygiene

- **Secrets:** clean. No credential-shaped literals in tracked files. `.env`
  gitignored (`.gitignore:13`), only `.env.example` tracked.
- **Internal leakage:** clean. No hits for `C:\dev`, `C:\Users`, `leaff`, or
  `_pmo`. Hits for "reviewer" and "handoff" are legitimate domain usage
  (release-evidence review), not internal-process leakage.
- **Banned framing:** clean. No "production-grade". The four `SOTA` hits are
  all *disclaimers* ("not SOTA", "lenient, not SOTA").
- **Links:** all 15 README markdown links resolve. Three bare code references
  (`experiments/…`, `reports/…`, `deployment/…`) omitted the
  `src/aerospace_prognostics/` prefix and pointed at nothing — **fixed**.

### Internal planning documents — corrected

An earlier draft of this report claimed seven internal planning documents were
"tracked and public but unlinked". **That was wrong**, and the error mattered:
it described a decision as outstanding when it had already been made.

Five of the seven are explicitly gitignored at `.gitignore:42-47`:
`Aerospace_Prognostics_Project_Plan.md`, `docs/restructure_plan.md`,
`docs/private_hosting_handoff.md`, `docs/project_checklist.md`, and
`docs/mlops_portfolio_positioning.md`. They exist only in the local working
copy. This matters most for the project plan, which names target employers and
frames the repository as a portfolio piece — content that would read oddly in a
public repo, and which was correctly kept out of one.

Only two were genuinely tracked, public, and unlinked:
`docs/product_roadmap.md` and `docs/phase3_cmapss_recommendation.md`. Both were
read in full: neither leaks anything, and the Phase 3 note is a genuine decision
record with an evidence table and an explicit "not certified PHM uncertainty"
boundary. **Resolved** by linking both from the README repository map — an
unlinked public document is strictly worse than a linked one, because a reviewer
who finds it by browsing is left wondering why it was hidden.

The general lesson for later audits in this series: check `git ls-files` before
calling a file public. Presence on disk is not presence in the repository.

## 6. Portfolio-level note: the claims discipline is inverted

This section is about the wider portfolio, not this repository. It is recorded here because
it was found while auditing this repository, and because it changes what "audited" should
mean for the rest of the series.

### What the sabotage sweep actually was

After this audit, a single check was run across the portfolio: break each repository's core
logic at runtime, run its test suite, and record whether anything goes red. Nine repositories
were checked this way. Eight passed.

**That was one dimension out of thirteen.** It tests exactly one property — whether a suite
would notice its own product breaking. It says nothing about a repository's claims, dataset
licensing, split methodology, selection hygiene, typing posture, secret hygiene, or README
accuracy. Every other section of this document represents work that was *not* done for those
repositories.

Eight repositories passing that check is **not** eight repositories audited, and the results
table it produced must not be read as if it were. Only this repository has been audited. The
one repository that failed the check — `marketing-effectiveness-lab`, where freezing the core
analytics produces a single red test and that test covers input validation rather than
behaviour — has not been audited either; it has one known defect and twelve unexamined
dimensions.

### The inversion

The sweep began from a prediction, which was wrong, and the way it was wrong is the useful
part.

The prediction: the portfolio's near-empty repositories would be the overclaimers. Eight of
them have between zero and two tracked Python files — an eight-line `__init__.py` and an
eight-line smoke test — while their READMEs open with substantive present-tense capability
claims ("a designed-and-verified control law", "a precisely-validated flight-dynamics core").
That looked like the clearest overclaim in the portfolio.

It is not. Every one of those eight carries a bold status line near the top of its README —
*"Status: pre-Gate-0. Nothing here is a result yet"* — and every one ships a `claims.md`
ledger whose only row reads *"No claims yet."* Several also carry explicit ground rules
disclaiming certification, hardware-in-the-loop, and real-flight relevance before any number
is published at all.

The mature repositories — the ones with real results, real readers, and real numbers in their
READMEs — had no claims ledger at all. This one did not, until this audit.

**So the discipline is present exactly where there is nothing to overclaim, and absent exactly
where the risk is.** The scaffolding template got it right; the repositories that predate the
template never received it, and those are precisely the repositories whose numbers someone
might actually rely on.

That is not a criticism of the scaffolds. It is an argument that they contain a solved problem
the mature repositories still have. The fix is to backport the pattern, not to invent one:
`claims.md` in this repository is the worked example, written against numbers this repository
can actually evidence.

### What this does not license

No ledger row has been written for any other repository. A single sabotage check is not
grounds to certify another repository's claims, and doing so would repeat in miniature the
error this whole audit exists to catch — asserting more than the evidence carries.

## What this repository may and may not claim

**May claim.** That it implements an end-to-end PHM MLOps envelope with real
release evidence; that its splits are leakage-guarded and temporal; that its
reported metrics are reproducible from the recorded commands; that the HGB
policy substantially beats a trivial floor on C-MAPSS; that its ESA-ADB
event-wise baselines follow the protocol's detection tier with stated
deviations; that it found and self-reported a leakage bug in its own scorer;
and that its test suite contains a genuine skill regression — an estimator that
stopped predicting would turn CI red, demonstrated by sabotage rather than
asserted.

**May not claim.** Anything about real fleets, real airframes, real spacecraft
operations, or the safety of any maintenance decision. C-MAPSS results are
evidence about *modelling on simulated degradation*, not about engines.
Nothing here is airworthiness or certification evidence. It may not claim a
clean single-shot held-out estimate on C-MAPSS, because the rolling window was
test-selected and the test set has been consulted repeatedly. It may not claim
a full ESA-ADB leaderboard result — only the detection tier is computed and
official resampling is not applied, which the repository already says. It may
not claim mypy strict, or repo-wide type checking.

## Reproducing this audit

```bash
uv run pytest                                    # test count
uv run pytest -k naive_floor                     # the skill test specifically
uv run ruff check .                              # lint
uv run mypy                                      # scoped type gate
uv run mypy src/aerospace_prognostics --ignore-missing-imports   # full cost
git grep -nI "C:\\\\dev\|C:\\\\Users\|leaff\|_pmo"               # path leakage
git grep -nIi "production-grade\|state-of-the-art"               # framing
git check-ignore -v .env                                          # env hygiene
uv run aerospace-prognostics cmapss-naive-baseline --data-dir data/raw/cmapss --subset FD001
```
