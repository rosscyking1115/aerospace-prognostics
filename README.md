# Aerospace Prognostics

Predicting when aircraft engines and spacecraft hardware are heading for failure — and
checking whether the scores those predictions are graded on are themselves correct.

**The question.** A model scores badly on a benchmark. Is the model weak, or is the
*scoring* wrong? The second possibility is rarely checked. A low score looks like honesty,
and nobody audits a number that flatters them less than it should.

**The answer, here, was the scoring.** This project's spacecraft anomaly detector first
reported a recall of `0.24`. Auditing the evaluation showed the scorer was counting events
from the training period — events the model was never asked to predict — as missed
detections. Score only the window the model was actually given, which is what the
benchmark's protocol requires, and the honest figure is `0.42`.

| ESA-ADB Mission 1, event-wise detection at threshold 5 | Recall |
| --- | ---: |
| first reported | `0.236842` |
| after restricting scoring to the test window | `0.415385` |

Precision was `1.000` before and after; the corrected F0.5 is `0.780`. The correction
*raises* a published number, which is exactly why it is recorded rather than quietly
adopted — see [docs/public_results.md](docs/public_results.md). The leakage guard that came
out of it is now a typed, on-by-default error in the evaluation library, so the same
mistake fails loudly instead of producing a plausible number.

> **Status: active reference implementation, not a product.** Every figure comes from a
> public benchmark dataset. Nothing here is airworthiness evidence, a certification claim,
> or input to any real maintenance or flight-safety decision. Productization is frozen,
> there is no hosted instance of anything, and there is no support commitment. Released
> under the [MIT Licence](LICENSE); dataset terms are separate and are recorded in
> [docs/datasets.md](docs/datasets.md).

[![CI](https://github.com/rosscyking1115/aerospace-prognostics/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rosscyking1115/aerospace-prognostics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## What this is for

Two audiences. Someone evaluating whether a prognostics result can be trusted, who wants
to see the evaluation audited rather than asserted. And someone who has to *ship* a
predictive-maintenance model, who wants the surfaces around it — serving, release gating,
evidence, monitoring — rather than another notebook.

**How this relates to other work.** Most predictive-maintenance repositories are a
notebook and a leaderboard score on NASA's C-MAPSS turbofan dataset. This one keeps C-MAPSS
only as a familiar baseline anyone can cross-check, puts its anomaly work on ESA-ADB, and
wraps both in a production envelope. ESA-ADB is an under-used benchmark, carried here
through its real protocol at the detection tier, with no comparison set — nothing in this
repository establishes how much prior work exists on it, so read that as a description of
what was built, not a claim about the literature. The evaluation layer
proved general enough to extract: it now lives as
**[telemeval](https://github.com/rosscyking1115/telemeval)**, a standalone Apache-2.0
library, and this repository is its reference pipeline, consuming it as a dependency.

## How the result was reached

A robust z-score baseline, fitted on nominal training points only, scored event-wise on a
chronological test window with communication gaps excluded — ESA-ADB's own default
benchmark table. Two threshold policies are compared: a fixed cutoff and one selected on
the last three months of training data. The protocol was locked against the ESA-ADB paper,
the official benchmark code, and the official evaluator script *before* any loader or
scorer was written.

Full method, both missions, and the fixed-versus-validation comparison:
[docs/phase3_esa_adb_intake.md](docs/phase3_esa_adb_intake.md).

## Why the result is trustworthy

- **The evaluation was audited, and the audit is published.** Including the correction
  above, which moved a number in this project's favour.
- **Deviations are stamped on the artifacts, not buried.** Only the event-wise detection
  tier of ESA-ADB's metric hierarchy is computed and the official resampling is not
  applied, so every artifact carries *"event-wise detection only — not a full ESA-ADB
  leaderboard claim."*
- **There is no comparison set yet.** The only comparison in this repository is one
  threshold policy against another, both from the same baseline. Nothing here reproduces a
  published ESA-ADB result or sits beside one.
- **Negative results are reported as results.** The classical model still beats the deep
  model on C-MAPSS, and the README says so rather than cherry-picking.
- **Selection hygiene is stated, including where it is imperfect.** One C-MAPSS
  hyperparameter was selected on the official test set; that is disclosed in the ledger
  rather than left for a reader to find.
- **Every published number has a row in [claims.md](claims.md)** naming what produced it,
  what it may support, and what about it is disclosed-but-unresolved.
- **The test suite contains a skill check, not only plumbing.** The learned baseline must
  beat a naive floor on a fixture built so a constant predictor cannot win, so a model that
  lost its predictive power turns CI red.

## Reproduce it

No downloads required. Tiny fixture data generates a complete local evidence bundle and
seeds the console database:

```powershell
uv sync --dev
uv run aerospace-prognostics quickstart-cmapss-demo
uv run aerospace-prognostics app-init-db
uv run streamlit run src/aerospace_prognostics/app/streamlit_app.py
```

This writes dashboard, release, provenance, artifact-inspection, model-card, validation,
benchmark and SBOM artifacts under `artifacts/quickstart_cmapss`. The ESA-ADB and C-MAPSS
results need their datasets downloaded locally first — see
[docs/first_run.md](docs/first_run.md) for every entry path and
[docs/quickstart.md](docs/quickstart.md) for what the fixture evidence contains.

## What is in this repository

| | |
| --- | --- |
| [claims.md](claims.md) | The claims ledger: every published number, what produced it, what it may claim, what is unresolved |
| [docs/public_results.md](docs/public_results.md) | All benchmark results with limitations, including the correction above |
| [docs/datasets.md](docs/datasets.md) | Dataset provenance, licences, and what benchmark data does and does not entitle this repo to claim |
| [docs/engineering_envelope.md](docs/engineering_envelope.md) | Serving, release gating, evidence artifacts, architecture, and the local Docker stack |
| [docs/first_run.md](docs/first_run.md) | Console-only, Compose, and read-only demo first-run paths |
| [docs/architecture.md](docs/architecture.md) | System boundaries, evidence flow, runtime modes, security controls |
| [docs/deployment.md](docs/deployment.md) | Model packaging, serving, release evidence, promotion workflow |
| [docs/cmapss_domain_primer.md](docs/cmapss_domain_primer.md) | What the C-MAPSS benchmark represents, and why RMSE alone is the wrong metric |
| [docs/command_catalog.md](docs/command_catalog.md) | Every research, deployment, and app command |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, tooling posture, dependency-advisory policy |
| [docs/house_style_departures.md](docs/house_style_departures.md) | Where this repo departs from the shared documentation standard, and why |

Research logs and decision records — C-MAPSS baselines and deep experiments, SMAP/MSL
anomaly work, uncertainty and monotonicity, the ESA-ADB protocol intake, the licence
posture, and the engineering roadmap — are all under [`docs/`](docs/).
