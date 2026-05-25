# Aerospace Predictive Maintenance & Anomaly Detection — Project Building Plan

**Project working title:** A Prognostics & Health Management (PHM) pipeline for aerospace telemetry — Remaining Useful Life prediction and spacecraft anomaly detection with calibrated uncertainty.

**Author:** Ross
**Goal:** A serious multi-month portfolio piece. Primary objective is genuine domain learning; secondary objective is a credible artefact recognisable to aerospace/space-industry engineers (Airbus, Boeing, ESA-aligned suppliers, KP Labs, etc.).
**Estimated duration:** ~16 weeks (4 phases), part-time alongside other work.
**Last research check:** 25 May 2026. All datasets, tools, methods and standards below were verified against current sources at that date — see the *Sources & currency notes* section.

---

## 0. Executive summary

This project builds a two-track Prognostics & Health Management system:

1. **Track A — Remaining Useful Life (RUL) prediction** on turbofan engine degradation data (NASA C-MAPSS, then N-CMAPSS). This is the *aircraft* side.
2. **Track B — Anomaly detection** on real spacecraft telemetry (NASA SMAP/MSL as a learning baseline, then the ESA Anomaly Detection Benchmark, ESA-ADB, as the serious target). This is the *spacecraft* side.

The differentiator — and the part that connects directly to your physics-aware ML dissertation — is **physics-informed constraints** and **uncertainty quantification (UQ)**. The deliverable is a benchmarked, well-documented pipeline plus an interactive dashboard and a written technical report.

> **Honest framing note.** C-MAPSS and SMAP/MSL are heavily studied; beating published state-of-the-art is *not* the goal and not realistic for a portfolio project. The value is in (a) genuine domain understanding, (b) sound engineering and reproducibility, (c) the physics-informed + uncertainty angle, and (d) understanding the operational decision the model feeds. Frame it that way throughout.

---

## 1. Domain primer (read before Phase 1)

A short shared vocabulary so the rest of the plan makes sense.

- **PHM (Prognostics & Health Management):** the engineering discipline of monitoring a system's health, diagnosing faults, and predicting future failures. The PHM Society is the main academic community; its annual conference proceedings are the best free source of current methods.
- **RUL (Remaining Useful Life):** how many cycles/hours/flights a component has left before it fails. A *regression* problem.
- **Anomaly detection:** flagging telemetry that deviates from normal behaviour. Usually *unsupervised* or *semi-supervised* because labelled failures are rare.
- **Condition-Based Maintenance (CBM):** maintenance triggered by observed condition rather than fixed schedule. Predictive maintenance is the data-driven evolution of CBM.
- **Turbofan basics:** the engine has a fan, low-pressure and high-pressure compressors, a combustor, and high/low-pressure turbines. Sensors measure shaft speeds, temperatures and pressures at each stage (e.g. exhaust gas temperature, core speed). Degradation shows up as gradual drift in these readings.
- **Telemetry:** the continuous stream of sensor and housekeeping data a spacecraft sends to ground. Channels cover power, thermal, attitude, comms, payload.
- **Asymmetric cost:** in aerospace, a *missed* failure is far costlier than a *false alarm*. Models and metrics must reflect this — NASA's C-MAPSS scoring function deliberately penalises late (overestimated RUL) predictions more heavily than early ones.

---

## 2. Tooling & environment (verified current)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+; pin one tested minor version for releases | Carries over from your dissertation work. |
| Deep learning | PyTorch 2.x | Industry + research standard; matches your MACE/NequIP experience. |
| Classical ML | scikit-learn, XGBoost / LightGBM | For Phase 1 baselines. |
| Data | NumPy, pandas, Polars, Parquet; DuckDB optional | N-CMAPSS and ESA-ADB are large; use lazy loading and columnar storage where possible. |
| Time-series UQ | TorchUncertainty / Laplace / MAPIE / native MC-Dropout / deep ensembles | See Phase 3. |
| Experiment tracking | Weights & Biases or MLflow | Non-negotiable for a serious portfolio piece — reproducibility is a stated industry concern. |
| Config management | Hydra or pydantic-settings | Avoids hard-coded hyperparameters. |
| Dashboard | Streamlit (fast) **or** Next.js + FastAPI (stronger portfolio artefact) | Given your TypeScript/Next.js skills and solo-founder goals, the Next.js route doubles as a product demo. |
| Packaging / reproducibility | uv or Poetry, Docker, pre-commit, ruff | Demonstrates engineering maturity. |
| Security / supply chain | Dependabot, `pip-audit`, SBOM, pinned GitHub Actions, least-privilege CI permissions | Lightweight but industry-aligned hygiene from day one. |
| Testing | pytest | Unit-test the data pipeline and metrics at minimum. |
| Version control | Git + GitHub, conventional commits | Public repo is part of the deliverable. |

**Hardware:** your MSI Pulse 15 (RTX 4070) is sufficient for C-MAPSS and SMAP/MSL. N-CMAPSS and ESA-ADB are larger; budget for either longer local runs or occasional cloud GPU (Colab/Kaggle free tiers, or a short paid spot instance).

---

## 3. Datasets (verified current)

| Dataset | Track | Role | Source |
|---|---|---|---|
| **C-MAPSS** (FD001–FD004) | A | Phase 1–2 learning baseline; simulated turbofan run-to-failure, 21 sensors + 3 operational settings. | NASA Prognostics Data Repository / IEEE DataPort / Kaggle mirrors. |
| **N-CMAPSS** | A | Phase 3 step-up; larger, with realistic flight-condition profiles. | NASA Prognostics Data Repository. |
| **SMAP / MSL** | B | Phase 2–3 learning baseline for spacecraft anomaly detection; real telemetry from the SMAP satellite and MSL "Curiosity" rover, anonymised, ~82 channels, expert-labelled anomalies. | NASA JPL `telemanom` GitHub repo / Kaggle mirror. |
| **ESA-ADB** (ESA Anomaly Detection Benchmark) | B | Phase 3 serious target; the current standard. Real telemetry from 3 ESA missions, 224 channels, 821 telecommands, and 1430 annotated events (157 anomalies, 716 rare nominal events, 401 communication gaps, 156 invalid segments) — the largest public spacecraft telemetry anomaly benchmark. The default benchmark uses Mission1/Mission2; Mission3 is present but excluded from the default benchmark split. | `kplabs-pl/ESA-ADB` GitHub + Zenodo/OpenReview records. |

> **Important currency note on SMAP/MSL.** SMAP/MSL is still useful as a *learning* dataset, but the research community has documented flaws in it (trivial anomalies, labelling issues, evaluation pitfalls). ESA-ADB was created specifically to address these and is now the credible benchmark. **Use SMAP/MSL to learn, use ESA-ADB to be taken seriously.** Acknowledging this distinction in your write-up itself signals domain awareness.

> **Important modelling note on ESA-ADB.** ESA-ADB is anonymised, which is necessary for release but limits true physics-informed spacecraft modelling. Keep the strongest physics-informed contribution on C-MAPSS/N-CMAPSS, and use operational constraints, robust statistics, uncertainty, latency, and explainability as the credibility layer for ESA-ADB.

---

## 4. Phase plan

Each phase ends in a concrete, demonstrable deliverable. Treat the deliverable as a gate — do not start the next phase until it is met.

### Phase 1 — Domain grounding & classical baseline (Weeks 1–4)

**Objective:** Genuinely understand turbofan degradation and the PHM problem, then build a defensible classical baseline.

**Learning tasks**
- Read the original C-MAPSS dataset documentation and the Saxena et al. damage-propagation paper that introduced it.
- Learn what each of the 21 sensors physically measures and why some are flat/uninformative in certain subsets.
- Read 2–3 recent PHM Society papers on C-MAPSS RUL to understand how results are *reported* (scoring function, RMSE, the piecewise-linear RUL target convention capping early-life RUL around 120–130 cycles).

**Build tasks**
- Reproducible data pipeline: ingestion, per-subset handling (FD001–FD004 differ in operating conditions and fault modes), normalisation, sliding-window sequence generation.
- Exploratory analysis: sensor drift visualisation, correlation, operating-regime clustering.
- Feature engineering + gradient-boosted baseline (XGBoost/LightGBM) for RUL regression.
- Implement the **NASA asymmetric scoring function** and RMSE as evaluation metrics. Implement them as tested functions.

**Deliverable:** Working baseline with reported RMSE + NASA score on all four FD subsets, a clean repo skeleton, and an EDA notebook. A short written "what I learned about turbofans" section — this is the genuine-domain-learning checkpoint.

---

### Phase 2 — Deep learning models & spacecraft track kickoff (Weeks 5–9)

**Objective:** Build sequence models for RUL, and stand up the spacecraft anomaly-detection track.

**Track A — RUL deep models**
- 1D-CNN over sensor windows.
- LSTM / BiLSTM (BiLSTM is a common strong baseline in current C-MAPSS literature).
- Optional Temporal Convolutional Network (TCN) baseline if time permits.
- A Transformer-based model (attention over the sensor sequence).
- Honest benchmarked comparison vs the Phase 1 baseline, reported as a results table mirroring the literature. Use the experiment tracker for every run.

**Track B — Anomaly detection baseline**
- Set up SMAP/MSL via the `telemanom` repo.
- Reproduce a forecasting-plus-dynamic-threshold LSTM approach (the Hundman et al. method) as your baseline — reproducing a known result is a strong learning exercise.
- Add a reconstruction-based approach (autoencoder or Transformer reconstruction error) for comparison.
- Include simple robust baselines (PCA reconstruction, Isolation Forest, robust z-score/median absolute deviation thresholds). Recent time-series anomaly-detection work shows deep models can look better than they are under weak metrics.
- Learn the evaluation subtleties: point-adjusted F1 and why it can be misleading; precision/recall trade-offs under rare anomalies.

**Deliverable:** A benchmarked model-comparison table for Track A, and a working reproduced anomaly-detection baseline for Track B with honest metrics. Short write-up of why naive F1 on SMAP/MSL is not enough — sets up Phase 3.

---

### Phase 3 — The differentiator: physics-informed + uncertainty + ESA-ADB (Weeks 9–14)

**Objective:** Make the project genuinely yours by adding physics-informed constraints and uncertainty quantification — the themes from your graphene dissertation — and graduate to the real ESA benchmark.

**3a — Physics-informed constraints (Track A)**
- Encode **monotonic degradation**: RUL should be non-increasing over a unit's life; a health indicator should not improve. Add a penalty term to the loss for violations.
- Optionally encode smoothness / bounded-rate-of-change constraints.
- This is conceptually the same idea as physics-constrained ML in your dissertation — embedding known physical behaviour into the loss rather than hoping the model learns it. Physics-informed and Bayesian physics-informed neural networks for RUL are an active 2025–2026 research area, so this is current, not dated.

**3b — Uncertainty quantification (both tracks)**
- Implement at least two of: MC-Dropout, deep ensembles, a Bayesian neural network, or **conformal prediction** intervals.
- Produce *calibrated prediction intervals*, not just point estimates. For RUL, "fails in 50 ± 5 cycles" is actionable; "± 80" is not.
- Evaluate calibration explicitly (e.g. coverage of prediction intervals, interval width, calibration curves). For conformal prediction, use temporal calibration splits or rolling/online conformal methods where appropriate; the usual exchangeability assumption is weakened by time dependence and drift, so report empirical coverage honestly.

**3c — Graduate to ESA-ADB (Track B)**
- Move from SMAP/MSL to ESA-ADB. Use its official hierarchical evaluation pipeline and its operator-defined requirements/metrics rather than inventing your own.
- Expect your models to *not* fully satisfy operator requirements — the benchmark authors found this is true even of standard algorithms. Reporting that honestly is itself a credible result.
- Report data-loading, preprocessing, and evaluation choices carefully: use Parquet/lazy loading where useful, verify checksums, never commit large raw telemetry, and document the default Mission1/Mission2 benchmark split.

**Deliverable:** Physics-informed RUL model with calibrated uncertainty, benchmarked against your Phase 2 models; anomaly detection results on ESA-ADB using its official pipeline. This is the intellectual core of the portfolio piece.

---

### Phase 4 — Productisation, write-up & dissemination (Weeks 14–16)

**Objective:** Turn the work into something a recruiter or engineer can absorb in minutes and trust in depth.

**Build tasks**
- Dashboard: a fleet view — engines/satellites listed with predicted RUL, confidence bands, and anomaly flags; drill-down into per-unit sensor traces. Streamlit for speed, or Next.js + FastAPI to double as a product/portfolio demo.
- Polish the repo: README with results table, architecture diagram, reproducibility instructions, Docker image, licence, tests passing in CI (GitHub Actions).

**Write-up tasks**
- Technical report (PDF): problem, datasets, methods, results, *and a "path to deployment" section* — certification considerations, data/concept drift, monitoring, the false-alarm vs missed-failure cost asymmetry, and what would be needed for real operational use.
- A blog post / Ainsight newsletter piece — accessible narrative version. Good for your Substack and your public profile.
- Optional: a short demo video or GIF of the dashboard.

**Deliverable:** Public GitHub repo, deployed/recordable dashboard, technical report PDF, and a published article.

---

## 5. Methods reference (current as of May 2026)

**RUL prediction (Track A)** — classical: gradient-boosted trees on engineered features. Deep: 1D-CNN, TCN, LSTM/BiLSTM, Transformer/attention models, hybrid CNN-LSTM. Advanced: physics-informed neural networks (PINNs), Bayesian PINNs for joint RUL + uncertainty.

**Anomaly detection (Track B)** — robust statistical baselines (PCA reconstruction, robust thresholds, Isolation Forest), forecasting-based methods (LSTM forecast + dynamic/nonparametric thresholding), reconstruction-based methods (autoencoders, VAEs, Transformer reconstruction), and prior/attention-informed Transformer models. Note that the ESA-ADB authors stress that standard algorithms still fall short of operator needs — there is genuine open ground here.

**Uncertainty quantification** — MC-Dropout, deep ensembles, Bayesian neural networks, conformal prediction. Conformal prediction is attractive because it is lightweight and can give finite-sample coverage under exchangeability; for telemetry time series, use temporal/online variants and always validate empirical coverage under drift.

---

## 6. Aerospace / space-industry standards & practices to be aware of

You will not *certify* anything in a portfolio project, but referencing the right standards shows you understand the operational context — and that is exactly what separates a portfolio piece from coursework.

- **PHM Society** — the main community for prognostics methods and metrics; its proceedings are free and current.
- **ISO 13374** — condition monitoring & diagnostics of machines: the reference data-processing architecture (data acquisition → manipulation → state detection → health assessment → prognostics → advisory generation). Map your pipeline onto these blocks in the report.
- **SAE / aerospace assurance standards** — for software in airborne systems, the relevant assurance frameworks are **ARP4754B** (development of civil aircraft and systems), **ARP4761A** (safety assessment), and **DO-178C** (software considerations in airborne systems). You are not following these formally, but a paragraph noting *where* an ML prognostics tool would sit relative to them, and why certifying ML is hard (non-determinism, data dependence), is a strong signal.
- **EASA AI guidance** — the European Union Aviation Safety Agency has published an AI roadmap and AI Concept Paper Issue 2 on trustworthiness, learning assurance, explainability, and Level 1/2 ML applications. Citing this shows you know ML in aerospace is a live regulatory topic.
- **ECSS** (European Cooperation for Space Standardization) — the space-segment equivalent. Reference **ECSS-E-ST-40C Rev.1** (software), **ECSS-Q-ST-80C Rev.2** (software product assurance), **ECSS-E-ST-80C** (space system cybersecurity), and the ECSS machine-learning handbook where relevant.
- **Operational metric realism** — always report with the asymmetric cost in mind, and prefer benchmark-defined metrics (NASA C-MAPSS score; ESA-ADB's official pipeline) over ad-hoc ones.

---

## 7. Security, robustness & efficiency considerations

A "top industry standard" project addresses these explicitly, even if lightly:

- **Data integrity & provenance** — record dataset versions and checksums; telemetry pipelines must detect corrupted or dropped data.
- **Dataset handling** — keep raw NASA/ESA/JPL data out of Git; use documented download scripts, checksums, `data/` gitignore rules, and optionally DVC or Hugging Face Datasets only if it simplifies reproducibility.
- **Adversarial / drift robustness** — note the risk of concept drift (a fielded model degrades as the fleet ages) and sensor spoofing; describe monitoring and retraining triggers.
- **Reproducibility as a security property** — pinned dependencies, seeded runs, Docker, CI. In safety-critical contexts, reproducibility is a trust requirement, not a nicety.
- **Software supply chain** — align lightly with NIST SSDF: Dependabot for dependency updates, `pip-audit` in CI, CycloneDX SBOM generation, pinned GitHub Actions by SHA for release workflows, explicit `GITHUB_TOKEN` permissions, artifact attestations/provenance for packaged outputs, and a stretch target of SLSA Build Level 2+ for releases.
- **Efficiency** — report inference latency and model size; on-board anomaly detection has hard compute/power budgets. ESA-ADB's authors explicitly check real-time feasibility — mirror that by reporting whether your model could run within a realistic budget.
- **Explainability** — for any anomaly flag, surface *which channels* drove it. Operators will not act on a black-box alarm. Attention weights or feature-attribution methods help here.
- **No secrets in the repo** — standard hygiene: no API keys, no credentials; use environment variables.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Scope creep across two tracks | Track A is the spine; Track B Phase 2 baseline is the minimum. ESA-ADB (3c) is "stretch" — droppable without breaking the project. |
| Chasing leaderboard scores | Pre-commit to the honest framing in Section 0. Contribution = engineering + physics/UQ angle. |
| N-CMAPSS / ESA-ADB compute cost | Start small (C-MAPSS, SMAP/MSL) locally; use cloud GPU only for the large-scale step. |
| Benchmark evaluation pitfalls | Use official scoring (NASA score, ESA-ADB pipeline); read the "evaluation flaws" literature before reporting. |
| Large raw data in GitHub | Ignore raw data by default; use download scripts, checksums, and documented storage locations. |
| Overclaiming physics-informed results on anonymised spacecraft data | Keep physics-informed constraints central to Track A; describe ESA-ADB constraints as operational/statistical rather than physical unless a channel mapping supports a stronger claim. |
| Dissertation time conflict | The physics-informed/UQ overlap means project and dissertation reinforce each other — schedule them as complementary, not competing. |

---

## 9. Definition of done

The project is complete when:

1. A public, documented, tested GitHub repo reproduces all results.
2. RUL models are benchmarked on C-MAPSS (min.) with the NASA score, including a physics-informed model with calibrated uncertainty.
3. An anomaly-detection model runs on SMAP/MSL (min.) or ESA-ADB (stretch) with benchmark-appropriate metrics.
4. An interactive dashboard presents predictions, uncertainty and anomaly flags.
5. A technical report includes a credible "path to deployment" section referencing the relevant standards.
6. CI covers linting, tests, dependency audit, and basic supply-chain hygiene.
7. A public write-up (Ainsight) communicates the work to a general technical audience.

---

## 10. Sources & currency notes

All items below were checked on 25 May 2026 and reflect the current state of the field.

- **C-MAPSS** remains the canonical turbofan RUL benchmark; 2025 publications (e.g. BiLSTM with change-point detection; CNN-LSTM hybrids) continue to use it and report via RMSE + the NASA asymmetric score. The piecewise-linear RUL target (cap ~120–130 cycles) is standard practice.
- **N-CMAPSS** is the current larger, more realistic successor; 2025 master's-level work uses it with NASA's asymmetric scoring function.
- **SMAP/MSL** (NASA JPL `telemanom`) is still widely used but has **documented flaws** (trivial anomalies, labelling and evaluation issues) flagged in 2024 re-evaluation papers — hence treated here as a learning dataset only.
- **ESA-ADB** (European Space Agency Anomaly Detection Benchmark, KP Labs + ESA/ESOC) is the **current credible standard** for spacecraft telemetry anomaly detection: real data from 3 ESA missions, 224 channels, 821 telecommands, and 1430 annotated events, with an operator-designed hierarchical evaluation pipeline. Public material confirms an OpenReview submission and public code/data; avoid claiming final proceedings acceptance unless separately confirmed.
- **Physics-informed & Bayesian PINNs for RUL with uncertainty quantification** are an active 2025–2026 research area (e.g. physics-constrained Bayesian neural networks in *Reliability Engineering & System Safety* 2026; PHM Society 2025 papers on data-efficient, uncertainty-aware PINN RUL). This confirms the differentiator is current, not dated.
- **Transformer-based time-series anomaly detection** continues to advance through 2025 (prior-informed dual-attention, time-invariant, decomposition-based variants), so a Transformer model in Phase 2 reflects the current method landscape.
- **Standards** — PHM Society proceedings, ISO 13374, ARP4754B/ARP4761A/DO-178C, EASA's AI roadmap/concept papers, and ECSS software/software-assurance/cybersecurity standards remain the relevant reference frameworks for contextualising aerospace and space prognostics work.
- **Software supply chain** — NIST SSDF, SLSA, CycloneDX SBOMs, GitHub artifact attestations, Dependabot, and `pip-audit` are current, practical controls to show industry-grade engineering hygiene without pretending this is a certified aerospace product.

**Source anchors for the May 2026 audit**
- NASA PCoE Data Repository: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- ESA-ADB OpenReview paper: <https://openreview.net/forum?id=FYEGPuUrpo>
- ESA-ADB GitHub: <https://github.com/kplabs-pl/ESA-ADB>
- Hundman et al. SMAP/MSL anomaly detection paper: <https://arxiv.org/abs/1802.04431>
- EASA AI Concept Paper Issue 2: <https://www.easa.europa.eu/en/document-library/general-publications/easa-artificial-intelligence-concept-paper-issue-2>
- SAE ARP4754B: <https://saemobilus.sae.org/standards/arp4754b-guidelines-development-civil-aircraft-systems>
- SAE ARP4761A: <https://saemobilus.sae.org/standards/arp4761a-guidelines-conducting-safety-assessment-process-civil-aircraft-systems-equipment>
- ECSS published standards list: <https://ecss.nl/list-of-published-ecss-standards-long/>
- NIST SSDF SP 800-218: <https://csrc.nist.gov/pubs/sp/800/218/final>
- SLSA v1.2: <https://slsa.dev/spec/v1.2/about>
- CycloneDX SBOM capabilities: <https://cyclonedx.org/capabilities/>
- GitHub artifact attestations: <https://docs.github.com/en/actions/concepts/security/artifact-attestations>
- `pip-audit`: <https://github.com/pypa/pip-audit>
- MAPIE conformal prediction documentation: <https://mapie.readthedocs.io/en/latest/>

> Re-run a currency check at the start of Phase 3 — the ESA-ADB benchmark is explicitly described by its authors as evolving, with new missions and updated annotations planned.
