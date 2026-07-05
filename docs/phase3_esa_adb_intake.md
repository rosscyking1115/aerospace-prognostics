# Phase 3 ESA-ADB Protocol Intake

This note starts the ESA Anomaly Detection Benchmark (ESA-ADB) milestone after
the C-MAPSS Phase 3 recommendation freeze. Its purpose is to lock the benchmark
protocol before writing loader, evaluator, or model code.

## Source Ledger

Primary sources checked on 2026-07-05:

- ESA-ADB paper: <https://arxiv.org/abs/2406.17826>
- Official benchmark code: <https://github.com/kplabs-pl/ESA-ADB>
- Official evaluator script:
  <https://github.com/kplabs-pl/ESA-ADB/blob/main/scripts/reevaluate.py>
- Official anomaly type inference script:
  <https://github.com/kplabs-pl/ESA-ADB/blob/main/scripts/infer_anomaly_types.py>
- Official ESA-specific metrics:
  <https://github.com/kplabs-pl/ESA-ADB/blob/main/timeeval/metrics/ESA_ADB_metrics.py>
- ESA Anomaly Dataset v2 Zenodo record:
  <https://zenodo.org/records/15237121>
- Original dataset DOI used by the paper and README:
  <https://doi.org/10.5281/zenodo.12528696>

The original Zenodo record now points to a newer dataset version. Use the v2
record for any new raw-data intake, while recording the original DOI because the
paper and official README cite it.

## What ESA-ADB Is

ESA-ADB combines three pieces:

- ESA Anomaly Dataset telemetry from three ESA missions, with Mission1 and
  Mission2 used for the benchmark and Mission3 excluded from benchmark scoring;
- an evaluation pipeline designed around spacecraft-operator priorities;
- baseline algorithm results implemented through a modified TimeEval workflow.

The dataset is large by this project's current anomaly-detection standards:
Zenodo v2 publishes `ESA-Mission1.zip`, `ESA-Mission2.zip`, and
`ESA-Mission3.zip`, totalling about 11.6 GB compressed. The official README
recommends a much larger working disk budget for full experiment artifacts, plus
Linux or WSL2, Docker, Conda, and substantial RAM. We should therefore treat
ESA-ADB as a protocol-first integration, not as a casual local data import.

## Protocol Requirements

- Dataset versioning:
  - Use Zenodo v2 DOI `10.5281/zenodo.15237121` for new downloads.
  - Record original DOI `10.5281/zenodo.12528696` for paper/README traceability.
  - Verify raw archives with Zenodo MD5 values before extraction.
- Data governance:
  - Keep all ESA raw archives, extracted mission folders, preprocessed TimeEval
    files, and generated experiment results out of Git.
  - Record the dataset license as `CC BY 3.0 IGO`.
  - Keep the ESA-ADB benchmark code license separate from the dataset license;
    the official code repository is MIT-licensed.
- Mission scope:
  - Benchmark implementation starts with Mission1 and Mission2 only.
  - Mission3 may be useful for exploration, but it is not part of ESA-ADB
    benchmark claims.
- Splits:
  - Preserve chronological splits: first half training, second half test.
  - Use the last three months of the training half as validation.
  - Never fit thresholds, standardization, or contamination levels on test
    rows.
  - Do not assume training data is anomaly-free; ESA-ADB explicitly contains
    anomalies in training and validation.
- Channels:
  - Distinguish target channels from non-target channels.
  - Evaluate only target channels.
  - Use non-target channels and telecommands only as support inputs.
  - Full-set runs may use priority-3 telecommands; lightweight runs do not
    include telecommands.
- Lightweight subsets:
  - Mission1 lightweight subset: channels `41-46`.
  - Mission2 lightweight subset: channels `18-28`.
  - Regenerate `anomaly_types.csv` when evaluating a lightweight subset because
    anomaly types can change when only a subset of channels is considered.
- Labels and event types:
  - Official evaluation expects mission-style structures including `labels.csv`
    and `anomaly_types.csv`.
  - Default benchmark tables evaluate all events except communication gaps.
  - Rare nominal events are treated as detections to handle in the default
    benchmark setup; anomaly-only analysis is a separate comparison mode.
- Preprocessing:
  - Respect the official zero-order-hold resampling policy.
  - Preserve the correction that prevents point anomalies from disappearing
    during resampling.
  - Use the mission target frequencies from the paper: Mission1 `0.033 Hz` and
    Mission2 `0.056 Hz`.
  - Standardize per channel using nominal training points after resampling,
    except where the official preprocessing excludes a channel type.
- Metrics:
  - Do not reuse this repo's SMAP/MSL point-wise or point-adjusted F1 as an
    ESA-ADB claim.
  - Use binary detections and time-domain metrics, not threshold-agnostic AUC
    metrics or sample-index timing.
  - Preserve the hierarchy:
    - corrected event-wise `F0.5` for false-alarm-sensitive event detection;
    - subsystem-aware and channel-aware `F0.5` for affected-source diagnosis;
    - event-wise alarming precision for repeated alarms;
    - ADTQC for detection timing;
    - modified affiliation-based `F0.5` for range/proximity quality.

## README-Paper Differences And Gaps

- Dataset version gap:
  - README and paper cite DOI `10.5281/zenodo.12528696`.
  - Zenodo now exposes v2 at `10.5281/zenodo.15237121`.
  - Decision: use v2 for new intake, and record both DOIs in manifests.
- Runtime gap:
  - The official README documents full experiment reproduction but does not give
    a lightweight one-command smoke path for this project's Windows workspace.
  - Decision: first implementation should create a manifest/downloader and an
    evaluator contract before attempting full official reproduction.
- Output contract gap:
  - The README says official experiment results are written under `results`, but
    the exact minimal prediction/detection file contract should be confirmed
    from the official TimeEval integration code before we write our own outputs.
  - Decision: done for the fixture layer. The official evaluator reads
    `labels.csv`, joins `anomaly_types.csv` by `ID`, casts score columns to
    binary `uint8`, passes global metrics a `Timestamp, Score` series, and
    passes channel-aware metrics a dictionary of channel names to
    `Timestamp, Score` series. A full result writer still waits for the
    lightweight Mission1 smoke path.

## Implemented Evaluator Contract Fixture Layer

The current code does not vendor the full official TimeEval stack. Instead it
adds fixture-tested helpers that prepare the same minimal objects consumed by
the official ESA-ADB evaluator:

- `read_esa_adb_evaluator_labels` reads `labels.csv`, parses `StartTime` and
  `EndTime`, and joins `Category`, `Dimensionality`, `Locality`, and `Length`
  from `anomaly_types.csv`.
- `group_esa_adb_binary_events` mirrors the official run grouping for binary
  detections: positive runs are closed-open until the next timestamp, except a
  run that reaches the final sample ends closed at the final timestamp.
- `build_esa_adb_metric_inputs` validates aligned per-channel
  `Timestamp, Score` predictions, builds the global max-score series used by
  event-wise metrics, and keeps the channel dictionary used by channel-aware
  metrics.

Fixture coverage now protects:

- labels and anomaly-type joins;
- missing anomaly-type IDs;
- binary event grouping;
- global and channel prediction input shapes;
- non-binary scores and misaligned channel timestamps.

## Smallest Protocol-Correct First Run

The first implementation milestone should not train a new deep model. It should
prove that this project can ingest and evaluate ESA-ADB without changing the
benchmark contract.

Recommended first run:

1. Done: add an ESA-ADB source manifest command that records:
   - dataset DOI/version,
   - raw archive names,
   - expected MD5 values,
   - local paths under ignored `data/raw/esa_adb/`,
   - official repository URL and commit/ref used for protocol decisions.
2. Done: add a no-download validator for locally supplied archives, starting
   with `ESA-Mission1.zip` from Zenodo v2.
3. Done: inspect only the minimal official evaluator interface needed to
   understand the binary-detection output contract.
4. Done: build tiny fixture tests for labels, anomaly types, event grouping, and
   metric-input shape before touching the full dataset.
5. Next: run a real lightweight Mission1 path only after the fixture evaluator
   contract is green:
   - preprocess Mission1 with the official script;
   - select channels `41-46`;
   - regenerate subset anomaly types;
   - score a simple official-compatible baseline;
   - write JSON and Markdown evidence that reports the official metric hierarchy.

Mission1 lightweight is the preferred first real run because it has only six
channels and exercises the point-anomaly preservation requirement. Mission2
lightweight should follow as the second run because it stresses rare nominal
events and the anomaly-versus-rare-nominal distinction.

## Implemented Source Gate

Write the source manifest:

```powershell
uv run aerospace-prognostics esa-adb-source-manifest --output-json artifacts/data/esa_adb_source_manifest.json
```

Validate a locally supplied Mission1 archive without downloading anything:

```powershell
uv run aerospace-prognostics esa-adb-verify-archives `
  --archive-dir data/raw/esa_adb `
  --manifest artifacts/data/esa_adb_source_manifest.json `
  --missions Mission1 `
  --output-json artifacts/data/esa_adb_mission1_archive_validation.json
```

This source gate must pass before extraction, preprocessing, evaluator
integration, or model work.

## Do Not Do Yet

- Do not download all three mission archives as a prerequisite for the first
  implementation slice.
- Do not claim ESA-ADB benchmark performance from SMAP/MSL metrics.
- Do not collapse rare nominal events into ordinary negatives unless a separate
  anomaly-only analysis explicitly documents that policy.
- Do not tune thresholds or standardization on test rows.
- Do not add a deep ESA-ADB model before the official evaluator contract and
  lightweight baseline path are proven.
