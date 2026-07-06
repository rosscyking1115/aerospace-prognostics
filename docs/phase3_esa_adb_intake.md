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

## Implemented Event-Wise Detection Scoring Layer

`data/esa_adb_scoring.py` adds the scoring slice above the evaluator-contract
fixtures, all offline and fixture-tested:

- `ESA_ADB_LIGHTWEIGHT_CHANNELS` / `lightweight_channel_numbers` codify the
  official lightweight subsets as channel numbers (no on-disk naming scheme is
  assumed until confirmed against real archives).
- `robust_zscore_detections` is a simple robust z-score baseline that emits the
  official `Timestamp, Score` binary-detection contract. "Official-compatible"
  refers to the emitted output contract, not to an official ESA-ADB detector.
- `score_esa_adb_event_wise` scores sample-based event-wise detection from the
  metric inputs: a labelled event is detected when any positive sample lands in
  its `[StartTime, EndTime]` interval; a predicted run is a false alarm when
  none of its samples land in any event. `beta=0.5` weights precision above
  recall to match ESA-ADB's false-alarm sensitivity.
- `build_esa_adb_event_wise_evidence` / `write_esa_adb_event_wise_evidence`
  emit JSON and Markdown evidence stamped with an explicit scope caveat.

**Scope guardrail.** This layer computes only the *event-wise detection* top of
the ESA-ADB metric hierarchy. It does not yet reproduce ADTQC detection timing,
affiliation-based proximity, or subsystem-aware diagnosis. Every artifact is
labelled "protocol-shaped detection evidence, pending official-evaluator
cross-check; not a full ESA-ADB leaderboard claim." Do not quote these numbers
as an ESA-ADB benchmark result.

Score prepared inputs without any download (labels, anomaly types, and a
directory of per-channel `Timestamp, Score` CSVs):

```powershell
uv run aerospace-prognostics esa-adb-mission-score `
  --mission Mission1 `
  --lightweight `
  --labels-csv data/raw/esa_adb/Mission1/labels.csv `
  --anomaly-types-csv data/raw/esa_adb/Mission1/anomaly_types.csv `
  --predictions-dir artifacts/esa_adb/mission1_predictions `
  --output-json artifacts/esa_adb/mission1_event_wise.json `
  --output-markdown artifacts/esa_adb/mission1_event_wise.md
```

## First Real Mission1 Result

The lightweight Mission1 baseline has now run on real ESA Anomaly Dataset
telemetry (`data/esa_adb_mission1.py`, command `esa-adb-mission1-run`). This is
the first ESA-ADB number in the project, and it is deliberately a conservative
baseline, not a leaderboard entry.

Run setup:

- Target channels 41-46 (all `Target=YES` in `channels.csv`), which share an
  identical native ~30s time grid, so they are already aligned.
- Chronological 50/50 split: ~7.69M train samples, ~7.69M test samples; the
  test window opens `2006-12-22`.
- Robust z-score baseline (median/MAD, threshold 5.0) fit on nominal training
  points only (training rows outside every labelled anomaly), then applied to
  the test window. No thresholds or standardization touch test rows.
- Events filtered to those overlapping the test window; `Communication Gap`
  events excluded, matching the default benchmark table.

Result (event-wise detection only):

| Metric | Value |
| --- | ---: |
| Test-window events | 65 |
| Detected events | 27 |
| Missed events | 38 |
| Predicted alarms | 86 |
| False alarms | 0 |
| Event-wise precision | 1.000000 |
| Event-wise recall | 0.415385 |
| Event-wise F0.5 | 0.780347 |

Reading: the baseline is precise but low-recall — it raises no false alarms on
the test window yet catches under half of the labelled events, which is the
expected profile of a conservative robust-threshold detector and leaves clear
headroom for a real model.

Honest deviations from full official reproduction, recorded in the artifact
provenance:

- Only event-wise detection is scored; ADTQC timing, affiliation-based
  proximity, and subsystem-aware/channel-aware diagnosis are not yet computed.
- The official zero-order-hold resampling to the Mission1 target frequency is
  not applied; the baseline scores on the native aligned grid.
- The number is therefore protocol-shaped detection evidence, not an ESA-ADB
  leaderboard claim, and must not be quoted as one.

Reproduce (after extracting the mission folder locally; raw data stays out of
Git):

```powershell
uv run aerospace-prognostics esa-adb-mission1-run `
  --archive data/raw/esa_adb/ESA-Mission1 `
  --output-json artifacts/esa_adb/mission1_lightweight_event_wise.json `
  --output-markdown artifacts/esa_adb/mission1_lightweight_event_wise.md
```

The command also accepts `--archive path/to/archive.zip` directly; on a
memory-constrained machine, extracting the six channel files first avoids
decompressing large zip members in the same process as the model stack.

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
5. Partially done: the offline scoring layer that a real lightweight Mission1
   run will use is now implemented and fixture-tested (see "Implemented
   Event-Wise Detection Scoring Layer" below):
   - lightweight subset channel numbers are codified (`Mission1` `41-46`,
     `Mission2` `18-28`);
   - a simple robust z-score baseline emits the official binary-detection
     contract;
   - event-wise detection precision/recall/F0.5 is scored from the
     evaluator-contract metric inputs;
   - JSON and Markdown evidence is written with explicit scope caveats.
   - Done on local data: the real lightweight Mission1 baseline now runs on the
     supplied ESA Anomaly Dataset archive (see "First Real Mission1 Result"
     below).

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
