# Datasets, provenance, and what this repository may claim

Every number in this repository comes from a **public benchmark dataset**. No part of it
touches real in-service aircraft, real spacecraft operations, or any operator's
maintenance system.

## Provenance

| Dataset | Provenance | Terms as recorded |
| --- | --- | --- |
| NASA C-MAPSS turbofan degradation | NASA Prognostics Data Repository (PCoE). *Simulated* engine degradation, not measured engines. Source URL recorded in [phase1_cmapss_baseline_results.md](phase1_cmapss_baseline_results.md) | US Government work; NASA attaches no formal open-source licence. Cite Saxena & Goebel (2008) |
| SMAP / MSL spacecraft telemetry | NASA/JPL, released with the Telemanom work. Ingested from the Telemanom S3 release (`s3-us-west-2.amazonaws.com/telemanom/data.zip`), recorded in the local download metadata; the Kaggle mirror is a documented fallback only | No formal dataset licence attached by the publisher; the Telemanom *code* is Apache-2.0. Treat as research-use. Cite Hundman et al. (2018) |
| ESA-ADB spacecraft telemetry | European Space Agency Anomaly Dataset | Data: `CC BY 3.0 IGO`. Benchmark *code*: MIT. Attribution required — the attribution text is in [license_posture.md](license_posture.md) |

None of these datasets are redistributed here; each is downloaded locally. Raw telemetry
and generated model artifacts stay out of Git — keep datasets under `data/` or another
documented local path, and record source URLs and checksums when adding download scripts.

## What that entitles this repository to claim

Benchmark data supports claims about *method* — that a pipeline is reproducible, that a
split is leakage-free, that one estimator beats another under a stated protocol. It
supports **no** claim about real fleets, real airframes, real spacecraft, or the safety of
any maintenance decision. C-MAPSS in particular is *simulation output*: a result on it is
evidence about modelling, not about engines.

Words like "fleet", "operator console", "operations", and "deployable" in this repository
describe the **shape of the software** — the surfaces a real PHM system would need — not a
deployment, a certification, or an operational qualification. This is not airworthiness
evidence and must not be used as input to any real maintenance or flight-safety decision.

Per-number scope, and what each number is and is not allowed to support, is in the claims
ledger: [../claims.md](../claims.md).

## Downloading SMAP/MSL when the S3 archive is unavailable

Telemanom's current README points users to the Kaggle-hosted SMAP/MSL archive. If the
legacy public S3 archive is unavailable, download
`patrickfleith/nasa-anomaly-detection-dataset-smap-msl` to
`data/raw/downloads/smap_msl_telemanom.zip`, then rerun `smap-msl-download`; the command
imports that local archive without downloading it again.

## Related

- [license_posture.md](license_posture.md) — current open-source posture and the ESA-ADB
  attribution text.
- [phase3_esa_adb_intake.md](phase3_esa_adb_intake.md) — the ESA-ADB protocol lock,
  including dataset versioning and the two Zenodo DOIs.
