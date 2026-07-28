# License Posture

Current posture:

- Repository code is distributed under the MIT License.
- Raw NASA/JPL/ESA datasets are not redistributed by this repository.
- Generated model artifacts, downloaded telemetry, and large experiment outputs
  remain ignored by Git.
- Dataset-specific licenses and source records stay documented separately from
  the repository code license.

This license keeps the code straightforward to inspect, reuse, and fork, while
the project remains a reference implementation rather than a commercial PHM
product.

## Dataset terms and required attribution

None of these datasets is redistributed by this repository; each is downloaded
locally by the documented commands. Terms as recorded:

### ESA-ADB (European Space Agency Anomaly Dataset)

Data is licensed **`CC BY 3.0 IGO`**, which *requires attribution*. The ESA-ADB
benchmark code is separately MIT-licensed. Any publication, figure, or derived
result built on this data must carry:

> Contains modified European Space Agency Anomaly Dataset (ESA-ADB) material,
> © European Space Agency, licensed under CC BY 3.0 IGO
> (https://creativecommons.org/licenses/by/3.0/igo/).

### NASA C-MAPSS turbofan degradation

US Government work from the NASA Prognostics Data Repository (PCoE). NASA
attaches no formal open-source licence; the repository requests citation:

> A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation Data
> Set, NASA Ames Prognostics Data Repository, NASA Ames, Moffett Field, CA.

### NASA/JPL SMAP/MSL telemetry

Released with the Telemanom work and ingested from the Telemanom S3 archive.
No formal dataset licence is attached by the publisher; the Telemanom *code* is
Apache-2.0. Treat the data as research-use and cite:

> K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom (2018).
> Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
> Thresholding.

Where a licence is described here as "not formally attached", that is a
statement about what the publisher provides, not a grant of any right by this
repository.
