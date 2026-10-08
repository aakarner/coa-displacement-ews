# Production application of reviewed property locations

Implemented October 7, 2026, under [decision 0016](../decisions/0016-reviewed-case-property-locations.md).
Recent-filing comparisons cover **April 2, 2025–April 1, 2026**, not calendar
2026. The analytical cutoff remains April 1, 2026.

## Applied corrections

All **71 reviewed filings** now use their verified property's existing unit
cell in both the paired snapshots and annual outcome panel. These are movements
of accepted filings; original addresses, geocodes, case exclusions and unit
counts are unchanged. No other case assignments changed.

| Reviewed property | Filings | Original cell | Production unit cell |
|---|---:|---:|---:|
| Bell Southpark — Springs | 53 | 6381 | 6670 |
| Bell Southpark — Meadows | 1 | 6381 | 6974 |
| Bridge at Asher | 6 | 6381 | 6973 |
| Bridge at Monarch Bluffs | 11 | 6965 | 6964 |

Bell's overlapping apartment numbers are resolved at the individual-case level.
One Springs case retains the conflicting apartment numbers from the export;
its court plaintiff/DBA, shared street address and corroborating property
evidence support the property without establishing the exact apartment. The
production case ledger retains `property_apartment_conflict = TRUE` for that
case. Neither source apartment number is corrected.

Monarch's destination is the project containing land account 533185 and
residential improvement account 975264. The improvement's existing units supply
the reference cell. No housing is attributed to the neighboring restaurant.

## Measurements

| Measure | Before | After |
|---|---:|---:|
| Recent filings in covered City cells | 11,784 | 11,784 |
| Eligible Part 1 cells | 2,675 | 2,675 |
| Recent filings in eligible Part 1 cells | 10,508 | 10,579 |
| Cells with filings and fewer than 20 units | 84 | 82 |
| Recent filings in those low-unit cells | 405 | 334 |
| Unresolved low-unit filings in the original 41-cell audit | 217 | 146 |

The promoted housing surface remains **511,725.3605 units** across the grid;
the unit files match their pre-change hashes exactly. This includes estimated
counts under the existing hierarchy. Proposed count reconciliations for Springs,
Asher and Monarch have not been applied. Domain and Ben White remain separate
review work, as do the other residual cases.

The six cells whose filing counts change are:

| Cell | Before filings | After filings | Existing whole-cell units |
|---:|---:|---:|---:|
| 6381 | 60 | 0 | 1 |
| 6670 | 0 | 53 | 330.86 |
| 6964 | 7 | 18 | 361.29 |
| 6965 | 11 | 0 | 0 |
| 6973 | 7 | 13 | 461 |
| 6974 | 27 | 28 | 219 |

## Substantive results

Part 1 retains the same seven profiles and all 2,675 eligible cells. After
refitting, **2,627 cells (98.2%) retain their substantive profile**, while 48
change. Numeric cluster IDs were remapped after reviewing the new centroids
and profile anchors; their numeric ordering has no substantive meaning.

| Profile | Before cells | After cells |
|---|---:|---:|
| Lower Measured Pressure | 904 | 903 |
| Higher Rents / Lower Vulnerability | 662 | 685 |
| Nearby Amenity Activity | 137 | 137 |
| Corporate Ownership + Vulnerability | 359 | 360 |
| Selected 311 + Vulnerability | 239 | 244 |
| Demolition-Permit Concentration | 263 | 236 |
| Eviction-Filing Concentration | 111 | 110 |

Holding the preceding classifier's centroids and standardization fixed changes
only **one** assignment. The larger refitted change includes changes in cluster
boundaries; it does not imply that eviction evidence changed in 48 cells or
that demolition data changed. The eviction profile's mean filing rate is
14.8 per 100 units; every member has at least one recent mapped filing.

Two neighborhood population-plurality labels change: North Shoal Creek moves
from Corporate Ownership + Vulnerability to Higher Rents / Lower Vulnerability;
West University moves from Selected 311 + Vulnerability to Corporate Ownership
+ Vulnerability. These are plurality descriptions, not uniform neighborhood
conditions.

Part 2 retains **2,503 common cells**. Its 2025 baseline centroids and assignments
are identical, so its existing baseline interpretation remains valid. One 2026
assignment under that fixed baseline changes; 2025-to-2026 fixed-model
transitions increase from 905 to 906. The later refit, interpretation tables and
figures were rebuilt. Annual Part 3 counts and forward labels were also rebuilt;
partial 2026 remains partial and model training remains paused.

## Implementation and validation

`config/eviction_property_reviews.json` pins the local input bundle under
`data/reviewed_eviction_properties/batch1_20261007/`. That immutable bundle holds
the 71 decisions and 57 evidence files, including the court screenshots and
review snapshots. Private records remain ignored by Git. The original intake
JSON records retain their historical review state; the production assignment
ledgers and application audit show the implemented result.

`R/eviction_property_reviews.R` validates the evidence and compiles case-specific
links against the current operational unit reference. The shared crosswalk
includes those links alongside automatic polygon matching. Changed evidence,
dates, courts, reliable-address sets, geocodes or reviewed unit cells fail
validation. An unreviewed case sharing an address cannot inherit a correction.
The original case resolver and City/county/court restrictions remain in force.
The targets dependency graph connects review inputs to the shared crosswalk,
paired eviction snapshots and current measurements; the annual panel consumes
the same crosswalk.

Validation covers exact annual/paired assignment agreement for all 71 cases,
preservation of every other case's assignment, original exclusions and annual
totals, unchanged source/unit-file hashes, apartment-flag retention, and rejection
of stale evidence or new address variants. Independent checks reproduce rates,
eligibility, cluster assignments and interpretation from the saved products.
The Part 1 fit uses the full 100 gap bootstraps and 100 stability replicates.
Its model/lock audit passes all 19 checks.

Detailed comparisons and the baseline are local under
`output/property_review_production/`. Reproduce the comparison with
`Rscript scripts/audits/reviewed_property_locations.R`. Tests specific to this
repair are `tests/test_reviewed_property_geography.R` and
`tests/test_reviewed_property_outputs.R`, alongside the existing property,
ambiguity, measurement and cluster output checks. Future changes should create
a newly reviewed input batch; do not edit generated assignments directly.
