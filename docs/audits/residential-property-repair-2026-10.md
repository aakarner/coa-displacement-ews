# Residential inventory and eviction/property geography repair

Implemented October 2, 2026, under [decision 0015](../decisions/0015-residential-property-geography.md).
Filing comparisons below use **April 2, 2025–April 1, 2026**, not calendar 2026.
The analytical cutoff remains April 1, 2026.

## What changed

The Williamson supplement now searches the available inventory for active
C3/C5 accounts with positive living area and independent residential evidence.
It no longer relies solely on residential words in the county descriptions.
Exact-parcel City apartment/condominium use or documented housing review is
required; C3/C5 alone does not establish housing. The CORNER STORE account
R081218 is an explicit commercial negative control.

The repair adds **35 active accounts**. The existing final land-use review gives
34 positive unit counts and excludes one from the operational denominator.
These new accounts contribute approximately 9,498 estimated units. Other
recomputed estimates change slightly, giving a net gridwide increase of 9,468.
Most recovered counts remain estimates under the existing hierarchy; property
verification does not turn them into independently confirmed unit counts.

The original audit's 18 omitted polygon IDs are not 18 independent active
developments. Six are reference accounts. Explicit certified NON-REF links and
the reviewed Lakeline Station link consolidate these into their active housing
accounts. Seventeen of the 18 audited footprints are now represented by 15
active accounts. Caliza (R500219) remains outside the supplement's point-in-grid
selection: its representative point lies outside the grid even though some
filing addresses lie inside the property and City. Its nine audited filings
remain a boundary/geography review item.

Verified filing addresses now use the same operational reference cell as their
residential property. Every reliable address for a case must agree on one
property and one unit cell. Original coordinates, original cells and decisions
are preserved. The same crosswalk is consumed by paired snapshots and annual
counts. Original ambiguity exclusions, accepted case totals, source/court/City
coverage and the 20-unit threshold remain unchanged.

## Staged results

“Inventory only” rebuilds housing and all dependent measurements while keeping
accepted filings in their original geocode cells. The combined repair adds the
verified property reassignment.

| Measure | Before | Inventory only | Combined repair |
|---|---:|---:|---:|
| Operational units across the grid, rounded | 502,257 | 511,725 | 511,725 |
| Eligible Part 1 cells | 2,660 | 2,675 | 2,675 |
| Recent filings in eligible Part 1 cells | 9,528 | 9,716 | 10,508 |
| Recent filings in covered City cells | 11,784 | 11,784 | 11,784 |
| Covered cells with filings and fewer than 20 units | 170 | 154 | 84 |
| Recent filings in those low-unit cells | 1,614 | 1,365 | 405 |

Across all covered City cells, 2,422 recent cases move to verified unit cells,
8,207 already share their verified unit cell, and 1,155 retain their accepted
original cell without a verified property link. Verification does not require
moving a case. The excluded cases remain excluded and do not suppress cells.

Of the **1,219 filings in the original 41-cell audit**, **1,002 (82.2%)** now
have at least 20 units in their analytical cell:

- 776 align with already-present housing in another cell.
- 226 gain recovered Williamson housing support: 191 stay in their original
  cell and 35 move to the active property's reference cell.
- 217 remain in ten low-unit cells: 134 need further account/residential-use
  reconciliation, 71 have points just outside county parcel polygons, three
  have existing unit references outside the grid, and nine are at Caliza.

Those 217 are retained and flagged. They are not assigned to nearby housing
merely to create a denominator. The broader total of 405 also includes low-unit
filings outside the original 41-cell cohort.

## Part 1 substantive changes

There are 16 newly eligible cells and one lost cell, a net gain of 15. Cell 2326
loses eligibility because restoring its housing reveals insufficient usable
ownership evidence; its units rise from 72 to approximately 366. No coverage
requirement is relaxed to retain it.

Of the 2,659 cells eligible both before and after, **2,514 (94.5%) retain the same
substantive profile** and 145 change. Labels were reviewed against the new
profiles; numeric cluster IDs were remapped because they have no substantive
ordering.

| Substantive profile | Before cells | Repaired cells |
|---|---:|---:|
| Lower Measured Pressure | 913 | 904 |
| Higher Rents / Lower Vulnerability | 689 | 662 |
| Nearby Amenity Activity | 129 | 137 |
| Corporate Ownership + Vulnerability | 359 | 359 |
| Selected 311 + Vulnerability | 250 | 239 |
| Demolition-Permit Concentration | 235 | 263 |
| Eviction-Filing Concentration | 85 | 111 |

The eviction-concentration profile grows while its unweighted mean filing rate
falls from **27.9 to 14.7 per 100 units**. Every member still has a recent mapped
filing. This is a comparison of refitted groups, not a same-property rate change.
The seven recognizable profiles remain, with materially less denominator
distortion in the filing measure.

Holding the previous cluster centers and standardization fixed, the
inventory-only measurements change seven common-cell assignments; the combined
measurements change 93. Current component scores are rebuilt in both stages,
as required by Part 1. Refitting the clusters produces the 145 changes above.
Eight neighborhood population-plurality labels change: Coronado Hills,
Georgian Acres, North Shoal Creek, South Lamar, South River City, Tech Ridge,
University Hills and West University. These are plurality descriptions, not
uniform neighborhood conditions.

Part 2's common sample increases from **2,490 to 2,503 cells**. Its 2025 and
2026 models and semantic interpretations were rebuilt independently of Part 1.
The annual Part 3 panel and forward labels were rebuilt using the same case
assignments; partial 2026 does not become a completed outcome year. Part 3
model training remains paused.

## Evidence, validation and reproduction

New evidence/configuration:

- `config/wcad_residential_evidence_reviews.csv`
- `config/wcad_residential_geometry_reviews.csv`
- `R/wcad_residential_evidence.R`
- `R/eviction_property_geography.R`

Local detailed outputs are ignored by Git:

- `output/property_geography/`: operational unit references, parcel/account
  links, address crosswalk, unmatched review flags and input hashes.
- `output/residential_geography_repair/`: preserved baseline, inventory-only
  snapshot, three-stage comparison, all 41 cells after repair, residual queue,
  recovered accounts and semantic transitions.
- `output/part3/eviction_property_assignment_ledger.csv`: annual case decisions.

The rebuild order is unit calibration → Census unit validation → unit source
linking → project grouping → count models → Williamson validation → integration
→ promotion → corporate/unit features → property crosswalk. Rebuild dependent
ownership snapshots/index, ACS rent crosswalks/demographics and 311 snapshots.
Then rebuild paired evictions → current measurement/features → Part 1 fit,
reviewed labels, frozen model, maps and audits → Part 2 matrix, fits, reviewed
interpretation and figures. Rebuild the annual panel and forecast labels.

`EWS_EVICTION_GEOGRAPHY=raw` is the explicit diagnostic switch used for the
inventory-only snapshot. Production runs omit it. Source stamps and crosswalk
hashes prevent silently consuming stale products. Run
`scripts/audits/residential_property_repair.R` after the complete rebuild to
regenerate the comparison from the preserved local snapshots.

Validation covers commercial negative controls and reference-account
deduplication; exact unit reconciliation; accepted-case and ambiguity
preservation; annual/paired assignment agreement; counts, missingness and rate
formulas; current eligibility and component scores; and independent frozen
cluster reproduction. The Part 1 fit retains 100 gap bootstraps and 100
stability replicates. Raw court records and geocodes are not rewritten.
