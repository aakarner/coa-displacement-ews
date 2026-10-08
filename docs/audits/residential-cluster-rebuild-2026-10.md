# Residential repairs and full-purpose grid rebuild — October 7, 2026

Production current measures, Part 1 clusters, paired Part 2 measures/clusters,
and their maps have been rebuilt. All 50 R/Python test scripts pass. This closes
the deferred rebuild described in the residential follow-up and residual-repair
reports. The analysis cutoff remains April 1, 2026; the adopted municipal boundary
remains the April 29, 2026 snapshot chosen by the user.

## What changed

The batch incorporates the previously staged geocode-precision rule, Ocotillo
address review, source-year ownership reconciliation/reference reviews, 793 Oak
Ranch homes (+791 net units), 676 additional manufactured homes, and the Olivine
and Bennett property-reference corrections. Earlier Bell/Asher/Monarch and other
reviewed case corrections remain in force. Verified case and unit locations
share property geography; ambiguous cases remain unassigned without suppressing
otherwise usable cells. The 20-unit threshold is unchanged.

Sources and the individual decisions remain in the
[follow-up report](residential-followup-2026-10.md),
[residual-repair report](residential-residual-repairs-2026-10.md), and
[earlier closeout](residential-geography-closeout-2026-10.md).
The Ben White count remains an explicitly provisional 170-unit proxy with its
[documented bed-count source](ben-white-provisional-units-2026-10.md).

The grid is the union of the old resolution-9 surface and all cells intersecting
the adopted full-purpose boundary with positive area. It adds 923 cells and
covers approximately 33.43 square kilometers of City land omitted by the old
surface. Independent geometry validation finds no uncovered City area. Every
original numeric ID, H3 index, geometry and reference coordinate is preserved.
New IDs are permanently registered in `config/hex_id_registry.csv`.

| Measure | Previous production | Completed rebuild |
| --- | ---: | ---: |
| Computational H3 cells | 7,027 | 7,950 |
| City-center cells | 6,060 | 6,196 |
| Part 1 eligible cells | 2,677 | 2,693 |
| Part 2 common eligible cells | 2,505 | 2,515 |
| Promoted units aggregated across the computational grid | 512,415.31 | 519,390.75 |
| Recent assigned filings | 11,784 | 11,722 |
| Recent assigned filings on cells below 20 units | 215 | 106 |
| Cells containing those low-unit filings | 77 | 55 |

The filing window is **April 2, 2025–April 1, 2026**, not calendar 2026. These are
comparisons to the immediately preceding production run; earlier report stages
have different baselines. The total unit increase combines 1,467 net recovered
homes and 5,508.44 existing estimated units newly captured by the expanded grid.
The grid total includes retained out-of-scope cells and is not a City housing
estimate. Eight recent assigned filings now fall on newly added cells; two were
already assigned elsewhere and move to a verified property reference, so this
is six additional assigned filings relative to the repaired old-grid staging run.

## Why expanded coverage does not mean every added cell is clustered

Analytical eligibility still requires the projected cell reference point inside
the City, sufficient residential units, and all required source support. Of the
923 additions, 787 fail the center-based City rule. Among the 136 inside, 125
have fewer than 20 units, five fail the demolition comparison coverage gate,
one fails the eviction count gate, one fails selected-311 coverage, one lacks
complete components, and three enter Part 1 (IDs 7310, 7313 and 7517).
These are the first failing eligibility gates; they are not mutually exclusive
causes of all missing information.

The resulting computational surface fully covers the municipal polygon while
whole-cell analytical inclusion still uses the existing center rule. Admitting
all boundary-intersecting cells would be a separate measurement decision.
See [decision 0020](../decisions/0020-full-purpose-h3-grid.md).
Both Part 2 vintages use the same expanded geography, preserving the fixed
geography comparison. The new surface adds no new court-source coverage: Hays
and uncovered Williamson precincts remain explicit missing data.

## Substantive cluster effects

Cluster numbers are nominal and were matched by centroid profiles before
comparison. All seven familiar profiles remain supported by their feature
centers and raw event counts. Among 2,676 cells classified in both production
versions, **2,606 (97.4%) retain their profile**; 70 change profile and 50 change
qualitative concern tier. Seventeen cells become eligible and one ceases to be
eligible, for a net gain of 16. The removed cell, 1571, no longer has a usable
rent series after the expanded-grid ACS allocation; it is retained in the audit
surface with an explicit missing-component exclusion.

| Part 1 profile | Previous cells | Rebuilt cells |
| --- | ---: | ---: |
| Lower Measured Pressure | 900 | 904 |
| Higher Rents / Lower Vulnerability | 687 | 656 |
| Nearby Amenity Activity | 137 | 145 |
| Corporate Ownership + Vulnerability | 357 | 359 |
| Selected 311 + Vulnerability | 244 | 239 |
| Demolition-Permit Concentration | 240 | 268 |
| Eviction-Filing Concentration | 112 | 122 |

The largest profile flows are 18 cells from Higher Rents / Lower Vulnerability
to Demolition-Permit Concentration, 14 from Higher Rents / Lower Vulnerability
to Lower Measured Pressure, and 11 from Lower Measured Pressure to Demolition.
These compare corrected analytical versions, not neighborhood change over time.
Every cell in the rebuilt eviction profile has recent mapped filings, averaging
13.7 per 100 units. Every demolition-profile cell has a recent residential permit.

The selected seven-domain solution has average silhouette 0.2212 (previously
0.2215) and mean subsample adjusted Rand index 0.9591 (previously 0.9517).
The 100 gap bootstraps and 100 stability replicates were retained. The seven
clusters remain a substantive typology choice; this is not evidence of seven
objectively distinct risk levels or calibrated displacement probabilities.

Part 2 retains all 2,505 previous common-sample cells and adds ten. Matching
profiles across model versions, 12 shared cells change their 2025 profile and
14 change their fixed-definition 2026 assignment. Within the rebuilt paired
analysis, 913 of 2,515 cells change profile between dates (36.3%), compared with
916 of 2,505 before (36.6%). The qualitative overall comparison is therefore
little changed. The rebuilt Part 2 baseline eviction profile averages 11.7
filings per 100 units; Part 1 and Part 2 have different dates/samples and models.

## Remaining limits

There are still **106 recent assigned filings in 55 City-center cells below
20 units**: 56 filings in 26 zero-unit cells and 50 in 29 positive-unit cells.
They remain in the case ledger and audit counts, but these cells do not enter
clustering. This residual does not establish that each filing is mislocated:
it includes genuinely small residential concentrations and still-unresolved
property/unit evidence. Coarse-only and ambiguous cases remain separately
flagged and unassigned.

The prior deferred items remain deferred: Hudson Miramont's split-City parcel
needs defensible residential allocation, Maravilla needs mixed-use review,
39 Oak Ranch records and three suspect home records are withheld, and shared
park/phase reference points remain an explicit aggregation convention. Current
ownership evidence is not backfilled into an unsupported historical year.
This rebuild does not resolve the separate demolition-methodology review.

## Validation and reproducibility

All **50 `tests/test_*.R` and `tests/test_*.py` scripts pass**, including raw-source
checksum verification, current/paired source manifests, case conservation,
annual/paired assignment parity, all 144 manually reviewed case assignments,
reviewed unit/property totals, fixed-period feature reconstruction, clustering
and concern interpretation checks, and source-year ownership sensitivities.
Independent spatial tests verify full City coverage, every original ID/geometry,
and unchanged original county/JP assignments. Both principal maps were visually
inspected after regeneration.

Tests formerly tied to 7,027 cells or obsolete repair-stage outputs now validate
the new grid contract or explicitly replay the preserved historical endpoint.
Case-conservation tests permit only documented precision changes, newly covered
locations, and verified new-grid property references; other precise assignments
remain protected. Older preservation inventories release an explicit list of
authorized derived replacements while retaining raw-source checksum checks.

The old production products are preserved locally under
`output/residential_cluster_rebuild_20261007/before/`. The same directory's parent
contains `rebuild_comparison.json`, matched profile transitions and sizes,
newly/no-longer eligible cells, changed case assignments, remaining low-unit
cells, centroid-label mappings and `test_results.json`. Case-level artifacts
remain local/ignored. `scripts/audits/cluster_rebuild_20261007.R` reproduces the
before/after comparison from these archived endpoints.

The rebuild used production scripts in dependency order with existing source
caches. No new City boundary vintage or geocoding cache fill was introduced.
Raw unit-estimator inputs/models were retained; reviewed promotion overlays were
applied. Current and paired selected domains, annual eviction/demolition outcomes,
forecast labels, cluster models, diagnostics and maps were regenerated. Part 3
ML fitting remains paused; legacy nonselected feature artifacts were not all
refreshed, and this does not claim a complete `targets` cache refresh.

Boundary provenance: local `data/BOUNDARIES_jurisdictions_20260429.geojson`
(SHA-256 `f020a2441f63150746a80616e83ae163c47839d43b5b248357aeffc5d9d5b39f`),
from the [City of Austin jurisdiction dataset](https://data.austintexas.gov/Locations-and-Maps/Jurisdiction-Austin/3pzb-6mbr/about).
County and effective-dated JP references reuse the original official source
polygons with matching source hashes; their regenerated metadata record the
expanded-grid hash. `output/hex_grid_manifest.json` pins the adopted boundary,
ID registry, computational/analytical rules, grid hash and independent counts.
