# Part 2 paired data processing: corrected common sample

- **Regenerated:** September 10, 2026, with scored-window eviction eligibility.
- **Cutoffs:** April 1, 2025 and April 1, 2026.
- **Design:** Retrospective reconstruction with fixed current geography/units,
  one-year source-vintage shifts and earlier-frozen component scores.
- **Measurement contract:** part2-fixed-components-v2.
- **Status:** Corrected matrix and downstream cluster comparison complete;
  Part 1 separately refreshed and Part 3 ML paused. Prior outputs overwritten.

## Result

The two identically ordered, complete matrices contain **2,351 common cells:
2,210 Travis and 141 Williamson**. The audit table retains all **14,054 rows:
7,027 cells at each date**. Hays is not in the analysis sample because
equivalent eviction records are unavailable.

The common cells contain approximately **363,949 fixed promoted residential
units**, not separate annual housing counts. There are **156 boundary-straddling
cells**. Event points must be inside the fixed current city boundary, but
denominators retain full-hex areas and units: these are not exact city-clipped
rates.

Every included index has its complete fixed recipe at both dates. There are
**zero changes in component availability among included cells**. Rent uses
block-group series for 1,456 common cells and tract series for 895, with the
chosen level held fixed across both snapshots and all six vintages.

## How the sample narrows

These are sequential, mutually exclusive exclusions. A cell failing several
checks appears at its first failure; separate flags retain every reason.

| Check | Additional excluded | Cells remaining |
| --- | ---: | ---: |
| Canonical audit grid | — | 7,027 |
| Center inside fixed current Austin FULL boundary | 967 | 6,060 |
| At least 20 fixed promoted units | 2,786 | 3,274 |
| Ownership common-support screen at both dates | 54 | 3,220 |
| Selected 311 coverage screen at both dates | 7 | 3,213 |
| Demolition source coverage at both dates | 0 | 3,213 |
| Eviction scored-window coverage and ambiguity at both dates | 186 | 3,027 |
| Amenity retrospective usability | 0 | 3,027 |
| Every required component of all seven indices at both dates | 676 | **2,351** |

Zero additional exclusions does not mean universal source coverage; preceding
gates remove overlapping unsupported cells. The 186 primary eviction
exclusions comprise 155 Travis, 25 Williamson and six Hays cells. Relative to
the preceding screen, 51 cells regain full paired eligibility; none are lost.

## Complete fixed recipes and per-stream availability

These are usable indices on the full audit grid, **not** final sample counts.

| Index | Required terms | April 2025 available | April 2026 available | Available both |
| --- | ---: | ---: | ---: | ---: |
| Rent pressure | 3 | 4,623 | 4,623 | 4,623 |
| Demographic vulnerability | 5 | 4,164 | 4,217 | 4,150 |
| Demolition pressure | 3 | 6,033 | 6,033 | 6,033 |
| Eviction pressure | 2 | 3,123 | 3,135 | 3,083 |
| Selected 311 pressure | 3 | 3,267 | 3,267 | 3,267 |
| Ownership pressure | 3 | 3,227 | 3,227 | 3,227 |
| Amenity change | 3 category scores | 7,027 | 7,027 | 7,027 |

Composites no longer average whatever terms happen to be available. A missing
required term makes its index unavailable. Rent retains level, growth and
acceleration through an all-six-vintage reliable BG/tract hierarchy.
Vulnerability requires all five components. Demolition, ownership and amenity
definitions remain unchanged, with full recipes enforced.

Evictions now average recent mapped filings per 100 fixed units and the signed
change from the previous 12-month rate. Expanding-history recent share and
percentage change remain raw diagnostics only. Selected 311 averages recent
rate, density and signed rate change; two thirds of the weight therefore
reflects current activity.

Unchanged terms retain original earlier p1/p99 bounds. Signed changes use
symmetric earlier bounds: eviction ±9.811020 filings per 100 units and 311
±28.75 requests per 100 units. Zero change scores 50. Zero events with zero
change can therefore yield an eviction index of 25; these are relative
indices, not probabilities with a literal no-risk zero.

## Source coverage and the narrower ambiguity window

Ownership's 20-common-unit and 95%-unit/parcel coverage screen remains intact.
Its source-agreement and certified-only sensitivity screens retain 3,196 and
3,063 cells respectively before the other domain requirements; neither replaces
the main reconstruction.

Integrated Travis and Williamson JP1/JP2 records provide continuous court
source coverage in 5,977 of 6,060 fixed-city cells. The remaining 54 Hays and
29 Williamson JP3 cells lack supplied filings. Scored-window localizable
uncertainty removes another 132/123 cells, leaving 5,845/5,854 cells with usable
mapped counts. Positive unit support and a complete score reduce those to the
3,123/3,135 finite eviction indices above.

The screen now covers only April 2, 2023–April 1, 2025 and April 2, 2024–April 1,
2026 respectively. Missing/conflicting dates potentially inside those windows
still mask candidate cells; older-only issues do not. Raw case records,
geocoding, units and spatial support were not changed. Entirely
unlocated records remain separate court/window QA; they cannot establish a
City-only match-rate denominator.

On the final **same 2,351 cells**, recent mapped filings are **5,708/6,059**,
selected requests **14,129/15,560**, and recent 24-month demolition permits
**819/993**. The later previous-window filing count equals the earlier recent
count, 5,708. These are counts for the covered subset, not all Austin events or
completed displacement.

## Limitations carried forward

The full source grid is retained so exclusions are visible. Adjacent ACS
releases overlap; six-vintage reliability and common ownership support use
retrospective information. Keeping rent's geographic level fixed is not
historical boundary harmonization. Amenities are retrospectively usable but
not proven exhaustive. 311 remains a coordinate-required selected-request
universe with a conservative geographic screen. Unknown evidence is not zero,
and the covered sample is not claimed to represent the entire city.

## Verification and artifacts

Synthetic and persisted-output tests cover frozen scoring, signed changes,
zero versus missing, strict composite requirements, source boundaries,
coverage, fixed support, paired keys, rent selection, identical matrix IDs and
exclusion reconciliation. The matrix verifies six domain manifests and nested
source pins, checks sources unchanged during assembly, and independently
reconciles the corrected composite values.

The final matrix integration test passes **35 manifest checks** and verifies
**886 non-Part1/non-Part2 output/figure artifacts unchanged** after the approved
Part 1 rebuild. Its as-run preservation inventory is retained. Part2 outputs were
intentionally overwritten, not included in an old-run preservation archive.

Primary artifacts under output/part2/matrix/:

- part2_features_paired.rds/.csv: full grid/date table.
- part2_eligibility_by_hex.rds/.csv: all exclusion and availability flags.
- part2_analysis_matrix_2025-04-01.rds/.csv and 2026-04-01 counterparts.
- Exclusion, county and index summaries; source registry and run manifest.

See [methods and reproduction](../methods/historical-feature-matrix.md),
[readiness](part2-historical-readiness-2026-09.md) and the
[completed corrected cluster comparison](part2-cluster-comparison-2026-09.md).
