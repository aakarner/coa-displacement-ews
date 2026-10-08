# Part 2 paired data processing: corrected common sample

The run statistics below precede the October residential/property repair.
The current paired matrix contains 2,503 cells; see the
[repair audit](residential-property-repair-2026-10.md).

- **Regenerated:** September 11, 2026, with ambiguous cases unassigned and candidate cells retained.
- **Cutoffs:** April 1, 2025 and April 1, 2026.
- **Design:** Retrospective reconstruction with fixed current geography/units,
  one-year source-vintage shifts and earlier-frozen component scores.
- **Measurement contract:** part2-fixed-components-v2.
- **Status:** Corrected matrix and downstream cluster comparison complete;
  Part 1 separately refreshed and Part 3 ML paused. Prior outputs overwritten.

## Result

The two identically ordered, complete matrices contain **2,490 common cells:
2,348 Travis and 142 Williamson**. The audit table retains all **14,054 rows:
7,027 cells at each date**. Hays is not in the analysis sample because
equivalent eviction records are unavailable.

The common cells contain approximately **411,741 fixed promoted residential
units**, not separate annual housing counts. There are **163 boundary-straddling
cells**. Event points must be inside the fixed current city boundary, but
denominators retain full-hex areas and units: these are not exact city-clipped
rates.

Every included index has its complete fixed recipe at both dates. There are
**zero changes in component availability among included cells**. Rent uses
block-group series for 1,554 common cells and tract series for 936, with the
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
| Eviction scored-window source coverage at both dates | 30 | 3,183 |
| Amenity retrospective usability | 0 | 3,183 |
| Every required component of all seven indices at both dates | 693 | **2,490** |

Zero additional exclusions does not mean universal source coverage; preceding
gates remove overlapping unsupported cells. Removing the ambiguity veto restores
139 cells relative to the preceding 2,351-cell sample; no cells are lost.

## Complete fixed recipes and per-stream availability

These are usable indices on the full audit grid, **not** final sample counts.

| Index | Required terms | April 2025 available | April 2026 available | Available both |
| --- | ---: | ---: | ---: | ---: |
| Rent pressure | 3 | 4,623 | 4,623 | 4,623 |
| Demographic vulnerability | 5 | 4,164 | 4,217 | 4,150 |
| Demolition pressure | 3 | 6,033 | 6,033 | 6,033 |
| Eviction pressure | 2 | 3,241 | 3,241 | 3,241 |
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

Eviction rate bounds are refitted on the expanded 2025 domain support
(p1=0, p99=21.07632 per 100 units). Other domains retain their reviewed
bounds. Signed changes use symmetric earlier bounds: eviction ±11.18147
filings per 100 units and 311
±28.75 requests per 100 units. Zero change scores 50. Zero events with zero
change can therefore yield an eviction index of 25; these are relative
indices, not probabilities with a literal no-risk zero.

## Source coverage and ambiguity audits

Ownership's 20-common-unit and 95%-unit/parcel coverage screen remains intact.
Its source-agreement and certified-only sensitivity screens retain 3,196 and
3,063 cells respectively before the other domain requirements; neither replaces
the main reconstruction.

Integrated Travis and Williamson JP1/JP2 records provide continuous court
source coverage in 5,977 of 6,060 fixed-city cells. The remaining 54 Hays and
29 Williamson JP3 cells lack supplied filings. The 132/123 covered candidate
cells affected by ambiguity retain usable mapped counts. All 5,977 covered
cells now have proxy counts; the 20-unit rate floor leaves 3,241 finite
eviction indices at each date.

The screen now covers only April 2, 2023–April 1, 2025 and April 2, 2024–April 1,
2026 respectively. Ambiguous cases, including missing/conflicting dates, remain flagged and
unassigned. Candidate cells retain valid mapped counts and audit flags, without
suppression. Raw records, geocodes, units and spatial support are unchanged.
Entirely unlocated records remain separate court/window QA; they cannot
establish a City-only match-rate denominator.

On the final **same 2,490 cells**, recent mapped filings are **8,446/9,163**,
selected requests **15,977/17,240**, and recent 24-month demolition permits
**859/1,038**. The later previous-window filing count equals the earlier recent
count, 8,446. These are counts for the covered subset, not all Austin events or
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

The matrix integration audit checks current manifest hashes and retained
as-run preservation evidence. Decision 0014 authorizes replacing the annual
eviction panel as well as current/paired derived results; unrelated source
and output files remain protected. Original dated hashes are not rewritten.

Primary artifacts under output/part2/matrix/:

- part2_features_paired.rds/.csv: full grid/date table.
- part2_eligibility_by_hex.rds/.csv: all exclusion and availability flags.
- part2_analysis_matrix_2025-04-01.rds/.csv and 2026-04-01 counterparts.
- Exclusion, county and index summaries; source registry and run manifest.

See [methods and reproduction](../methods/historical-feature-matrix.md),
[readiness](part2-historical-readiness-2026-09.md) and the
[completed corrected cluster comparison](part2-cluster-comparison-2026-09.md).
