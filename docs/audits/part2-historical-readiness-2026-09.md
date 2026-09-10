# Part 2 historical comparison: current readiness

- **Updated:** September 10, 2026, after scored-window eviction screening and regeneration.
- **Earlier/later cutoff:** April 1, 2025 / April 1, 2026.
- **Design:** Retrospective reconstruction, advancing source vintages one year.
  Adjacent ACS five-year releases overlap; they are not independent annual surveys.
- **Scope:** Seven cluster inputs and historical cluster comparison. Part 3 ML paused.

## Bottom line

All seven inputs, the common matrix and the corrected cluster comparison are
complete. **2,351 cells (2,210 Travis, 141 Williamson)** have all required
components at both dates under fixed recipes, geography level and weights.
**35.0% change fixed clusters; fixed/refitted 2026 assignments agree 95.5%.**
No included cell changes component availability.

Readiness does not imply complete Austin coverage or an operational predictive
model. Missing courts, uncertain locations, ownership gaps and unreliable ACS
remain excluded rather than zero-filled. See the
[common-sample audit](part2-paired-matrix-processing-2026-09.md) and
[corrected results](part2-cluster-comparison-2026-09.md).
Failed Part2 results were overwritten in place, not archived.

## Seven-feature readiness table

Availability is per stream on the 7,027-cell audit grid, before intersecting
all eligibility requirements. Generic Part1 outputs do not substitute for
these paired historical reconstructions.

| Feature | Earlier inputs | Later inputs | Corrected implementation and availability | Remaining limitation |
| --- | --- | --- | --- | --- |
| Amenity change | Two 18-month windows ending April 2025 | Windows shifted one year | Three complete category scores; 7,027 usable at both dates; earlier scoring frozen. Source processing finds 1,340/1,447 eligible three-county openings, 99.85%/99.79% geocoded. | Retrospective historical completeness is unverified; nearby exposure is not a unique event count. |
| Corporate ownership | 2024 Travis, Hays and Williamson ownership evidence | 2025 evidence | Three common-parcel components; 3,227 indices at both dates; at least 20 common units and 95% unit/parcel coverage. | Common support is retrospective; source-agreement and certified-only sensitivity flags remain. |
| Selected 311 pressure | Adjacent 12-month windows ending April 2025 | Windows shifted one year | Recent rate, density and signed rate change; all three required; 3,267 indices at both dates, within 6,020 geographically screened count-usable cells. | Coordinate-required selected Code Officer requests; historical FULL-only screen is conservative, not proof of complete requests. |
| Demolition pressure | Adjacent 24-month windows ending April 2025 | Windows shifted one year | Existing three-term recipe with full components required; 6,033 indices at both dates; dated jurisdiction coverage checked throughout windows. | Permits are not completed demolitions or displaced households; 27 fixed-city cells unsupported by the conservative coverage contract. |
| Eviction pressure | Adjacent 12-month windows ending April 2025 | Windows shifted one year | Recent filings/100 units plus signed rate difference; 3,123/3,135 indices, 3,083 both. Only the scored 24 months gate coverage/ambiguity. Expanding-history recent share and percentage change are not scored. | Mapped-filing proxy; Hays and Williamson JP3 records missing, plus localized ambiguity and entirely unlocated records. |
| Rent pressure (ACS) | 2013, 2018, 2023 releases | 2014, 2019, 2024 releases | Level, growth and acceleration all required. Reliable BG across all six vintages, else reliable tract across all six, else unavailable at both dates. 4,623 supported both: 3,018 BG and 1,605 tract. | Geographic level fixed, but historical boundaries not fully harmonized; retrospective selection uses both snapshots' reliability. |
| Demographic vulnerability (ACS) | 2019–2023 ACS, labeled 2023 | 2020–2024 ACS, labeled 2024 | All five components required; 4,164/4,217 indices, 4,150 both; fixed allocation, appropriate median fallback and common 2024 dollars. | Overlapping releases and sampling uncertainty; unavailable required components exclude the index. |

## Exact event windows

Both listed endpoints are included. Event windows shift, not merely the label
on an existing annual summary.

| Stream | April 2025 previous | April 2025 recent | April 2026 previous | April 2026 recent |
| --- | --- | --- | --- | --- |
| 311 and evictions | Apr 2, 2023–Apr 1, 2024 | Apr 2, 2024–Apr 1, 2025 | Apr 2, 2024–Apr 1, 2025 | Apr 2, 2025–Apr 1, 2026 |
| Demolitions | Apr 2, 2021–Apr 1, 2023 | Apr 2, 2023–Apr 1, 2025 | Apr 2, 2022–Apr 1, 2024 | Apr 2, 2024–Apr 1, 2026 |
| Amenities | Apr 2, 2022–Oct 1, 2023 | Oct 2, 2023–Apr 1, 2025 | Apr 2, 2023–Oct 1, 2024 | Oct 2, 2024–Apr 1, 2026 |

Eviction coverage/ambiguity is screened over each snapshot's two scored years:
April 2, 2023–April 1, 2025 and April 2, 2024–April 1, 2026. Older-only issues
do not remove a cell. Supplied court periods cover those windows; incomplete
calendar-year 2026 is not a missing April 1 comparison. The geographic gaps
are 54 Hays and 29 Williamson JP3 cells within the 6,060 fixed-city-center grid.

311 retains the existing three versioned Code Officer intake types, not the
newer linked structure-condition series whose sustained coverage starts inside
the earlier window. Raw-query completeness applies only to requests with
coordinates. Forty fixed-city cells fail the conservative historical FULL-only
assumption: 27 ETJ, 12 LTD and one 2MILE.

The demolition source's latest qualifying issue date is March 31, 2026; its
observed-through contract is April 1. Latest event date is not source coverage
end. Effective-dated coverage supports 6,033 current-city cells, including 65
that shift from LTD to FULL in 2022 under the permit source's contract.

## Shared measurement and analysis requirements

| Requirement | Implemented |
| --- | --- |
| Fixed study support | Same canonical hex grid, current FULL city-center mask, promoted units and areas; exact point-level city filters and edge flags. |
| Complete paired sample | Same 2,351 IDs at both dates; every required component present; all exclusions retained on the full audit grid. |
| Fixed scoring | Original earlier bounds for unchanged terms; symmetric earlier bounds for signed event-rate changes, with zero change at 50. No later rescaling or row-specific reweighting. |
| Comparable ACS | 2024 dollars; fixed allocation support; full-series reliable BG/tract choice for rent; all five vulnerability components. |
| Fixed baseline comparison | New retrospective 2025 k=7 baseline with earlier mean/SD standardization frozen for 2026. Part1 benchmark untouched. |
| Separate structural check | Independent 2026 refit, matched to coeval fixed-2026 assignments; 20 optimizer fits and 70 paired random/spatial holdouts. |
| Qualitative interpretation | Reviewed Low/Moderate/High/Very high profile mapping pinned to the corrected baseline hash; numeric IDs not treated as ordinal. |
| Preservation | Non-Part2 outputs/figures verified unchanged; failed Part2 products replaced in place. No ML modeling or commit/push. |

A zero-event/zero-change eviction composite can equal 25 because its signed
change component is neutral at 50. These are relative feature indices, not
displacement probabilities. Likewise, a cluster-tier change is an interpretable
qualitative transition, not an estimate of residents displaced.

## Next useful step

The scoped proof of concept is complete. Review representative large concern
transitions, especially eviction and demolition cases, against raw event counts
and locations before adopting operational labels or action thresholds. A new
data search, fixed-ACS branch or ML model is not required to do that.

Land-value pressure, transactions, CoStar rents, additional vulnerability
variants and the full landlord-mapper migration remain outside this scope.
Ownership already uses the pinned upstream classifier and reviewed 2024/2025
imports.

## Evidence and reproduction

- Ownership: [methods](../methods/historical-ownership.md) and
  [two-year Williamson integration](williamson-2024-ownership-integration-2026-09.md).
- Amenities: [methods](../methods/historical-amenities.md) and
  [processing audit](amenity-historical-processing-2026-09.md).
- ACS: [methods](../methods/historical-acs.md),
  [processing audit](acs-historical-processing-2026-09.md) and
  [integrated tract-fallback test findings](part2-rent-tract-fallback-2026-09.md).
- Events: [methods](../methods/historical-events.md) and
  [processing audit](events-historical-processing-2026-09.md).
- Evictions/matrix: [methods](../methods/historical-feature-matrix.md) and
  [common-sample audit](part2-paired-matrix-processing-2026-09.md).
- Clusters: [methods](../methods/historical-cluster-comparison.md) and
  [corrected results](part2-cluster-comparison-2026-09.md).

Reproducible code, raw caches, paired domain products and nested checksum
manifests remain in their existing project locations. Independent checks
validate full-component scoring, source selection, sample construction,
assignments, robustness and figures. The final matrix checks 35 manifest
entries and 960 preserved non-Part2 artifacts; the downstream cluster audit
checks 1,404 declarations including that preservation set.
