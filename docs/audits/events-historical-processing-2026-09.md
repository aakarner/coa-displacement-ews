# Part 2 paired event processing audit

**Updated:** September 10, 2026 (eviction eligibility); 311/demolition remain
the September 8 reconstructions. **Comparison:** April 1, 2025 versus April 1,
2026, reconstructed retrospectively on fixed geography. **Result:** corrected
v2 311, demolition and eviction indices are processed and scored from the
existing local sources. No new acquisition or geocoding was needed. At the
user's direction, these outputs replace the provisional run at the same paths;
the prior run is not archived.

## Results

All outputs retain the 7,027-cell computational grid. All three streams use the
same fixed 6,060 city-center cells within that grid, with additional historical
coverage screens. Counts below are for usable mapped-count cells, which may
include cells with fewer than 20 units; the rate-based indices additionally
require the unit floor. These are not counts for every part of Austin.

| Diagnostic | April 2025 | April 2026 |
| --- | ---: | ---: |
| 311 cells with usable mapped counts/geographic coverage | 6,020 | 6,020 |
| 311 cells with a complete usable index | 3,267 | 3,267 |
| Selected 311 requests in recent 12-month window | 21,397 | 22,713 |
| Selected 311 requests in previous 12-month window | 24,017 | 21,397 |
| Usable 311 cells with zero recent mapped requests | 2,898 | 2,831 |
| 311 cells with a usable ≥20-unit rate | 3,267 | 3,267 |
| 311 cells with a usable signed rate change | 3,267 | 3,267 |
| 311 cells with defined legacy percentage change (diagnostic only) | 3,233 | 3,122 |
| Demolition cells with a usable index | 6,033 | 6,033 |
| Residential-demolition permits in recent 24-month window | 1,224 | 1,450 |
| Residential-demolition permits in previous 24-month window | 1,870 | 1,434 |
| Usable demolition cells with zero recent permits | 5,464 | 5,437 |
| Eviction cells passing source coverage before ambiguity screening | 5,977 | 5,977 |
| Eviction cells with usable mapped counts after ambiguity screening | 5,845 | 5,854 |
| Eviction cells with a complete usable index | 3,123 | 3,135 |
| Mapped filings in recent 12-month window | 8,406 | 8,920 |
| Mapped filings in previous 12-month window | 7,433 | 8,357 |
| Usable eviction cells with zero recent mapped filings | 4,651 | 4,658 |

The shared April 2, 2024–April 1, 2025 311 window has exactly the same counts
when treated as the earlier snapshot's recent period and the later snapshot's
previous period. This checks the one-year shift directly.

These are **feature-specific coverage counts**, not the final seven-feature
cluster sample. The separate common-sample assembly intersects residential
eligibility, ownership support, source coverage and every required component
at both cutoffs. In particular, the reduction from 6,020 to 3,267 usable 311
indices reflects the complete fixed recipe and its unit floor, not missing
newly acquired requests or a change to source coverage.

## Coverage and interpretation

All streams use fixed current FULL-purpose city membership, at cell-center
and event-point levels. The 311 and demolition stages replay historical
jurisdiction evidence across the entire comparison intervals. The April 29,
2026 boundary is deliberately held fixed for this retrospective exercise;
it is not claimed to be the exact boundary known at either cutoff.

- Demolitions retain the existing source-coverage rule for continuously
  resolved FULL/LTD/2MILE jurisdiction. Twenty-seven otherwise city-center
  cells fail this screen with ETJ evidence and remain `NA`.
- 311 uses a conservative FULL-only geographic assumption. Forty city-center
  cells fail it: 27 ETJ, 12 LTD and one 2MILE. They are **not** labeled as proven
  absences of 311 service. Query completeness and verified coverage of all
  requests are separately represented; the latter remains false.
- Eviction coverage and ambiguity are now checked only over the two scored
  years, not history since January 2022. Historical totals remain diagnostics.
  Missing/conflicting dates potentially in-window still mask cells. Source coverage
  includes the supplied Travis JP1–5 and Williamson JP1/JP2 evidence; Hays,
  unsupported court geography and unresolved coverage remain unknown.
  Localizable uncertainty masks 132/123 cells at the two cutoffs. Entirely
  unlocated cases remain in court/window QA rather than suppressing all cells
  in a court. This is a mapped-filing proxy, not complete filing incidence.
- Another 967 grid cells lie outside the fixed city-center footprint and have
  unavailable indices. Observed mapped counts remain in audit fields, rather
  than being confused with usable modeled counts.
- There are 909 boundary-straddling city-center cells. Full-hex areas and fixed
  promoted-unit denominators are retained, so edge-cell rates/densities are
  hex-scale approximations, not exact city-clipped rates.

The existing three Code Officer intake descriptions are retained, including
department-name transitions. The narrower structure-condition linkage series
is not substituted. A 311 zero means no qualifying mapped requests observed
in the pinned extract, not no complaints, violations or displacement.
Demolition counts describe issued permits, not confirmed completed demolition.

## Source and location QA

The 311 cache has 148,469 unique requests, no duplicate IDs, no invalid event
dates and no ambiguous hex assignments. Fifty records have unusable
coordinates, including 14/15 in the two recent windows. Unknown locations are
not asserted to lie outside Austin. Complete pagination applies only to the
coordinate-required source query; omissions from that query are not measured.

Recent 311 counts progress from 21,547/22,863 mapped to the grid, to
21,439/22,751 with points inside the exact city boundary, to 21,397/22,713 after
the study-cell and historical geography screens. Before those footprint filters,
both later snapshot windows reproduce the canonical Part 1 counts in every
hex. Differences are therefore attributable to explicit footprint/coverage
rules, not an unnoticed change in the intake definition or date windows.

The demolition extract has 14,124 rows, of which 12,139 are unique residential
demolition permits. There are no duplicate IDs, selection-rule disagreements,
invalid coordinates, missing descriptions or ambiguous hex assignments in
this source. Exact-window event classification reconciles all source permits
to their mapped or excluded dispositions.

The later demolition windows have 1,477 previous and 1,484 recent permits
mapped to the raw grid, exactly matching the existing Part 1 raw counts.
The previous usable count becomes 1,434 after excluding 40 grid-mapped points
outside the exact city, one point in a non-selected center cell and two permits
in unsupported historical coverage. The recent usable count becomes 1,450
after excluding 32 outside-city points and two in non-selected center cells.

Evictions reuse the existing prepared Travis and Williamson filing records
and reviewed geocode registries. The case resolver still deduplicates county/
court-namespaced filings, checks consistent dates and locations, and retains
unresolved evidence rather than choosing an arbitrary row. Both dates use
inclusive April 2–April 1 recent and previous 12-month windows. Historical
totals, percentage changes and expanding-history recent shares remain raw
diagnostics; they no longer determine the eviction index recipe. See the
[eviction and feature-matrix methods](../methods/historical-feature-matrix.md)
for the narrowed screening window and unchanged location-quality contract.

## Scoring and validation

Every corrected index uses a complete fixed recipe, with equal weights and no
available-component averaging:

- **311:** recent selected requests per 100 fixed units, recent request density
  per square kilometer, and signed recent-minus-previous request-rate change
  per 100 of the same units, each weighted one third. Rate and density retain
  two thirds of the weight on current activity; they are not independent
  signals. Both rate components require at least 20 fixed units.
- **Evictions:** recent mapped filings per 100 fixed units and signed
  recent-minus-previous filing-rate change per 100 of those same units, each
  weighted one half. Both require at least 20 fixed units. Percentage change
  and expanding-history recent share are excluded from the index.
- **Demolitions:** recent residential-demolition permit density,
  `max(log1p(recent permits) - log1p(previous permits), 0)`, and recent
  total-demolition-description permit density, each weighted one third.
  The third component is a residential-permit subset, not all demolition
  activity. Both rolling windows must pass coverage checks.

All unchanged component bounds retain the previously reviewed 2025 1st/99th
percentiles, using R quantile type 7. They were loaded before overwriting the
saved scaling and verified unchanged. The new signed rate differences use
`B = q99(abs(2025 rate change))`, clip at `[-B, B]`, and map linearly to 0–100.
The observed bounds are **28.75** selected requests and **9.811020** mapped
filings per 100 units, respectively. Later observations use the same bounds.

Zero signed change scores **50**, including a flagged degenerate `B=0` case;
unknown inputs remain `NA`. Thus 0→0, 0→1 and 1→0 event counts all have defined
rate changes when coverage and units are valid. A missing required term makes
the entire index `NA`; every included row has the same terms and weights.
Unchanged components retain their zero-score policy for a degenerate range.
With the observed zero lower bounds, zero events and zero change yield an
eviction index of 25 or a 311 index of 16.67, not zero. These are relative
composites, not displacement probabilities or literal no-risk-zero measures.

Scaling objects use `part2-event-scaling-v2` or `part2-index-scaling-v2` and
`fixed_components_v2`. Features explicitly record required/available term
counts and completeness. Snapshot summaries distinguish
`usable_rate_change_hexes` from `usable_legacy_percent_change_hexes`; the
legacy percentage-change field is still undefined at zero previous counts
but does not affect the corrected index.

The checks pass:

- Synthetic tests for inclusive date endpoints, duplicate conflicts, spatial
  ambiguities, coverage transitions, valid zero versus unknown, denominator
  thresholds, zero-safe signed changes, neutral 50, strict completeness and
  preserved/frozen scoring.
- Independent 311 output audit: **38** manifest checksum checks plus spatial,
  date, raw-count, denominator, missingness and scoring checks.
- Independent demolition output audit: **47** manifest checksum checks plus
  permit reconciliation, windows, coverage, counts, bounds and changes.
- Independent eviction output audit: **67** manifest checksum entries plus
  independently reproduced counts/rates, exact windows, source coverage,
  ambiguity masks, fixed support, signed bounds and full-recipe scoring.
- Cross-stream audit: **85** manifest checksum checks, compatible canonical
  integer hex IDs, equal city footprints/areas, independently recomputed
  components and index changes, and **298 protected preexisting output files
  unchanged**. The historical audit baseline's superseded Part 2 files are
  excluded from that preservation check because their replacement was
  explicitly authorized; original Part 1/Part 3 files remain protected.

Before/after checks verified identical raw measures, geography, source coverage,
unit support and unchanged component bounds. Full local-only 311 and eviction
reruns also reproduced the corrected v2 feature and scaling objects exactly.
These domain builders do not fit clusters or ML models and do not overwrite
current Part 1 or Part 3 outputs. The downstream corrected cluster comparison
is a separate stage.

## Artifacts and downstream work

See [historical event methods](../methods/historical-events.md) for commands,
exact windows, coverage assumptions and artifact definitions. The main paired
tables are `output/part2/311/311_features_paired.rds` and
`output/part2/demolitions/demolition_features_paired.rds`, plus
`output/part2/evictions/eviction_features_paired.rds`; each folder also
contains its `.csv`, changes, coverage/event audits, frozen scaling and manifest.

The three manifest statuses are `paired_311_features_complete_v2`,
`paired_demolition_features_complete_v2` and
`paired_eviction_features_complete_v2`. All input, code and output pins verify.
No raw-source acquisition facts or raw-source checksums changed during this
measurement correction; processing-code and output checksums were refreshed.

The [readiness table](part2-historical-readiness-2026-09.md),
[paired-matrix audit](part2-paired-matrix-processing-2026-09.md) and
[cluster audit](part2-cluster-comparison-2026-09.md) document the completed
seven-feature assembly and subsequent comparison separately. Their common
sample is not the same as any one event stream's coverage count. ML remains
paused; the canonical Part 1 benchmark remains separate.
