# Paired amenity processing: September 2026

- **Executed:** September 8, 2026
- **Status:** Completed for the agreed retrospective Part 2 proof of concept
- **Scope:** Amenity event reconstruction, geocoding and paired hex features only
- **Methods:** [Historical amenities](../methods/historical-amenities.md)

## Result

Comparable amenity features now exist for April 1, 2025 and April 1, 2026 on
the same 7,027-cell grid. The earlier cutoff's six component scoring bounds are
held fixed for the later cutoff. No cluster model, complete feature-readiness
table or ML model was built in this step. Existing Part 1 outputs were not
overwritten.

| Measure | April 1, 2025 | April 1, 2026 |
| --- | ---: | ---: |
| Eligible opening events after exact-alias deduplication | 1,340 | 1,447 |
| Openings in previous 18-month window | 656 | 696 |
| Openings in recent 18-month window | 684 | 751 |
| Geocoded opening events | 1,338 | 1,444 |
| Geocoding match rate | 99.85% | 99.79% |
| Matched by Census | 1,158 | 1,239 |
| Matched by ArcGIS | 180 | 205 |
| Unmatched opening events | 2 | 3 |
| Events contributing positive exposure to at least one hex | 727 | 784 |
| Events directly inside the computational hex grid | 684 | 743 |
| Hexes with nonzero recent exposure | 2,067 | 2,041 |
| Complete hex feature rows | 7,027 | 7,027 |
| Exact duplicate-ID aliases removed | 14 | 16 |

Event and geocoding totals refer to eligible candidates in the **three counties**,
not to businesses inside Austin alone. Only events within 800 meters of a hex
centroid contribute to its exposure. The 800-meter neighborhood can include
businesses outside city limits. Direct grid containment also should not be
interpreted as an exact city-boundary test.

Three distinct source addresses remain unmatched across the two vintages:
Dave and Buster's (source city Austin), Whiskey Ridge (Driftwood), and Taco Clem
1626 LLC (Kyle, later vintage only). ArcGIS returned scores of 84, below the
unchanged acceptance threshold of 90. Their point locations and spatial scope
remain unknown; they were not treated as confirmed outside the study area.

On the fixed grid, the amenity index increases in 1,218 cells, decreases in
1,148, and is unchanged in 4,661 (numerical tolerance 1e-9). Its between-vintage
Spearman correlation is approximately 0.672. These are feature changes only,
not cluster movement, displacement measurements or causal findings. Many cells
have no eligible opening nearby in either window.

## Source handling

The February 8, 2025 archived full export matches its previously recorded
SHA-256 checksum, all 300,879,159 bytes, 1,375,129 data rows and 33 columns. Its
normalized three-county/core-category subset has 5,808 unique IDs. The new live
extract has 6,417 unique IDs, with source modification September 7, 2026.

The unrestricted source reconciliation has 5,304 shared IDs, 504 archive-only
IDs and 1,113 live-only IDs. The single ledger retains coherent archive records
for shared IDs and supplements live-only IDs. Across all retained records,
960 shared IDs have 1,045 field differences; these are exported rather than
silently applied to historical events. Source absence is not interpreted as
closure.

Among the union of eligible, deduplicated events in the two windows, four shared
IDs have changed first-sale dates, four changed address numbers, and five changed
ZIPs. The four date changes are all within the same assigned windows, so they
do not change window membership here. Other changes include business names,
street text, permit dates and closures. Retaining archive values avoids treating
later moves or renaming as a relocation of the original opening. It also means
later corrections are not automatically preferred.

The 2025 reconstruction retains eight eligible IDs absent from the unrestricted
live source, plus 70 live-only supplemental events after alias deduplication.
The 2026 window has no eligible archive-only IDs and 574 live-only supplemental
events. Exact same-opening taxpayer aliases are counted once in both vintages.

## Correction to the earlier feasibility tallies

The [coverage audit](amenity-historical-coverage-2026-09.md) reported 1,292
archive-eligible events and 461 calendar-2024 events. Those numbers are reproduced
exactly by omitting only the existing home-business exclusion, as are all nine
of its calendar-2024 county/category counts. The implemented shared production
classifier applies that exclusion: **1,278** archive-eligible events in the
earlier 36-month window and **459** in calendar 2024, before alias deduplication.
The original feasibility code is not preserved, so this is an exact numerical
reconciliation of the published totals rather than a direct inspection of that
earlier implementation.

The earlier phrase “archive-only” also referred to independently eligible ID
sets, not necessarily IDs absent from an unrestricted current download. Under
production eligibility, 69 IDs are archive-only eligible; 61 of them still exist
in the live file (60 now have permit dates after the cutoff, and one fails the
cafe-name rule), leaving eight physically absent IDs. These distinctions explain
the apparent discrepancy without attributing it to unverified source churn.

Final baseline arithmetic is 1,278 archive-eligible records plus 76 eligible
live-only additions, minus 14 exact duplicate aliases, yielding 1,340 openings.

## Validation and limitations

- Shared classification exactly reproduced the saved Part 1 fields for all
  7,883 source candidates before Part 2-specific reconciliation and alias rules.
- Refactored scoring reproduced all existing Part 1 category/index scores on
  7,027 cells; the original Part 1 artifacts were not rewritten.
- Synthetic classification tests cover window boundaries, permit dates,
  retained closed openings, category exclusions, source/ID conflicts and aliases,
  including missing fingerprint fields and different units, names or dates.
- Scoring tests cover frozen bounds, zero-range behavior, coverage contracts and
  unassessed corroboration. All six actual baseline ranges are nondegenerate.
- The completed-output audit passed 20 SHA-256 checks, full-grid alignment,
  paired output equality, score recomputation, exact frozen-reference equality,
  date eligibility, unique opening IDs and correct differences. Unmatched
  spatial flags and unassessed corroboration remain unknown.
- Pinned raw source reuse succeeded offline; the target graph and runner parse,
  and `git diff --check` passes.

The reconstruction is usable for the agreed proof of concept, but exhaustive
source completeness is not established (`amenity_window_complete = NA`).
Mixed-beverage and food-inspection corroboration was not rebuilt because it
does not affect this opening index. The next overall workflow step remains
assembling the other paired features and the full readiness table; this run
does not establish their readiness.

## Artifacts

- `output/part2/amenities/amenity_features_paired.rds` and `.csv`
- `output/part2/amenities/amenity_feature_changes_by_hex.csv`
- `output/part2/amenities/amenity_snapshot_summary.csv`
- `output/part2/amenities/amenity_paired_spatial_qa.csv`
- `output/part2/amenities/amenity_run_manifest.json`
- Date-specific candidates, event/alias/geocoding/source QA and scoring bounds
- Tracked `config/amenity_historical_sources.json` and reproducible source/run scripts

Large raw snapshots and generated artifacts are ignored by Git. Preserve them
in project data backups; the source registry alone cannot recreate a mutable
live extract byte-for-byte after it changes.
