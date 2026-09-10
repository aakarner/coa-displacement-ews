# Part 2 paired ACS processing audit

**Run date:** September 8, 2026. **Status:** corrected v2 outputs, regenerated
in place. **Scope:** ACS rent pressure and demographic vulnerability only, for
the retrospective April 1, 2025 / April 1, 2026 comparison.

## Result

Both ACS indices are processed on the fixed 7,027-cell audit grid. Rent now
requires a reliable, consistently sourced six-vintage series; both indices
require every component before receiving a score. The corrected outputs replace
the earlier incomplete-component results, without archiving that failed run.

The eight missing raw extracts were acquired: block-group/tract rent for 2013,
2018 and 2023, plus 2023 block-group demographics and tract fallback medians.
Existing later-vintage caches were reused. No additional ACS acquisition is
required for this proof of concept.

| Diagnostic | April 2025 snapshot | April 2026 snapshot |
| --- | ---: | ---: |
| Current ACS release label | 2023 | 2024 |
| Rent release triplet | 2013 / 2018 / 2023 | 2014 / 2019 / 2024 |
| Audit-grid hexes | 7,027 | 7,027 |
| Complete rent series / three-component rent index | 4,623 | 4,623 |
| Selected block-group series | 3,018 | 3,018 |
| Selected tract fallback series | 1,605 | 1,605 |
| Neither source supports the complete rent series | 2,404 | 2,404 |
| Complete five-component vulnerability index | 4,164 | 4,217 |
| Both complete ACS indices | 3,197 | 3,241 |
| Median income available | 6,876 | 6,882 |
| Median income passes 30% relative-MOE screen | 3,058 | 3,005 |
| Income estimate present but its MOE missing | 303 | 465 |
| Hexes with unallocated count fields retained as NA | 2,029 | 2,029 |

Across both dates, **3,196 cells have both complete indices**—equivalently, all
eight ACS components. There are 4,150 cells with all five vulnerability inputs
at both dates. The component availability pattern is unchanged in 6,944 cells
and changes in 83. These counts describe the computational grid, **not the final eligible residential
or commonly covered seven-feature clustering sample**.

The rent index requires level, growth and acceleration; vulnerability requires
low income, renters, poverty, rent burden and low college attainment. Scores
use fixed equal weights and are `NA` if any required component is missing;
there is no averaging of whichever terms happen to be available. For example,
2,597/2,599 cells have only one observed vulnerability component, generally an
assigned income median. These cells now correctly have no vulnerability index.
Median assignment does not establish usable count allocation or percentage
denominators for the remaining components. Missing evidence is not zero risk.

## Fixed rent-source selection

For each hex, the processor evaluates both block-group and tract candidates
for all six releases: **2013, 2014, 2018, 2019, 2023 and 2024**. A candidate is
reliable only when its estimate is finite and positive, its MOE is finite and
nonnegative, and `MOE / estimate <= 0.30`.

1. Use block groups if all six block-group candidates pass.
2. Otherwise use tracts if all six tract candidates pass.
3. Otherwise leave all three rent components and the rent index missing at
   both dates.

The geographic level is fixed across both snapshots and their growth periods;
there is no per-vintage mixing or level-only fallback. Actual source GEOIDs,
estimates and MOEs are retained for each vintage. The rule uses both dates'
data, consistent with retrospective reconstruction—not an earlier-date-only
forecasting exercise. The earlier
[rent fallback audit](part2-rent-tract-fallback-2026-09.md) records the diagnostic
that motivated this rule; its historical test-cohort counts are not the current
full-grid availability figures above.

## Comparability and corrections

- Both dates use the same hexes, fixed current parcel support and 2020 block
  ancillary weights. Actual ACS count estimates still update by release.
- Income and rent, including dollar-valued MOEs, use a common **2024-dollar**
  base. Recent and previous rent growth intervals are exactly five years.
- The original eight earlier-snapshot 1st/99th-percentile scoring bounds are
  preserved verbatim and applied to both dates. They were **not refitted** on
  the corrected source-selection cohort, isolating the effects of the recipe
  correction from a change in normalization.
- Missing counts remain `NA`, not zero. A defensive paired allocator also
  withholds partially observed positive-weight contributions, but no such
  missing source-count or count-MOE contributions occurred in this run.
- Estimates and MOEs use the same selected block-group or tract source.
  Previously a missing block-group MOE could be replaced by a tract MOE even
  while retaining the block-group estimate. The correction changes uncertainty
  fields, not the demographic median estimates. Rent now applies the complete
  six-vintage rule above, so selecting a tract series can change rent estimates
  as well as uncertainty fields.

The demographic helper likewise stops borrowing inappropriate MOEs for 417
later income, 37 rent and 43 home-value hexes. These demographic-product median
estimates are unchanged. Income reliability is a reported diagnostic, not a
new vulnerability scoring restriction.

The fixed block/parcel allocation support does not harmonize historical ACS
source boundaries. A shared geographic level—or an unchanged GEOID—does not
guarantee an unchanged source polygon. Many hexes can share the same tract rent
series; they are not independent rent observations.

Adjacent ACS releases overlap in four of five survey years. This is the agreed
retrospective data-refresh comparison; no claim of independent annual change
or statistical significance is made. See [methods](../methods/historical-acs.md)
for the exact source, dollar and component definitions, including retained
legacy education categories.

## Validation and preservation

- Demographic source coverage is 1,206 block groups (Hays 116, Travis 766,
  Williamson 324) and 471 tracts in each current release.
- The older 2013/2018 rent releases each contain 903 block groups and 332 tracts;
  2023/2024 each contain 1,206 block groups and 471 tracts. Every rent-vintage
  crosswalk supplies a source assignment for all 7,027 hexes, which does not
  imply that every assigned source publishes a usable median or MOE.
- All 24 demographic count allocations conserve expected in-grid totals to
  floating-point precision: maximum absolute differences below `4e-9`.
  Percentage bounds pass with floating-point tolerance.
- Synthetic tests cover coherent estimate/MOE provenance, partial missing
  counts, valid zero counts, exact 30% reliability boundaries, missing vintages
  and MOEs, duplicate candidate rejection, fixed source selection, frozen
  normalization, common dollars, and strict three-/five-component scoring.
- The paired integration audit passes **84 SHA-256 checks**, confirms the grid,
  cutoffs and dollar bases, independently recomputes both scores, checks change
  flags and verifies allocation conservation. A separate promoted-rent test
  independently reconstructs candidates from the **12 raw rent extracts** and
  existing dominant-source crosswalks, then verifies selection, provenance,
  inflation conversion, growth, acceleration and unsupported-series `NA`s.
- All four focused tests pass: `test_acs_snapshot_scoring.R`,
  `test_acs_snapshot_outputs.R`, `test_part2_rent_fallback.R` and
  `test_part2_rent_fallback_outputs.R`.
- Before/after checksums confirm all **37 protected inputs and canonical
  outputs** are unchanged, including the 16 preexisting generic ACS/support
  files, raw caches and source crosswalks. The earlier shared estimate/MOE
  correction can affect an intentional future Part 1 rebuild; the fixed-series
  selection and strict paired scoring are Part 2-specific. This run did not
  replace the Part 1 benchmark outputs.

Reproduction commands and artifact definitions are in the
[historical ACS methods](../methods/historical-acs.md). The principal artifacts
are under `output/part2/acs/`:

- `acs_features_paired.rds`, `acs_feature_changes_by_hex.csv` and
  `acs_snapshot_summary.csv` provide the scored pair and availability audit.
- `acs_rent_source_candidates.rds` retains all 84,324 source candidates;
  `acs_rent_source_selection.rds` records the 7,027 fixed source decisions;
  `acs_rent_fixed_series_features.rds` contains the 14,054 hex/date rent rows.
- Date-specific `acs_rent_by_hex_vintage.rds` and
  `acs_rent_trends_by_hex.rds` have been regenerated from those selections,
  including estimate/MOE provenance. Old trend files are not used as inputs.
- `acs_scaling.rds` preserves the earlier bounds, and `acs_run_manifest.json`
  records schema version 2 and status `paired_acs_features_complete_v2`.

`Rscript scripts/part2/build_acs_snapshots.R --assemble-only` reproduces this
cached rebuild using the existing demographic products, raw rent extracts,
crosswalks and normalization reference; it does not download data.

Raw data and generated outputs follow the repository's existing local/ignored
storage policy; scripts, tests and documentation preserve the recipe.

## What remains

ACS processing is complete for the agreed retrospective pair. These corrected
indices feed the common seven-feature matrix and historical cluster analysis;
their full-grid availability does not determine that analysis's final eligible
sample. The [readiness table](part2-historical-readiness-2026-09.md) tracks the
other streams and shared eligibility rules. This ACS rebuild does not resume
the paused ML work.
