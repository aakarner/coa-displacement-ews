# Part 2 rent-series tract-fallback sensitivity

**Run date:** September 8, 2026. **Status:** test findings retained as a dated
method note; the hierarchy is now integrated into the corrected Part 2 build.
The superseded experimental runner and duplicated generated outputs have been
retired. Current results are in the regenerated ACS and cluster audits.
Related: [issue #14](https://github.com/aakarner/coa-displacement-ews/issues/14).

## Question and result

Can reliable tract rent histories preserve rent growth and acceleration where
the assigned block-group histories fail the current uncertainty screen?

Yes. Within the **then-current 2,898-cell** diagnostic cohort, a geographic
level held fixed across both snapshots supports **2,346 complete rent pairs
(81.0%)**, compared with **1,614 (55.7%)** under the existing source-selection
and reliability rules.

| Diagnostic, on the same 2,898 cells | Hexes |
| --- | ---: |
| Selected complete block-group histories | 1,441 |
| Selected complete tract fallback histories | 905 |
| Total supported by the hierarchy | 2,346 |
| Neither complete history passes | 552 |
| Newly supported relative to the existing reliable pairs | 734 |
| Previously reliable pairs lost when per-vintage mixing is disallowed | 2 |

The net gain is **732 cells**, or **25.3 percentage points**. The two losses are
not a contradiction: the existing pipeline can combine block-group and tract
estimates independently by vintage, whereas neither complete source series
passes for those two cells. The all-tract alternative, without preferring block
groups, supports 2,235 cells; the hierarchy preserves useful block-group detail
and also retains cells whose block-group series passes but whose tract series
does not.

These are rent-coverage results on a fixed diagnostic cohort, **not a newly
approved seven-index cluster sample**. Requirements in other domains remain
separate, including complete five-component vulnerability.

The supported cells contain **88.4% of the fixed residential units** in the
existing cohort. By county, support is 2,207 of 2,755 Travis cells and 139 of
143 Williamson cells; the remaining gaps are 548 and four cells, respectively.
Requiring all five vulnerability components at both dates as well retains
**2,300 cells** (out of the existing 2,898), before other measurement changes.

## Exact test

1. Reuse the twelve cached block-group/tract rent extracts and the existing
   per-vintage dominant-source assignments. No acquisition or spatial allocation
   is rerun. The assigned medians and their MOEs come from the same source.
2. For each hex, evaluate all six required releases: 2013, 2014, 2018, 2019,
   2023, and 2024. An estimate passes when it is positive and finite, its MOE is
   finite and nonnegative, and MOE divided by estimate is at most 0.30. This is
   the project's existing input-precision screen, not a significance test for
   rent growth or acceleration.
3. Select block-group data if **all six** pass. Otherwise select tract data if
   **all six** pass. Otherwise select neither. Never mix geographic levels
   within a trend or between the two snapshots.
4. Recalculate current rent, recent annualized growth, and acceleration in
   common 2024 dollars. April 2025 uses 2013/2018/2023; April 2026 uses
   2014/2019/2024. Growth is `100 * log(new real rent / old real rent) / 5`;
   acceleration is recent growth minus the preceding five-year growth rate.
5. For diagnostic score comparisons only, reuse the existing earlier-snapshot
   component bounds and average exactly three component scores with equal
   weights. Unsupported cells have three missing components and a missing
   index at both dates. There is no level-only fallback or available-component
   reweighting in the alternative.

The source choice uses reliability information from both snapshots. That is
appropriate for this retrospective comparison, but is not an operational rule
that claims to have known the later release when constructing the earlier
snapshot.

## Geography: what this fixes and what it does not

Of the 905 tract-fallback cells, **none changes its assigned tract GEOID between
corresponding snapshot roles**: 2013 to 2014, 2018 to 2019, or 2023 to 2024.
Among the 1,441 selected block-group cells, one changes current-role GEOID
(2023 to 2024); the earlier and middle role pairs do not change IDs.

However, **477 tract-fallback cells** use more than one tract GEOID *within each
long historical profile*. Across both geographic levels this affects 1,564
selected cells. These historical source changes must not be described as
477 cells switching tracts during the one-year snapshot refresh.

Holding the geographic level fixed does **not** harmonize historical Census
boundaries. An unchanged GEOID also does not prove an unchanged polygon or
statistical population. The test retains the existing vintage-specific spatial
assignments and records source IDs; boundary harmonization is not claimed.
Tract fallback trades finer spatial detail for usable temporal evidence. Hexes
sharing a tract do not thereby have independently measured hex-level rents.

## Effects on the rent measures

On the **same 2,346 supported cells**, the middle 90% of annual snapshot changes
in the diagnostic rent index spans **-16.8 to +15.4 points** under the fixed
three-component hierarchy, versus **-24.5 to +23.9** under the existing index.
The median change is approximately -1 point under either rule. Less extreme
movement is not itself a validation criterion: tract smoothing and genuine
differences in the source estimates also contribute.

For the 1,612 cells supported by both the legacy reliability rule and the new
hierarchy, old/new index rank correlations are **0.967 in 2025** and **0.981 in
2026**. Changes are substantially larger among newly supported cells, where the
comparison can replace an incomplete old composite with a full three-component
one. The audit keeps these groups separate.

The previously discussed **hex 378** illustrates the mechanism. Its old index
fell from 43.0 to 19.2 while contracting from three components to rent level
alone. The alternative uses tract data for all six releases and retains all
three components at both dates: its diagnostic score moves from **35.3 to
25.9**. Current real tract rent moves from $1,438 to $1,425, recent annualized
real growth from -1.09% to -2.68%, and acceleration from -2.04 to -6.03 percentage
points. The remaining score change follows those measured inputs, not the
disappearance of two index terms; it is not proof of a statistically significant
annual change in underlying rents.

Among the 552 unsupported cells, the tract candidates have an invalid or
missing positive rent estimate in at least one vintage for 96 cells, a missing
or invalid MOE for 210, and a relative MOE exceeding 0.30 for 467. These groups
overlap. Thus the tract fallback recovers substantial information without
waiving the reliability screen for the remaining cases.

## Interpretation and implementation decision

The coverage gain supported retaining rent change and adopting this hierarchy
as part of the measurement correction. The 552 unsupported cells in this
diagnostic cohort stay
explicitly unsupported in this three-component specification. A uniform
level-plus-growth alternative could be a separate test if excluding those
cells proves too costly; it must not be substituted selectively by hex.

The rent-only test did not correct eviction/311 formulas or validate the
earlier cluster-switching and qualitative-risk results. The subsequent
integrated rebuild includes those measurement corrections. The subsequent
scored-window eviction screen yields a 2,351-cell cluster comparison; see the
[corrected results](part2-cluster-comparison-2026-09.md).
Part 3 ML remains paused.

## Current implementation and validation

Run from the repository root:

```sh
Rscript tests/test_part2_rent_fallback.R
Rscript scripts/part2/build_acs_snapshots.R --assemble-only
Rscript tests/test_part2_rent_fallback_outputs.R
```

Current artifacts live in `output/part2/acs/`: `acs_rent_source_candidates.rds`,
`acs_rent_source_selection.rds`, `acs_rent_fixed_series_features.rds`, the paired
features, date-specific vintages/trends, and the version-2 run manifest. The
independent fallback-output test now validates these promoted products directly
from the original twelve Census extracts, not the retired experiment directory.

The helper's synthetic tests cover reliability thresholds, missing and invalid
inputs, incomplete histories, source consistency, inflation adjustment and
trend arithmetic. The independent output audit reconstructs the hierarchy
from raw caches, checks component/score calculations and coverage, and verifies
input and output preservation. The initial sensitivity verified **1,143 existing analytical
outputs unchanged**, including the earlier ACS, matrix, and cluster artifacts.
The initial synthetic suite passed, and the independent output audit passed **244
assertions** across seven test blocks, including county-level count conservation
and hashes for 20 inputs, two implementation files, and 14 audit artifacts.
