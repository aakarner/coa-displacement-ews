# Part 2 historical cluster comparison: corrected results

- **Regenerated:** September 10, 2026, after narrowing eviction eligibility to
  each snapshot's two scored annual windows.
- **Pair:** April 1, 2025 / April 1, 2026, retrospectively reconstructed.
- **Common sample:** 2,351 cells: 2,210 Travis and 141 Williamson.
- **Model:** Seven complete equally standardized indices; k=7; 2025 scaling
  frozen for later observations. Measurement: part2-fixed-components-v2.
- **Status:** Features, models, interpretation, robustness and figures replaced
  in place. Part 1 separately refreshed; Part 3 ML remains paused.

## Main finding

**The narrower eviction screen expands coverage without materially changing
the broad historical findings.**

| Question | Current result | Interpretation |
| --- | --- | --- |
| Do cells move with 2025 definitions fixed? | 822 of 2,351 move (35.0%); 1,529 stay (65.0%). | Updated features cross fixed cluster boundaries. |
| Do fixed/refitted 2026 classifications agree? | 2,245 agree (95.5%); 106 differ (4.5%), after coeval label matching. | Most later classifications survive the separate refit. |
| Do concern tiers rise or fall? | 335 rise (14.2%), 262 fall (11.1%), 1,754 retain their tier (74.6%). | Modest upward tilt under the reviewed interpretation. |

The preceding 2,300-cell run had 34.9% movement and 95.5% fixed/refitted
agreement. All its eligible cells remain; 51 are added. Raw window counts and
geocodes are unchanged. Gains arise from older-only ambiguities no longer
invalidating a scored window; actual source-court coverage is unchanged.
See output/part1/eviction_window_change_summary.csv for the decision audit.

These transitions are **not displacement of 35% of places or people**.
Movers contain approximately 118,369 fixed estimated residential units, 32.5%
of common-sample units; these are not displaced units. Temporal fixed ARI is
0.391; fixed/refitted 2026 ARI is 0.897. The combined baseline-to-later-refit
comparison has 788 movers and ARI 0.411; it mixes temporal and definition change.

## Measurement and eligibility

All seven indices retain complete fixed recipes; zero included cells change
component availability. Rent uses one reliable BG history across all six ACS
vintages or one tract history: 1,456 included cells use BGs and 895 tracts.
Ownership retains the common-parcel 20-unit/95% screen. Other domain rules
are unchanged.

Evictions retain recent filings per 100 fixed units and signed rate change,
equal halves. Coverage and potentially localizable ambiguity checks now span
April 2, 2023–April 1, 2025 and April 2, 2024–April 1, 2026 respectively.
Missing/conflicting dates potentially in-window still mask candidate cells;
older-only issues do not. History totals/recent shares remain unscored
diagnostics. No court records or geocodes were added.

Part 2 retains reviewed rate bounds; its signed-change bound is recalculated
from expanded 2025 domain support (±9.811020 per 100 units), then frozen for
2026. Part 1 recalibrates on 2026 support. Before/after masks and unchanged raw
counts isolate the eligibility gain; model refits also refresh sample-based
standardization and cluster boundaries.

## Profiles and qualitative concern

New C1–C7 IDs are nominal and differ from the preceding fit. Labels were
reviewed against baseline centroids/raw events and hash-pinned before computing
tier transitions. They are not calibrated probabilities.

| ID | Reviewed profile | Concern | 2025 | 2026 fixed | 2026 aligned refit |
| --- | --- | --- | ---: | ---: | ---: |
| C1 | Eviction-filing concentration | Very high | 86 | 75 | 72 |
| C2 | Corporate ownership and vulnerability | Moderate | 290 | 272 | 276 |
| C3 | Lower measured pressure | Low | 820 | 780 | 805 |
| C4 | Higher/rising rents, lower vulnerability | Low | 559 | 543 | 585 |
| C5 | Demolition-permit concentration | High | 236 | 282 | 268 |
| C6 | Selected 311 activity and vulnerability | Moderate | 191 | 246 | 215 |
| C7 | Nearby amenity activity | Moderate | 169 | 153 | 130 |

Every baseline C1 cell has a recent mapped filing; its mean rate is 15.4 per
100 units, versus at most 1.8 in other profiles. Every baseline C5 cell has
a recent residential demolition permit. Low concern does not mean no vulnerable
residents or displacement. Filings/permits do not establish completed events.

Of 822 switchers, 225 change profile within the same tier, 354 move one tier,
and 243 move two or more tiers. These are ordinal steps, not equal amounts of
risk. On the same common sample, recent mapped filings total 5,708/6,059;
the shared-window counts reconcile exactly.

## Robustness and limits

Twenty 100-start fits at each date check optimization. Conditional holdouts
are rerun with frozen component scoring and training-sample cluster scaling:

| Holdout design | Replicates per date | Median held-out ARI 2025 | Median held-out ARI 2026 |
| --- | ---: | ---: | ---: |
| Random 20% of cells | 50 | 0.981 | 0.971 |
| Whole H3 resolution-7 regions | 20 | 0.959 | 0.932 |

Spatial 10th-percentile ARIs are 0.792/0.873. Unevaluated recovery stays
missing. Same-model subsets with at least 50/100 units have movement of
34.9%/34.7%, versus 35.0% overall; these are not threshold-specific refits or
proof against small-count/geocoding error.

Limitations remain: retrospective paired support, overlapping ACS releases,
uncovered Hays/Williamson JP3 filings, coordinate-required 311 extraction,
uncertain amenity completeness, and 156 boundary-straddling included cells
with whole-hex denominators. This is not a complete Austin census or forecast.

## Verification and artifacts

Independent tests reproduce raw-to-index rules, coverage/ambiguity masks,
paired eligibility, scaling, assignments, alignment, transitions, profiles,
contributions and robustness. Forty optimizer fits and 140 date-specific
holdout result sets are audited. Interpretation has 280 checks and 90 checksum
pins. Maps have source/output hash and dimension checks.

Source data and Part 3 products remain unchanged. Preservation inventories
describe what each stage left untouched during execution; the approved later
Part 1 rebuild is not a preservation failure. Superseded derived results are
overwritten, not archived. See the [changelog](../../CHANGELOG.md).

- [Comparison maps](../../figures/part2/part2_cluster_comparison_maps.png).
- [Fixed transitions](../../figures/part2/part2_cluster_transition_heatmap.png).
- [Profile comparison](../../figures/part2/part2_cluster_profile_heatmap.png).
- [Methods](../methods/historical-cluster-comparison.md) and
  [paired-matrix audit](part2-paired-matrix-processing-2026-09.md).

Machine-readable results remain under output/part2/clusters/ and
output/part2/interpretation/. No commit, push or website deployment was made.
