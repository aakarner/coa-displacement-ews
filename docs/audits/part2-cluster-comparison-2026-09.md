# Part 2 historical cluster comparison: corrected results

The run statistics below precede the October residential/property repair.
See the [current repair audit](residential-property-repair-2026-10.md) for the
updated 2,503-cell sample. Current generated models and figures use that repair.

- **Regenerated:** September 11, 2026, with ambiguous eviction cases unassigned
  and candidate cells retained.
- **Pair:** April 1, 2025 / April 1, 2026, retrospectively reconstructed.
- **Common sample:** 2,490 cells: 2,348 Travis and 142 Williamson.
- **Model:** Seven complete equally standardized indices; k=7; 2025 component
  scales and cluster standardization frozen for 2026. Measurement:
  part2-fixed-components-v2; eviction eligibility: rolling_scored_24_months_v2.
- **Status:** Features, models, interpretation, robustness and figures replaced
  in place. Part 1 separately refreshed; Part 3 ML remains paused.

## Main finding

Removing the ambiguity veto expands coverage while the broad comparison
results remain similar to the preceding fit.

| Question | Current result | Interpretation |
| --- | --- | --- |
| Do cells move with 2025 definitions fixed? | 867 of 2,490 move (34.8%); 1,623 stay (65.2%). | Updated features cross fixed cluster boundaries. |
| Do fixed/refitted 2026 classifications agree? | 2,376 agree (95.4%); 114 differ (4.6%), after coeval label matching. | Most later classifications survive the separate refit. |
| Do concern tiers rise or fall? | 342 rise (13.7%), 266 fall (10.7%), 1,882 retain their tier (75.6%). | Modest upward tilt under the reviewed interpretation. |

The preceding 2,351-cell run had 35.0% movement and 95.5% fixed/refitted
agreement. All its eligible cells remain; 139 are added. Raw accepted filing
counts, geocodes, units and source-court coverage are unchanged. Ambiguous
cases remain unassigned. See the [decision audit](eviction-ambiguity-policy-2026-09.md).

These transitions are not displacement of 34.8% of places or people.
Movers contain approximately 132,292 fixed estimated residential units, 32.1%
of common-sample units; these are not displaced units. Temporal fixed ARI is
0.393; fixed/refitted 2026 ARI is 0.898. The combined baseline-to-later-refit
comparison has 836 movers and ARI 0.409; it mixes temporal and definition change.

## Measurement and eligibility

All seven indices retain complete fixed recipes; zero included cells change
component availability. Rent uses one reliable BG history across all six ACS
vintages or one tract history: 1,554 included cells use BGs and 936 tracts.
Ownership retains the common-parcel 20-unit/95% screen. Other domain rules
are unchanged.

Evictions use recent filings per 100 fixed units and signed rate change,
equal halves. Source coverage spans April 2, 2023–April 1, 2025 and April 2,
2024–April 1, 2026 respectively. Potentially in-window ambiguous cases produce
candidate-cell audit flags, never a count or eligibility veto. History totals
and recent shares remain unscored diagnostics. No records or geocodes were added.

Both eviction component scales are refitted on expanded 2025 domain support:
current-rate p1/p99 bounds are 0 and 21.07632; the signed-change bound is
±11.18147 per 100 units. They remain frozen for 2026. Part 1 fits its separate
2026 reference. Changed assignments also reflect refreshed sample-based
standardization and cluster boundaries.

## Profiles and qualitative concern

C1–C7 IDs are nominal and differ from the preceding fit. Existing profile
names and concern criteria were reviewed against baseline centroids and raw
events, then hash-pinned before computing tier transitions. They are not
calibrated probabilities.

| ID | Reviewed profile | Concern | 2025 | 2026 fixed | 2026 aligned refit |
| --- | --- | --- | ---: | ---: | ---: |
| C1 | Selected 311 activity and vulnerability | Moderate | 209 | 243 | 231 |
| C2 | Eviction-filing concentration | Very high | 77 | 78 | 81 |
| C3 | Nearby amenity activity | Moderate | 174 | 166 | 131 |
| C4 | Lower measured pressure | Low | 841 | 821 | 827 |
| C5 | Demolition-permit concentration | High | 243 | 295 | 279 |
| C6 | Corporate ownership and vulnerability | Moderate | 353 | 322 | 327 |
| C7 | Higher/rising rents, lower vulnerability | Low | 593 | 565 | 614 |

Every baseline C2 cell has a recent mapped filing; its mean rate is 20.8 per
100 units, versus at most 3.0 in other profiles. Every baseline C5 cell has a
recent residential demolition permit. Low concern does not mean no vulnerable
residents or displacement. Filings and permits do not establish completed events.

Of 867 switchers, 259 change profile within the same tier, 366 move one tier,
and 242 move two or more tiers. These are ordinal steps, not equal amounts
of risk. On the same common sample, recent mapped filings total 8,446/9,163;
the shared-window counts reconcile exactly.

## Robustness and limits

Twenty 100-start fits at each date check optimization. Conditional holdouts
use frozen component scoring and training-sample cluster standardization:

| Holdout design | Replicates per date | Median held-out ARI 2025 | Median held-out ARI 2026 |
| --- | ---: | ---: | ---: |
| Random 20% of cells | 50 | 0.975 | 0.967 |
| Whole H3 resolution-7 regions | 20 | 0.964 | 0.932 |

Spatial 10th-percentile ARIs are 0.909/0.883. Unevaluated recovery stays
missing. Same-model subsets with at least 50/100 units have movement of
34.9%/34.7%, versus 34.8% overall; these are not threshold-specific refits
or evidence ruling out small-count or geocoding error.

Limitations remain: retrospective paired support, overlapping ACS releases,
uncovered Hays/Williamson JP3 filings, coordinate-required 311 extraction,
uncertain amenity completeness, and 163 boundary-straddling included cells
with whole-hex denominators. This is not a complete Austin census or forecast.

## Verification and artifacts

Independent tests reproduce raw-to-index rules, coverage and ambiguity flags,
paired eligibility, scaling, assignments, alignment, transitions, profiles,
contributions and robustness. Forty optimizer fits and 140 date-specific
holdout result sets are audited. Interpretation has 280 checks and 90 checksum
pins. Figures have input/output hash and dimension checks and were visually
inspected.

Source evidence remains unchanged. The annual eviction panel and forward
labels are intentionally rebuilt under decision 0014. As-run preservation
inventories retain their original hashes; authorized replacement of derived
products is distinguished from unrelated changes. Superseded generated
results are overwritten. See the [changelog](../../CHANGELOG.md).

- [Comparison maps](../../figures/part2/part2_cluster_comparison_maps.png).
- [Fixed transitions](../../figures/part2/part2_cluster_transition_heatmap.png).
- [Profile comparison](../../figures/part2/part2_cluster_profile_heatmap.png).
- [Methods](../methods/historical-cluster-comparison.md) and
  [paired-matrix audit](part2-paired-matrix-processing-2026-09.md).

Machine-readable results remain under output/part2/clusters/ and
output/part2/interpretation/. No website deployment is part of this change.
