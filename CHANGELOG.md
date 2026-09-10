# Analytical changelog

Major design decisions and consequential corrections, newest first. This is a
short running record, not a copy of every model run. Current methods describe
what the code does; decision records explain why; current audit reports give
the results and validation. Git records code changes.

## 2026-09-10 — Harmonize Part 1 and Part 2 measurement

Follow-up correction: eviction eligibility now checks only the two annual
windows that enter each score, not all history since January 2022. The pair
requires April 2023–April 2025 and April 2024–April 2026 respectively. Retain
potentially in-window missing/conflicting dates and all other safeguards;
older-only ambiguity no longer removes a cell. Source filing counts and
geocodes are unchanged. Downstream results replace the preceding run in place.
The change restores 60/109 usable eviction scores in 2025/2026. Full Part 1
eligibility rises from 2,456 to 2,557 (+101); paired Part 2 rises from 2,300 to
2,351 (+51), with no eligibility losses. Current profiles remain recognizable
(eviction group 81→87); Part 2 movement is 35.0% and fixed/refitted agreement
95.5%. The compact audit is output/part1/eviction_window_change_summary.csv.

- Apply the corrected complete-component recipes to the current Part 1 model,
  not only the historical comparison. Retire available-component averaging and
  the old eviction percentage-change/expanding-history-share recipe.
- Part 1 uses all eligible cells at the April 1, 2026 cutoff. Rent requires a
  coherent reliable 2014/2019/2024 block-group history, otherwise a coherent
  tract history. Ownership uses jointly known parcels in 2025 only, with the
  same 20-unit and 95% parcel/unit evidence thresholds.
- Part 2 additionally holds rent source level and ownership parcel support
  fixed across the paired dates and requires both dates to pass. Cells may
  drop out; missing evidence is not zero pressure.
- Fit current Part 1 component bounds and cluster standardization on 2026
  evidence. Part 2 retains its 2025 reference scales for temporal comparison.
  Shared formulas do not imply identical samples, scores, or cluster IDs.
- Replace superseded generated outputs in place. Preserve raw source vintages,
  reviewed evidence, current artifacts and checksum provenance needed for
  reproduction; do not create another archive directory for each failed run.
- First harmonized refit, before the window relaxation above: 2,456 cells,
  including all 2,300 then-current Part 2 cells plus
  156 current-only eligible cells. Refresh labels, maps and neighborhood
  summaries; retire 24 obsolete generated selection/island-review artifacts.
  Seven clusters remain provisional; old spatial-holdout findings are not
  transferred to the new fit. See the [current audit](docs/audits/part1-harmonized-measurement-2026-09.md).
- ML remains paused. See [decision 0013](docs/decisions/0013-harmonized-measurement.md)
  and [current measurement methods](docs/methods/current-measurement.md).

## 2026-09-08 — Correct the historical measurement design

- Require complete fixed recipes at both dates; missing subcomponents no longer
  silently change weights. Retain rent level, growth and acceleration through
  an all-six-vintage reliable BG/tract hierarchy.
- Use recent event rates and signed rate differences for evictions; remove the
  expanding-history recent share. Replace 311 percentage change with signed
  rate change. Zero signed change scores 50, not a literal no-risk zero.
- Regenerate the 2025/2026 comparison in place: 2,300 eligible cells; 34.9%
  fixed-definition movement; 95.5% fixed/refitted 2026 agreement.
- Review qualitative concern labels against the corrected profiles, separately
  from nominal cluster IDs. See the [current Part 2 audit](docs/audits/part2-cluster-comparison-2026-09.md).

## Earlier decisions

The [decision index](docs/decisions/README.md) records the grid, unit surface,
ACS allocation, seven-domain architecture, seven-cluster choice, update design
and neighborhood summaries. Source investigation and import evidence are
indexed in [the documentation guide](docs/README.md). Those records predate
this single chronological log; their dates are not being reconstructed here.

## Maintenance rule

Add a dated entry when the measurement, analytical universe, denominator,
interpretation or update design changes materially. Link to current methods
and the relevant decision. Routine reruns and cosmetic edits need no entry.
An as-run preservation check is evidence that a run did not mutate its inputs;
it is not a permanent prohibition on an explicitly approved later rebuild.
