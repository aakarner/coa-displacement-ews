# Current Part 1 measurement

Part 1 reconstructs the April 1, 2026 snapshot on all eligible current cells.
It shares the corrected component formulas with Part 2, without importing the
paired eligibility mask. Source reconstructions currently live under
output/part2/ because that is where they were first assembled; their storage
location does not impose a two-date requirement on Part 1.

## Rules

- **Rent:** choose BG if all three 2014/2019/2024 estimates are positive finite,
  have finite nonnegative MOEs, and MOE/estimate is at most 0.30. Otherwise use
  tract if all three pass. Otherwise all rent terms remain missing. Never mix
  geographic levels within a history. Compute level, annualized real log
  growth and growth acceleration in 2024 dollars. Part 2 adds the analogous
  2013/2018/2023 requirements and fixes source level across all six vintages.
- **Vulnerability:** require all five existing components—low income, renters,
  poverty, rent burden and low educational attainment—from the 2024 ACS.
  Missing allocated counts are not zero. Income MOE is a diagnostic, not an
  additional eligibility screen in this specification.
- **Ownership:** use 2025 evidence for corporate and financialized ownership
  on jointly known parcels, not the intersection with 2024. Require at least
  20 observed units and at least 95% coverage of both fixed units and parcels.
  Shares use observed units/parcels; corporate-unit density uses whole-hex area.
  Unknown parcels remain outside those numerators/denominators and coverage is
  explicit. Part 2 restricts the observed parcel cohort to both years.
- **Evictions:** recent 12-month mapped filings per 100 fixed promoted units
  and recent-minus-previous rate change, each half the score. Retain the
  reviewed source/ambiguity rules; historical share and percentage change are
  diagnostics only, not scored terms.
  Coverage and localizable ambiguity checks span only April 2, 2024–April 1,
  2026, the two scored years. Older-only issues do not exclude a current cell;
  missing dates and conflicting dates potentially in-window still do.
- **311:** recent selected-request rate, density and signed rate change, each
  one third. Rate and density intentionally put two thirds of the weight on
  current activity. Only the configured Code Officer intake types are used.
- **Demolitions/amenities:** retain the corrected Part 2 definitions and exact
  windows, requiring all terms/category scores.

All seven indices require complete fixed recipes. Component clipping uses the
same p1/p99 method as Part 2, but the current reference is estimated from
2026 domain support. Signed event changes use ±q99(abs(change)), mapping zero
to 50 (also the flagged degenerate-scale value). Missing stays missing. A
zero-filing/zero-change eviction score can therefore be 25, not literal no risk.
Part 1 subsequently standardizes the seven indices using its eligible 2026
sample. Part 2 retains both component bounds and standardization from 2025.

## Eligibility and source scope

Require current FULL-purpose city-center membership, at least 20 fixed units,
current ownership evidence, usable event coverage, retrospective amenity
usability and every component of all seven indices. Do not require the cell
to pass the 2025 snapshot. Keep all 7,027 audit cells and explicit exclusions.

The dated event-domain reconstructions are reused as raw measures and rescored
for the current reference. They are not downloaded or geocoded again. Hays and
Williamson JP3 filing gaps, the coordinate-required 311 query, uncertain
amenity completeness and whole-hex boundary denominators remain limitations.
The retrospective current snapshot is not a leakage-safe forecast backtest.

## Reproduction

With the reviewed local source reconstructions available, run:

```sh
Rscript scripts/features/build_current_measurement.R
Rscript scripts/features/build_current_features.R
Rscript scripts/audits/features.R
Rscript scripts/part1/fit_baseline_clusters.R
```

Then review the newly fitted profiles before updating the label configuration,
freezing the model and regenerating maps, validation and neighborhood summaries.
The targets graph includes the current-measurement dependency, so the ordinary
feature builder cannot silently restore the superseded selected recipes.

Current source decisions, component scales, eligibility and checksum manifest
are under output/part1/measurement/. Canonical Part 1 outputs are overwritten
in place, not copied to another dated run directory. See the
[changelog](../../CHANGELOG.md) and [decision 0013](../decisions/0013-harmonized-measurement.md).
