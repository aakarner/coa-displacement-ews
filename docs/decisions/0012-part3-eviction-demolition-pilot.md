# 0012: Pilot Joint Eviction and Demolition Forecasts

- **Status:** Accepted architecture
- **Decision date:** September 4, 2026

## Context

Part 3 was originally documented as four displacement-proxy outcomes at 1-,
3-, and 5-year horizons. Implementing every combination at once would mix
outcomes with different coverage and readiness while providing no benchmark
for whether a joint model benefits from shared information. Eviction filings
and residential demolitions are concrete event outcomes and are the most useful
initial test cases.

## Decision

Scope the initial Part 3 pilot to eviction filings and residential demolitions
at 1- and 3-year horizons. This creates four distinct predictions rather than a
single composite outcome.

Use December 31 of year `T` as the annual forecast origin. Count events in
calendar year `T + 1` for the 1-year label and calendar years `T + 1` through
`T + 3` for the 3-year label. Define evictions as unique filed cases regardless
of later disposition and demolitions as unique issued residential-demolition
permits. Model counts with task-specific historical unit exposure rather than
using a current denominator for earlier origins.

Compare separate outcome-by-horizon models with a joint multi-task,
multi-horizon candidate that can share relationships while retaining an
outcome-specific prediction for each horizon. Use the separate models as
required benchmarks. Promote the joint candidate only if common historical
backtests show improvement for at least one output without materially degrading
another.

Defer rent growth and county-adjusted real land-value growth until the pilot is
evaluated. Do not include a 5-year horizon in the pilot.

## Consequences

The modeling dataset needs four labels per forecast origin, plus an explicit
completeness indicator for each label. A covered hex-period with no event is a
zero; an uncovered or incomplete period remains missing. Separate evaluation
by outcome and horizon preserves differences in source coverage, event rates,
calibration, and false-negative performance even if the joint candidate shares
information internally.

Separate the fixed prediction geography from time-varying source coverage. The
7,027-cell H3 grid remains the computational frame, while the Part 3 study
geography is the 6,060 cells whose projected point-on-surface falls inside the
exact April 29, 2026 City of Austin FULL polygon. Filing and permit coordinates
must themselves pass the exact boundary test; a point outside the City cannot
enter merely because its H3 cell straddles the boundary. Eviction uncertainty
from multiple plausible in-study hexes or mixed in-study/out-of-study addresses
makes the affected candidate hex-years unavailable.

Historical source coverage is still evaluated at the panel date. Demolition
coverage is reconstructed from effective-dated City jurisdiction baselines and
actions. FULL, LTD, and 2MILE states are supported, while generic ETJ coverage
remains conservatively unavailable pending source confirmation. A demolition
label is available only when each year in its future window has resolved,
supported coverage for the entire observed annual interval; a midyear
jurisdiction change cannot create a full-year zero. The historical replay
determines source observability inside the fixed study geography rather than
changing the prediction boundary from year to year.

The eviction outcome combines Travis JP1-JP5 with the supplied Williamson JP1
and JP2 records. Williamson is not treated as countywide: each hex-year is
linked to the applicable official 2012-2021 or 2022-present precinct map and is
covered only when that JP source spans the period. Within the fixed current-full
subset, the supplied Williamson source covers 128 cells in 2020-2021 and 266
cells from 2022 onward. Twenty-nine current JP3 cells remain missing; none of
the selected cells is assigned to JP4. Hays remains missing. A filing geocoded
outside the exact current FULL polygon, its source county, or its effective
filing-court precinct cannot enter the outcome count.

Williamson court addresses use a cached four-stage cascade: conservative local
matching to public county address points, the City of Austin public ArcGIS
locator, the U.S. Census Bureau `Public_AR_Current` batch geocoder, and a
targeted authenticated ArcGIS World refinement. The active registry retains
the first qualifying local/City/Census result, then replaces only targeted
unresolved or Census-interpolated rows that pass the stricter World point gate.
The external requests contain only an opaque address ID and cleaned address
string, not defendant names or case numbers.
Address-bearing requests, responses, provider registries, review detail, and
resumable caches stay under ignored `output/` paths. Routine runs validate and
reuse the frozen caches. An unmatched record can still be outside provider
coverage or an in-City geocoding failure. Williamson zeroes therefore mean no
reliably located in-study filing in a source-covered hex-year, and geocoding and
linkage QA must accompany modeling and sensitivity analysis. The Part 1
eviction feature remains frozen to its Travis-only input until a future
re-baseline decision.

Applying the fixed April 29, 2026 boundary to all historical years estimates
events for today's City footprint. It does not establish whether a location was
inside Austin on the original filing or permit date. Answering that different
question would require a contemporaneous, time-varying municipal boundary.

The shorter scope allows the outcome-panel and temporal-validation machinery to
be tested before adding outcomes with less-comparable histories. Deferral does
not remove rent or land-value growth from the broader research agenda.

## Revisit When

Revisit after the common backtests are complete, when additional historical
coverage materially changes the number of usable forecast origins, or before
adding rent growth, land-value growth, or a longer horizon. Reject or revise the
joint structure if it degrades an individual output, is poorly calibrated, or
cannot preserve explicit missing-outcome coverage.

## Implementation

The active horizons are set in
[`R/analysis_config.R`](../../R/analysis_config.R), and the pilot/deferred
outcome roles are versioned in
[`config/forecast_outcomes.csv`](../../config/forecast_outcomes.csv). Current
data readiness and modeling intent are described in
[`docs/methods/analytical-workflow.md`](../methods/analytical-workflow.md) and
[`data/README.md`](../../data/README.md). The complete outcome panels are built
by
[`scripts/part3/build_eviction_outcome_panel.R`](../../scripts/part3/build_eviction_outcome_panel.R)
and
[`scripts/data/demolitions_panel.R`](../../scripts/data/demolitions_panel.R).
The historical demolition-coverage reconstruction is implemented in
[`R/demolition_coverage_history.R`](../../R/demolition_coverage_history.R), with
versioned source URLs and checksums in
[`config/demolition_jurisdiction_sources.csv`](../../config/demolition_jurisdiction_sources.csv).
Eviction source periods and the effective-dated Williamson JP-to-hex reference
are versioned in
[`config/eviction_sources.csv`](../../config/eviction_sources.csv) and
[`config/williamson_jp_hex_assignment.csv`](../../config/williamson_jp_hex_assignment.csv).
The public address reference is downloaded by
[`scripts/data/williamson_address_reference_download.R`](../../scripts/data/williamson_address_reference_download.R),
and the active ArcGIS World refinement follows the resumable batch and
authentication design already used for Travis records in
[`scripts/data/evictions_prepare.R`](../../scripts/data/evictions_prepare.R).
The local Williamson stage is implemented in
[`scripts/data/williamson_evictions_geocode_local.R`](../../scripts/data/williamson_evictions_geocode_local.R),
the City locator stage is implemented in
[`scripts/data/williamson_evictions_geocode_coa.R`](../../scripts/data/williamson_evictions_geocode_coa.R),
the Census fallback and base cascade are implemented in
[`scripts/data/williamson_evictions_geocode_census.R`](../../scripts/data/williamson_evictions_geocode_census.R),
and the targeted World refinement is implemented in
[`scripts/data/williamson_evictions_geocode_arcgis.R`](../../scripts/data/williamson_evictions_geocode_arcgis.R).
[`scripts/part3/build_forecast_labels.R`](../../scripts/part3/build_forecast_labels.R)
creates and validates the four forward-label tasks. Model fitting remains
unimplemented pending a leakage-safe historical predictor panel.
