# Austin Displacement Early Warning System

This repository implements the three-part analytical architecture in the
March 2026 City of Austin proposed methods report:

1. **Classify today's risk.** Use current displacement proxies,
   socioeconomic vulnerability, and smoke signals to estimate a baseline
   neighborhood typology.
2. **Update risk over time.** Hold the baseline definitions fixed, assign each
   new data vintage to the nearest established cluster, and report transitions,
   assignment confidence, and global drift.
3. **Predict future risk.** Use historical smoke signals, vulnerability, and
   prior pressure to forecast evictions and demolitions at 1- and 3-year
   horizons in the initial pilot. Rent and land-value growth are deferred
   candidate outcomes.

Major design changes are recorded in the [analytical changelog](CHANGELOG.md).
The [current measurement contract](docs/methods/current-measurement.md) governs
the harmonized Part 1/Part 2 recipes. Part 3 ML work is currently paused.

The unit of analysis is an H3 resolution 9 hexagon. The repository retains a
7,027-cell computational grid generated from the 2021 Census Austin place
polygon. The Part 3 pilot uses a fixed subset of 6,060 cells whose projected
point-on-surface falls inside the exact April 29, 2026 City of Austin
full-purpose boundary.

## Three-County Study Area

Austin's full-purpose municipal boundary extends into Travis, Williamson, and
Hays Counties. The fixed current-full Part 3 geography and 2024 ACS allocation
produce the following approximate distribution:

| County | Austin area, square miles | Center-selected H3 cells | Allocated population, people | Allocated housing, units |
| --- | ---: | ---: | ---: | ---: |
| Travis | 268.6 (93.3%) | 5,711 | 881,953 (93.0%) | 439,264 (93.5%) |
| Williamson | 13.7 (4.8%) | 295 | 65,032 (6.9%) | 30,129 (6.4%) |
| Hays | 5.5 (1.9%) | 54 | 909 (0.1%) | 287 (0.1%) |
| **Total** | **287.8** | **6,060** | **947,894** | **469,680** |

Area is calculated from the intersection of county boundaries with Austin's
full-purpose boundary. Population and housing are sums of 2024 ACS 5-year
estimates allocated to the center-selected H3 cells through the project's
Census-block and residential-parcel dasymetric method and attributed to the
source block group's county. Counts are rounded, and the shares will change
when the source or boundary vintage changes. The full 7,027-cell computational
grid remains available for Part 1, Part 2, and boundary QA; it is not itself the
Part 3 study universe.

Although most of Austin is in Travis County, the analysis seeks comparable
data for all three counties wherever a domain is critical to classification or
forecasting. When an equivalent source is not available, the pipeline retains
the gap as missing or as an explicit coverage flag rather than treating it as
a zero. Parts 1 and 2 now use Travis and Williamson JP1/JP2 eviction evidence.
The Part 3 outcome panel uses the same jurisdictions with effective-dated
precinct coverage. Within the fixed current-full subset, the supplied
Williamson source covers 128 cells in 2020-2021 and 266 cells from 2022 onward;
29 current JP3 cells and all 54 Hays cells remain explicit gaps.

Williamson address linkage uses a cached local public-address match, then the
City of Austin public ArcGIS locator, then the U.S. Census Bureau batch
geocoder. A final authenticated ArcGIS World refinement revisits only
City-relevant unresolved or Census-interpolated records. External requests
carry only an opaque ID and cleaned address string; address-bearing requests,
responses, reviews, and caches remain in ignored `output/` paths. The active
Part 3 panel consumes the augmented four-stage registry. Routine pipeline runs
reuse the frozen caches without network access; an intentional ArcGIS cache
fill requires separate authorization. The resumable World stage follows the
pattern already used for the earlier Travis eviction records in
`scripts/data/evictions_prepare.R`.

## Pipeline

The canonical dependency graph is [`_targets.R`](_targets.R). Run:

```bash
Rscript 00_requirements.R
Rscript run_analysis.R
```

Or run a named target and everything upstream of it:

```bash
Rscript run_analysis.R part1_validation
Rscript run_analysis.R part2_baseline_assignment
Rscript run_analysis.R part3_forecast_readiness
```

Inspect the graph or pipeline state from R:

```r
targets::tar_visnetwork()
targets::tar_progress()
```

`{targets}` records the input files, code, configuration, and upstream artifacts
used by every stage. Updating a raw source or the analysis cutoff invalidates
only the affected branches. Part 3 additionally writes SHA-256 source and run
manifests so an adopted panel or label set can be tied to exact local inputs.

An existing reviewed workspace can initialize the metadata store once with
`EWS_TARGETS_ADOPT_EXISTING=true Rscript run_analysis.R`; subsequent runs should
leave that variable unset.

## Repository Structure

```text
_targets.R                    Canonical pipeline graph
_targets_review.R             Optional methodological review workflows
00_requirements.R             Project package bootstrap
01_create_hex_grid.R          Base H3 geography
run_analysis.R                Small wrapper around targets::tar_make()

R/
  analysis_config.R           Shared vintage and method settings
  pipeline.R                  Script-stage orchestration helpers
  cluster_assignment.R        Frozen-centroid Part 2 assignment
  eviction_panel.R            Complete eviction outcome-panel helpers
  eviction_coverage.R         County/JP/year eviction source coverage
  williamson_eviction_ingest.R Report-aware Williamson workbook parsing
  demolition_panel.R          Complete demolition outcome-panel helpers
  demolition_coverage_history.R Effective-dated jurisdiction replay and QA
  forecast_labels.R           Forward-label construction and QA
  forecast_spec.R             Part 3 contract/readiness checks
  acs_dasymetric.R            Dasymetric ACS allocation helpers
  unit_count_*.R              Parcel unit-count modeling helpers

scripts/
  data/                       Source-specific processing stages
  data/unit_counts/           Canonical parcel unit hierarchy
  features/                   Shared hex-level feature construction
  part1/                      Baseline clustering and maps
  part3/                      Outcome-panel and forecast-label construction
  audits/                     Routine production QA
  reviews/                    Optional re-baseline sensitivity analyses
  exploratory/                Non-pipeline research

config/
  311_smoke_signal_types.csv  Exact 311 descriptions used in Part 1
  feature_dictionary.csv      Feature domains, roles, and missingness rules
  forecast_outcomes.csv       Part 3 displacement-proxy outcomes
  eviction_sources.csv        Court files and declared source periods
  williamson_jp_hex_assignment.csv Versioned historical/current JP coverage
  demolition_jurisdiction_sources.csv Versioned coverage-source URLs and hashes
  amenity_cluster_labels.csv  Display labels for the selected Part 1 solution

docs/                         Methods, decision records, and dated audits
data/                         Local inputs and cached public extracts
output/                       Derived data artifacts
figures/                      Static and interactive outputs
```

## Current Analytical State

The selected Part 1 solution is the seven-cluster amenity-augmented typology.
`output/part1/baseline_cluster_model.rds` freezes:

- the analysis vintage and H3 resolution;
- the exact feature schema;
- baseline means and standard deviations;
- the seven centroids;
- substantive cluster labels;
- baseline distance and boundary thresholds.

`output/part1/baseline_cluster_validation.csv` checks the complete Part 1
feature contract, labels, scaling, centroids, population coverage, and exact
frozen-model reassignment. The accompanying summary, canonical assignments,
and runtime manifest preserve reproducibility. Future feature vintages will use
the same assignment function.

Run-specific coverage, diagnostics, and the assignment checksum are generated
in `output/part1/baseline_cluster_summary.csv`. The rationale for the current
seven-cluster choice is retained in
[`docs/decisions/0008-select-seven-clusters.md`](docs/decisions/0008-select-seven-clusters.md),
with the corrected 2,557-cell fit documented in the
[current audit](docs/audits/part1-harmonized-measurement-2026-09.md).
The August spatial-holdout findings describe the superseded specification,
not validation of this refit. The current baseline remains provisional pending
partner review and now includes covered Williamson JP1/JP2 eviction evidence.

Part 3 does not yet train a model. The initial pilot now has validated annual
eviction and residential-demolition panels plus four forward labels: each
outcome at 1- and 3-year horizons. Separate outcome-by-horizon models will be
required benchmarks for a joint multi-task candidate; the joint candidate is
retained only if backtesting shows a benefit without materially degrading
another output. Rent and land-value growth are deferred, and five-year
forecasts are outside the pilot. The current
`output/part3/forecast_readiness.csv` confirms that the outcome panels and
labels pass their contracts and identifies the leakage-safe historical
predictor panel as the next requirement. The former same-year cluster
classifiers were removed from the active tree because they do not implement the
methods-report architecture; Git history preserves them.

## Implemented Evidence

The versioned feature dictionary is
[`config/feature_dictionary.csv`](config/feature_dictionary.csv). The current
source roles are:

| Domain | Current source and status | Analytical role |
| --- | --- | --- |
| Demographics | ACS 5-year estimates allocated with Census-block controls and residential-parcel support | Part 1 vulnerability |
| Rent | ACS gross-rent vintages are the citywide backbone; CoStar is coverage-limited enrichment | ACS is a Part 1 input; CoStar is a sensitivity input and rent growth is deferred from the Part 3 pilot |
| Evictions | Travis County JP1-JP5 and Williamson County JP1-JP2 filing extracts support current/historical indices and annual/forward-label outputs | Parts 1 and 2 use covered filings; Part 3 uses precinct- and year-specific coverage. Current Williamson JP3 and Hays gaps remain explicit throughout |
| Demolitions | Austin issued construction permits are filtered to residential demolitions and paired with effective-dated City jurisdiction baselines and actions to build the annual outcome panel | Part 1 displacement proxy and Part 3 pilot outcome; inside the fixed current-full study geography, a historical period is covered only while FULL, LTD, or 2MILE source coverage is resolved continuously, so partial-year jurisdiction changes do not create annual zeroes |
| Land value | Hays, Travis, and Williamson appraisal histories | Part 1 sensitivity input; growth is deferred from the Part 3 pilot |
| 311 requests | Austin 311 code-enforcement intake requests selected by exact versioned descriptions | Part 1 smoke signal; general 311 activity is excluded |
| Corporate ownership and property sales | County appraisal corporate-ownership classifications plus available deed and sales histories | Current corporate ownership is a Part 1 input; corporate-ownership change and transaction measures are sensitivity inputs |
| Amenity change | Texas Comptroller openings, with mixed-beverage and Austin inspection corroboration | Part 1 input in the selected amenity-augmented specification |

See [`data/README.md`](data/README.md) for source files, coverage limits, and
generated artifacts.

## Data and Secrets

Large raw appraisal files and API caches remain local and are excluded from
Git. Austin's public 311 endpoint can be queried anonymously. Optional
authenticated access uses environment variables that must never be committed:

```bash
export AUSTIN_DATA_API_KEY="..."
export AUSTIN_DATA_API_SECRET="..."
```

See [`docs/README.md`](docs/README.md) for the documentation index and
[`data/README.md`](data/README.md) for source details and coverage limits.
