# Quick Start

## 1. Install Requirements

From the repository root:

```bash
Rscript 00_requirements.R
```

Packages are installed into the version-specific project library under
`.r-library/`.

## 2. Configure the Analysis Vintage

Defaults live in `R/analysis_config.R`. Historical reruns can override them
without editing code:

```bash
export EWS_ANALYSIS_AS_OF_DATE="2026-04-01"
export EWS_ACS_CURRENT_YEAR="2024"
export EWS_APPRAISAL_CURRENT_YEAR="2025"
export EWS_AMENITY_CLUSTER_K="7"
export EWS_BASELINE_CLUSTER_SPECIFICATION="amenity_augmented"
```

Austin's public 311 endpoint supports anonymous access. Optional authenticated
access can be configured with:

```bash
export AUSTIN_DATA_API_KEY="..."
export AUSTIN_DATA_API_SECRET="..."
```

Refresh the ignored City Neighborhood Reporting Area snapshot before producing
the neighborhood summaries:

```bash
Rscript scripts/data/download_neighborhood_reporting_areas.R
```

## 3. Inspect the Pipeline

```r
targets::tar_manifest(fields = c(name, command))
targets::tar_visnetwork()
```

The graph is divided into source processing, shared feature construction,
Part 1 baseline estimation, Part 2 fixed-cluster assignment, and Part 3
forecast preparation.

The Part 3 pilot is scoped to 1- and 3-year eviction and residential-demolition
outputs. Its model comparison will use separate outcome-by-horizon benchmarks
and a joint multi-task candidate. Rent growth, land-value growth, and five-year
forecasts are outside the initial pilot.

Before rebuilding Part 3, retain the reviewed Travis eviction geocode input at
`output/eviction_addresses_geocoded.csv`. It is deliberately ignored because
it contains address-level material. Part 3 retains the 7,027-cell computational
grid but models only the 6,060 cells whose projected point-on-surface is inside
the exact April 29, 2026 City of Austin FULL polygon. Filing coordinates must
also pass an exact point-in-polygon test rather than merely landing in a
boundary-straddling H3 cell.

For an initial Williamson run or an intentional cache refresh, build the grid,
download the county's public address points, prepare the court files, and fill
the local, City, and Census stages first. Then fill the authenticated ArcGIS
World stage using one of the methods below.

```bash
Rscript run_analysis.R hex_grid
WILLIAMSON_ADDRESS_REFERENCE_NETWORK=true \
  Rscript scripts/data/williamson_address_reference_download.R
Rscript run_analysis.R prepared_williamson_evictions
Rscript scripts/data/williamson_evictions_geocode_local.R
WILLIAMSON_EVICTION_COA_NETWORK=true \
  Rscript scripts/data/williamson_evictions_geocode_coa.R
WILLIAMSON_EVICTION_CENSUS_NETWORK=true \
  Rscript scripts/data/williamson_evictions_geocode_census.R
```

The public-reference download reads no court file and sends no eviction
address. The local match also makes no address-bearing network request. The
City of Austin public ArcGIS locator, Census batch geocoder, and ArcGIS World
stage transmit an opaque address ID and cleaned address string, but not a
defendant name, case number, or other court field. They write address-bearing
requests, responses, provider registries, review detail, and resumable caches
only under ignored `output/` paths. Later runs validate and reuse the caches
with the network flags unset. The active cascade keeps a conservative local
match first, a qualifying City locator point second, and a qualifying Census
address-range result third. ArcGIS World then revisits only unresolved City
candidates and Census-interpolated records that the City locator places in
full-purpose Austin. It writes the augmented registry consumed by Part 3,
requests stored results, and therefore requires an authorized ArcGIS credential
and account usage when filling missing caches. With an API key or
noninteractive credential, fill its cache explicitly with:

```bash
WILLIAMSON_EVICTION_ARCGIS_NETWORK=true \
  Rscript scripts/data/williamson_evictions_geocode_arcgis.R
```

With only `ARCGIS_CLIENT`, launch R interactively, set the same network flag
and `WILLIAMSON_EVICTION_ARCGIS_AUTH_METHOD=code`, then source the script and
paste the browser authorization code at the R prompt. Routine pipeline runs do
invoke the stage with networking disabled: all valid caches are reused, and a
missing cache stops the build rather than silently dropping the refinement.

Within the fixed current-full subset, Williamson source coverage is 128 cells
in 2020-2021 and 266 cells from 2022 onward. The 29 current JP3 cells and all 54
Hays cells remain missing, rather than becoming zeroes. Applying one current
boundary to every outcome year produces history for today's City footprint; it
does not reconstruct whether each location was inside Austin on the event date.
Effective-dated JP boundaries and historical City jurisdiction states still
determine court and permit-source coverage.

Also download the two public City GIS files named in
`config/demolition_jurisdiction_sources.csv` to their configured local paths.
The demolition processor verifies their SHA-256 checksums before using them.
The versioned county and Williamson JP crosswalks under `config/` are committed
and do not require a live geography download.

## 4. Run

For an existing workspace that already contains reviewed outputs, adopt those
files into a new `{targets}` metadata store once:

```bash
EWS_TARGETS_ADOPT_EXISTING=true Rscript run_analysis.R
```

This records the current artifacts without rerunning stages whose expected
outputs already exist. Do not keep this variable set for routine work.

Run the complete currently implemented pipeline:

```bash
Rscript run_analysis.R
```

Run only a final artifact and its prerequisites:

```bash
Rscript run_analysis.R part1_validation
Rscript run_analysis.R part1_neighborhood_summary
Rscript run_analysis.R part2_baseline_assignment
Rscript run_analysis.R part3_forecast_readiness
```

After updating one raw data source, rerun the same command. `{targets}` checks
the source manifests and rebuilds only affected downstream artifacts.

## 5. Diagnose

```r
targets::tar_progress()
targets::tar_meta(fields = c(name, time, error))
```

To force one branch to rebuild:

```r
targets::tar_invalidate(part1_cluster_analysis)
targets::tar_make(part1_baseline_model)
```

## Key Outputs

The CARTO basemaps in the cluster and high-risk-island maps require
`CARTO_BASEMAP_API_KEY` in the environment or the project-root `.Renviron`
(Git-ignored). Both scripts load that file when the key is not already set.
The generated HTML includes the browser basemap key, so restrict it to
`aakarner.github.io` in CARTO's dashboard. To refresh the published cluster map,
run `Rscript scripts/part1/visualize_baseline_clusters.R`, then publish the updated
`site/index.html` through the GitHub Pages workflow. Force-refresh the browser
if old watermarked tiles remain cached.

- `output/hex_features.rds`: current shared feature surface.
- `output/amenity_cluster_sensitivity.rds`: Part 1 cluster diagnostics.
- `output/part1/baseline_cluster_model.rds`: frozen Part 1 model.
- `output/part1/baseline_cluster_validation.csv`: Part 1 lock checks.
- `output/part1/baseline_cluster_summary.csv`: presentation-run metrics.
- `output/part1/baseline_cluster_assignments.csv`: canonical labeled results.
- `output/part1/neighborhood_cluster_composition.csv`: population- and
  housing-unit-weighted cluster shares by Neighborhood Reporting Area.
- `output/part1/neighborhood_cluster_summary.csv`: neighborhood plurality,
  majority, coverage, and population/housing agreement fields.
- `output/part2/baseline_fixed_cluster_assignments.csv`: Part 2 self-check.
- `output/eviction_filings_complete_by_hex_year.csv`: complete covered and
  uncovered eviction outcome panel.
- `output/williamson_eviction_geocode_local_qa.csv`,
  `output/williamson_eviction_geocode_coa_qa.csv`, and
  `output/williamson_eviction_geocode_qa.csv`: aggregate local, City-locator,
  Census-cascade, and exact current-boundary QA.
- `output/williamson_eviction_geocode_arcgis_qa.csv`: aggregate World-stage
  quality-gate and before/after City-linkage QA.
- `output/part3/eviction_source_geography_qa.csv`: exact City, county, and
  effective-JP quarantine checks.
- `output/demolition_permits_by_hex_year.csv`: complete residential-demolition
  outcome panel with continuous, effective-dated jurisdiction coverage and
  explicit partial-year flags.
- `output/part3/demolition_historical_coverage_by_hex_year.csv`: replayed annual
  interval coverage with transition, end-state, and ambiguity metadata.
- `output/part3/demolition_coverage_current_snapshot_qa.csv` and its summary:
  comparison with the April 29, 2026 current snapshot.
- `output/part3/eviction_demolition_forecast_labels_long.rds`: canonical
  long-form 1- and 3-year pilot labels.
- `output/part3/forecast_label_qa.csv`: label availability, event counts, and
  zero-event checks by origin, outcome, and horizon.
- `output/part3/*_source_manifest.csv` and
  `output/part3/forecast_label_run_manifest.csv`: exact SHA-256 provenance for
  the pilot panels and labels.
- `output/part3/forecast_readiness.csv`: pilot/deferred outcome scope and
  outcome-panel, label, and predictor-panel status.
- `figures/03e_amenity_clusters_interactive.html`: interactive baseline map.
- `figures/03g_neighborhood_cluster_plurality.png`: neighborhood map shaded by
  the population-plurality cluster.
- `output/part1/high_risk_island_hex_summary.csv`: high- and very-high-risk
  hexes whose immediate classified neighbors are predominantly low risk,
  including their leading domains and property-influence diagnostics.
- `figures/03h_high_risk_islands_interactive.html`: interactive review map for
  the high-risk island candidates.

The processors under `scripts/data/` remain runnable for focused debugging, but
normal analysis runs should go through `_targets.R` so their dependencies and
vintages are recorded.

## Optional Methodological Reviews

Expensive sensitivity and model-selection checks are intentionally separate
from routine production execution. After the required production inputs exist,
run the review pipeline with its own metadata store:

```r
targets::tar_make(
  script = "_targets_review.R",
  store = "_targets_review"
)
```

To run only the Austin Code complaint linkage audit:

```r
targets::tar_make(
  code_complaint_audit,
  script = "_targets_review.R",
  store = "_targets_review"
)
```

To run the high-risk-island and property-driver review (including the Code
complaint linkage audit it depends on):

```r
targets::tar_make(
  high_risk_island_review,
  script = "_targets_review.R",
  store = "_targets_review"
)
```

These reviews support an explicit re-baseline decision; they are not required
when rebuilding the currently approved method or assigning a new Part 2
vintage.

## Part 2 historical ownership comparison

With the current promoted parcel/unit surface built and the pinned sibling
`landlord-mapper` checkout and ownership output available:

```sh
Rscript scripts/part2/build_ownership_snapshots.R
```

This isolated command writes to `output/part2/ownership/` and preserves Part 1
features. The `part2_ownership_snapshots` target provides the equivalent stage
with upstream dependencies. Start with `ownership_county_qa.csv`,
`ownership_hex_change.csv`, and `ownership_snapshot_manifest.json`; select
`comparison_ready` for screened common-parcel comparisons. Williamson uses
the pinned certified reports plus same-year GIS evidence for 2024 and 2025; place all
ZIPs at the paths in `config/williamson_ownership_sources.json`. No current-owner
fallback is used. Also inspect `ownership_source_variant_summary.csv` and the
certified-only/source-agreement sensitivity tables, including the
2024-certified-only variant isolating the latest supplement. See the
[`historical ownership method`](methods/historical-ownership.md) and
[`updated run audit`](audits/williamson-2024-ownership-integration-2026-09.md) before clustering.
