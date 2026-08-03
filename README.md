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
   prior pressure to forecast future evictions, demolitions, rent growth, and
   land-value growth at 1-, 3-, and 5-year horizons.

The unit of analysis is an H3 resolution 9 hexagon covering Austin.

## Three-County Study Area

Austin's full-purpose municipal boundary extends into Travis, Williamson, and
Hays Counties. The current project geography and 2024 ACS allocation produce
the following approximate distribution:

| County | Austin area, square miles | Allocated population, people | Allocated housing, units |
| --- | ---: | ---: | ---: |
| Travis | 268.6 (93.3%) | 902,754 (93.0%) | 448,186 (93.4%) |
| Williamson | 13.7 (4.8%) | 67,073 (6.9%) | 31,061 (6.5%) |
| Hays | 5.5 (1.9%) | 1,095 (0.1%) | 403 (0.1%) |
| **Total** | **287.8** | **970,922** | **479,650** |

Area is calculated from the intersection of county boundaries with Austin's
full-purpose boundary. Population and housing are 2024 ACS 5-year estimates
allocated through the project's Census-block and residential-parcel
dasymetric method and attributed to the source block group's county. Counts
are rounded, and the shares will change when the source vintage changes.

Although most of Austin is in Travis County, the analysis seeks comparable
data for all three counties wherever a domain is critical to classification or
forecasting. When an equivalent source is not available, the pipeline should
retain the gap as missing or as an explicit coverage flag rather than treating
it as a zero. Travis-only eviction filings are the principal current exception.

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
only the affected branches.

The lightweight source-manifest targets intentionally check file metadata on
every run; unchanged manifests do not rebuild downstream stages.

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
  forecast_spec.R             Part 3 outcome/readiness checks
  acs_dasymetric.R            Dasymetric ACS allocation helpers
  unit_count_*.R              Parcel unit-count modeling helpers

scripts/
  data/                       Source-specific processing stages
  data/unit_counts/           Canonical parcel unit hierarchy
  features/                   Shared hex-level feature construction
  part1/                      Baseline clustering and maps
  audits/                     Routine production QA
  reviews/                    Optional re-baseline sensitivity analyses
  exploratory/                Non-pipeline research

config/
  311_smoke_signal_types.csv  Exact 311 descriptions used in Part 1
  feature_dictionary.csv      Feature domains, roles, and missingness rules
  forecast_outcomes.csv       Part 3 displacement-proxy outcomes
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
with detailed spatial-holdout evidence in the corresponding
[`dated audit`](docs/audits/part1-cluster-selection-2026-08.md). The current
baseline remains provisional pending partner review; Travis-only eviction data
are its principal geographic coverage caveat.

Part 3 does not yet train a model. Its current output,
`output/part3/forecast_readiness.csv`, records the historical hex-year panels
still required for the four future displacement-proxy outcomes. The former
same-year cluster classifiers were removed from the active tree because they do
not implement the methods-report architecture; Git history preserves them.

## Implemented Evidence

The versioned feature dictionary is
[`config/feature_dictionary.csv`](config/feature_dictionary.csv). The current
source roles are:

| Domain | Current source and status | Analytical role |
| --- | --- | --- |
| Demographics | ACS 5-year estimates allocated with Census-block controls and residential-parcel support | Part 1 vulnerability |
| Rent | ACS gross-rent vintages are the citywide backbone; CoStar is coverage-limited enrichment | ACS is a Part 1 input; CoStar is a sensitivity input and coverage flag |
| Evictions | Travis County JP filing extracts are processed to current and hex-year outputs | Part 1 displacement proxy; historical panel requires Part 3 validation |
| Demolitions | Austin issued construction permits are filtered to residential demolitions | Part 1 displacement proxy; the Part 3 hex-year outcome artifact remains to be built |
| Land value | Hays, Travis, and Williamson appraisal histories | Part 1 sensitivity input; historical panel requires Part 3 validation |
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
