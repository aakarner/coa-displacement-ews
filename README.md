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

## Pipeline

The canonical dependency graph is [`_targets.R`](_targets.R). Run:

```bash
Rscript 00_requirements.R
Rscript run_analysis.R
```

Or run a named target and everything upstream of it:

```bash
Rscript run_analysis.R part1_baseline_model
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

The seven lightweight source-manifest targets intentionally check file
metadata on every run; unchanged manifests do not rebuild downstream stages.

An existing reviewed workspace can initialize the metadata store once with
`EWS_TARGETS_ADOPT_EXISTING=true Rscript run_analysis.R`; subsequent runs should
leave that variable unset.

## Repository Structure

```text
_targets.R                    Canonical pipeline graph
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
  audits/                     QA and sensitivity analyses
  exploratory/                Non-pipeline research

config/
  feature_dictionary.csv      Feature domains, roles, and missingness rules
  forecast_outcomes.csv       Part 3 displacement-proxy outcomes
  amenity_cluster_labels_k6.csv

data/                         Local inputs and cached public extracts
output/                       Derived data artifacts
figures/                      Static and interactive outputs
```

## Current Analytical State

The selected Part 1 solution is the six-cluster amenity-augmented typology.
`output/part1/baseline_cluster_model.rds` freezes:

- the analysis vintage and H3 resolution;
- the exact feature schema;
- baseline means and standard deviations;
- the six centroids;
- substantive cluster labels;
- baseline distance and boundary thresholds.

`output/part2/baseline_fixed_cluster_assignments.csv` checks that the frozen
model exactly reproduces the original Part 1 assignments. Future feature
vintages will use the same assignment function.

Part 3 does not yet train a model. Its current output,
`output/part3/forecast_readiness.csv`, records the historical hex-year panels
still required for the four future displacement-proxy outcomes. The former
same-year cluster classifiers were removed from the active tree because they do
not implement the methods-report architecture; Git history preserves them.

## Data and Secrets

Large raw appraisal files and API caches remain local and are excluded from
Git. API credentials must be supplied through environment variables, never
committed:

```bash
export AUSTIN_DATA_API_KEY="..."
export AUSTIN_DATA_API_SECRET="..."
```

See [`data/README.md`](data/README.md) for source details,
[`WORKFLOW.md`](WORKFLOW.md) for the analytical graph,
[`CLUSTER_METHODOLOGY.md`](CLUSTER_METHODOLOGY.md) for Parts 1 and 2, and
[`UNIT_COUNT_MODELING.md`](UNIT_COUNT_MODELING.md) for the parcel-unit hierarchy.
