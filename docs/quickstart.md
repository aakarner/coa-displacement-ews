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
export EWS_AMENITY_CLUSTER_K="6"
export EWS_BASELINE_CLUSTER_SPECIFICATION="amenity_augmented"
```

For the Austin 311 source:

```bash
export AUSTIN_DATA_API_KEY="..."
export AUSTIN_DATA_API_SECRET="..."
```

## 3. Inspect the Pipeline

```r
targets::tar_manifest(fields = c(name, command))
targets::tar_visnetwork()
```

The graph is divided into source processing, shared feature construction,
Part 1 baseline estimation, Part 2 fixed-cluster assignment, and Part 3
forecast preparation.

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
Rscript run_analysis.R part1_visualizations
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

- `output/hex_features.rds`: current shared feature surface.
- `output/amenity_cluster_sensitivity.rds`: Part 1 cluster diagnostics.
- `output/part1/baseline_cluster_model.rds`: frozen Part 1 model.
- `output/part2/baseline_fixed_cluster_assignments.csv`: Part 2 self-check.
- `output/part3/forecast_readiness.csv`: historical outcome-panel status.
- `figures/03e_amenity_clusters_interactive.html`: interactive baseline map.

The processors under `scripts/data/` remain runnable for focused debugging, but
normal analysis runs should go through `_targets.R` so their dependencies and
vintages are recorded.
