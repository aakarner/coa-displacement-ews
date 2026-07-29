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
  311_smoke_signal_types.csv  Exact 311 descriptions used in Part 1
  feature_dictionary.csv      Feature domains, roles, and missingness rules
  forecast_outcomes.csv       Part 3 displacement-proxy outcomes
  amenity_cluster_labels_k6.csv

docs/                         Current guidance and dated audit reports
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

`output/part1/baseline_cluster_validation.csv` checks the complete Part 1
feature contract, labels, scaling, centroids, population coverage, and exact
frozen-model reassignment. The accompanying summary, canonical assignments,
and lock manifest preserve the reviewed presentation run. Future feature
vintages will use the same assignment function.

The locked in-progress run classifies 3,261 hexes, representing 92.1% of
allocated population and 93.6% of allocated housing units. At the substantively
selected `k = 6`, average silhouette width is 0.245 and repeated-subsample
stability is 0.907. These are presentation results, not a formally adopted
baseline; the Travis-only eviction source remains the principal geographic
coverage caveat.

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
