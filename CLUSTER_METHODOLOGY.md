# Parts 1 and 2: Cluster Methodology

## Purpose

Part 1 discovers a baseline typology of current displacement pressure and
vulnerability. Part 2 tracks change by assigning later observations to those
same definitions. Part 2 does not rerun k-means.

## Part 1 Inputs

The current selected specification contains seven equally scaled conceptual
indices:

- citywide rent pressure;
- demographic vulnerability;
- residential demolition pressure;
- eviction pressure;
- 311 pressure;
- corporate ownership pressure;
- amenity change pressure.

CoStar availability is represented within the rent construction rather than
used to remove non-CoStar hexes. `config/feature_dictionary.csv` records source
coverage and missingness rules. Land-value and transaction indices remain
available for sensitivity work and should be reconsidered before the baseline
is formally adopted by project partners.

## Selecting the Solution

`scripts/part1/fit_baseline_clusters.R` evaluates:

- average silhouette width;
- gap statistics;
- cluster size balance;
- repeated 80% subsample stability;
- geographic coherence through mapped assignments.

These statistics inform, but do not mechanically replace, substantive review.
The current shared solution uses `k = 6`, configured in
`R/analysis_config.R`, with tentative labels in
`config/amenity_cluster_labels_k6.csv`.

## Frozen Baseline Artifact

`output/part1/baseline_cluster_model.rds` contains:

- the baseline cutoff and H3 resolution;
- ordered feature names;
- baseline means and standard deviations;
- selected centroids;
- cluster labels and concern levels;
- the distance metric;
- cluster-specific 95th-percentile baseline distances;
- the baseline 10th-percentile nearest-versus-second-nearest margin.

The quantiles are configuration values, not universal uncertainty thresholds.
They provide transparent baseline reference points that can be reviewed before
operational deployment.

## Fixed Assignment

For each later vintage, feature `j` is transformed using its baseline mean and
standard deviation:

```text
z_j(t) = (x_j(t) - baseline_mean_j) / baseline_sd_j
```

The hex is assigned to the centroid with minimum Euclidean distance. Assignment
confidence is:

```text
1 - nearest_distance / second_nearest_distance
```

A value near zero indicates that the hex lies close to a boundary. A separate
flag identifies observations farther from their assigned centroid than 95% of
that cluster's baseline members.

Assignment files retain every H3 cell. Cells outside the residential
eligibility rule or missing a required cluster feature receive an explicit
`assignment_status` instead of disappearing from the output.

## Validation

Part 1 validation includes spatial holdout stability, silhouette and gap
diagnostics, mapped geographic review, and expert/community interpretation.
Part 2 validation includes:

- exact baseline self-reproduction;
- distance-to-centroid distributions;
- boundary flags;
- transition tables between vintages;
- global drift monitoring.

The current baseline self-check is written to
`output/part2/baseline_fixed_cluster_assignment_summary.csv`.
