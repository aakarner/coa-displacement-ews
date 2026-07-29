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

ACS rent is the selected citywide rent domain. CoStar coverage and its separate
rent-pressure index remain profile and sensitivity fields; missing CoStar data
neither remove a hex nor get backfilled with zero. `config/feature_dictionary.csv`
records source coverage and missingness rules. Land-value and transaction
indices remain available for sensitivity work and should be reconsidered
before the baseline is formally adopted by project partners.

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

## In-Progress Presentation Snapshot

The locked run uses an analysis cutoff of April 1, 2026. It classifies 3,261
hexes containing 92.1% of allocated population and 93.6% of allocated housing
units. At `k = 6`, average silhouette width is 0.245, repeated-subsample
adjusted Rand index is 0.907, the smallest cluster has 178 hexes, and the
largest has 1,226.

| Cluster | Tentative interpretation | Concern | Hexes |
| --- | --- | --- | ---: |
| 1 | Demolition-Led Redevelopment | Very high - physical | 307 |
| 2 | Corporate Ownership + Vulnerability | High - structural | 426 |
| 3 | Lower-Pressure / Watch | Low | 1,226 |
| 4 | High-Cost / Lower-Vulnerability | Moderate - ambiguous | 659 |
| 5 | Eviction + Vulnerable Renters | Very high - immediate | 465 |
| 6 | Amenity-Led Emerging Pressure | High - rising | 178 |

Silhouette favors `k = 3`, while gap statistics favor larger solutions.
`k = 6` is therefore a substantive choice that preserves interpretable
variation, not a mechanical statistical optimum. About 6.6% of population in
eligible hexes is in Hays- or Williamson-dominant cells without equivalent
eviction-filing coverage; this remains a presentation caveat and a blocker to
formal baseline adoption.

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

Current automated Part 1 validation includes repeated random-subsample
stability, silhouette and gap diagnostics, exact frozen-model reproduction,
mapped geographic review, and a version/runtime lock manifest. The proposed
geographically structured holdout and expert/community interpretation remain
review steps rather than completed diagnostics.
Part 2 validation includes:

- exact baseline self-reproduction;
- distance-to-centroid distributions;
- boundary flags;
- transition tables between vintages;
- global drift monitoring.

The current baseline lock checks are written to
`output/part1/baseline_cluster_validation.csv`; the compact metrics and
assignment checksum are in `output/part1/baseline_cluster_summary.csv`.
