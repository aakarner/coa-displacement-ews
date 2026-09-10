# Parts 1 and 2: Cluster Methodology

## Purpose

Part 1 discovers a baseline typology of current displacement pressure and
vulnerability. Part 2 tracks change by assigning later observations to those
same definitions. Its primary fixed-assignment comparison does not rerun
k-means. The separate retrospective April 2025/2026 proof of concept also
refits the later centers as a **structural-stability sensitivity**, without
replacing the fixed-assignment results. Its harmonized common-sample baseline
is distinct from the current Part 1 model. Both now use the same corrected
component recipes, but Part 1 uses current-only eligibility and 2026 reference
scales; Part 2 requires paired support and freezes its 2025 scales. See the
[current measurement contract](current-measurement.md) and
[historical comparison methods](historical-cluster-comparison.md).

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

### Pressure Variables

Six of the seven cluster inputs describe displacement pressure or an earlier
smoke signal. Each is a unitless index from 0 to 100, with a larger value
indicating more of the conditions represented by that index. The component
measures are placed on a common scale before they are averaged, so a variable
measured in dollars does not outweigh one measured as an event rate. The six
completed indices are then standardized equally for clustering.

| Cluster input | Role | Key source and vintage | Construction |
| --- | --- | --- | --- |
| **Citywide rent pressure** (`rent_pressure_citywide_index`) | Displacement proxy | ACS 5-year median gross rent for Hays, Travis, and Williamson Counties, using vintages ending in 2014, 2019, and 2024 | Equal thirds: inflation-adjusted 2024 median rent, annualized real log growth from 2019 to 2024, and acceleration relative to 2014-2019 growth. Require all three reliable estimates at the dominant residential block group; otherwise all three at the tract; otherwise the index is missing. Never mix geography levels or average block-group medians. |
| **Residential demolition pressure** (`demolition_pressure_index`) | Displacement proxy | City of Austin issued construction permits through April 1, 2026 | Keeps issued permits classified as residential demolition and compares April 2, 2024-April 1, 2026 with the preceding 24 months. Combines recent demolition density, positive change between the two periods, and recent density of permits whose description identifies a total demolition. An issued permit indicates authorized activity, not necessarily a completed demolition. |
| **Eviction pressure** (`eviction_pressure_index`) | Displacement proxy | Travis County and Williamson JP1/JP2 filing records through April 1, 2026 | Compares April 2, 2025-April 1, 2026 with the preceding 12 months. Equal halves: recent mapped filings per 100 fixed promoted units and signed recent-minus-previous rate change. Require at least 20 units and observed source coverage. Hays and Williamson JP3 gaps remain missing, not zero. Percentage change and the expanding-history recent share are not scored. |
| **Selected 311 pressure** (`sr_311_pressure_index`) | Smoke signal | Austin 311 records from January 1, 2020 through April 1, 2026 | Uses only the three versioned code-officer intake descriptions in `config/311_smoke_signal_types.csv`, not all 311 activity. Equal thirds: latest-12-month requests per 100 fixed units, requests per square kilometer, and signed rate change from the preceding 12 months. Rate and density deliberately give current activity two-thirds weight. A request records reported concern, not a verified violation. |
| **Corporate ownership pressure** (`ownership_pressure_index`) | Smoke signal | Reviewed 2025 county appraisal ownership snapshots for Hays, Travis, and Williamson Counties | Equal thirds: corporate share of observed units, corporate units per whole-hex square kilometer, and financialized share of observed parcels. Use parcels with both classifications known in 2025; require at least 20 observed units and 95% unit and parcel coverage. No 2024 gate in Part 1. Corporate ownership does not necessarily identify a large institutional investor. |
| **Amenity change pressure** (`amenity_change_index`) | Smoke signal | Texas Comptroller permitted sales-tax locations, with mixed-beverage and Austin food-inspection records used for corroboration; events are truncated at April 1, 2026 | Compares openings during October 2, 2024-April 1, 2026 with the preceding 18 months. Measures distance-weighted exposure within 800 meters for cafes, full-service restaurants, and drinking places. Each category combines its recent opening level with positive change, and the three category scores receive equal weight. The index measures selected openings rather than overall amenity density. |

The seventh input, `demographic_vulnerability_index`, is intentionally not
listed as a pressure variable. It describes who may be more vulnerable if
pressure occurs. It gives equal weight to lower household income, renter share,
poverty, rent burden, and lower educational attainment. This distinction lets
the clusters identify places where similar market or event pressure may have
different consequences for current residents.

Every index requires its complete fixed recipe, including all five vulnerability
components. Missing components never redistribute weights. Signed event changes
are centered at 50 for zero change, so a composite is not a probability or a
literal zero-to-maximum risk scale. Part 1 requires current city-center membership,
20 fixed units, current evidence coverage and all seven complete indices.

## Selecting the Solution

`scripts/part1/fit_baseline_clusters.R` evaluates:

- average silhouette width;
- gap statistics;
- cluster size balance;
- repeated 80% subsample stability;
- geographic coherence through mapped assignments.

When the baseline is reconsidered, the optional review in
`scripts/reviews/part1_cluster_selection.R` adds matched 20% random and spatially
blocked holdout tests. It runs through `_targets_review.R`, not the routine
production graph.

The blocked tests hold out whole H3 parent regions at resolutions 8 and 7. For
each replicate, feature means, standard deviations, and k-means centroids are
estimated from the remaining hexes only. Held-out hexes are then assigned to
their nearest estimated centroid and compared with their full-sample cluster.
This is more conservative than random thinning because neighboring hexes are
kept together rather than split between training and evaluation samples.

The audit also reports assignment margins, cluster-specific recovery, and the
share of hexes in each cluster with an observed eviction, demolition, selected
311, ownership-change, or amenity-opening event. These checks matter because a
high composite index can reflect growth or relative timing even when a current
raw count is zero. They prevent a cluster's tentative name from implying that
every member has the named event.

No single diagnostic mechanically selects k. The current shared solution uses
`k = 7`, configured in `R/analysis_config.R`, with tentative display labels in
`config/amenity_cluster_labels.csv`. The decision rationale is in
[`docs/decisions/0008-select-seven-clusters.md`](../decisions/0008-select-seven-clusters.md),
and the current corrected fit is documented in the
[September harmonization audit](../audits/part1-harmonized-measurement-2026-09.md).
The [August selection audit](../audits/part1-cluster-selection-2026-08.md)
is superseded for current numerical results. Its spatial holdouts have not been
rerun under the corrected specification; seven clusters remain provisional.
Current metrics are generated in `output/part1/baseline_cluster_summary.csv`
rather than copied into this evergreen methods document.

## Neighborhood Summaries

The downstream neighborhood summary does not refit or reinterpret the cluster
model. Each resolution-9 H3 cell is assigned to the City of Austin Neighborhood
Reporting Area containing its center. Within each reporting area, the pipeline
sums allocated population and promoted residential units for every cluster and
for unclassified cells.

The thematic neighborhood map uses the cluster containing the largest share of
classified allocated population. A neighborhood is marked as a **majority**
only when that share exceeds 50 percent; otherwise it is a **plurality**.
Unclassified population is excluded from selecting the plurality but remains
visible in the composition and coverage tables. Housing-unit plurality is
reported separately so population and housing summaries are not assumed to
agree.

## Frozen Baseline Artifact

`output/part1/baseline_cluster_model.rds` contains:

- the baseline cutoff and H3 resolution;
- ordered feature names;
- baseline means and standard deviations;
- current component reference scales and a pinned measurement manifest;
- selected centroids;
- cluster labels and concern levels;
- a centroid- and label-hash-pinned interpretation review;
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

Routine Part 1 validation includes repeated random-subsample stability,
silhouette and gap diagnostics, exact frozen-model reproduction, mapped
geographic review, and a version/runtime manifest. Spatially blocked holdouts,
cluster-specific recovery, sparse-signal prevalence, and candidate crosswalks
are optional re-baseline reviews run through `_targets_review.R`. Expert and
community interpretation remains a review step rather than a completed
technical diagnostic.

Part 2 validation includes:

- exact baseline self-reproduction;
- distance-to-centroid distributions;
- boundary flags;
- transition tables between vintages;
- global drift monitoring.

The current baseline checks are written to
`output/part1/baseline_cluster_validation.csv`; the compact metrics and
assignment checksum are in `output/part1/baseline_cluster_summary.csv`.
