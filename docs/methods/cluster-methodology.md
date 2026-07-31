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

### Pressure Variables

Six of the seven cluster inputs describe displacement pressure or an earlier
smoke signal. Each is a unitless index from 0 to 100, with a larger value
indicating more of the conditions represented by that index. The component
measures are placed on a common scale before they are averaged, so a variable
measured in dollars does not outweigh one measured as an event rate. The six
completed indices are then standardized equally for clustering.

| Cluster input | Role | Key source and vintage | Construction |
| --- | --- | --- | --- |
| **Citywide rent pressure** (`rent_pressure_citywide_index`) | Displacement proxy | ACS 5-year median gross rent for Hays, Travis, and Williamson Counties, using vintages ending in 2014, 2019, and 2024 | Combines inflation-adjusted 2024 median rent, annualized real growth from 2019 to 2024, and acceleration relative to 2014-2019 growth. Growth and acceleration are included only when all three vintages meet the ACS reliability rule. Each hex receives the median from its dominant residential block group, with tract fallback; block-group medians are never averaged. |
| **Residential demolition pressure** (`demolition_pressure_index`) | Displacement proxy | City of Austin issued construction permits through April 1, 2026 | Keeps issued permits classified as residential demolition and compares April 2, 2024-April 1, 2026 with the preceding 24 months. Combines recent demolition density, positive change between the two periods, and recent density of permits whose description identifies a total demolition. An issued permit indicates authorized activity, not necessarily a completed demolition. |
| **Eviction pressure** (`eviction_pressure_index`) | Displacement proxy | Travis County Justice of the Peace filing records through April 1, 2026 | Compares April 2, 2025-April 1, 2026 with the preceding 12 months. Combines recent unique filings per 100 promoted residential units, percentage change between periods, and the share of all observed filings occurring recently. Rates require at least 20 residential units. Equivalent Hays and Williamson filings are not yet integrated. |
| **Selected 311 pressure** (`sr_311_pressure_index`) | Smoke signal | Austin 311 records from January 1, 2020 through April 1, 2026 | Uses only the three versioned code-officer intake descriptions in `config/311_smoke_signal_types.csv`, not all 311 activity. Combines selected requests during the latest 12 months per 100 residential units, requests per square kilometer, and change from the preceding 12 months. A request records reported concern, not a verified violation. |
| **Corporate ownership pressure** (`ownership_pressure_index`) | Smoke signal | Current county appraisal ownership records, primarily the 2025 certified or current rolls for Hays, Travis, and Williamson Counties | Combines the percentage of residential units owned by corporate entities, corporate-owned residential units per square kilometer, and the percentage of residential parcels associated with financialized owners. Corporate ownership means ownership by a company or other legal entity; it does not necessarily identify a large institutional investor. |
| **Amenity change pressure** (`amenity_change_index`) | Smoke signal | Texas Comptroller permitted sales-tax locations, with mixed-beverage and Austin food-inspection records used for corroboration; events are truncated at April 1, 2026 | Compares openings during October 2, 2024-April 1, 2026 with the preceding 18 months. Measures distance-weighted exposure within 800 meters for cafes, full-service restaurants, and drinking places. Each category combines its recent opening level with positive change, and the three category scores receive equal weight. The index measures selected openings rather than overall amenity density. |

The seventh input, `demographic_vulnerability_index`, is intentionally not
listed as a pressure variable. It describes who may be more vulnerable if
pressure occurs. It gives equal weight to lower household income, renter share,
poverty, rent burden, and lower educational attainment. This distinction lets
the clusters identify places where similar market or event pressure may have
different consequences for current residents.

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

The locked run uses an analysis cutoff of April 1, 2026. It classifies 3,250
hexes containing 92.0% of allocated population and 93.6% of allocated housing
units. At `k = 6`, average silhouette width is 0.245, repeated-subsample
adjusted Rand index is 0.916, the smallest cluster has 177 hexes, and the
largest has 1,216. These results use the July 31 City land-use validation
repair to the promoted residential unit surface.

| Cluster | Tentative interpretation | Concern | Hexes |
| --- | --- | --- | ---: |
| 1 | Lower-Pressure / Watch | Low | 1,216 |
| 2 | High-Cost / Lower-Vulnerability | Moderate - ambiguous | 658 |
| 3 | Amenity-Led Emerging Pressure | High - rising | 177 |
| 4 | Corporate Ownership + Vulnerability | High - structural | 422 |
| 5 | Demolition-Led Redevelopment | Very high - physical | 307 |
| 6 | Eviction + Vulnerable Renters | Very high - immediate | 470 |

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
