# Part 2 retrospective cluster comparison

## Two different questions

This proof of concept compares April 1, 2025 with April 1, 2026 using the
same commonly covered cells and seven complete, fixed-recipe indices. The
current sample count is reported in the regenerated analysis audit. It answers
two distinct questions:

1. **Do places change category when the category definitions stay fixed?**
   Fit the 2025 baseline, then assign 2026 observations to those same centers.
   This is the primary temporal comparison.
2. **Would the categories themselves change if we fitted them again?**
   Separately refit the 2026 centers and compare that partition with the fixed
   2026 assignments. This is a structural-stability sensitivity, not a
   replacement for the primary comparison.

This reconstructed 2025 baseline is distinct from the existing Part 1 model.
It uses harmonized sources and a different common sample. It is not a
reproduction of Part 1 or a new prediction model; Part 3 ML remains paused.

## Fixed inputs and geometry

The [paired matrix](historical-feature-matrix.md) has already frozen the earlier
component clipping and 0–100 scoring bounds. This analysis does not rebuild or
rescale those underlying indices. It estimates each index's mean and sample
standard deviation **on the 2025 common sample**, then applies those same
parameters to both dates:

```text
standardized index = (index − 2025 mean) / 2025 sample standard deviation
```

Every hex receives equal weight in clustering. Residential units describe the
amount of housing in each group but are not clustering weights. All seven
indices are equally standardized. Even the separate 2026 refit retains the
2025 scaling: only the centers move, not the relative weighting of inputs.

The fixed grid, current-city footprint, unit support, source-coverage exclusions
and adjacent overlapping ACS releases remain as documented in the data methods.
The result describes a covered subset of Austin, not the entire city.

## Model fitting and labels

The proof of concept retains **k=7**, consistent with the selected Part 1
specification, without claiming that seven is newly optimized for this sample.
At each date, the script runs 20 independently seeded Lloyd k-means fits, each
with 100 starts and a 500-iteration limit, and retains the smallest within-cluster
sum of squared distances. Tied best runs select the first seed deterministically.
The complete seed list, fits and optimization checks are saved.

Labels **C1–C7 are neutral identifiers**, not an ordering of risk, severity or
expected displacement. The old Part 1 names and colors are not imported.
Profiles in original 0–100 units and standardized units support subsequent
substantive review; technical clustering alone does not validate a label.

The later refit initially has arbitrary labels. An exact one-to-one matching
maximizes shared-cell overlap with the **coeval fixed-2026 assignments**.
It does not maximize staying in the 2025 groups. Tied optimal permutations are
reported and resolved lexicographically. Matched agreement is therefore the
best agreement under relabeling, not a separate validation statistic.

## Movement, structural change and uncertainty flags

Later observations are assigned to their nearest 2025 center in Euclidean
standardized-feature space. The baseline must reproduce its own training
assignments exactly. The outputs retain all 7,027 audit cells; excluded cells
have missing assignments and their original exclusion reason.

The primary transition table compares baseline 2025 with fixed 2026. A separate
table compares fixed 2026 with aligned refit 2026. A third, supplementary table
compares baseline 2025 with aligned refit 2026, combining both kinds of change;
it must not be interpreted as movement against fixed definitions.

Agreement is summarized by the same-label share, adjusted Rand index (ARI),
cluster sizes and cluster-specific overlap/Jaccard. ARI is label-invariant:
one means identical partitions; values around zero indicate agreement near its
chance-adjusted reference, not zero shared labels. No p-values or independence
claims are attached to these descriptive comparisons.

For each fixed assignment, separation margin is
`1 − nearest distance / second-nearest distance`. A small value means the cell
is near a category boundary; it is **not a probability**. Low-margin flags use
the earlier sample's 10th percentile. A separate flag marks distances exceeding
the earlier assigned cluster's 95th-percentile distance. These are baseline
reference flags, not calibrated confidence or displacement-risk thresholds.

For changing cells, the analysis records which standardized features changed
most. It also decomposes the change in squared-distance preference for the new
versus old center into feature contributions. That arithmetic explains a model
boundary crossing; it does not identify a cause of neighborhood displacement.

## Conditional robustness checks

Optimizer checks compare the 20 full-sample seeded fits at each date with that
date's best fit. Sampling checks use identical held-out cells at both dates:

- **50 random-cell replicates:** about 80% fit / 20% held out.
- **20 spatial replicates:** hold out whole resolution-7 H3 parent regions,
  approximately 20% of the eligible cells.

Each replicate fits centers on its training cells. Label correspondence is
learned using training cells only and then applied to held-out cells. Reported
held-out agreement, ARI and cluster recovery are conditional on the frozen
feature scoring and 2025 standardization. They test robustness to changing
which places inform the centers, **not out-of-sample predictive accuracy** or
the probability that an assignment is true.

Resampled earlier models also assign both dates, allowing a check of how often
each held-out cell's exact fixed transition, or simply switch/stay status,
is reproduced. Per-cell denominators are retained. A cell never held out in a
scheme has zero evaluations and missing recovery, not zero robustness. Spatial
checks help reveal instability that random removal of neighboring cells can
miss, but 20 coarse-block replicates do not exhaust geographic uncertainty.

## Interpretation safeguards

The corrected measurement contract requires every component at both dates;
no included cell changes its effective index definition or weights. Eviction
pressure no longer uses expanding-history recent share or percentage change.
Selected 311 also uses signed rate differences instead of percentage changes.
Rent retains level, growth and acceleration through the reliable full-series
BG/tract hierarchy, with geographic level fixed across both snapshots.

Sparse relative scores require particular care. Cluster profiles include the
share of cells with positive observed recent mapped filings, demolition permits,
selected requests, and exposure to nearby amenity openings. A high composite
does not imply that every member has the named event. Amenity catchments overlap:
summing per-hex amenity counts counts **hex–event exposure links**, not unique
openings. Eviction, demolition and selected-311 counts have disjoint assigned
hex locations in this common sample, subject to their documented source limits.

Neither changes in clusters nor improved statistical stability establish that
individual residents were displaced, that corporate ownership caused an event,
or that the typology is ready for operational risk prediction. A separate,
explicitly qualitative Low/Moderate/High/Very high concern interpretation can
be attached after reviewing the corrected cluster profiles. The interpretation
configuration is pinned to the baseline-centroid hash so that a new fit cannot
silently inherit old numeric-ID meanings. Tier steps are ordinal, not equal
amounts of risk. Community interpretation and calibration remain separate.

## Reproduction and outputs

With the verified paired data present, run from the repository root:

```sh
Rscript tests/test_part2_clusters.R
Rscript scripts/part2/analyze_cluster_comparison.R
Rscript tests/test_part2_cluster_outputs.R
Rscript scripts/part2/visualize_cluster_comparison.R
Rscript tests/test_part2_cluster_figures.R
Rscript scripts/part2/analyze_cluster_concern.R
Rscript tests/test_part2_concern_outputs.R
```

Analysis artifacts live under `output/part2/clusters/`, including the frozen
model/scaling/thresholds, full-grid assignments, transition and profile tables,
alignment, per-feature contributions, event prevalence, robustness replicates,
per-cell diagnostics and checksum manifest. Figures and their independent
manifest live under `figures/part2/`. The qualitative interpretation and
small-count/unit-support diagnostics live under `output/part2/interpretation/`.
Existing Part1/Part3 outputs and non-Part2 figures are preserved and checked;
the failed Part2 run is overwritten in place, not archived. The separate analysis
runner does not alter the routine production graph or commit/push changes.
