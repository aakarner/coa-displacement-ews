# Part 1 Cluster-Selection Audit: August 2026

## Scope

This dated audit records the evidence used to supersede the six-cluster
in-progress checkpoint with the amenity-augmented seven-cluster Part 1
baseline. It is decision evidence, not a production pipeline dependency.

The analysis used 3,250 eligible hexes and seven equally standardized domain
indices with an April 1, 2026 cutoff. Candidate solutions covered k = 2 through
12, with focused substantive review of k = 5 through 8.

## Review Design

The audit compared silhouette, gap statistic, cluster size, existing repeated
80% subsampling, assignment margins, sparse-event prevalence, spatial behavior,
and candidate profiles. It added 100 matched 20% holdouts for each of:

- randomly selected hexes;
- complete H3 resolution-8 parent blocks; and
- complete H3 resolution-7 parent blocks.

For each holdout, scaling and centroids were estimated from the remaining
geography. Held-out hexes were assigned to the nearest estimated centroid and
compared with the full-sample solution after optimal label matching.

## Stability Results for k = 7

| Holdout design | Median ARI | 10th percentile ARI | Worst ARI | Median matched agreement |
| --- | ---: | ---: | ---: | ---: |
| Random hexes | 0.975 | 0.953 | 0.919 | 98.9% |
| H3 resolution-8 blocks | 0.964 | 0.936 | 0.874 | 98.5% |
| H3 resolution-7 blocks | 0.949 | 0.906 | 0.854 | 97.8% |

All seven clusters appeared in every blocked replicate. The smallest
amenity-led cluster was the least stable individual group; under resolution-7
blocking its 10th-percentile Jaccard overlap was 0.772 and recall was 0.788.

## Selection Evidence

At k = 7, average silhouette was 0.254, existing repeated-subsample ARI was
0.969, the smallest cluster contained 159 hexes, and 11.7% of assignments had a
nearest-versus-second-nearest margin below 0.10. The corresponding k = 6
silhouette and repeated-subsample ARI were 0.245 and 0.916. Its resolution-7
blocked 10th-percentile ARI was 0.773, compared with 0.906 at k = 7.

The highest-eviction k = 7 cluster contained a positive latest-12-month filing
count in every member hex. At k = 6, 17.2% of the highest-eviction cluster had
zero filings because the solution combined eviction-dominant and high-311
profiles. The k = 8 solution did not add a comparable improvement: silhouette
fell to 0.201 and assignment margins weakened.

Silhouette alone favored the much coarser k = 3, while the gap statistic kept
increasing at larger k. Seven clusters were therefore selected through joint
statistical and substantive review, not a mechanical rule.

## Selected Snapshot

| Cluster | Tentative interpretation | Concern category | Hexes |
| --- | --- | --- | ---: |
| 1 | Lower Current Pressure | Low | 1,138 |
| 2 | High-Cost / Lower-Vulnerability | Low | 627 |
| 3 | Amenity-Led Emerging Pressure | Moderate | 159 |
| 4 | Corporate Ownership + Vulnerability | Moderate | 396 |
| 5 | 311 + Vulnerable Renters | Moderate | 369 |
| 6 | Demolition-Led Redevelopment | High | 285 |
| 7 | Eviction + Vulnerable Renters | Very high | 276 |

The selected sample represented 92.0% of allocated population and 93.6% of
allocated housing units. The categories are interpretive and are not estimated
probabilities.

## Caveats

Eviction filings currently cover Travis County only. Approximately 6.6% of the
population in cluster-eligible hexes is in Hays- or Williamson-dominant cells
without equivalent filing coverage. Spatial holdouts test geographic recovery
of the current feature structure; they do not establish temporal stability or
validate future-year assignments.

## Reproduction

The optional review pipeline is separate from routine production execution:

```r
targets::tar_make(
  script = "_targets_review.R",
  store = "_targets_review"
)
```

Detailed generated tables are written under `output/part1/`; figures use the
`03f_cluster_selection_` prefix. The implementation was committed as `21e1845`
and discussed in [GitHub issue #10](https://github.com/aakarner/coa-displacement-ews/issues/10).
