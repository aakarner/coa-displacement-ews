# 0009: Assign Later Vintages to Frozen Baseline Definitions

- **Status:** Accepted architecture
- **Decision period:** July 2026
- **Recorded retrospectively:** August 3, 2026

## Context

Rerunning k-means whenever new data arrive would redefine the clusters and make
apparent transitions difficult to distinguish from a changed model. Recomputing
standardization alone would also move the baseline definitions.

## Decision

Freeze the Part 1 feature names, means, standard deviations, centroids, labels,
and baseline confidence references. For routine Part 2 updates, construct the
same features for the new vintage, apply the frozen transformations, and assign
each eligible hex to its nearest frozen centroid. Do not rerun cluster selection
during routine updates.

## Consequences

Changes in membership represent movement relative to a stable baseline.
Distance, margin, out-of-envelope flags, and global drift must be monitored
because sufficiently large data or structural change may eventually require an
explicit re-baseline.

## Revisit When

Initiate a separate re-baseline review when feature definitions change, source
coverage changes materially, or assignment and drift diagnostics show that the
frozen typology no longer describes the city adequately.

## Implementation

The frozen artifact is created by `freeze_baseline_cluster_model()` in
[`R/cluster_assignment.R`](../../R/cluster_assignment.R). Part 2 remains only
partially implemented.
