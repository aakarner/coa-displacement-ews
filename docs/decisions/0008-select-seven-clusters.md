# 0008: Select Seven Part 1 Clusters

- **Status:** Provisional baseline
- **Decision period:** August 1, 2026
- **Recorded retrospectively:** August 3, 2026
- **Supersedes:** Six-cluster in-progress presentation checkpoint

## Context

The six-cluster checkpoint combined an eviction-dominant group with a distinct
high-311 and high-vulnerability group whose observed eviction filings were
often zero. Random subsampling also could overstate stability because nearby
hexes share spatial structure.

## Decision

Use the amenity-augmented seven-cluster solution for the current Part 1
baseline. Treat the labels and low-to-very-high concern categories as
interpretive descriptions rather than probabilities.

## Consequences

The selected solution distinguishes `Eviction + Vulnerable Renters` from
`311 + Vulnerable Renters`. It improves detailed-solution silhouette,
subsample stability, spatially blocked lower-tail performance, and assignment
confidence relative to the six-cluster checkpoint. It remains provisional
pending partner interpretation and the stated geographic coverage caveats.

## Revisit When

Revisit during a formal Part 1 re-baseline, after major feature or source
changes, or if spatially blocked stability, cluster size, assignment confidence,
or substantive interpretation deteriorates.

## Evidence and Implementation

See [`docs/audits/part1-cluster-selection-2026-08.md`](../audits/part1-cluster-selection-2026-08.md)
and [GitHub issue #10](https://github.com/aakarner/coa-displacement-ews/issues/10).
The current value is in [`R/analysis_config.R`](../../R/analysis_config.R), and
display labels are in
[`config/amenity_cluster_labels.csv`](../../config/amenity_cluster_labels.csv).
