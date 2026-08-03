# 0005: Use One Composite Input Per Part 1 Domain

- **Status:** Accepted
- **Decision period:** July-August 2026
- **Recorded retrospectively:** August 3, 2026

## Context

The available sources produce many correlated component measures. Sending all
components directly into k-means would give domains with more available fields
more influence and would blur the distinction between observed displacement,
earlier smoke signals, and resident vulnerability.

## Decision

Represent each selected conceptual domain once through a 0-100 composite index,
then standardize the seven domain indices equally for clustering. The current
domains are citywide rent pressure, demographic vulnerability, residential
demolition pressure, eviction pressure, selected 311 pressure, corporate
ownership pressure, and amenity change pressure.

Treat rent, demolition, and eviction as displacement proxies; 311, corporate
ownership, and amenity change as smoke signals; and demographics as
vulnerability. Retain component and sensitivity variables in the feature table
without allowing them to enter the distance calculation separately.

## Consequences

No domain is overweighted simply because it has more component measures. The
composites are relative indices, not probabilities, and cluster interpretation
must consider both the index and its underlying raw-event prevalence.

## Revisit When

Revisit when land-value, transaction, corporate-ownership change, race and
ethnicity, or another proposed domain has complete enough coverage and a clear
conceptual role for a formal sensitivity test.

## Implementation

The machine-readable feature roles are in
[`config/feature_dictionary.csv`](../../config/feature_dictionary.csv). Current
construction is documented in
[`docs/methods/cluster-methodology.md`](../methods/cluster-methodology.md).
