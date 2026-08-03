# 0007: Use ACS Rent Citywide and CoStar Only Where Observed

- **Status:** Accepted for Part 1
- **Decision period:** July 2026
- **Recorded retrospectively:** August 3, 2026

## Context

CoStar provides useful and potentially timely multifamily rent information but
does not cover the full housing market or every EWS hex. Treating missing
CoStar coverage as zero or requiring it for cluster eligibility would
systematically remove places without qualifying developments.

## Decision

Construct the citywide Part 1 rent domain from ACS five-year median gross rent
vintages ending in 2014, 2019, and 2024. Retain `costar_present` and the separate
CoStar rent-pressure measure as coverage and sensitivity fields. Do not impute
CoStar absence as zero and do not use it to exclude a hex from the citywide
cluster sample.

## Consequences

Part 1 preserves citywide coverage but relies on lagged survey estimates for
rent. CoStar remains valuable for profiling covered multifamily areas and for a
future real-time update layer whose interpretation explicitly conditions on
coverage.

## Revisit When

Revisit when a new ACS vintage is available, CoStar coverage changes, or a
formal conditional update strategy is validated without conflating data
availability with rent pressure.

## Implementation

Current rent construction is documented in
[`docs/methods/cluster-methodology.md`](../methods/cluster-methodology.md); field
roles and missingness are in
[`config/feature_dictionary.csv`](../../config/feature_dictionary.csv).
