# 0011: Use Neighborhood Reporting Areas for Cluster Summaries

- **Status:** Accepted
- **Decision date:** August 18, 2026

## Context

The EWS needs population- and housing-unit-weighted summaries of Part 1 cluster
membership for recognizable neighborhoods. The City publishes both
Neighborhood Reporting Areas and Neighborhood Planning Areas, but the two
layers serve different purposes and cover different portions of Austin.

## Decision

Use City of Austin Neighborhood Reporting Areas for neighborhood summaries.
Assign each H3 cell to the reporting area containing its center. Report every
cluster plus unclassified population and housing units. Use the cluster with
the largest share of classified population for the thematic map, distinguishing
a share above 50 percent as a majority from a smaller plurality.

## Consequences

The 103 Reporting Areas are uniquely named and essentially non-overlapping.
Using the current full-purpose and EWS surfaces, they contain about 98.6 percent
of classified population and housing units. Of those 103 areas, 102 intersect
the current full-purpose boundary; Whisper Valley is outside it and is therefore
not shown in the current map. Neighborhood Planning Areas cover only about 51.6
percent of classified population and 58.0 percent of classified housing units
and contain repeated subdistrict records, making them unsuitable as the primary
citywide reporting geography.

Hex-center assignment is transparent and preserves each H3 result without
introducing another areal interpolation. Cells outside the reporting layer
remain explicit rather than being assigned to the nearest neighborhood.

## Revisit When

Revisit when the City updates the Reporting Area geography, when the analytical
boundary changes, or if uncovered population or housing support grows
materially.

## Implementation

The source refresh is
[`scripts/data/download_neighborhood_reporting_areas.R`](../../scripts/data/download_neighborhood_reporting_areas.R).
Summary and map construction are in
[`scripts/part1/summarize_neighborhood_clusters.R`](../../scripts/part1/summarize_neighborhood_clusters.R).
