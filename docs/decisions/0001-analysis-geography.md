# 0001: Use Resolution-9 H3 Hexes

- **Status:** Accepted
- **Decision period:** Before July 2026
- **Recorded retrospectively:** August 3, 2026

## Context

The EWS combines parcel, permit, service-request, eviction, and establishment
locations with Census estimates published for larger areas. A common geography
is required, but larger zones would conceal local variation in point-based
signals and parcel conditions.

## Decision

Use H3 resolution-9 cells as the common analytical grid. Retain cells needed to
represent the project geography, then apply residential eligibility and feature
completeness rules rather than assuming every grid cell must be classified.

## Consequences

The fine grid preserves local variation and supports repeatable spatial joins.
It also creates sparse denominators and makes ACS allocation more uncertain at
the cell level. Unclassified and boundary cells must therefore remain explicit.

## Revisit When

Reconsider the resolution if sensitivity at the next larger H3 level materially
improves reliability without erasing policy-relevant patterns, or when the
project boundary/grid construction is repaired or expanded.

## Implementation

The current value is `h3_resolution = 9L` in
[`R/analysis_config.R`](../../R/analysis_config.R). Grid construction is in
[`01_create_hex_grid.R`](../../01_create_hex_grid.R).
