# 0004: Use Residential Dasymetric Allocation for ACS Estimates

- **Status:** Accepted
- **Decision period:** July 2026
- **Recorded retrospectively:** August 3, 2026

## Context

Most ACS demographics are published for block groups or tracts rather than EWS
hexes. Straight land-area allocation would place residents and housing on
roads, parks, industrial land, and other nonresidential areas.

## Decision

Use 2020 Census blocks as controls within ACS source geographies. Within each
block, distribute additive estimates among hexes using appraisal-reported
residential floor area attached to parcel points. Where positive floor area is
unavailable, use promoted units and then residential parcel count. Use a Census
block point only when the block contains no residential parcel point.

Use the Census block population pattern for population variables and its
housing pattern for housing, tenure, rent burden, and occupied-housing
variables. Assign non-additive medians from the dominant residential block
group, with tract fallback rather than averaging medians.

## Consequences

Allocated values preserve source-area totals and better follow residential
development, but they remain modeled small-area estimates. Floor-area, unit,
and parcel fallbacks are scaled into a common allocation weight; the scaling is
not an assumed dwelling size.

## Revisit When

Revisit when building footprints, address-level housing inventories, or better
occupied-unit locations become available, and when group-quarters or mixed-use
sensitivity analyses indicate material bias.

## Implementation

See the ACS allocation section of [`data/README.md`](../../data/README.md) and
`scripts/data/acs_demographics.R`.
