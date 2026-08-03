# 0010: Exclude Unsupported Units on Exclusively Nonresidential City Land

- **Status:** Accepted
- **Decision period:** July 31, 2026
- **Recorded retrospectively:** August 3, 2026

## Context

Some projects entered the multifamily candidate universe through zoning or
another broad parcel signal even though neither appraisal current-use evidence
nor the City Land Use Inventory indicated residential use. Direct counts and
appraisal multifamily codes sometimes disagreed with the City inventory, so the
City classification could not be treated as ground truth in every case.

## Decision

Before promotion, exclude a modeled or fallback project when an exact parcel-ID
match places it exclusively on nonresidential City land inside the full-purpose
boundary and no appraisal multifamily code supplies contrary current-use
evidence. Retain direct and deterministic counts, and retain conflicting
appraisal multifamily cases for review rather than automatically deleting them.

## Consequences

The July 2026 application removed 6,587 unsupported estimates from 123 projects
while preserving source disagreements for audit. City land use validates
current classification; it does not provide unit counts.

## Revisit When

Revisit when the City inventory is updated, exact parcel linkage changes, or
reviewed evidence resolves retained appraisal/City conflicts.

## Implementation

See the land-use validation section of
[`docs/methods/unit-count-modeling.md`](../methods/unit-count-modeling.md) and
`scripts/data/unit_counts/promote_integration.R`.
