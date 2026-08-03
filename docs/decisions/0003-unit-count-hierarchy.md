# 0003: Construct Projects and Use a Unit-Count Evidence Hierarchy

- **Status:** Accepted
- **Decision period:** July 2026
- **Recorded retrospectively:** August 3, 2026

## Context

One appraisal row is not necessarily one dwelling or one physical property.
Multifamily developments may span several parcels, while a project total may be
repeated on multiple appraisal accounts. Direct counts are incomplete, so some
projects require deterministic rules or estimates.

## Decision

First group related parcel and appraisal records into physical projects using
high-confidence source links or a normalized-address and proximity rule. Select
one project total through an explicit hierarchy: reviewed direct counts,
documented or deterministic counts, validated floor-area estimates, and then
targeted fallbacks. Do not average conflicting sources or repeat a project total
on every parcel.

Use main or living floor area for the selected multifamily model because it is
more comparable to residential unit counts than total improvement area.

## Consequences

Every promoted count retains its source and method. Project construction avoids
double counting but introduces linkage decisions that require audit tables.
Modeled and fallback totals remain distinguishable from observed counts.

## Revisit When

Revisit when counties expose more direct unit fields, reviewed project sources
expand substantially, project-linkage audits identify systematic errors, or a
new model materially improves held-out validation.

## Implementation

See [`docs/methods/unit-count-modeling.md`](../methods/unit-count-modeling.md)
and `scripts/data/unit_counts/`.
