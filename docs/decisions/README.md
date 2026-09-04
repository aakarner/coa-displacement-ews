# Analytical Decision Records

This directory records consequential analytical choices without making their
history part of the executable pipeline. Current code and configuration remain
the source of truth for what the pipeline computes. These records explain why
the current choices were made, what alternatives remain plausible, and what
evidence should trigger reconsideration.

These records were initially reconstructed from the methods documents,
configuration, audit reports, GitHub issues, and Git history on August 3, 2026.
The stated decision period is intentionally approximate when no contemporaneous
record established an exact date.

## Index

| ID | Decision | Status |
| --- | --- | --- |
| [0001](0001-analysis-geography.md) | Use resolution-9 H3 hexes for the analytical grid | Accepted |
| [0002](0002-residential-unit-surface.md) | Use the parcel-derived unit surface operationally and ACS as an independent comparison | Accepted |
| [0003](0003-unit-count-hierarchy.md) | Group parcel records into projects and select unit counts through an evidence hierarchy | Accepted |
| [0004](0004-acs-dasymetric-allocation.md) | Allocate additive ACS estimates with residential dasymetric weights | Accepted |
| [0005](0005-part1-feature-architecture.md) | Give each Part 1 conceptual domain one equally standardized composite input | Accepted |
| [0006](0006-rate-eligibility-threshold.md) | Require 20 promoted residential units for cluster eligibility and unit-rate measures | Provisional |
| [0007](0007-rent-data-strategy.md) | Use ACS rent citywide and CoStar only where its coverage is observed | Accepted for Part 1 |
| [0008](0008-select-seven-clusters.md) | Use seven clusters for the current Part 1 baseline | Provisional baseline |
| [0009](0009-fixed-baseline-updates.md) | Assign later vintages to frozen Part 1 definitions | Accepted architecture |
| [0010](0010-city-land-use-validation.md) | Exclude unsupported modeled units on exclusively nonresidential City land | Accepted |
| [0011](0011-neighborhood-reporting-areas.md) | Use City Neighborhood Reporting Areas for neighborhood cluster summaries | Accepted |

## What Qualifies

A decision record is warranted when a choice materially affects the analytical
universe, geography, denominator, interpretation, model structure, or update
behavior and when a reasonable alternative could have been selected. Source
downloads, routine code repairs, package changes, and presentation edits belong
in source inventories, issues, commits, or release notes instead.

## Status Terms

- **Accepted:** current production behavior.
- **Accepted for Part 1:** current baseline behavior that may differ in later
  EWS stages.
- **Provisional:** current behavior that still requires planned sensitivity or
  partner review.
- **Accepted architecture:** approved workflow design whose later stages may
  not yet be fully implemented.
- **Superseded:** retained for history and linked to its replacement.

## Record Template

Each record should state its status, approximate decision period, context,
decision, consequences, revisit triggers, and current implementation. Detailed
run-specific statistics belong in `docs/audits/`, not in the decision record.
