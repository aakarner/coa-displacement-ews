# 0006: Require 20 Residential Units for Rate-Based Analysis

- **Status:** Provisional
- **Decision period:** July 2026
- **Recorded retrospectively:** August 3, 2026

## Context

Eviction and selected 311 rates become unstable when a small number of events
is divided by very few estimated dwellings. The same unit surface defines the
residential support required for the primary cluster sample.

## Decision

Require at least 20 promoted residential units for primary cluster eligibility
and for unit-denominated rates. Preserve raw counts and area-based measures so
that excluded rates are not confused with zero observed activity.

## Consequences

The threshold reduces extreme small-denominator rates but leaves some
residential hexes unclassified. Because parcel units are partly modeled, the
threshold can also make eligibility sensitive to unit-count revisions.

## Revisit When

Complete and document sensitivity analyses using plausible lower and higher
thresholds, including effects on classified population, cluster profiles, and
assignment stability.

## Implementation

The current value is `minimum_residential_units_for_rates = 20L` in
[`R/analysis_config.R`](../../R/analysis_config.R).
