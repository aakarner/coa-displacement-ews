# 0002: Use Parcel Units Operationally and ACS as a Comparison

- **Status:** Accepted
- **Decision period:** July 2026
- **Recorded retrospectively:** August 3, 2026

## Context

County appraisal records approximate a property census but report residential
unit counts inconsistently. ACS housing estimates are survey estimates for
larger geographies and gain additional uncertainty when allocated to hexes.
The two systems measure housing through different methods and need not agree at
small spatial scales.

## Decision

Use the reviewed parcel-derived unit surface for operational denominators,
residential eligibility, ownership exposure, and spatial allocation weights.
Use ACS housing estimates as an independent benchmark and diagnostic. Do not
force parcel totals to match ACS and do not replace parcel zeros merely because
ACS allocation suggests housing.

## Consequences

The EWS preserves the stronger location information in property records while
making modeled parcel counts and ACS disagreement visible. Neither source is
treated as error-free, and aggregate agreement does not establish local
accuracy.

## Revisit When

Revisit if a more authoritative citywide unit inventory becomes available, if
parcel unit assumptions change materially, or if the ACS allocation audit
reveals systematic geographic bias.

## Implementation

The current hierarchy is described in
[`docs/methods/unit-count-modeling.md`](../methods/unit-count-modeling.md).
Comparison evidence is retained in
[`docs/audits/parcel-acs-unit-audit-2026-07.md`](../audits/parcel-acs-unit-audit-2026-07.md).
