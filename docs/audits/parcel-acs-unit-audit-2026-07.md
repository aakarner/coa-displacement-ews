# Parcel and ACS Housing-Unit Audit

**Audit vintage:** July 31, 2026. The values below describe the promoted
parcel-unit surface after the City land-use validation and the contemporaneous
ACS allocation. Rerun the named audit scripts before using the figures to
describe a later pipeline vintage.

## Purpose

This audit tests disagreement between the parcel-derived housing-unit surface
and ACS total housing units before either source is used to expand clustering
eligibility. Run:

```r
source("scripts/audits/parcel_acs_housing_units.R")
source("scripts/audits/populated_zero_unit_hexes.R")
```

The scripts are diagnostic. They do not change unit estimates, feature
engineering, rate denominators, eligibility, or cluster assignments.

## Current Populated Zero-Unit Residual

`scripts/audits/populated_zero_unit_hexes.R` evaluates the promoted canonical
surface after project modeling, strict direct-project integration, and the
City land-use repair. Of 7,027 study hexes, 2,958 have zero parcel units, but
2,662 of those also have zero allocated population. The land-use repair changed
the population-weighted residual as follows:

| Measure | Before land-use repair | Current promoted surface |
|---|---:|---:|
| Zero-unit hexes | 2,954 | 2,958 |
| Populated zero-unit hexes | 293 | 296 |
| Population in zero-unit hexes | 29,822 | 35,243 |
| Mapped parcel units | 508,843 | 502,257 |
| Mapped ACS housing units | 479,614 | 479,650 |
| Parcel excess over ACS | 6.1% | 4.7% |

The 296 residual hexes contain 12,546 allocated ACS housing units. This does
not imply that 12,546 parcel units should be backfilled: most ACS evidence in
these cells was placed through Census-block point fallback after no unit-parcel
support was found.

The mutually exclusive audit categories are:

| Category | Hexes | Population | ACS housing units |
|---|---:|---:|---:|
| Reviewed TCAD land-only parcels with MF zoning | 10 | 203 | 162 |
| Multifamily signal not selected for modeling | 2 | 4,885 | 8 |
| Cross-county project count relocated | 1 | 237 | 153 |
| Other improved zero-unit parcel | 2 | 149 | 58 |
| Reviewed nonresidential exclusions only | 2 | 36 | 2 |
| ACS point fallback without a unit parcel | 154 | 22,672 | 10,564 |
| No full-parcel centroid support | 17 | 3,938 | 1,418 |
| Population with fewer than five ACS housing units | 107 | 3,123 | 181 |
| Other zero-unit parcel | 1 | 1 | 1 |

The earlier candidate-only run omitted 16 strict selected project labels
totaling 4,148 units. Target `unit_integration` now applies all 892 selected
direct project totals before unresolved-candidate predictions. The revised
audit compares selected and allocated totals at the project level and finds
zero direct-project integration mismatches. This project-level check matters
for linked projects whose total is legitimately concentrated on the member
parcel carrying the building area.

The Williamson review exposed a version mismatch between the broad parcel
input and the current certified property roll. The certified roll identifies
407 active, uniquely addressed residential parcels with positive living area
inside the study grid that were absent from the input. A reviewed legacy
geometry supplies the location for one additional active certified property.
Target `unit_calibration` now adds all 408 as one-unit records, including 16
corporate-owned parcels. This resolves all 84 initially flagged proxy records
and reduces the full-residential-proxy gap count to zero.

The ten former multifamily-signal cases contain 18 TCAD records. The property
profile confirms `landOnly = 1`, zero improvement units, and zero improvement
area for every record; the multifamily signal came from zoning alone. They
remain at zero units and are now recorded as reviewed land-only parcels.

The remaining residual is therefore dominated by ACS point fallback,
low-housing-control population, or geometry support limitations. Those cases
should remain zero unless independent parcel or project evidence is found.

`scripts/audits/populated_zero_unit_hexes.R` writes:

- `output/populated_zero_unit_hex_audit.rds/.csv`;
- category, jurisdiction, transition, and summary tables;
- parcel, direct-project, exclusion, and full-parcel review tables; and
- `figures/populated_zero_unit_hex_audit.png`.

The sections below document the refreshed canonical comparison from
`scripts/audits/parcel_acs_housing_units.R`.

## Comparable Totals

On the exact H3 grid:

| Unit surface | Units |
|---|---:|
| Raw parcel values | 804,160 |
| Primary parcel calibration | 519,440 |
| Promoted parcel hierarchy | 502,257 |
| ACS 2024 five-year total housing | 479,650 |
| ACS aggregate 90% MOE | 5,726 |
| Conservative parcel calibration | 441,178 |

The promoted parcel surface is 22,607 units, or 4.7%, above the allocated ACS
five-year estimate. The comparison is sensitive to parcel-unit assumptions:
the ACS estimate lies between the conservative and primary parcel surfaces.

Inside the Austin full-purpose boundary, the promoted parcel total is 507,653
units. That is 2.1% below the retained 2024 one-year ACS city benchmark
of 518,574. Citywide agreement therefore does not establish local agreement.

## Threshold Agreement

Using 20 units as a diagnostic threshold:

| Agreement class | Hexes |
|---|---:|
| Both sources at or above 20 | 3,149 |
| ACS only at or above 20 | 293 |
| Parcel only at or above 20 | 133 |
| Both sources below 20 | 3,452 |

Of the 426 discordant hexes, 285 remain robustly discordant after considering
the full parcel calibration range and the ACS 90% interval:

- 196 robust ACS-only hexes;
- 97 uncertainty-sensitive ACS-only hexes;
- 89 robust parcel-only hexes; and
- 44 uncertainty-sensitive parcel-only hexes.

## ACS-Only Findings

The 293 ACS-only hexes contain 22,333 ACS housing units and approximately
46,011 residents. A mutually exclusive diagnostic classification identifies:

| Provisional explanation | Hexes | ACS units |
|---|---:|---:|
| No matched residential parcel | 101 | 11,141 |
| Majority Census-block point fallback | 21 | 1,840 |
| Other parcel undercoverage | 171 | 9,351 |

The previously excluded mixed-use multifamily set no longer forms a distinct
ACS-only class after direct project totals and validated model estimates are
promoted. Across all ACS-only cases, 122 hexes receive a
majority of their Census-block housing control from the no-parcel point
fallback; this overlaps the categories above and is a spatial-allocation
warning rather than proof that ACS is wrong.

## Parcel-Only Findings

The 133 parcel-only hexes contain 21,167 promoted parcel units but only 1,072
allocated ACS housing units and approximately 3,390 residents.

| Provisional explanation | Hexes | Parcel units |
|---|---:|---:|
| Recent construction supplies at least half of parcel units | 76 | 18,053 |
| No allocated ACS housing or population | 3 | 570 |
| Other parcel-only pattern | 54 | 2,544 |

- No parcel-only units are attributed to the earlier multifamily floor-area
  estimation method in these hexes.
- 17,855 units, or 84%, are associated with buildings dated 2020 or later.
- 10,018 units are associated with buildings dated 2024 or later.
- 2,179 direct CoStar units occur in this group.
- 34 parcel-only hexes have zero allocated ACS population.

These findings are consistent with a mixture of recent construction, ACS
temporal lag, parcel overestimation, and spatial matching or allocation errors.

## Block-Group Comparison

For 536 block groups with at least 95% of their area inside Austin:

| Parcel variant | Parcel/ACS aggregate ratio | Within ACS MOE |
|---|---:|---:|
| Conservative | 0.969 | 63.4% |
| Promoted | 1.118 | 55.0% |
| Primary | 1.163 | 54.3% |

The conservative surface agrees better locally but falls below the direct
citywide benchmark. The promoted surface is close to the citywide benchmark
but still has more local overestimation. ACS disagreement remains a diagnostic
and monitoring concern rather than a reason to force local reconciliation.

## Implications

A one-way ACS fallback remains indefensible because it would repair suspected
parcel undercounts while retaining suspected parcel overcounts. Continued
governance of the promoted surface should:

1. Preserve reported parcel units and direct CoStar unit matches.
2. Preserve explicit review flags for mixed-use multifamily records.
3. Verify recent parcel-only developments against completion or occupancy
   evidence.
4. Retain the held-out stratified project model only for in-domain multifamily
   candidates and preserve explicit fallbacks elsewhere.
5. Use ACS block-group estimates and MOEs as independent diagnostics on uncertain
   parcel units rather than exact replacement totals.
6. Reconcile the aggregate result to a common-date citywide benchmark without
   forcing the parcel and ACS surfaces to agree.
7. Keep zero-unit residuals explicit where neither parcel nor project evidence
   supports housing.

Canonical clustering eligibility now uses the promoted parcel point estimate
at the 20-unit threshold. ACS estimates and MOEs remain independent diagnostics
rather than a fallback denominator.
