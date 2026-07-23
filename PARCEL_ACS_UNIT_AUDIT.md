# Parcel and ACS Housing-Unit Audit

## Purpose

This audit tests disagreement between the parcel-derived housing-unit surface
and ACS total housing units before either source is used to expand clustering
eligibility. Run:

```r
source("02o_audit_parcel_acs_housing_units.R")
```

The script is diagnostic. It does not change unit estimates, feature
engineering, rate denominators, eligibility, or cluster assignments.

## Comparable Totals

On the exact H3 grid:

| Unit surface | Units |
|---|---:|
| Raw parcel values | 805,334 |
| Primary parcel calibration | 531,312 |
| Current targeted parcel calibration | 515,844 |
| ACS 2024 five-year total housing | 477,409 |
| ACS aggregate 90% MOE | 5,718 |
| Conservative parcel calibration | 447,890 |

The current parcel surface is 38,435 units, or 8.1%, above the allocated ACS
five-year estimate. The comparison is sensitive to parcel-unit assumptions:
the ACS estimate lies between the conservative and primary parcel surfaces.

Inside the Austin full-purpose boundary, the current targeted parcel total is
521,085 units. That is 0.5% above the retained 2024 one-year ACS city benchmark
of 518,574. Citywide agreement therefore does not establish local agreement.

## Threshold Agreement

Using 20 units as a diagnostic threshold:

| Agreement class | Hexes |
|---|---:|
| Both sources at or above 20 | 3,117 |
| ACS only at or above 20 | 322 |
| Parcel only at or above 20 | 127 |
| Both sources below 20 | 3,460 |

Of the 449 discordant hexes, 322 remain discordant after considering the full
parcel calibration range and the ACS 90% interval:

- 225 robust ACS-only hexes;
- 97 uncertainty-sensitive ACS-only hexes;
- 97 robust parcel-only hexes; and
- 30 uncertainty-sensitive parcel-only hexes.

## ACS-Only Findings

The 322 ACS-only hexes contain 28,222 ACS housing units and approximately
54,987 residents. A mutually exclusive diagnostic classification identifies:

| Provisional explanation | Hexes | ACS units |
|---|---:|---:|
| No matched residential parcel | 104 | 11,704 |
| Mixed-use multifamily estimate excluded | 34 | 6,071 |
| Majority Census-block point fallback | 19 | 1,314 |
| Other parcel undercoverage | 165 | 9,133 |

The 34 mixed-use hexes contain 38 excluded parcel records with 14,623 raw
units. Those raw values are not automatically valid, but the records are a
high-priority review set. Across all ACS-only cases, 124 hexes receive a
majority of their Census-block housing control from the no-parcel point
fallback; this overlaps the categories above and is a spatial-allocation
warning rather than proof that ACS is wrong.

## Parcel-Only Findings

The 127 parcel-only hexes contain 12,663 targeted parcel units but only 1,033
allocated ACS housing units and approximately 3,310 residents.

| Provisional explanation | Hexes | Parcel units |
|---|---:|---:|
| Recent construction supplies at least half of parcel units | 61 | 7,937 |
| Multifamily floor-area estimates supply at least half | 15 | 2,414 |
| No allocated ACS housing or population | 4 | 521 |
| Other parcel-only pattern | 47 | 1,791 |

- 8,072 parcel units, or 64%, are multifamily floor-area estimates.
- 7,938 units, or 63%, are associated with buildings dated 2020 or later.
- 3,581 units are associated with buildings dated 2024 or later.
- 2,135 direct CoStar units occur in this group.
- 34 parcel-only hexes have zero allocated ACS population.

These findings are consistent with a mixture of recent construction, ACS
temporal lag, parcel overestimation, and spatial matching or allocation errors.

## Block-Group Comparison

For 536 block groups with at least 95% of their area inside Austin:

| Parcel variant | Parcel/ACS aggregate ratio | Within ACS MOE |
|---|---:|---:|
| Conservative | 0.969 | 63.6% |
| Current targeted | 1.126 | 54.9% |
| Primary | 1.165 | 54.5% |

The conservative surface agrees better locally but falls below the direct
citywide benchmark. The current surface agrees citywide but has more local
overestimation. Neither should be accepted as the final hex denominator without
reconciliation.

## Implications

A one-way ACS fallback is not defensible because it would repair suspected
parcel undercounts while retaining suspected parcel overcounts. The next unit
surface should:

1. Preserve reported parcel units and direct CoStar unit matches.
2. Review the excluded mixed-use multifamily records.
3. Verify recent parcel-only developments against completion or occupancy
   evidence.
4. Replace global multifamily square-foot assumptions with a held-out,
   property-level prediction model.
5. Use ACS block-group estimates and MOEs as soft constraints on uncertain
   parcel units rather than exact replacement totals.
6. Reconcile the aggregate result to a common-date citywide benchmark.
7. Produce a unit estimate and uncertainty interval for every hex.

Eligibility can then be based on whether the reconciled lower bound, point
estimate, or upper bound crosses 20 units. The primary clustering rule should
be chosen only after comparing the population and spatial consequences of
those three definitions.
