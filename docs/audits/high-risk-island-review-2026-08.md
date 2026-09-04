# High-Risk Island and Property-Driver Review

**Review date:** August 2026  
**Part 1 analysis vintage:** April 1, 2026  
**Status:** Diagnostic review; does not change the approved Part 1 clusters

## Purpose

This review identifies high-risk hexes that look unlike their immediate
surroundings and asks whether their observed pressure is concentrated at one
or a few residential properties. These places may represent genuine localized
displacement pressure. They may also reveal sparse event counts, geocoding near
a hex boundary, or another feature that deserves closer review before the
result is interpreted.

The review uses the frozen seven-cluster Part 1 model. It does not re-estimate
the clusters, change any source data, or create a new risk category.

## Island Screen

A hex is an island candidate when all three conditions are met:

1. It belongs to Cluster 6 (high concern) or Cluster 7 (very high concern).
2. At least four of its six possible immediate H3 neighbors are classified.
3. At least two-thirds of those classified immediate neighbors belong to
   Cluster 1 or Cluster 2, the two low-concern clusters.

The second ring of neighbors, population- and housing-unit-weighted neighbor
shares, and the multivariate contrast between each candidate and its neighbors
are retained for interpretation, but they do not determine eligibility. Every
qualifying candidate is presented equivalently; the review does not create an
additional priority ranking.

## Property Attribution

Residential parcel records are connected to the physical-project identifiers
already used by the unit-count workflow. A physical project may contain one or
more appraisal parcels that represent the same development. The review assigns
property-level evidence as follows:

- **Eviction filings:** normalized street address first; otherwise the nearest
  residential parcel within 50 meters when the house number also agrees.
- **Austin Code complaint cases:** the exact residential parcel link produced
  by the Code complaint audit. These cases are a subset of the broader 311
  smoke-signal input, so their property attribution is deliberately
  conservative.
- **Residential demolition permits:** exact appraisal parcel through the City
  land-use crosswalk when possible, then normalized address, then the same
  conservative nearest-parcel rule.
- **Corporate ownership:** the final promoted residential units and
  financialized-owner parcel counts used to construct the clustered ownership
  feature. These totals are required to reconcile exactly by hex.

Rent pressure and demographic vulnerability come from area-level data and
cannot be attributed to an individual property in this review. Amenity change
is also not treated as a property-level signal.

For each property, the review calculates its share of the matched eviction,
Code complaint, demolition, and ownership evidence in the hex. It reports both
the largest share of any one signal and a pressure-weighted share across the
four property-attributable domains. This distinction prevents a single event
in an otherwise weak domain from being described as the main driver of the
whole risk profile.

## Counterfactual Test

For the leading project in each candidate hex, the review sets that project's
matched eviction, Code complaint, demolition, and ownership evidence to zero.
It does not remove the building, residential units, or population. The affected
pressure indexes are recalculated using the original citywide normalization
bounds, and the hex is reassigned to the nearest frozen Part 1 centroid.

This is an influence diagnostic, not a causal estimate. A changed assignment
means that the current classification is sensitive to evidence associated with
that project. It does not prove that the property caused displacement or that
the original classification is incorrect.

## Results

- 113 of the 561 high- or very-high-concern hexes meet the island screen:
  47 Cluster 6 hexes and 66 Cluster 7 hexes.
- The candidates contain an estimated 26,853 residents and 12,335 residential
  units. These totals describe the candidate hexes; they are not counts of
  displaced people or units.
- The most elevated domain is rent pressure in 40 candidates, demographic
  vulnerability in 25, eviction pressure in 23, demolition pressure in 22,
  311 pressure in 2, and corporate ownership pressure in 1.
- One project contributes at least half of the pressure-weighted,
  property-attributable evidence in 77 candidates. A project contributes at
  least half of one individual signal in 107 candidates, often because the
  underlying event count is sparse.
- Setting the leading project's attributable signals to zero lowers the
  assigned concern category in 96 candidates. This high sensitivity should be
  interpreted alongside the event counts, matching method, and area-level rent
  and demographic conditions shown in the property table.
- The review matches 81.5% of candidate-hex eviction filings, 80.3% of linked
  Code cases, and 97.9% of residential demolition permits across their full
  available histories. In the current feature windows, 81.6% of evictions,
  80.3% of the broader 311 smoke-signal count, and all residential demolitions
  are attributed to projects.
- 35 candidates have at least one event point in a different hex from its
  matched residential parcel. No candidate contains a physical project that
  itself crosses a hex boundary. These cases are flagged because geocoding
  placement near a cell edge may contribute to the apparent island pattern.

## Interpretation and Limits

An island is a prompt for investigation, not evidence that the hex is miscoded.
The strongest candidates for property review are those with a high weighted
property share, a counterfactual concern reduction, and no spatial-attribution
flag. Cases led by rent pressure or demographic vulnerability may remain
substantively important even when a matched property changes the cluster,
because those area-level conditions are not removed by the counterfactual.

The Code complaint match describes only complaints linked in the current
post-2023 Code case dataset. It does not provide complete historical attribution
for every 311 smoke-signal request. Unmatched events remain in the hex-level
cluster features but are not assigned to a property. Property names and owners
should therefore be used as review leads, not definitive attribution.

## Outputs

- `output/part1/high_risk_island_hex_summary.csv`: one row per candidate hex.
- `output/part1/high_risk_island_top_properties.csv`: the three leading
  projects per candidate.
- `output/part1/high_risk_island_property_drivers.csv`: full project-level
  attribution table.
- `output/part1/high_risk_island_counterfactuals.csv`: before-and-after pressure
  indexes and frozen-model assignments.
- `output/part1/high_risk_island_attribution_coverage.csv`: reconciliation of
  property-attributed evidence to clustered hex totals.
- `output/part1/high_risk_island_attribution_qa.csv`: source-level matching QA.
- `output/part1/high_risk_island_unmatched_events.csv`: events that could not be
  linked conservatively.
- `figures/03h_high_risk_islands.png`: static review map.
- `figures/03h_high_risk_islands_interactive.html`: interactive review map.

The screen and matching thresholds are stored in
`config/high_risk_island_parameters.csv`. The reproducible review target is
`high_risk_island_review` in `_targets_review.R`.
