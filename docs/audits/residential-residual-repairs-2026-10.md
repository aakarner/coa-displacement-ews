# Apartment references and manufactured-home recovery — October 7, 2026


## Status and scope

This batch extends the [residential follow-up](residential-followup-2026-10.md).
Code, reviewed configurations and private source evidence are prepared. The
staged parcel copy is a diagnostic denominator surface, not a replacement for
the final promotion/validation pipeline. Changes
are **staged only**: no canonical measures, production clusters or maps have
been rebuilt. The full test suite remains deferred at the user's request.
The filing window in this audit is April 2, 2025–April 1, 2026, not calendar 2026.

## Repairs

| Property | Repair | Denominator treatment |
| --- | --- | --- |
| The Olivine, 3201 Century Park Blvd | Move account 549351 from a point outside its parcel (cell 2682) to a verified internal reference (2683). | Preserve the existing modeled 317.494778 units and model provenance. |
| The Bennett, 7301 S IH35 | Improvement account 942518 explicitly links to parent 942517. Its raw situs ZIP 78752 produced a point about 18.06 km north, in cell 5326. Move the reference into the southern property, cell 6789. | Preserve 271 existing units; retain the erroneous raw source ZIP for audit. |
| Capitol View, 1308 Thornberry Rd | Recover 101 active individual manufactured-home accounts. | Shared park reference, cell 3917. Withhold one duplicate-serial account. |
| Pecan Park, 5701 Johnny Morris Rd | Recover 207 phase-one and 81 phase-two home accounts. Match 12 reviewed filings by their unique county home space and phase. | Separate phase references, cells 1594 and 1606. No broad street-address alias across phases. |
| Village Park, 2705 Hoeke Ln | Recover 99 active home accounts. | Shared park reference, cell 4028. Withhold two accounts competing for the same space. |
| Trails of Oak Hill | Recover 173 active home accounts with individually matched City house-address points. Review six filings at 6008 Oleander Trl against the exact home account. | Individual home references across the park, not one park-wide centroid. |
| Bel Aire, 841 Airport Blvd | Recover 15 active home accounts. | Shared park reference, cell 5058. Partial inventory: do not substitute the City's registered-space capacity for verified homes. |

The **676 added homes** were absent from the existing and staged Oak Ranch
inventory. They remain individual unit and ownership records. Of these, 503 use
an explicitly labelled shared park/phase reference because individual dwelling
coordinates are not established; 173 have individual City address points.
Park references are a documented spatial aggregation convention, not a claim
that every home occupies that coordinate. Verified filings and denominators use
the same reference. The 20-unit threshold is unchanged.

All 676 homes have independently classified 2025 ownership. In 2024, 571 have
source-year evidence and 105 remain unknown; park ownership is never inherited
by the homes. No 2025/2026 ownership backfill is used for missing 2024 records.

The primary operator addresses are [The Olivine](https://www.theolivineaustin.com/)
and [The Bennett](https://www.thebennettaustin.com/). County evidence is the
retained 2025 TCAD special export, including active profiles, situs records,
serials, park codes and explicit account links. City address evidence comes
from the [Address Points with SubAddresses service](https://awgisadaptor.austintexas.gov/awgisago/rest/services/AGO/Address_Points_with_SubAddresses/MapServer).
Retrieved records and SHA-256 checksums are retained with the private evidence.

## Conflicts and exclusions retained

Six Pecan Park case reviews involve phase-two spaces whose City subaddress
record uses the shared phase-one parcel/address point. City subaddresses thus
corroborate registered unit identity, not phase geography. Each case matches
one active county home/space and county park-phase code; the original filing
geocode also falls inside that county phase polygon. These conflicts are
explicitly recorded in the private case review and staged phase-check table.
No exact dwelling coordinate is inferred from the shared address point.

Hudson Miramont (now Panorama Villas), 8818 Travis Hills Dr, remains unresolved.
[Austin Energy's property list](https://services.austintexas.gov/edims/document.cfm?id=184161)
and the [operator's December 30, 2025 acquisition announcement](https://www.weidner.com/blog/2025/12/30/weidner-apartment-homes-acquires-panorama-villas-apartments-in-southwest-austin-tx/)
support 276 apartments. However, only about **52.3% of parcel 103824** lies inside
Austin's full-purpose boundary, and its county reference point lies outside.
The upstream point-based City filter explains the omission; this is not an
unexplained residential-classification omission. Land-area share does not tell
us the share of occupied units. Do not add all 276, or multiply 276 by 52.3%,
until building/unit geography establishes the appropriate City allocation.

The Olivine's previous point lay in the neighboring 13601 Elm Ridge Lane
property. Rebuilding the staged crosswalk withdraws 40 address-level false
Olivine links (across all source dates); these addresses retain their original
filing geography unless separately verified. They are not dropped or treated
as Olivine cases. This is why an address-only patch was insufficient.

Maravilla at the Domain remains a separate mixed-use/facility review. Three
suspect home accounts in this batch, 39 deferred Oak Ranch records, and the
coarse-geocode recovery queue remain unresolved. Low-unit filings are not all
errors: genuine small inventories, boundary cases and uncertain locations
remain possible.

## Measured staged effect

| Metric | Previous staged batch | With this batch |
| --- | ---: | ---: |
| Accepted mapped recent filings | 11,716 | 11,716 |
| Filings in cells with fewer than 20 units | 161 | **108** |
| Such cells with filings | 65 | **56** |
| Total grid residential units | 513,206.313 | **513,882.313** |

This batch removes **53 filings (33%)** from the below-threshold category,
without dropping any additional filings or creating any newly below-threshold
cases. All 11,716 retained filings keep the same assigned cells as the previous
stage: the gains here primarily repair the denominator and confirm property
links, rather than move the filings.

The 47 targeted residual filings at the two apartments and five park addresses
now have adequate cell denominators. Six other filings also cross the threshold
because of the added homes in their cells; this is not independent verification
of those other filings' property identity. The staged crosswalk withdraws six
recent false Olivine case links while retaining their original cells.

Remaining: **58 filings in 27 zero-unit cells**, plus **50 filings in 29 cells
with positive counts below 20**. These are open review candidates, not grounds
for suppressing whole cells or inflating unit counts. Against current unchanged
production, the combined staged batches reduce the below-threshold count from
215 to 108; some earlier reductions reflect rejection of coarse geocodes, so
that combined improvement must not all be described as recovered housing.

Machine-readable results and case-level changes are in
`tmp/residential_residual_repair_20261007/`: `combined_diagnostic.json`,
`batch_comparison.json`, `threshold_change_summary.csv`,
`threshold_changes_since_previous_stage.csv`, `case_phase_checks.csv`, and
`withdrawn_olivine_links_recent.csv`. Canonical units, geography, ledger and
measurement SHA-256 hashes match the pre-batch snapshot.

## Implementation and reproducibility

- `config/residual_property_reviews.json` adds 676 one-home reviews and two
  geometry-only apartment reviews. Geometry-only reviews reject count drift
  and preserve the estimation method and confidence.
- `R/reviewed_unit_properties.R` loads both manufactured-home batches, retains
  home-level ownership, and supports explicit park/phase grouping and location
  precision. City-boundary validation explicitly selects Austin full-purpose
  jurisdiction.
- Five narrow single-property address aliases join the existing Ocotillo
  alias. Pecan phases and the six Trails cases use 18 case-specific reviews,
  bringing the total to 144.
- The geography builder now supports an explicit `tmp/` staging destination
  and staged unit surface. Rule version 4 recognizes reviewed park/phase
  groups while preserving individual ownership/unit accounts. A full staged
  address crosswalk removes obsolete automatic links as well as adding new
  ones. No nearest-property heuristic is introduced.

Evidence is retained under
`data/reviewed_unit_properties/residual_20261007/` and
`data/reviewed_eviction_properties/residual_20261007/` (private/ignored).
Preparation scripts are `scripts/data/prepare_residual_park_inventory.py`,
`scripts/audits/residual_park_geography.R`,
`scripts/data/prepare_residual_park_ownership.py`, and
`scripts/data/prepare_residual_property_integration.py`.

Reproduce the focused staging replay, after the previous Oak Ranch stage:

```sh
Rscript scripts/audits/residual_property_integration.R
Rscript scripts/data/build_eviction_property_geography.R tmp/residential_residual_repair_20261007/geography tmp/residential_residual_repair_20261007/staged_promoted.rds
Rscript scripts/audits/residual_property_replay.R
Rscript scripts/audits/residual_property_comparison.R
```

Focused checks cover unit/coordinate conservation, rejection of changed unit
counts and evidence, unchanged unrelated accounts, duplicate inventory,
source-year home ownership, consistent coordinates for shared-property
aliases, and county phase containment. Production-output checks are updated
for the planned 144 reviewed cases but intentionally await the final rebuild.
