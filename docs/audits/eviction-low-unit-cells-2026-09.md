# Audit of 41 cells with low estimated units and substantial eviction filings

Follow-up: the [October repair](residential-property-repair-2026-10.md) resolves
unit/geography support for 1,002 of these 1,219 filings. It also refines the
18 omitted polygon IDs into reference and active accounts. The original
diagnostic evidence below is retained as the pre-repair record.

Audit date: September 11, 2026. Diagnostic only: no production filing assignments,
unit counts, eligibility thresholds, or clusters changed in this investigation.

The cohort is the 41 source-covered city cells with fewer than 20 operational
residential units and at least 10 accepted recent filings. They contain 1,219
unique filings for **April 2, 2025–April 1, 2026**, not calendar 2026. This is a
targeted investigation, not an estimate of error prevalence across the city.

The main finding is that the low cell denominators often reflect a mismatch in
how properties are mapped, while a separate upstream selection rule omits
substantial Williamson residential properties.

| Evidence category | Unique filings | Share |
|---|---:|---:|
| Same parcel ID; its units are counted in another cell | 697 | 57.2% |
| Related residential account at the same property; units counted in another cell | 79 | 6.5% |
| Williamson residential parcel omitted from the input and promoted unit files | 235 | 19.3% |
| Filing point just outside mapped parcel boundaries; assignment needs review | 71 | 5.8% |
| Other parcel/account or residential-use issue requiring review | 134 | 11.0% |
| Matched unit reference point falls outside the analysis grid | 3 | 0.2% |
| **Total** | **1,219** | **100.0%** |

Cases were deduplicated after all address and parcel joins. Categories are
mutually exclusive at case level. Individual cells can have multiple categories.

## Units already exist, but are counted elsewhere

For **776 filings across 23 cells**, the property already has operational units
in another cell. In 697 cases, the filing point falls inside a county polygon
whose parcel ID directly matches the unit file. For the other 79, at 8110 Blue
Goose Road, the mapped parent polygon is 956770 and the residential account is
956771. Its reference point is inside that polygon, its address agrees, and its
300 units are counted in cell 1217.

Examples below report operational estimates, not independently verified totals.

| Filing cell | Filings on the property | Current units in filing cell | Property / address | Operational property units | Cell receiving those units |
|---|---:|---:|---|---:|---|
| 1806 | 107 | 1 | Orbit, 8900 N Interstate 35 | 243.8 | 1807 |
| 174 | 84 | 0 | 8021 N FM 620 | 327.5 | 175 |
| 176 | 86 | 0 | 7655 N FM 620 / 11350 Four Points Drive | 571.0 | 203 |
| 1220 | 79 | 0 | Alta Blue Goose, 8110 Blue Goose Road | 300 | 1217 |
| 4979 | 57 | 1 | 5800 Techni Center Drive | 291.3 | 4992 |
| 812 | 40 | 1 | Nichols Park, 5001 Convict Hill Road | 200 | 669 |

![Two examples of filings and units in different cells](../../output/eviction_low_unit_audit/filings_units_different_cells.png)

The implementation explains this: `scripts/data/corporate_ownership.R` creates
one point from each parcel record's longitude and latitude, places it in a hex,
and sums all that record's units there. Eviction filings use separately geocoded
defendant-address points. A development can span multiple hexes, so both locations
can be on the correct property and still produce incompatible cell rates.

This is an understandable simplifying assumption for aggregating parcel records,
but it was not reconciled with the independently geocoded numerator. Correcting
it matters for the neighboring cells too: they can receive the units without the
corresponding filings.

Unit magnitude can also require review. Orbit has 336 units in the cached URO
inventory and CoStar series, but its current model-based parcel estimate is
243.8. Its URO record is unlinked. Resolving the spatial mismatch alone would not
resolve that count discrepancy. Source inventories must be reconciled at project
level rather than copied into every affected cell.

## Williamson selection omissions

**235 filings across 12 cells fall on 18 Williamson parcel IDs absent from both
the upstream residential input and the promoted unit file.** The county's
certified records contain these properties, their floor area, and recognizable
development names. Examples include The Asher, The Loretta, Lakeline Station,
Hunters Chase East/West, Parkside, and Bexley Whitestone. Cached housing or URO
inventories corroborate several of them, including 377 units at The Asher,
137 at The Loretta, and 128 at Lakeline Station; those counts still need normal
source and project reconciliation before promotion.

The upstream generator is the sibling repository's
`williamson-parcel-pull.R`, particularly its residential classification around
lines 402–409. It accepts a few literal property-type descriptions and parcel-use
codes, or words such as “APARTMENTS”, “APTS”, “CONDO”, and “DUPLEX” in address,
legal-description, comment, or DBA fields. In the inspected certified rows,
`PropertyTypeDesc` contains values such as `C3` and `C5`, rather than the literal
descriptions that branch expects.

The audit reproduced the text rule for all 22 Williamson parcel IDs containing
filings in this cohort: **all 18 absent IDs fail the rule; all four present IDs
pass it**. Names such as “ASHER” and “THE LORETTA” do not identify housing to this
rule. The separate EWS unit-estimation model cannot recover a property that
never entered its input universe. This is strong evidence of a systematic
selection defect, not merely an inaccurate floor-area-to-units estimate.

The city-scope explanation was checked: every filing coordinate in this cohort
lies inside the cached Austin full-purpose boundary. These are not simply
out-of-city filings landing in boundary hexes.

## Remaining evidence and limits

* **71 filings in cells 6381 and 6965:** three geocode locations are about
  5.5–8.2 metres outside the nearest county parcel polygon. Matching address
  candidates have hundreds of units in other cells, but a nearest-neighbor
  match was not promoted as a confirmed assignment.
* **24 filings in cells 3767 and 3769:** the containing parent parcels are
  classified as Mobile Homes in the cached city land-use inventory. Those
  parent IDs are absent from the unit file; one footprint contains two other
  one-unit accounts. Reconcile park, pad, and individual-home accounts before
  deciding the missing housing total.
* **60 filings in cell 6260:** the point falls on parcel 291453 at 2101 E Ben
  White Boulevard, absent from the unit file and classified as Office in the
  city inventory. Residential use, possible source-address problems, and the
  inventory classification require review. The audit does not establish that
  these should be ordinary residential-unit filings.
* **31 filings in cell 3485:** three nearby addresses fall on parent parcel
  774333, classified Commercial. Housing inventories identify The Villages at
  The Domain at 11011 Domain Drive with 436–438 units, but those inventory
  records are unlinked. This is a concrete mixed-use/parent-account linkage
  candidate; it is not yet a verified unit allocation.
* **18 filings in cell 3319:** 9009 FM 620 resolves inside parcel 498142,
  classified Meeting and Assembly. A residential record at 9009 N Ranch Road
  620, parcel 498141, has 446.4 operational units in cell 3321. Resolve the
  address alias and the adjacent-parcel discrepancy before changing assignment.
* **One filing in cell 602:** parcel 1010063 lacks a resolved residential-account
  link in this audit.
* **Three filings in cell 6929:** the matched 2001 E Slaughter Lane parcel has
  294 units, but its reference point falls outside the analysis grid. Its
  geometry and boundary treatment need review.

All 1,219 filings have address-level, subaddress-level, or local apartment-point
geocodes. That precision is evidence against a general ZIP-centroid explanation,
not proof every address is correct. For all 139 locations with available display
coordinates, those coordinates equal the selected geocode coordinates to
numerical precision: switching between routing and display coordinates would
not fix this cohort. Three of the 152 distinct geocode locations fall outside
the cached parcel polygons; the rest intersect a polygon.

## Recommended repair sequence

1. Repair residential candidate selection, starting with the 18 identified
   Williamson accounts and a broader audit of the same classification pattern.
   Use documented county classifications and corroborated housing inventories;
   do not treat all commercial records as housing. Preserve exclusions and
   source provenance, and avoid double-counting parent and individual accounts.
2. Make the filing numerator and housing denominator use a common property or
   project geography. A practical first sensitivity run would link verified
   filings to the same reviewed property reference used for the units, while
   retaining original geocodes and link confidence. Buildings could provide a
   better allocation for large developments when suitable data exist. Do not
   redistribute units according to filing intensity or add the entire complex's
   units to every cell it touches.
3. Resolve the remaining boundary, mobile-home, and land-use conflicts as
   explicit review cases. Keep genuinely ambiguous filings unassigned under the
   already-approved policy; do not suppress surrounding cells.
4. Reconcile project totals and unit mass, then compare cell counts, rates,
   eligibility, and Part 1 profiles across the repaired and current versions.
   Keep the 20-unit floor during that comparison. Removing it would expose the
   numerator/denominator mismatch rather than repair it.

## Reproduction and evidence

Run the local audit scripts in this order: `eviction_low_unit_cells.R`,
`eviction_low_unit_spatial.R`, `eviction_low_unit_sources.R`,
`eviction_low_unit_polygon_accounts.R`, `eviction_low_unit_jurisdictions.R`,
`eviction_low_unit_summary.R`, and `eviction_low_unit_figures.R`, all under
`scripts/audits/`.

Detailed evidence is in the ignored `output/eviction_low_unit_audit/` directory.
It includes an all-41-cell table, per-case categories, source-address and polygon
links, unit-account evidence, jurisdiction checks, and the Williamson filter
reproduction. Case identifiers and apartment-level source addresses remain local.

The audit verified 1,219 unique assigned case IDs, reconciled the parcel-point
unit aggregation to the current full feature table within 1e-6, and checked that
every accepted geocode falls in its existing assigned hex. Those checks validate
the diagnosis against current outputs; they do not independently validate every
court address, source inventory count, or land-use classification.

## All 41 cells

| Cell | Current units | Filings | Evidence / filing count |
|---|---:|---:|---|
| 1806 | 1 | 107 | Units counted in another cell: same parcel ID: 107 |
| 176 | 0 | 86 | Units counted in another cell: same parcel ID: 86 |
| 174 | 0 | 84 | Units counted in another cell: same parcel ID: 84 |
| 1220 | 0 | 79 | Units counted in another cell: verified related account: 79 |
| 6260 | 0 | 60 | Other parcel/account linkage requires review: 60 |
| 6381 | 1 | 60 | Geocode just outside parcel boundary: review: 60 |
| 4979 | 1 | 57 | Units counted in another cell: same parcel ID: 57 |
| 3698 | 0 | 46 | Units counted in another cell: same parcel ID: 46 |
| 1472 | 0 | 42 | Units counted in another cell: same parcel ID: 42 |
| 6929 | 0 | 41 | Unit reference point outside grid: 3; Units counted in another cell: same parcel ID: 38 |
| 812 | 1 | 40 | Units counted in another cell: same parcel ID: 40 |
| 2527 | 13 | 33 | Williamson residential parcel omitted: 33 |
| 3485 | 0 | 31 | Other parcel/account linkage requires review: 31 |
| 1286 | 0 | 27 | Units counted in another cell: same parcel ID: 27 |
| 2327 | 3 | 24 | Williamson residential parcel omitted: 24 |
| 3217 | 8 | 22 | Units counted in another cell: same parcel ID: 3; Williamson residential parcel omitted: 19 |
| 3580 | 4 | 22 | Units counted in another cell: same parcel ID: 22 |
| 6401 | 18 | 22 | Units counted in another cell: same parcel ID: 22 |
| 2325 | 0 | 20 | Williamson residential parcel omitted: 20 |
| 2515 | 0 | 20 | Williamson residential parcel omitted: 20 |
| 2516 | 8 | 20 | Williamson residential parcel omitted: 20 |
| 2459 | 0 | 19 | Williamson residential parcel omitted: 19 |
| 2510 | 13 | 19 | Williamson residential parcel omitted: 19 |
| 2532 | 0 | 19 | Williamson residential parcel omitted: 19 |
| 3319 | 11 | 18 | Other parcel/account linkage requires review: 18 |
| 2508 | 1 | 16 | Williamson residential parcel omitted: 16 |
| 3731 | 1 | 16 | Units counted in another cell: same parcel ID: 16 |
| 2822 | 11 | 15 | Units counted in another cell: same parcel ID: 15 |
| 2835 | 0 | 14 | Units counted in another cell: same parcel ID: 14 |
| 3769 | 0 | 14 | Other parcel/account linkage requires review: 14 |
| 663 | 0 | 13 | Units counted in another cell: same parcel ID: 13 |
| 2509 | 0 | 13 | Williamson residential parcel omitted: 13 |
| 3261 | 0 | 13 | Williamson residential parcel omitted: 13 |
| 5303 | 0 | 13 | Units counted in another cell: same parcel ID: 13 |
| 602 | 0 | 12 | Other parcel/account linkage requires review: 1; Units counted in another cell: same parcel ID: 11 |
| 6409 | 0 | 11 | Units counted in another cell: same parcel ID: 11 |
| 6965 | 0 | 11 | Geocode just outside parcel boundary: review: 11 |
| 2555 | 1 | 10 | Units counted in another cell: same parcel ID: 10 |
| 3204 | 10 | 10 | Units counted in another cell: same parcel ID: 10 |
| 3322 | 0 | 10 | Units counted in another cell: same parcel ID: 10 |
| 3767 | 0 | 10 | Other parcel/account linkage requires review: 10 |
