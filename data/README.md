# Data Sources and Processing

This directory contains the external data used by the Early Warning System
(EWS), along with saved local copies of downloads and geocoding results. These
saved copies are called **caches** and avoid repeating slow or rate-limited
requests. Most raw records are excluded from Git because they are large,
licensed, or potentially sensitive.
The pipeline reads each source from a specific location declared in
[`_targets.R`](../_targets.R). The analytical role of each resulting variable
and whether it enters the main clusters, a sensitivity test, or a descriptive
profile is documented in
[`config/feature_dictionary.csv`](../config/feature_dictionary.csv).

From the repository root, install or confirm the required R packages and then
run the pipeline:

```bash
Rscript 00_requirements.R
Rscript run_analysis.R
```

The repository uses the R package `{targets}` to determine the correct
processing order. It also remembers which inputs produced each output, so a
changed source or method rebuilds only the affected parts of the analysis.

## Terms Used in This File

- **Residential unit** or **housing unit**: one dwelling intended for one
  household, such as a house, apartment, or condominium unit. A building can
  contain one unit or many units.
- **Parcel**: a mapped piece of land. County appraisal data often use one row
  per tax account, so a row may describe a parcel, an individual condominium,
  a building, or a bookkeeping account. A parcel row is not automatically one
  dwelling.
- **Project** or **development**: one real-world residential property after
  related parcel and appraisal-account records have been grouped. An apartment
  development may span several parcels, and one parcel may have several
  appraisal accounts.
- **Hex**: one cell in the common H3 spatial grid used by the EWS. Source data
  are assigned or allocated to these cells so different datasets can be
  compared at the same geography.
- **Target**: a named processing step managed by `{targets}`. Target names such
  as `unit_projects` identify reproducible steps, not additional datasets.
- **Unit source**: any dataset or documented rule that supplies evidence about
  the number of dwellings at a parcel or project. Examples include an
  appraisal field, a City inventory, or a reviewed property record.
- **Direct count**: a unit total reported for a specific project by a source
  judged reliable enough for the stated use. Direct counts are kept distinct
  from rule-based and modeled estimates.
- **Unit-count surface**: a parcel-level table with one selected unit estimate
  for each eligible residential record. Those estimates can be summed to
  projects, hexes, counties, or the full study area.
- **Promoted** or **canonical**: the version currently used by the main
  pipeline. Earlier alternatives remain available for comparison but do not
  supply the current clustering denominators.
- **Derived output**: a cleaned, linked, aggregated, or modeled file created
  from raw sources. Derived outputs usually live under `output/`.
- **Calibration**: applying and checking an initial rule or estimate against
  known cases before using it in the main analysis.
- **QA**: quality-assurance checks that report source coverage, missingness,
  duplicates, conflicts, or other conditions requiring review.
- **Sensitivity measure**: an alternative input used to test whether a result
  depends on a particular data choice. It is not part of the selected main
  model unless explicitly promoted.
- **Analysis cutoff**: the latest date allowed into a particular run. Records
  after that date are excluded even if the source download contains them.
- **Denominator**: the quantity used to turn a count into a rate or share. For
  example, residential units are the denominator for eviction filings per 100
  units.
- **MOE**: the margin of error published with an ACS survey estimate. It
  describes sampling uncertainty; it is not an error correction.

## Current Source Inventory

| Domain | Main sources | How the source is currently used |
| --- | --- | --- |
| Residential units | Hays, Travis, and Williamson appraisal records; reviewed project sources; floor-area estimates | Creates the promoted parcel-level count of dwellings used for eligibility, rates, ownership shares, and ACS allocation |
| ACS demographics | ACS 5-year estimates and 2020 Census blocks | Creates the current Part 1 demographic-vulnerability measures |
| Rent | ACS median gross-rent vintages; CoStar for matched properties only | ACS supplies the citywide Part 1 rent input; CoStar is a separate sensitivity measure and never fills missing citywide values |
| Evictions | Travis County Justice of the Peace filing extracts | Creates current and hex-year filing measures; equivalent Hays and Williamson records are not yet integrated |
| Demolitions | City of Austin issued construction permits | Creates the current Part 1 demolition measure; the Part 3 historical outcome panel remains to be built |
| 311 requests | City of Austin open-data API | Creates a Part 1 smoke signal from selected code-enforcement request types |
| Appraisal values | County appraisal histories for 2021-2025 | Creates parcel and hex trends in land and improvement values |
| Corporate ownership and sales | Current county ownership classifications plus available deed and sales histories | Measures current corporate ownership and partial changes in ownership and transaction activity |
| Amenity change | Texas Comptroller sales-tax locations with corroborating alcohol and food-establishment sources | Measures recent openings in selected amenity categories near each hex |

## Census/ACS Spatial Allocation

### Why Unit Counts Come First

Many EWS calculations need to know where housing is and approximately how many
dwellings are present. The parcel-based unit-count surface is built before the
ACS data for three reasons:

1. It helps identify which hexes have enough residential development to enter
   the cluster analysis.
2. It supplies denominators for measures such as eviction filings per 100
   units and the share of units under corporate ownership.
3. It provides evidence about where people and housing are located when ACS
   estimates must be allocated from larger Census geographies to smaller
   hexes.

Targets `unit_calibration`, `unit_validation`, `promoted_unit_surface`, and
`corporate_features` create this parcel-level picture of residential
development. The ACS targets, `acs_demographics` and `acs_rent_history`, then
use it as one spatial allocation input. If an appraisal roll, direct project
count, unit-count rule, or model changes, `{targets}` automatically rebuilds
the ownership denominators, ACS allocations, and other dependent outputs.

### Why ACS Estimates Must Be Allocated

The ACS does not publish estimates for EWS hexes. Most demographic variables
are available for block groups, which are larger than the hexes and can
contain both residential and nonresidential land. Assigning a block-group
estimate to every intersecting hex would duplicate people and housing.
Allocating by land area alone would place too much of the estimate on roads,
industrial land, parks, and other places where people do not live.

The pipeline therefore uses **dasymetric allocation**: it uses independent
information about the location of residential development to distribute
larger-area Census totals to the hex grid. This improves the allocation, but it
does not turn block-group survey estimates into directly observed hex-level
counts.

`scripts/data/acs_demographics.R` performs the allocation in two stages:

1. ACS block groups are the source areas containing the demographic
   estimates. Smaller 2020 Census blocks provide population and housing
   controls within each block group.
2. Within each Census block, the controlled totals are distributed among hexes
   according to mapped residential floor area. Where floor area is missing,
   the pipeline uses calibrated unit counts and then residential parcel count.
   A Census block point is used only when no residential parcel evidence is
   available.

Population variables use the Census block's population distribution. Housing,
tenure, rent burden, and population-in-occupied-housing variables use its
housing-unit distribution. Blocks outside the project grid remain in each
block-group denominator, so an edge block group is not incorrectly assigned
entirely to Austin.

Some ACS statistics cannot be divided or added. Median income, median gross
rent, and median home value are therefore not averaged across block groups.
Each hex receives the estimate from the block group containing the largest
share of its mapped residential development. A tract estimate is used only
when the block-group median is suppressed. The output retains the selected
source geography, Census identifier, residential share, allocation method, and
MOE so the choice can be audited. Historical rent estimates follow the same
rule. Poverty uses ACS table `C17002`, which is available at the block-group
level.

Raw ACS and decennial extracts are cached under ignored `data/raw_acs/` files.
Audit outputs include:

- `output/acs_dasymetric_block_hex_allocation.rds/.csv`;
- `output/acs_dasymetric_hex_bg_crosswalk.rds/.csv`;
- `output/acs_dasymetric_allocation_qa.csv`;
- `output/acs_rent_dominant_sources_by_hex_vintage.csv`; and
- `output/acs_rent_dasymetric_crosswalk_qa.csv`.

The downstream `output/amenity_cluster_population_coverage.csv` uses these
independently allocated ACS totals to report the population and housing shares
inside and outside the clustering sample. It does not use parcel-unit counts as
a proxy for population.

### Comparing Parcel and ACS Housing Counts

The parcel and ACS counts are independent measurements:

- The parcel surface assembles appraisal records, reported project totals,
  documented rules, and model estimates. It aims to approximate the number and
  location of physical dwellings.
- The ACS total is a survey estimate for a larger Census geography and time
  period. Its allocation to hexes adds spatial uncertainty to the published
  sampling uncertainty.

The pipeline does not force these totals to agree and does not use the ACS as
an automatic replacement when a parcel count is low. Agreement at the city
level can coexist with substantial local disagreement.

The July 2026 audit found 508,843 promoted parcel units and 479,614 allocated
ACS units on the exact study grid, a parcel excess of 6.1 percent. Inside the
Austin full-purpose boundary, meaning the area under the City's full municipal
jurisdiction, 514,241 parcel units were 0.8 percent below the retained 2024
one-year ACS city benchmark of 518,574. These comparisons use different
boundaries and ACS products; neither proves that individual hex estimates are
correct.

Run `scripts/audits/parcel_acs_housing_units.R` before changing the 20-unit
clustering eligibility rule or any rate denominator. It compares three parcel
surfaces:

- **Promoted**: the current selected hierarchy used by the pipeline.
- **Primary**: the higher pre-model estimate produced by the original parcel
  calibration.
- **Conservative**: a deliberately cautious lower alternative.

The audit identifies hexes where the parcel and ACS estimates agree, disagree
robustly, or differ only after considering model ranges and ACS MOEs. It is
diagnostic and does not alter unit estimates, eligibility, or cluster inputs.
See the
[July 2026 parcel/ACS audit](../docs/audits/parcel-acs-unit-audit-2026-07.md)
for interpretation and current totals.

The reconciliation audit creates:

- `output/parcel_acs_hex_unit_audit.csv`;
- `output/parcel_acs_discordant_hex_review.csv`;
- `output/parcel_acs_discordant_method_audit.csv`;
- `output/parcel_acs_county_unit_summary.csv`;
- `output/parcel_acs_block_group_unit_audit.csv`; and
- `output/parcel_acs_unit_audit_summary.csv`.

Run `scripts/audits/populated_zero_unit_hexes.R` to investigate hexes that
receive allocated ACS population but still have zero units in the promoted
parcel surface. A zero in this audit means that the parcel and project sources
do not support a positive count; it does not necessarily mean that nobody
lives there. The audit checks omitted source records, project totals,
eligibility exclusions, full county parcel maps, the ACS block-point fallback,
and jurisdiction boundaries. It writes `output/populated_zero_unit_*` tables
and `figures/populated_zero_unit_hex_audit.png`. It does not fill parcel gaps
with ACS estimates or change cluster eligibility.

## How Parcel-Based Unit Counts Are Built

County appraisal records are the broadest available inventory of property, but
they do not provide a consistent dwelling count. One apartment development may
appear as several parcel accounts, a condominium may appear as one account per
unit, and some records describe common areas or land rather than housing. The
unit workflow resolves these differences in six steps.

### 1. Identify Residential Appraisal Records

Target `unit_calibration`, implemented in
`scripts/data/parcel_units_calibrate.R`, starts with broad parcel extracts from
Hays, Travis, and Williamson Counties. For Williamson, local WCAD property and
parcel files are classified with `R/wcad_unit_eligibility.R`.
Nonresidential condominiums, reference-only master accounts, park and amenity
parcels, vacant transitional-commercial land, and other non-housing records
remain in audit tables but are excluded from the unit-count surface.

The Williamson input is also checked against the current certified appraisal
roll. The comparison found 408 active residential records inside the study
grid that needed to be added or linked through a reviewed geometry proxy.
`R/wcad_residential_supplement.R` adds those records under the existing WCAD
rules. Reviewed exceptions are recorded in
`config/residual_unit_parcel_reviews.csv`.

This repository owns the eligibility rules. It may read cached data produced
for the sibling `landlord-mapper` project, but it does not import or execute
that repository's code.

### 2. Collect Unit-Count Evidence

Target `unit_sources`, implemented in
`scripts/data/unit_counts/prepare_sources.R`, collects the available evidence:

- unit and building fields from Travis County Appraisal District (TCAD)
  property profiles;
- explicit account classifications from Hays and Williamson appraisal data;
- City Affordable Housing Inventory project totals; and
- City Universal Recycling Ordinance (URO) multifamily estimates.

A reported project total is stored once in a source table. Parcel links are
stored separately. This prevents a 200-unit development linked to three
parcels from being mistakenly counted as 600 units.

### 3. Build Real-World Projects

Target `unit_projects`, implemented in
`scripts/data/unit_counts/build_projects.R`, groups related parcel and
appraisal-account records into physical developments. Grouping requires strong
source or address-and-location evidence; owner name alone is not sufficient.
Complete condominium or small-multifamily account groups can be summed.
Conflicting direct sources are retained for review rather than averaged or
used to train the model. A development crossing a county line remains one
project even though both appraisal districts may contain records for it.

### 4. Estimate Unresolved Multifamily Projects

Some apartment projects have usable residential floor area but no reliable
reported unit total. Target `unit_models`, implemented in
`scripts/data/unit_counts/fit_models.R`, compares several ways to estimate
their units from floor area and building characteristics. Validation with
held-out projects determines which method is sufficiently accurate and whether
a candidate resembles the projects used for training. The selected model is
not applied to records outside that supported range.

### 5. Validate Williamson County Transfer

Target `williamson_validation`, implemented in
`scripts/data/unit_counts/validate_williamson.R`, tests whether the Travis-based
floor-area relationship transfers to Williamson. It treats WCAD rows as
appraisal accounts, not automatically as separate developments, and applies
the reviewed project groupings in
`config/williamson_project_groups.csv`.

Documented comparison counts come from
`config/williamson_unit_validation_sources.csv`. Austin Energy Green Building
(AEGB) and Texas Department of Housing and Community Affairs (TDHCA)
inventories provide limited additional coverage. A nonmatch means only that a
property is absent from that program inventory; it is never interpreted as
zero units. The HUD multifamily service is documented in the source manifest
but was not used after repeated timeouts.

### 6. Select One Count and Promote It

Target `unit_integration`, implemented in
`scripts/data/unit_counts/build_integration.R`, selects one count for each
project in the following order:

1. a reliable direct project total;
2. a separately documented project count;
3. the validated floor-area estimate for a project within the model's
   supported range; or
4. the earlier parcel-based estimate for a project requiring review or falling
   outside that range.

That earlier estimate is called the **targeted baseline**. It is retained as a
fallback and for comparison with the new hierarchy. Cross-county duplicates
and reviewed companion accounts are assigned to one parcel representation
before units are summed.

The integration was first written to `unit_shadow` files. Here, **shadow**
means a candidate surface evaluated beside the existing pipeline without
changing downstream results. After the comparison passed its validation
checks, target `promoted_unit_surface` made the new hierarchy the default and
wrote `output/residential_parcels_unit_promoted.rds`. Use
`EWS_UNIT_SURFACE=baseline` only to reproduce the historical pre-promotion
surface.

The current hierarchy, caveats, validation gates, and complete output list are
documented in
[`docs/methods/unit-count-modeling.md`](../docs/methods/unit-count-modeling.md).
Raw and compact source extracts remain ignored under
`data/raw_parcels/unit_sources/`.

## Displacement Proxies

In this project, a **displacement proxy** is a relatively concrete event or
change associated with displacement pressure, such as a rent increase,
eviction filing, demolition permit, or land-value increase. It is distinct
from a **smoke signal**, which is an earlier or less direct indication that
neighborhood conditions may be changing.

### Rent Pressure

The citywide Part 1 rent measure comes from target `acs_rent_history`. It uses
ACS 5-year median gross-rent estimates ending in 2014, 2019, and 2024. These
ending years are called **vintages**. Dollar amounts are adjusted to 2025
dollars so an increase reflects more than general inflation.

The rent-pressure index combines:

- the most recent rent level;
- the recent percentage change, when the ACS MOEs support interpreting that
  change; and
- acceleration, meaning whether rent growth became faster or slower in the
  more recent period.

Each hex receives the median from the block group containing the largest share
of its mapped residential development. A tract value is used only when the
block-group estimate is unavailable. Because medians are not additive, the
pipeline never averages block-group medians to create a hex median.

CoStar covers only multifamily properties that appear in its database. It is
therefore not used to fill missing citywide rent data. Files
`data/CoStarHistoric-clean.csv` and `data/geocoded_buildings.csv` create a
separate `costar_rent_pressure_index` for matched properties.
`costar_present` identifies hexes with a match, and missing CoStar values remain
missing rather than becoming zeros.

Key outputs are:

- `output/acs_rent_by_hex_vintage.rds/.csv`;
- `output/acs_rent_trends_by_hex.rds/.csv`; and
- the CoStar fields in `output/hex_features.rds`.

### Eviction Filings

Targets `prepared_evictions` and `eviction_features` process defendant records
from Travis County Justice of the Peace courts. The pipeline standardizes the
filing addresses, geocodes them, retains ArcGIS address matches scoring at
least 90, assigns the matched filings to hexes, and excludes records after the
configured analysis date.

The Part 1 measure compares the most recent 12 months with the preceding 12
months. It includes the recent filing rate per 100 promoted residential units
and the change between the two periods. Dividing by units makes a hex with 20
filings and 200 dwellings different from a hex with 20 filings and 2,000
dwellings. A zero is used only when the source covers the location and no
filing was observed; a rate requires a positive unit denominator.

The available filing extracts cover Travis County only. The locked Part 1
baseline therefore understates eviction pressure in the Hays and Williamson
portions of Austin. About 6.6 percent of the population in cluster-eligible
hexes is in cells whose parcel units are predominantly in those two counties.
This limitation must be stated when presenting the clusters and addressed in a
future eviction-data update.

Key outputs are:

- `output/eviction_filings_prepared_for_geocoding.csv`;
- `output/eviction_filings_by_hex_summary.rds/.csv`; and
- `output/eviction_filings_by_hex_year.csv`.

Eviction ingestion is implemented. The existing hex-year file still needs
completeness and outcome-definition validation before Part 3 forecasting.

### Residential Demolitions

Target `current_features` reads
`data/Issued_Construction_Permits_20260401.csv`, keeps geocoded permits
classified as residential, and excludes permits after the analysis date. It
compares the most recent 24 months with the preceding 24 months. The resulting
`demolition_pressure_index` represents the level and recent change in permitted
residential demolition activity. An issued permit indicates authorized
activity, not necessarily a completed demolition.

Part 3 still requires a dedicated, validated
`output/demolition_permits_by_hex_year.csv` historical outcome file. The
pipeline does not accept a generic `demolitions.csv` in its place.

### County Appraisal Values

The appraisal-value workflow tracks changes in land value that may signal
gentrification or redevelopment pressure. A **panel** is a table that follows
the same parcel or hex across multiple years.

`scripts/data/appraisal_history.R` runs after the residential parcel set has
been defined. Its source inventory is versioned in
`config/appraisal_sources.csv`; downloaded archives and matched county-year
files are cached under `data/raw_parcels/appraisal_history/`.

The script creates:

- `output/appraisal_values_by_parcel_year.rds/.csv`
- `output/appraisal_values_by_hex_year.rds/.csv`
- `output/appraisal_value_trends_by_hex.rds/.csv`
- `output/appraisal_panel_source_qa.csv`
- `output/appraisal_panel_spatial_qa.csv`
- `output/appraisal_county_land_values_by_account_year.rds`

Dollar values are converted to 2025 dollars using the Consumer Price Index for
All Urban Consumers (CPI-U). Missing source years remain missing rather than
being interpreted as zero value.
The selected Hays 2022 source is the local certified fixed-width roll; the
later post-certification export is retained only for an explicit source
comparison. Other Hays archives can be placed at
`data/raw_parcels/appraisal_history/hays/<year>/hays_<year>.zip` before rerunning.

`scripts/data/appraisal_adjusted_trends.R` separates parcel-specific change
from broad changes in county appraisal practice. It estimates the typical
annual county shift using stable real-property accounts with buildings, then
subtracts that background shift from each parcel's change. It creates:

- `output/appraisal_county_year_baselines.csv`
- `output/appraisal_adjusted_parcel_trends.rds`
- `output/appraisal_adjusted_trends_by_hex.rds/.csv`
- `output/appraisal_adjustment_qa.csv`
- `output/appraisal_land_area_fallback_qa.csv`

Land-value levels are expressed relative to parcel land area. Reported,
calibrated land area is preferred. Where Williamson acreage is missing, the
pipeline calculates area from the mapped parcel polygon and compares that
fallback with parcels that do report acreage.

## Smoke Signals

A **smoke signal** is an indirect or emerging indication of displacement
pressure. It may help distinguish changing places before a concrete
displacement outcome is fully visible, but it should not be interpreted as
proof that displacement occurred.

### Austin 311 Requests

Target `requests_311` queries Austin's Socrata endpoint `xwdj-i9he`, truncates
requests at the analysis cutoff, and retrieves only the exact code-enforcement
intake descriptions versioned in `config/311_smoke_signal_types.csv`. Follow-up
workflow records, general 311 activity, drainage, debris, water, park
maintenance, and other unrelated requests do not enter the Part 1 index.

The index combines three measures: selected requests per 100 residential
units, selected requests per square kilometer, and change from the preceding
period of equal length. The first adjusts for the amount of housing; the second
captures geographic concentration. A 311 request reflects reported concern
and access to the reporting system, not a verified code violation.

The selected request-level extract is cached under `data/raw_311/`, so later
feature or clustering runs do not need to restream it. Anonymous public access
is supported; `AUSTIN_DATA_API_KEY` and `AUSTIN_DATA_API_SECRET` are optional.
Key outputs are:

- `output/311_requests_by_hex_summary.rds/.csv`;
- `output/311_requests_by_hex_year.csv`;
- `output/311_service_request_counts.csv`;
- and `output/311_service_request_selection.csv`.

### Corporate Ownership and Transaction History

This workflow tracks current corporate ownership, recent entry by corporate
owners, and property-sale activity. **Corporate ownership** means that the
current appraisal owner was classified as a company or other legal entity
rather than a natural person. It does not by itself establish that the owner is
a large institutional investor.

Current corporate ownership is processed by
`scripts/data/corporate_ownership.R`. Run
`scripts/audits/ownership_transactions.R` before
`scripts/data/ownership_transactions.R`. The audit checks the following
sources against the exact eligible residential parcel set:

- cached Travis deed events from the sibling `landlord-mapper` repository;
- Hays annual `OWNER` exports and the latest `SALES` history;
- Williamson 2023-2024 certified owner reports and current owner data; and
- WCAD's `Sales History - Certified` dataset (`kdj3-9hpg`).

The WCAD sales download is stored at
`data/raw_parcels/williamson/wcad_sales_history_certified.csv`. Owner and sales
row-level data remain ignored; only aggregate coverage QA is written to
`output/`.

`scripts/data/ownership_transactions.R` builds parcel- and hex-level outputs:

- `output/ownership_transaction_features_by_parcel.rds`;
- `output/ownership_transaction_features_by_hex.rds/.csv`;
- `output/ownership_transaction_source_qa.csv`;
- `output/transaction_event_type_qa.csv`; and
- `output/transaction_source_year_qa.csv`.

Transaction pressure compares two 24-month periods. **Parcel turnover** is the
share of residential parcels with a qualifying deed or sale. **Unit exposure**
is the share of residential units located on those parcels. **Positive
acceleration** means that activity increased in the more recent period. The
latest complete common source cutoff is April 30, 2025, so the windows are May
1, 2021-April 30, 2023 and May 1, 2023-April 30, 2025. Travis uses warranty
deeds and special warranty deeds; Hays uses comparable warranty-deed variants.
WCAD's current certified sales-history table
omits 2022-2023 and nearly all 2024 events, so Williamson transaction pressure
is deliberately missing rather than treated as zero. The raw partial event
counts remain in QA outputs.

Corporate-ownership change focuses on **corporate entry**, meaning a transition
from a noncorporate owner to a corporate owner. Hays and Williamson use annual
2023-2025 owner records to identify these transitions. Recent Travis deed-party
names are blank; for Travis only, a corporate acquisition is inferred when a
parcel with a recent market deed is corporate-owned in the current appraisal
extract. **Corporate dispositions**, or sales out of corporate ownership,
cannot be measured consistently for Travis and are excluded.

The audit also documents that Williamson's 2021-2022 `ASMNT`
field named `OwnerQuickRefID` duplicates the parcel reference and is not a
comparable owner identity. Williamson corporate-ownership change measures
therefore begin in 2023.

### Amenity Change

This workflow measures recent openings in selected commercial categories that
may indicate neighborhood reorientation. It does not treat the mere presence
of amenities, or overall commercial density, as displacement pressure.

Run `scripts/audits/amenity_sources.R` before
`scripts/data/amenities.R`. The audit downloads and caches:

- Texas Comptroller permitted sales-tax locations active at any point in the
  prior 48 months (`3kx8-uryv`), including NAICS, first-sale date, and
  out-of-business date;
- Texas mixed-beverage reports (`naix-2893`) for alcohol-establishment
  corroboration; and
- Austin food-establishment inspections (`ecmv-9xxi`) for local
  corroboration only.

The versioned classification is `config/amenity_categories.csv`. The core
score uses cafes, full-service restaurants, and drinking places. Craft alcohol,
specialty food, and fitness remain exploratory categories. North American
Industry Classification System (NAICS) codes are standard industry
classifications. Because code 722515 also contains businesses unrelated to the
intended cafe concept, a name filter removes fast-food and other mismatches.

The processing script compares equal 18-month windows ending April 1, 2026,
batch geocodes unique addresses with the US Census geocoder, and measures
distance-weighted exposure within 800 meters. A nearby opening contributes more
than one near the edge of that radius. Each category is placed on a comparable
0-100 scale and the categories receive equal weight, so the more numerous
restaurant records cannot dominate the score. Closures remain visible in the
quality-control files but do not enter the first Part 1 index.

Generated raw API extracts and geocode caches remain ignored under
`data/raw_amenities/`. Aggregate outputs include:

- `output/amenity_source_audit.csv`;
- `output/amenity_category_year_qa.csv`;
- `output/amenity_window_change_qa.csv`;
- `output/amenity_geocoding_qa.csv`; and
- `output/amenity_change_features_by_hex.rds/.csv`.

## Part 3 Readiness

Part 3 forecasting is not yet implemented. The available source data should not
be confused with modeling-ready outcome panels. An **outcome panel** needs one
consistent observation for each hex and time period, a clear definition of the
event or change being predicted, and enough historical coverage for
validation. Target `part3_forecast_readiness` checks four proposed outcomes:

- eviction filings: the hex-year file exists; construct and validate the
  complete outcome panel;
- residential demolitions: build the historical hex-year file;
- rent growth: the ACS vintage file exists, with CoStar context; define and
  validate the forecasting outcome without treating missing CoStar as zero;
- land-value growth: the appraisal hex-year file exists; construct and
  validate a comparable county-adjusted outcome panel.

See [`config/forecast_outcomes.csv`](../config/forecast_outcomes.csv) and
`output/part3/forecast_readiness.csv` for the machine-readable specification and
current check.

## Data Privacy

Eviction records, owner names, and individual addresses can contain sensitive
or personally identifiable information. Raw and row-level inputs remain local
and ignored by Git. Share only reviewed aggregate outputs, and inspect every
export for address or name fields before release.
