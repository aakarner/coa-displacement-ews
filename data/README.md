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

## Raw Input Data Sources

The table below lists external records and reference series before the EWS
cleans, links, allocates, aggregates, or models them. It therefore excludes the
hex grid, estimated unit counts, corporate-owner classifications, pressure
indices, and other derived products. Geocoding services are also processing
tools rather than substantive evidence and are not listed as data sources.

| Provider | Raw input | Vintage or period used | What it contributes |
| --- | --- | --- | --- |
| U.S. Census Bureau TIGER/Line | Austin place boundary | 2021 geography | Defines the polygon used to generate the current H3 grid |
| City of Austin GIS | Full- and limited-purpose jurisdiction boundaries | April 29, 2026 snapshot | Supports current boundary comparisons, coverage audits, and map context; it does not redefine the existing grid |
| City of Austin Planning | Detailed Land Use Inventory | July 2026 download of the December 2025 inventory | Independently checks whether residential parcel records fall on land classified as single-family, duplex, three/fourplex, apartment/condominium, retirement housing, mixed use, or another use; it does not supply unit counts |
| Travis, Hays, and Williamson county appraisal districts | Current parcel and appraisal-account extracts plus annual certified appraisal rolls | Current residential-candidate extracts; annual value records for 2021-2025 | Supplies property location, use, owner of record, floor and land area, appraisal values, and any reported unit fields |
| County clerks and county appraisal districts | Dated deed and sale records plus historical owner files | Primarily 2021-2025, with different gaps by county | Supplies transaction events and owner histories used to measure turnover and changes in corporate ownership |
| City of Austin and reviewed housing-project sources | Affordable Housing Inventory, Universal Recycling Ordinance inventory, Austin Energy Green Building and TDHCA records, and reviewed project documents | Most recent locally available record for each project | Supplies additional reported or comparison unit counts for identified residential developments |
| U.S. Census Bureau | 2020 Decennial Census blocks | 2020 | Supplies small-area population and housing controls for allocating ACS estimates |
| U.S. Census Bureau | ACS 5-year demographic and housing estimates | 2024 for current demographics; 2014, 2019, and 2024 for rent history | Supplies demographic vulnerability, housing estimates, and the citywide rent series |
| CoStar | Historical multifamily property, unit, asking-rent, and vacancy records | Licensed historical extract available to the project | Supplies project evidence and a coverage-limited rent sensitivity measure; it is not a citywide rent source |
| Travis County Justice of the Peace courts | Eviction filing extracts | January 2020 through the April 1, 2026 analysis cutoff | Supplies observed eviction filings for Travis County |
| City of Austin Development Services | Issued construction permits | Records issued through the April 1, 2026 analysis cutoff | Supplies permitted residential demolitions within source coverage |
| City of Austin 311 | Code-enforcement service-request records | Requests through the April 1, 2026 analysis cutoff | Supplies selected housing-condition and code-related requests |
| Texas Comptroller and City of Austin | Permitted sales-tax locations, mixed-beverage reports, and food-establishment inspections | Recent 48-month source history; equal 18-month analysis windows ending April 1, 2026 | Supplies and corroborates openings in selected amenity categories |
| U.S. Bureau of Labor Statistics | CPI-U annual averages | Years corresponding to the configured ACS and appraisal vintages | Converts rent and appraisal-value measures to constant dollars |

The first county row describes **property snapshots**: what each appraisal
account contains and who owns it in a particular tax year. The following row
describes **event and history records**: when a deed or sale occurred and how
the owner changed. Some annual owner histories are constructed by comparing
appraisal snapshots, so the evidence overlaps even though the source products
and analytical purposes differ. The current residential-candidate files and
their initial corporate-owner classifications were generated from county data
in the sibling `landlord-mapper` repository; the dependency is described under
[Identify Residential Appraisal Records](#1-identify-residential-appraisal-records).

## Current Source Inventory

Sources are grouped below by their analytical role. Foundational sources define
the residential denominator and socioeconomic vulnerability. Displacement
proxies measure relatively concrete events or changes associated with
displacement pressure. Smoke signals measure earlier or less direct signs that
neighborhood conditions may be changing.

### Foundational Context and Vulnerability

| Domain | Main sources | How the source is currently used |
| --- | --- | --- |
| Residential units | Hays, Travis, and Williamson appraisal records; reviewed project sources; floor-area estimates | Creates the promoted parcel-level count of dwellings used for eligibility, rates, ownership shares, and ACS allocation |
| ACS demographics | ACS 5-year estimates and 2020 Census blocks | Creates the current Part 1 demographic-vulnerability measures |

### Displacement Proxies

| Domain | Main sources | How the source is currently used |
| --- | --- | --- |
| Rent | ACS median gross-rent vintages; CoStar for matched properties only | ACS supplies the citywide Part 1 rent input; CoStar is a separate sensitivity measure and never fills missing citywide values |
| Evictions | Travis County Justice of the Peace filing extracts | Creates current and hex-year filing measures; equivalent Hays and Williamson records are not yet integrated |
| Demolitions | City of Austin issued construction permits | Creates the current Part 1 demolition measure; the Part 3 historical outcome panel remains to be built |
| Appraisal values | County appraisal histories for 2021-2025 | Creates parcel and hex trends in land and improvement values |

### Smoke Signals

| Domain | Main sources | How the source is currently used |
| --- | --- | --- |
| 311 requests | City of Austin open-data API | Creates a Part 1 smoke signal from selected code-enforcement request types |
| Corporate ownership and sales | Current county ownership classifications plus available deed and sales histories | Measures current corporate ownership and partial changes in ownership and transaction activity |
| Amenity change | Texas Comptroller sales-tax locations with corroborating alcohol and food-establishment sources | Measures recent openings in selected amenity categories near each hex |

## Calculated Variables and Units

The tables below describe the main analysis-facing fields in
`output/hex_features.rds` and the promoted residential-unit surface. They omit
identifiers, geometry, source dates, provenance fields, and detailed
quality-control columns.

The pipeline calculates more variables than it sends to the cluster algorithm.
The additional fields are retained because they:

1. provide interpretable component measures behind each composite index;
2. supply denominators, geographic context, eligibility, and coverage checks;
3. allow analysts to inspect and explain why a hex receives a particular
   domain score; and
4. preserve candidate measures for sensitivity analysis, later updates, and
   Part 3 forecasting.

The locked six-cluster solution uses only the seven composite fields marked
with `*` below. Using one composite index for each conceptual domain prevents a
domain with many available component variables from receiving extra weight
simply because it has more columns. Before clustering, the seven indices are
standardized to a common mean and standard deviation. Component variables,
eligibility flags, and sensitivity indices remain in the feature table but do
not separately enter the clustering distance.

Unless stated otherwise, a count is the count within one hex. Fields beginning
with `pct_`, and coverage fields ending in `_pct`, are percentages measured
from 0 to 100. Growth and change fields ending in `_pct` are percentage changes
and can be negative. Fields explicitly described as shares are fractions from
0 to 1. The composite indices are unitless 0-100 relative scores; they are not
percentages, probabilities, or counts.

### Residential Context and Demographics

| Variable or variables | Unit | Meaning |
| --- | --- | --- |
| `area_km2` | square kilometers | Area of the H3 cell, used for density measures |
| `promoted_units` | dwellings per parcel representation | Selected direct or modeled unit count carried into the production parcel surface |
| `residential_parcels`, `residential_units` | parcels; dwellings | Residential parcel records and summed promoted dwellings in the hex |
| `residential_improvement_sqft`, `residential_land_sqft` | square feet | Summed appraisal-reported or calibrated improvement and land area |
| `residential_parcels_per_km2`, `residential_units_per_km2` | parcels or dwellings per square kilometer | Residential development density |
| `total_pop`, `population_in_occupied_housing` | people | Allocated ACS total population and population living in occupied housing |
| `total_housing_units`, `total_tenure`, `owner_occupied`, `renter_occupied`, `gross_rent_occupied` | housing units | Allocated total, occupied, owner-occupied, renter-occupied, and gross-rent-reporting housing counts |
| `white_nh`, `black_nh`, `asian_nh`, `hispanic` | people | Allocated race and ethnicity counts |
| `pct_white`, `pct_black`, `pct_asian`, `pct_hispanic`, `pct_poc` | percent of total population | Race and ethnicity composition; `pct_poc` is the share not classified as non-Hispanic White |
| `total_edu`, `less_than_hs`, `hs_grad`, `some_college`, `bachelors`, `graduate` | people age 25 and older | Allocated educational-attainment counts |
| `pct_college` | percent of people age 25 and older | Share with a bachelor's or graduate degree |
| `total_poverty_det`, `below_poverty`, `poverty_under_050`, `poverty_050_099` | people for whom poverty status is determined | Allocated poverty-universe and below-poverty counts |
| `poverty_rate` | percent | `below_poverty` divided by `total_poverty_det` |
| `rent_burden_30_34`, `rent_burden_35_39`, `rent_burden_40_49`, `rent_burden_50_plus`, `rent_burdened_30plus` | renter-occupied housing units | Counts by gross rent as a share of household income and their 30-percent-or-more total |
| `pct_renter`, `pct_owner`, `pct_rent_burden_30plus` | percent | Tenure and rent-burden rates |
| `median_income`, `median_rent`, `median_home_value` | dollars per year; dollars per month; dollars | ACS medians assigned from the dominant residential block group, with tract fallback |
| `rent_burden_proxy` | ratio from 0 upward | Annualized median rent divided by median household income |
| `demographic_vulnerability_index` `*` | unitless 0-100 index | Cluster input: equal-weight relative score for lower income, renter share, poverty, rent burden, and lower college attainment |
| `demographic_vulnerability_equity_index` | unitless 0-100 index | Sensitivity version that also includes `pct_poc` |
| ACS fields ending in `_moe` | same unit as the estimate | Published ACS margin of error allocated or assigned with its estimate |
| `*_relative_moe` | fraction from 0 upward | MOE divided by the corresponding estimate; used for reliability checks |
| `missing_feature_count`, `missing_feature_pct`, `sufficient_data`, `primary_cluster_eligible` | count; percent; logical flags | Feature completeness and whether the hex meets the current Part 1 eligibility rules |

### Displacement Proxies

| Category | Variable or variables | Unit and interpretation |
| --- | --- | --- |
| ACS rent | `acs_rent_current`, `acs_rent_current_real` | Median gross rent in source-year dollars per month and 2024 dollars per month |
| ACS rent | `acs_rent_growth_recent_annualized_pct`, `acs_rent_growth_prior_annualized_pct`, `acs_rent_growth_long_annualized_pct` | Inflation-adjusted annualized percent change per year |
| ACS rent | `acs_rent_acceleration_pp` | Recent minus prior annualized growth, in percentage points |
| ACS rent | `rent_pressure_citywide_index` `*` | Cluster input: unitless 0-100 index combining current real rent, reliable recent growth, and acceleration |
| CoStar rent | `costar_present` | Coverage flag: 1 for at least one matched CoStar property, otherwise 0 |
| CoStar rent | `rent_current`, `rent_psf_current`, `vacancy_pct_current` | Asking rent in dollars per unit per month, asking rent in dollars per square foot per month, and vacancy percent |
| CoStar rent | `rent_units_current`, `n_buildings_current` | CoStar inventory units and matched buildings |
| CoStar rent | `rent_change_recent`, `rent_change_total`, `costar_rent_growth_recent_annualized_pct`, `costar_rent_growth_long_annualized_pct` | Percent change or annualized percent change per year |
| CoStar rent | `rent_acceleration`, `costar_rent_acceleration_pp`; `rent_volatility` | Percentage-point change in growth; coefficient of variation of recent rent |
| CoStar rent | `costar_rent_pressure_index` | Unitless 0-100 sensitivity index; missing outside CoStar coverage |
| Evictions | `eviction_cases_total`, `eviction_cases_<year>`, `eviction_cases_latest_12mo`, `eviction_cases_previous_12mo` | Unique eviction filing cases |
| Evictions | `eviction_defendant_rows_total`, `eviction_final_status_cases_total`, `eviction_dismissed_cases_total` | Defendant rows or unique cases by recorded court status |
| Evictions | `eviction_cases_per_100_units`, `eviction_latest_12mo_per_100_units` | Filing cases per 100 promoted residential units |
| Evictions | `eviction_cases_total_density`, `eviction_cases_latest_12mo_density` | Filing cases per square kilometer |
| Evictions | `eviction_cases_latest_12mo_change_pct`; `eviction_recent_share` | Percent change between 12-month windows; recent cases as a 0-1 fraction of all observed cases |
| Evictions | `eviction_pressure_index` `*` | Cluster input: unitless 0-100 index combining the recent rate, change, and recent share |
| Demolitions | `demo_count_total`, `demo_count_<year>`, `demo_latest_24mo`, `demo_previous_24mo` | Issued residential demolition permits |
| Demolitions | `demo_total_demolition_count`, `demo_total_latest_24mo`, `demo_total_previous_24mo` | Permits whose descriptions indicate total demolition |
| Demolitions | `demo_density`, `demo_recent_density`, `demo_total_recent_density` | Permits per square kilometer |
| Demolitions | `demo_trend`, `demo_total_trend` | Difference in log-transformed counts between equal 24-month windows |
| Demolitions | `demolition_pressure_index` `*` | Cluster input: unitless 0-100 index combining recent residential-demolition density, positive change, and recent total-demolition density |
| Land value | `land_value_real_per_current_land_sqft` | Median 2025 dollars of appraised land value per square foot |
| Land value | `land_value_county_project_percentile_current` | Median parcel land-value-per-square-foot percentile within county, from 0 to 100 |
| Land value | `land_value_growth_long_county_adjusted_pct`, `land_value_growth_recent_county_adjusted_pct`, `land_value_growth_prior_county_adjusted_pct` | Average annual real log change after subtracting the county-year background shift, in percent per year |
| Land value | `land_value_acceleration_county_adjusted_pp` | Recent minus prior county-adjusted growth, in percentage points |
| Land value | `appraisal_adjusted_trend_parcel_coverage_pct`, `appraisal_current_level_parcel_coverage_pct` | Percent of relevant parcels with complete trend or current-level evidence |
| Land value | `land_value_pressure_index` | Unitless 0-100 sensitivity index combining current county percentile, long and recent adjusted growth, and acceleration |

### Smoke Signals

| Category | Variable or variables | Unit and interpretation |
| --- | --- | --- |
| 311 | `sr_311_total`, `sr_311_smoke_signal_total`, `sr_311_latest_12mo`, `sr_311_previous_12mo`, `sr_311_smoke_signal_latest_12mo`, `sr_311_smoke_signal_previous_12mo` | Selected service-request counts |
| 311 | `sr_311_code_related_total`, `sr_311_housing_condition_total`, `sr_311_tenant_distress_total`, `sr_311_nuisance_or_disorder_total` | Selected request counts by configured smoke-signal group |
| 311 | `sr_311_*_per_100_units`, `sr_311_*_density` | Requests per 100 promoted residential units or per square kilometer |
| 311 | `sr_311_*_change_pct`; `sr_311_smoke_signal_share` | Percent change between equal 12-month windows; smoke-signal requests as a 0-1 fraction of selected requests |
| 311 | `sr_311_pressure_index` `*` | Cluster input: unitless 0-100 index combining recent smoke-signal requests per 100 units, density, and change |
| Corporate ownership | `corporate_owned_parcels`, `corporate_owned_units`, `corporate_owned_imprv_sqft`, `corporate_owner_count`, `financialized_owner_parcels` | Parcels, promoted dwellings, square feet, distinct classified owners, and parcels |
| Corporate ownership | `pct_corporate_parcels`, `pct_corporate_units`, `pct_corporate_improvement_sqft`, `pct_financialized_owner_parcels` | Percent of the corresponding residential parcel, unit, or floor-area denominator |
| Corporate ownership | `corporate_owned_units_per_km2`, `corporate_owned_parcels_per_km2` | Corporate-owned dwellings or parcels per square kilometer |
| Corporate ownership | `ownership_pressure_index` `*` | Cluster input: unitless 0-100 index combining corporate unit share, corporate-unit density, and financialized-owner parcel share |
| Property transactions | `transaction_recent_count`, `transaction_previous_count`, `transaction_recent_parcels`, `transaction_previous_parcels` | Qualifying deed or sale events and affected parcels in equal 24-month windows |
| Property transactions | `transaction_recent_per_100_parcels`, `transaction_previous_per_100_parcels`, `transaction_recent_per_100_units` | Events per 100 eligible parcels or promoted residential units |
| Property transactions | `transaction_recent_unit_exposure_pct`, `transaction_rate_change_per_100_parcels`; `transaction_log_count_change` | Percent of units on transacting parcels; rate difference; difference in log-transformed counts |
| Property transactions | `transaction_pressure_index` | Unitless 0-100 sensitivity index; missing where the two transaction windows are incomplete |
| Corporate entry | `ownership_change_recent_count`, `corporate_acquisition_recent_count`, `corporate_disposition_recent_count`, `corporate_net_acquisition_recent_count` | Parcels with owner change, corporate entry, corporate exit, and net corporate entry |
| Corporate entry | `corporate_acquisition_recent_per_100_parcels`, `corporate_net_acquisition_recent_per_100_parcels` | Events per 100 parcels with comparable ownership history |
| Corporate entry | `corporate_acquisition_recent_unit_exposure_pct`; `corporate_acquisition_recent_share` | Percent of covered units exposed to corporate entry; corporate entries as a 0-1 fraction of observed ownership changes |
| Corporate entry | `ownership_change_index` | Unitless 0-100 sensitivity index combining entry rate, unit exposure, positive net entry, and entry share |
| Amenities | `count_<category>_recent`, `count_<category>_previous` | Unique opening events within 800 meters of the hex centroid, for cafes, full-service restaurants, and drinking places |
| Amenities | `<category>_recent`, `<category>_previous` | Distance-weighted opening exposure; one event at the centroid contributes 1 and its weight declines linearly to 0 at 800 meters |
| Amenities | `amenity_<category>_weighted_change`, `amenity_<category>_score` | Difference in weighted exposure; unitless 0-100 category score |
| Amenities | `amenity_recent_opening_events`, `amenity_previous_opening_events`; `amenity_recent_weighted_openings`, `amenity_previous_weighted_openings`, `amenity_weighted_opening_change` | Opening-event counts; summed distance-weighted event equivalents |
| Amenities | `amenity_change_index` `*` | Cluster input: unitless 0-100 index giving equal weight to cafe, full-service restaurant, and drinking-place scores |

Year-specific fields use the four-digit year in place of `<year>`. Amenity
fields use `cafe`, `full_service_restaurant`, or `drinking_place` in place of
`<category>`. Reliability and coverage fields determine whether a measure is
usable; missing values are not silently converted to zero outside documented
source coverage. The analytical role and missingness rule for each top-level
index are versioned in `config/feature_dictionary.csv`.

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
   using appraisal-reported improvement or living-area square footage attached
   to geocoded residential parcel points. These are parcel attributes, not
   mapped building footprints. The Travis input uses TCAD total improvement
   area. Hays uses reported living area with summed building-segment area as a
   fallback. Williamson uses reported living area, then residential floor area,
   and then building area.

Where a parcel lacks positive floor area, its promoted unit count supplies the
weight. If that is also unavailable, the residential parcel count supplies the
weight. Each fallback unit or parcel is multiplied by 1,000 so it can be
combined with square-footage weights; this is a scaling convention, not a
claim that every dwelling contains exactly 1,000 square feet. A Census block
point is used only when the block contains no residential parcel point.

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

After the July 31 City land-use validation, the current surface contains
502,257 promoted parcel units and 479,650 allocated ACS units on the exact
study grid, a parcel excess of 4.7 percent. Inside the
Austin full-purpose boundary, meaning the area under the City's full municipal
jurisdiction, 507,653 parcel units are 2.1 percent below the retained 2024
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

The three county inputs under `data/` are residential-candidate files generated
by county-specific scripts in the sibling `landlord-mapper` repository.
Travis records are already filtered using residential appraisal state codes or
single-family/multifamily zoning. Hays records are already filtered using
residential improvement or land state codes. The files also carry the upstream
corporate-owner and financialized-owner classifications used by the current
EWS pipeline.

Williamson records receive an additional EWS review because WCAD uses separate
accounts for some condominium, common-area, and reference properties. Target
`unit_calibration`, implemented in
`scripts/data/parcel_units_calibrate.R`, applies the local rules in
`R/wcad_unit_eligibility.R`, retains excluded records in audit tables, and
checks the input against the certified appraisal roll. The current supplement
adds 408 residential records that were missing or needed a reviewed geometry
link. Exceptions are recorded in
`config/residual_unit_parcel_reviews.csv`.

The EWS does not execute `landlord-mapper` code, but it currently depends on
these generated files and their embedded initial residential and ownership
classifications. [GitHub issue #5](https://github.com/aakarner/coa-displacement-ews/issues/5)
tracks moving the complete county parcel-ingestion and corporate-classification
workflow into this repository.

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
wrote `output/residential_parcels_unit_promoted.rds`.

Promotion also applies a narrow current-land-use safeguard. A modeled or
fallback project is removed from residential calculations when it is inside
Austin's full-purpose boundary, matches the City Land Use Inventory by exact
parcel identifier, is classified there only as nonresidential or group
quarters, and has no appraisal multifamily code. This rule addresses cases
that entered the candidate file because housing was allowed by zoning even
though no source confirmed a current residential use. The table retains each
removed row, its pre-validation estimate, and the exclusion reason for audit;
downstream housing totals, corporate-ownership denominators, ACS allocation,
and cluster eligibility exclude it. Direct and deterministic counts are not
removed automatically when the sources disagree.

Use `EWS_UNIT_SURFACE=baseline` only to reproduce the historical pre-promotion
surface.

The current hierarchy, caveats, validation gates, and complete output list are
documented in
[`docs/methods/unit-count-modeling.md`](../docs/methods/unit-count-modeling.md).
Raw and compact source extracts remain ignored under
`data/raw_parcels/unit_sources/`.

### 7. Check Residential Form Against the City Land Use Inventory

Target `land_use_unit_classification_audit`, implemented in
`scripts/audits/land_use_unit_classification.R`, compares the promoted parcel
surface with the City of Austin Detailed Land Use Inventory. It first links
records using county-aware parcel identifiers. Parcel points are matched to
City land-use polygons only where the identifiers do not link. This broad
comparison is diagnostic and never changes a unit count or production
classification. It is distinct from the narrower exact-identifier production
safeguard described above.

The comparison distinguishes **residential form** from **unit-count evidence**.
For example, a City `Apartment/Condo` code independently supports a multi-unit
classification, but it does not report how many units are present. Conversely,
a direct project total may be valid even where the City's primary-use code is
older or describes a mixed-use site incompletely. Disagreements are therefore
written to review tables rather than automatically resolved in favor of either
source.

Refresh the ignored City and Census snapshots deliberately with
`Rscript scripts/data/download_land_use_inventory.R`. Then run
`Rscript run_analysis.R land_use_unit_classification_audit`.

## Displacement Proxies

In this project, a **displacement proxy** is a relatively concrete event or
change associated with displacement pressure, such as a rent increase,
eviction filing, demolition permit, or land-value increase. It is distinct
from a **smoke signal**, which is an earlier or less direct indication that
neighborhood conditions may be changing.

### Rent Pressure

The citywide Part 1 rent measure comes from target `acs_rent_history`. It uses
ACS 5-year median gross-rent estimates ending in 2014, 2019, and 2024. These
ending years are called **vintages**. Dollar amounts are adjusted to 2024
dollars, the latest configured ACS vintage, so an increase reflects more than
general inflation.

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

Eviction ingestion is implemented: the pipeline turns the available Travis
County court extracts into annual counts of observed eviction filings by hex.
Before Part 3 forecasting, these observations must be expanded into a complete
panel, using zero only for a covered hex-year with no filing and retaining
uncovered places or periods as missing; partial years must also be handled
explicitly. The forecasting outcome must then specify whether it represents
filings or a later case disposition and which rental-unit denominator is used
to calculate rates.

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
