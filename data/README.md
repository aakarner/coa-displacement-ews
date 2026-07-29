# Data Directory

This directory contains local source files and ignored caches used by the
pipeline. It is not a generic drop folder: source paths are declared in
[`_targets.R`](../_targets.R), and feature roles are declared in
[`config/feature_dictionary.csv`](../config/feature_dictionary.csv).

Run the dependency graph from the repository root:

```bash
Rscript 00_requirements.R
Rscript run_analysis.R
```

Do not manually sequence the old numbered `02*` stages. `{targets}` runs the
source-specific scripts below in dependency order and rebuilds affected
downstream artifacts when an input, method, or analysis cutoff changes.

## Current Source Inventory

| Domain | Pipeline source | Current status |
| --- | --- | --- |
| Residential units | Hays, Travis, and Williamson appraisal records plus reviewed direct-unit sources and modeled estimates | Promoted parcel-unit surface |
| ACS demographics | ACS 5-year estimates, 2020 Census blocks, and residential-parcel support | Current Part 1 vulnerability |
| Rent | ACS gross-rent vintages; CoStar only where matched properties exist | ACS citywide Part 1 input; CoStar sensitivity input |
| Evictions | Travis County JP filing extracts | Current and hex-year outputs implemented |
| Demolitions | Austin issued construction permits | Current Part 1 feature implemented; Part 3 hex-year artifact pending |
| 311 | Austin 311 Socrata API | Current Part 1 smoke signal implemented |
| Appraisal values | County appraisal histories, 2021-2025 | Current trends and hex-year panel implemented |
| Ownership and sales | Current county ownership plus available deed and sales histories | Current ownership and partial change measures implemented |
| Amenity change | Texas Comptroller sales-tax locations with corroborating sources | Current sensitivity feature implemented |

## Census/ACS Spatial Allocation

Targets `unit_calibration`, `unit_validation`, `promoted_unit_surface`, and
`corporate_features` establish the residential support before
`acs_demographics` and `acs_rent_history` run. When a unit source changes,
`{targets}` rebuilds the dependent ownership denominators and ACS allocation
automatically.

`scripts/data/acs_demographics.R` uses ACS block groups as source zones and
2020 Census blocks as control zones. Within each Census block, population and
housing totals are distributed among project hexes in proportion to mapped
residential appraisal floor area. Missing floor area falls back to calibrated
units and then parcel count. A block point is used only when a Census block has
no residential parcel support. Person counts use block-population shares;
housing, tenure, rent-burden, and population-in-occupied-housing counts use
block-housing shares. Source-zone denominators include blocks outside the
project grid, preventing edge block groups from being pulled wholly into
Austin.

Medians are non-additive. Median income, gross rent, and home value come from
the dominant residential block group in each hex; a dominant tract is used
only when that block-group estimate is suppressed. Each median retains its
source geography, GEOID, residential share, assignment method, and MOE.
Historical rent vintages use the same rule. Poverty uses block-group-available
table `C17002` rather than tract-only detail table `B17001`.

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

Run `scripts/audits/parcel_acs_housing_units.R` before changing the clustering
eligibility rule or housing-unit denominator. The diagnostic compares the
promoted, primary, and conservative parcel estimates with ACS total housing
units on the exact H3 grid. It distinguishes robust source disagreements from
cases that cross the 20-unit threshold only because of parcel-model or ACS
sampling uncertainty. It does not alter the feature table.

The reconciliation audit creates:

- `output/parcel_acs_hex_unit_audit.csv`;
- `output/parcel_acs_discordant_hex_review.csv`;
- `output/parcel_acs_discordant_method_audit.csv`;
- `output/parcel_acs_county_unit_summary.csv`;
- `output/parcel_acs_block_group_unit_audit.csv`; and
- `output/parcel_acs_unit_audit_summary.csv`.

Run `scripts/audits/populated_zero_unit_hexes.R` to classify populated hexes that
still have zero canonical parcel units. It checks the unit-bearing parcel universe,
selected project counts, explicit eligibility exclusions, the three full
appraisal parcel maps, ACS block-point fallback, and Austin jurisdiction
context. It writes `output/populated_zero_unit_*` audit tables and
`figures/02u_populated_zero_unit_hex_audit.png`; it does not backfill units or
change eligibility.

## Residential Unit Source and Project Tables

`scripts/data/parcel_units_calibrate.R` treats the three county CSVs as broad candidate
parcel inputs. Before calibration, it uses the local WCAD raw property and
parcel files plus `R/wcad_unit_eligibility.R` to classify Williamson records.
Explicit nonresidential condominium, reference-only, park/amenity,
transitional-land, and other non-unit accounts are written to audit outputs and
removed from the production unit universe. The EWS pipeline does not execute or
import code from `landlord-mapper`.

The same stage uses `R/wcad_residential_supplement.R` to compare the broad
Williamson input with active certified residential records whose parcel
geometry falls inside the study grid. Certified records with positive living
area, unique parcel/address keys, and no existing input row are added under the
existing one-unit WCAD rules. Reviewed exceptions and land-only dispositions
are tracked in `config/residual_unit_parcel_reviews.csv`. The current repair
adds 408 records and writes `output/williamson_certified_residential_*` source
and audit tables.

Target `unit_sources` runs `scripts/data/unit_counts/prepare_sources.R`. It
reuses the WCAD eligibility helper, extracts a compact set of TCAD unit and
improvement fields from the configured property-profile input, applies explicit
Hays account classifications, and ingests City Affordable Housing Inventory
and Universal Recycling Ordinance records. Source totals and parcel links are
stored in separate tables so a project total cannot be counted once per linked
parcel.

Target `unit_projects` groups parcels conservatively, holds conflicting direct
sources out of training, sums complete appraisal account enumerations, and
writes strict labels and unresolved multifamily model candidates. Cross-county
properties remain one project but carry explicit county-membership fields.
Apartment signals from historical comments are distinguished from legal, DBA,
and use evidence.

Target `unit_models` compares global and stratified square-feet-per-unit ratios,
a negative-binomial GAM, and monotonic gradient boosting. It uses project,
spatial, and source holdouts and writes prediction intervals and transfer flags.
Predictions enter the production denominator only through
`promoted_unit_surface`.

Target `williamson_validation` treats appraisal rows as accounts rather than
automatically treating each row as a separate physical development and applies
the reviewed groupings in
`config/williamson_project_groups.csv`. It reads documented validation records
from `config/williamson_unit_validation_sources.csv`. It also checks the
limited-coverage AEGB and TDHCA inventories, records nonmatches without
interpreting them as zero units, and tests a main/living-area ratio
specification on the existing folds. HUD's official multifamily API is listed
in the coverage manifest but is not used after repeated service timeouts.

Target `unit_integration` gives strict selected direct project totals first
priority, then documented unresolved-project counts, the validated
main/living-area stratified estimate for comparable Williamson and other
in-domain candidates, and the targeted parcel total for review or out-of-domain
projects. Reviewed companion accounts and cross-county duplicates are allocated
to one configured parcel representation before aggregation. The resulting
parcel and hex files have `unit_shadow` in their names and remain validation
artifacts.

Target `promoted_unit_surface` checks every shadow parcel against the targeted
baseline, archives the pre-promotion analytical outputs once, and writes
`output/residential_parcels_unit_promoted.rds`. Downstream targets use that
surface by default; use `EWS_UNIT_SURFACE=baseline` only for an explicit
historical or bootstrap run.

The current hierarchy, caveats, validation gates, and complete output list are
documented in `UNIT_COUNT_MODELING.md`. Raw and compact source extracts remain
ignored under `data/raw_parcels/unit_sources/`.

## Displacement Proxies

### Rent Pressure

The citywide Part 1 rent feature comes from target `acs_rent_history`. It uses
ACS 5-year median gross-rent estimates for configured non-overlapping vintages
(currently 2014, 2019, and 2024), converts them to constant dollars, and
combines the current level, reliable recent change, and acceleration. Each hex
receives the median from its dominant residential block group, with tract
fallback for suppressed estimates; medians are never averaged across source
geographies.

CoStar is not the citywide backfill. `data/CoStarHistoric-clean.csv` and
`data/geocoded_buildings.csv` produce a separate
`costar_rent_pressure_index` only where matched properties exist.
`costar_present` records that coverage. Missing CoStar observations remain
missing, and Zillow rent data are not used.

Key outputs are:

- `output/acs_rent_by_hex_vintage.rds/.csv`;
- `output/acs_rent_trends_by_hex.rds/.csv`; and
- the CoStar fields in `output/hex_features.rds`.

### Eviction Filings

Targets `prepared_evictions` and `eviction_features` ingest the Travis County
JP defendant extracts, standardize and geocode filing addresses, assign filings
to hexes, and truncate records at `EWS_ANALYSIS_AS_OF_DATE`. The current feature
uses recent filings per 100 residential units and change from an equal prior
window. Confirmed zeros are retained; rates require a valid unit denominator.

Key outputs are:

- `output/eviction_filings_prepared_for_geocoding.csv`;
- `output/eviction_filings_by_hex_summary.rds/.csv`; and
- `output/eviction_filings_by_hex_year.csv`.

Eviction ingestion is implemented. The existing hex-year file still needs
completeness and outcome-definition validation before Part 3 forecasting.

### Residential Demolitions

Target `current_features` reads
`data/Issued_Construction_Permits_20260401.csv`, retains geocoded permits whose
mapped class is residential, truncates them at the analysis cutoff, and compares
equal recent and prior 24-month windows. The resulting
`demolition_pressure_index` is a current Part 1 displacement proxy.

Part 3 still requires a dedicated, validated
`output/demolition_permits_by_hex_year.csv` outcome artifact. There is no
generic `demolitions.csv` input contract.

### County Appraisal Values

**Purpose**: Track changes in land values that may signal gentrification pressure.

Run `scripts/data/appraisal_history.R` after the calibrated residential parcel
universe has been built. The source inventory is versioned in
`config/appraisal_sources.csv`; downloaded archives and matched county-year
extracts are cached under `data/raw_parcels/appraisal_history/`.

The script creates:

- `output/appraisal_values_by_parcel_year.rds/.csv`
- `output/appraisal_values_by_hex_year.rds/.csv`
- `output/appraisal_value_trends_by_hex.rds/.csv`
- `output/appraisal_panel_source_qa.csv`
- `output/appraisal_panel_spatial_qa.csv`
- `output/appraisal_county_land_values_by_account_year.rds`

Dollar values are expressed in 2025 CPI-U dollars for trend calculations.
Missing source years remain missing rather than being backfilled with zero.
The Hays 2022 production source is the local certified fixed-width roll; the
later post-certification export is retained only for an explicit source
comparison. Other Hays archives can be placed at
`data/raw_parcels/appraisal_history/hays/<year>/hays_<year>.zip` before rerunning.

Then run `scripts/data/appraisal_adjusted_trends.R`. It estimates each annual
county shift from stable improved real-property accounts, subtracts that shift
from target parcel changes, and creates:

- `output/appraisal_county_year_baselines.csv`
- `output/appraisal_adjusted_parcel_trends.rds`
- `output/appraisal_adjusted_trends_by_hex.rds/.csv`
- `output/appraisal_adjustment_qa.csv`
- `output/appraisal_land_area_fallback_qa.csv`

For the current level denominator, calibrated parcel land area is preferred.
Where Williamson acreage is absent, projected parcel-polygon area is used and
reconciled against the reported-acre subset in the QA output.

## Smoke Signals

### Austin 311 Requests

Target `requests_311` queries Austin's Socrata endpoint `xwdj-i9he`, truncates
requests at the analysis cutoff, assigns geocoded records to hexes, and
separately counts service-request types classified as plausible displacement
smoke signals. The Part 1 feature is normalized by residential units;
all-request counts remain available for context.

The API requires `AUSTIN_DATA_API_KEY` and `AUSTIN_DATA_API_SECRET`. Key outputs
are:

- `output/311_requests_by_hex_summary.rds/.csv`;
- `output/311_requests_by_hex_year.csv`; and
- `output/311_service_request_counts.csv`.

### Ownership and Transaction History

**Purpose**: Track current corporate ownership, ownership turnover, and parcel
transaction activity without confusing missing history with zero activity.

Current ownership is processed by `scripts/data/corporate_ownership.R`. Run
`scripts/audits/ownership_transactions.R` before
`scripts/data/ownership_transactions.R`. The audit checks the following sources
against the exact residential parcel universe:

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

Transaction pressure compares two equal 24-month windows and combines parcel
turnover, unit exposure, and positive acceleration. The latest complete common
source cutoff is April 30, 2025, so the windows are May 1, 2021-April 30, 2023
and May 1, 2023-April 30, 2025. Travis uses WD/SW deeds and Hays uses
warranty-deed variants. WCAD's current certified sales-history table
omits 2022-2023 and nearly all 2024 events, so Williamson transaction pressure
is deliberately missing rather than treated as zero. The raw partial event
counts remain in QA outputs.

Ownership change focuses on corporate entry. Hays and Williamson use annual
2023-2025 owner transitions. Recent Travis deed-party names are blank; for
Travis only, a corporate acquisition is inferred when a recent market-deed
parcel is corporate-owned in the current appraisal extract. Corporate
dispositions remain unavailable for Travis and are excluded from its index.

The audit also documents that Williamson's 2021-2022 `ASMNT`
field named `OwnerQuickRefID` duplicates the parcel reference and is not a
comparable owner identity. Williamson owner-change measures therefore begin in
2023.

### Amenity Change

**Purpose**: Measure dated commercial reorientation without treating static
amenity density as displacement pressure.

Run `scripts/audits/amenity_sources.R` before
`scripts/data/amenities.R`. The audit downloads and caches:

- Texas Comptroller permitted sales-tax locations active at any point in the
  prior 48 months (`3kx8-uryv`), including NAICS, first-sale date, and
  out-of-business date;
- Texas mixed-beverage reports (`naix-2893`) for alcohol-establishment
  corroboration; and
- Austin food-establishment inspections (`ecmv-9xxi`) for local
  corroboration only.

The versioned taxonomy is `config/amenity_categories.csv`. The core score uses
cafes, full-service restaurants, and drinking places. Craft alcohol,
specialty food, and fitness categories remain exploratory. A cafe name filter
removes fast-food and unrelated outlets assigned to NAICS 722515.

The processing script compares equal 18-month windows ending April 1, 2026,
batch geocodes unique addresses with the US Census geocoder, and measures
distance-weighted exposure within 800 meters. Category scores are normalized
separately and averaged with equal weight so the restaurant category cannot
dominate the amenity domain. Closures are retained in QA but do not enter the
first index.

Generated raw API extracts and geocode caches remain ignored under
`data/raw_amenities/`. Aggregate outputs include:

- `output/amenity_source_audit.csv`;
- `output/amenity_category_year_qa.csv`;
- `output/amenity_window_change_qa.csv`;
- `output/amenity_geocoding_qa.csv`; and
- `output/amenity_change_features_by_hex.rds/.csv`.

## Part 3 Readiness

Part 3 forecasting is not yet implemented. The available source data should not
be confused with modeling-ready outcome panels. Target
`part3_forecast_readiness` checks the four displacement-proxy outcomes:

- eviction filings: the hex-year artifact exists; construct and validate the
  complete outcome panel;
- residential demolitions: build the historical hex-year artifact;
- rent growth: the ACS vintage artifact exists, with CoStar context; define and
  validate the forecasting outcome without treating missing CoStar as zero;
- land-value growth: the appraisal hex-year artifact exists; construct and
  validate a comparable county-adjusted outcome panel.

See [`config/forecast_outcomes.csv`](../config/forecast_outcomes.csv) and
`output/part3/forecast_readiness.csv` for the machine-readable specification and
current check.

## Data Privacy

Eviction records, owner names, and individual addresses can contain sensitive
or personally identifiable information. Raw and row-level inputs remain local
and ignored by Git. Share only reviewed aggregate outputs, and inspect every
export for address or name fields before release.
