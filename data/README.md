# Data Directory

This directory is for user-provided data files that complement the automatically-downloaded Census/ACS data.

## Census/ACS Spatial Allocation

Run `02d_calibrate_parcel_units.R`, `02e_validate_unit_counts.R`, and
`02c_process_corporate_parcels.R` before the ACS scripts. The master
`02_process_data.R` script enforces this order.

`02f_process_acs_demographics.R` uses ACS block groups as source zones and
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

Run `02o_audit_parcel_acs_housing_units.R` before changing the clustering
eligibility rule or housing-unit denominator. The diagnostic compares the
targeted, primary, and conservative parcel estimates with ACS total housing
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

## Residential Unit Source and Project Tables

`02d_calibrate_parcel_units.R` treats the three county CSVs as broad candidate
parcel inputs. Before calibration, it uses the local WCAD raw property and
parcel files plus `R/wcad_unit_eligibility.R` to classify Williamson records.
Explicit nonresidential condominium, reference-only, park/amenity,
transitional-land, and other non-unit accounts are written to audit outputs and
removed from the production unit universe. The EWS pipeline does not execute or
import code from `landlord-mapper`.

Run `02p_prepare_unit_sources.R` after `02d`. It reuses the WCAD eligibility
helper, extracts a compact set of TCAD unit and improvement fields from the
configured property-profile input, applies explicit Hays account
classifications, and ingests City Affordable Housing Inventory and Universal
Recycling Ordinance records. Source totals and parcel links are stored in
separate tables so a project total cannot be counted once per linked parcel.

Run `02q_build_residential_projects.R` next. It groups parcels conservatively,
holds conflicting direct sources out of training, sums complete appraisal
account enumerations, and writes strict labels and unresolved multifamily model
candidates. Cross-county properties remain one project but carry explicit
county-membership fields. Apartment signals from historical comments are
distinguished from legal, DBA, and use evidence. Source hierarchy and model
candidate outputs remain a shadow analysis, although the shared WCAD
eligibility rule itself is now part of production preprocessing.

The current hierarchy, caveats, validation gates, and complete output list are
documented in `UNIT_COUNT_MODELING.md`. Raw and compact source extracts remain
ignored under `data/raw_parcels/unit_sources/`.

## Optional Data Files

### 1. Building Demolitions (`demolitions.csv`)

**Purpose**: Track building demolitions that directly displace residents.

**Format**:
```csv
demo_id,latitude,longitude,demo_date,building_type
1,30.267153,-97.743061,2021-06-15,Single Family
2,30.268422,-97.744318,2021-08-22,Multi-Family
3,30.269134,-97.745927,2021-10-03,Commercial
```

**Required Columns**:
- `demo_id`: Unique identifier for each demolition
- `latitude`: Latitude in decimal degrees (WGS84)
- `longitude`: Longitude in decimal degrees (WGS84)
- `demo_date`: Date of demolition (YYYY-MM-DD format)
- `building_type`: Type of building (e.g., "Single Family", "Multi-Family", "Commercial")

**Data Sources**:
- City building permit databases
- Demolition permit records
- Property assessment records

---

### 2. Rent Prices (`rent_prices.csv`)

**Purpose**: Track rent price changes over time to identify areas of rapid rent growth.

**Format**:
```csv
hex_id,date,median_rent
1,2021-01-01,1200
1,2021-04-01,1250
1,2021-07-01,1300
2,2021-01-01,1500
2,2021-04-01,1550
```

**Required Columns**:
- `hex_id`: Hexagon ID from the grid (generated by `01_create_hex_grid.R`)
- `date`: Date of observation (YYYY-MM-DD format)
- `median_rent`: Median rent in dollars

**Alternative Format** (if you have point-level data):
```csv
property_id,latitude,longitude,date,rent
1,30.267153,-97.743061,2021-01-01,1200
1,30.267153,-97.743061,2021-04-01,1250
```

**Data Sources**:
- Zillow Rent Index
- CoStar rental data
- Apartment listing websites (scraped with permission)
- Fair Market Rent (FMR) from HUD

---

### 3. Eviction Filings (`evictions.csv`) - COMING SOON

**Purpose**: Track eviction filings as an indicator of housing insecurity.

**Planned Format**:
```csv
eviction_id,latitude,longitude,filing_date,case_type,outcome
1,30.267153,-97.743061,2021-03-15,Non-payment,Granted
2,30.268422,-97.744318,2021-04-22,Lease violation,Dismissed
```

**Integration**: Add to `02_process_data.R` Section 7

---

### 4. County Appraisal Values

**Purpose**: Track changes in land values that may signal gentrification pressure.

Run `02i_process_appraisal_history.R` after the calibrated residential parcel
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

Then run `02j_process_appraisal_adjusted_trends.R`. It estimates each annual
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

---

### 5. Ownership and Transaction History

**Purpose**: Track current corporate ownership, ownership turnover, and parcel
transaction activity without confusing missing history with zero activity.

Current ownership is processed by `02c_process_corporate_parcels.R`. Run
`02k_audit_ownership_transactions.R` before
`02l_process_ownership_transactions.R`. The audit checks the following sources
against the exact residential parcel universe:

- cached Travis deed events from the sibling `landlord-mapper` repository;
- Hays annual `OWNER` exports and the latest `SALES` history;
- Williamson 2023-2024 certified owner reports and current owner data; and
- WCAD's `Sales History - Certified` dataset (`kdj3-9hpg`).

The WCAD sales download is stored at
`data/raw_parcels/williamson/wcad_sales_history_certified.csv`. Owner and sales
row-level data remain ignored; only aggregate coverage QA is written to
`output/`.

`02l_process_ownership_transactions.R` builds parcel- and hex-level outputs:

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

---

### 6. Amenity Change

**Purpose**: Measure dated commercial reorientation without treating static
amenity density as displacement pressure.

Run `02m_audit_amenity_sources.R` before
`02n_process_amenity_change.R`. The audit downloads and caches:

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

---

## Data Preparation Tips

### Geocoding Addresses

If your data has addresses instead of coordinates:

```r
library(tidygeocoder)

data_with_coords <- your_data %>%
  geocode(
    address = address_column,
    method = "osm",  # or "census", "google", etc.
    lat = latitude,
    lon = longitude
  )
```

### Aggregating to Hexagons

If you have point-level data to aggregate:

```r
# Load hex grid
hex_grid <- readRDS("output/hex_grid.rds")

# Convert your data to spatial
your_data_sf <- your_data %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

# Spatial join
aggregated <- hex_grid %>%
  st_join(your_data_sf) %>%
  group_by(hex_id) %>%
  summarise(
    count = n(),
    mean_value = mean(value_column, na.rm = TRUE)
  )
```

## Data Privacy Considerations

**Important**: When using sensitive data (evictions, individual addresses):
- Aggregate to hexagon level before sharing
- Never publish individual addresses or personally identifiable information
- Follow local data privacy regulations
- Consider differential privacy techniques for public releases

## Getting Help

If you have data in a different format:
1. See `02_process_data.R` for examples of data processing
2. Consult the main README.md for integration instructions
3. Open an issue on GitHub with your data format question
