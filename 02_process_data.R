################################################################################
# 02 - Process and Aggregate Data to Hexagonal Grid
################################################################################
#
# This script processes various data sources and aggregates them to the
# hexagonal grid cells. It handles:
# - Census/ACS demographic and socioeconomic data (fetched via tidycensus API)
# - Building demolitions
# - Rent prices over time
# - 311 service request summaries
# - Eviction filing records
# - Corporate ownership / residential parcel universe
# - Placeholder structure for future land value data
#
# WHY THIS MATTERS:
# Aggregating diverse data sources to a common spatial unit (hexagons) enables
# integrated analysis of displacement risk factors. The script includes robust
# error handling to work with or without external data sources.
#
# INPUTS:
#   - output/hex_grid.rds: Hexagonal grid from script 01
#   - data/Residential_Demolitions_dataset_*.csv (optional)
#   - Census API (via tidycensus; requires API key or uses synthetic fallback)
#
# OUTPUTS:
#   - output/hex_data.rds: Hexagonal grid with aggregated data
#     Contains: demographics, rent, demolitions, derived variables
#
# DEPENDENCIES:
#   - tidyverse, sf, tidycensus packages
#   - Census API key (optional; falls back to synthetic data if unavailable)
################################################################################

project_path <- function(...) {
  if (requireNamespace("here", quietly = TRUE)) {
    here::here(...)
  } else {
    file.path(getwd(), ...)
  }
}

# Source utilities (enables standalone execution; also sourced by run_analysis.R)
source(project_path("R", "utils.R"))

print_header("02 - PROCESSING AND AGGREGATING DATA")

# Configuration
OUTPUT_DIR <- project_path("output")
DATA_DIR <- project_path("data")
FIGURES_DIR <- project_path("figures")
ACS_YEAR <- 2024  # Most recent complete ACS 5-year estimates as of 4/26

# Create data directory if it doesn't exist
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

################################################################################
# Step 1: Load hexagonal grid ##############################################
################################################################################

print_progress("Loading hexagonal grid...")
hex_grid <- load_output(
  file.path(OUTPUT_DIR, "hex_grid.rds"),
  "hexagonal grid"
)

################################################################################
# Step 2: Process Census/ACS data #########################################
################################################################################

print_progress("Processing ACS demographic data via 02f_process_acs_demographics.R...")

source(project_path("02f_process_acs_demographics.R"))

hex_with_census <- load_output(
  file.path(OUTPUT_DIR, "acs_demographics_by_hex.rds"),
  "ACS demographic hex summary"
)

################################################################################
# Step 5: Process building demolitions data ###############################
################################################################################

print_progress("Processing building demolitions data...")

# Check if actual data exists
# demo_file <- file.path(DATA_DIR, "Residential_Demolitions_dataset_20260401.csv")
demo_file <- file.path(DATA_DIR, "Issued_Construction_Permits_20260401.csv")


demolitions <- read_csv(demo_file)
problems(demolitions)  # Check for any parsing issues
cut_rows <- unique(problems(demolitions)$row)


demolitions <- demolitions %>%
  filter(!row_number() %in% cut_rows,
         !is.na(Latitude),
         !is.na(Longitude)) %>%
  st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326, remove = FALSE)

ggplot(demolitions) +
  geom_sf() +
  ggthemes::theme_map() +
  labs(title = "Building Demolitions in Austin, TX")


# Aggregate demolitions to hex grid
# NOTE: We join to hex_grid (not hex_with_census) to count demolitions per hex
# independently before merging with other data sources
hex_with_demos <- hex_grid %>%
  st_join(demolitions, join = st_intersects) %>%
  group_by(hex_id) %>%
  summarise(
    demo_count_total = sum(!is.na(`Calendar Year Issued`)),
    demo_count_2020 = sum(`Calendar Year Issued` == 2020, na.rm = TRUE),
    demo_count_2021 = sum(`Calendar Year Issued` == 2021, na.rm = TRUE),
    demo_count_2022 = sum(`Calendar Year Issued` == 2022, na.rm = TRUE),
    demo_count_2023 = sum(`Calendar Year Issued` == 2023, na.rm = TRUE),
    demo_count_2024 = sum(`Calendar Year Issued` == 2024, na.rm = TRUE),
    demo_count_2025 = sum(`Calendar Year Issued` == 2025, na.rm = TRUE),
    demo_count_2026 = sum(`Calendar Year Issued` == 2026, na.rm = TRUE), 
    .groups = "drop"
  ) %>%
  st_drop_geometry()

# Join back to main hex data
hex_data <- hex_with_census %>%
  left_join(hex_with_demos, by = "hex_id") %>%
  mutate(across(starts_with("demo_count"), ~replace_na(., 0)))

################################################################################
# Step 6: Process rent price data ##############################################
################################################################################

print_progress("Processing rent price time series data...")

print_progress("Loading rent price data from file...")
rent_data <- read_csv("data/CoStarHistoric-clean.csv")

# Clean the rent_data data frame to remove the string "QTD" from the Period column
rent_data$Period <- gsub(" QTD", "", rent_data$Period)
rent_data$Period <- yq(rent_data$Period)

# Assign a unique identifier to each unique building location within rent_data such
# that every row corresponding to the same building has the same identifier. This will
# allow us to join the rent data to the hex grid later on without creating duplicate rows.
rent_data <- rent_data %>%
  group_by(`Building Address`) %>%
  mutate(building_id = cur_group_id()) %>%
  ungroup()

# Geocode building locations using tidygeocoder
# If the geocoding fails, use the `Building Name` column instead of the address

# Get unique buildings to geocode (avoid redundant API calls)
buildings <- rent_data %>%
  distinct(building_id, `Building Address`, `Building Name`, `Zip Code`) %>%
  mutate(full_address = paste(`Building Address`, "Austin", "TX", `Zip Code`, sep = ", "))

# Attempt 1: Geocode using full address
geocoded <- buildings %>%
  geocode(full_address, method = "osm", lat = latitude, long = longitude)

# Identify failures
failed <- geocoded %>%
  filter(is.na(latitude) | is.na(longitude))

if (nrow(failed) > 0) {
  # Attempt 2: Geocode using building name + zip code with the arcgis geocoder
  fallback <- failed %>%
    select(-latitude, -longitude) %>%
    mutate(
      fallback_address = paste(`Building Name`, `Zip Code`, "Austin, TX", sep = ", ")
    ) %>%
    geocode(fallback_address, method = "arcgis", lat = latitude, long = longitude) %>%
    select(-fallback_address)
  } else {
    fallback <- geocoded %>% slice(0)
  }

# How many are still missing after both attempts?
still_missing <- fallback %>%
  filter(is.na(latitude) | is.na(longitude))
  
# Combine successful results from both attempts
buildings_final <- geocoded %>%
  filter(!is.na(latitude) & !is.na(longitude)) %>%
  bind_rows(fallback)

# Create a simple plot of all identified building locations to visually inspect geocoding results
# Use small black points and overlay the spatial extent of `hex_grid` for reference 
ggplot(buildings_final) +
  geom_sf(data = hex_grid, fill = NA, color = grey(0.7)) +
  geom_point(aes(x = longitude, y = latitude), color = "red", size = 0.5) +
    ggthemes::theme_map() +
  labs(title = "Geocoded Building Locations with Hex Grid Overlay")


# Stored coordinates in a separate CSV for reference and to avoid re-geocoding in the future
write_csv(buildings_final, "data/geocoded_buildings.csv")
buildings_final <- read_csv("data/geocoded_buildings.csv")

# Join coordinates back to the full dataset
building_coords <- buildings_final %>%
  distinct(building_id, .keep_all = TRUE) %>%
  select(building_id, latitude, longitude)

# This is a many-to-many join because there are buildings in the dataset that have the same address
# but different names. We will eliminate duplicate rows after the join.
rent_data <- rent_data %>%
  left_join(
    building_coords %>% select(building_id, latitude, longitude),
    join_by(building_id)
  )

# Check on missing coordinate data in rent_data and stop the script if they are found
missing_coords <- rent_data %>%
  filter(is.na(latitude) | is.na(longitude))

if (nrow(missing_coords) > 0) {
  stop("Missing coordinate data found in rent_data")
}

# Create a spatial data frame for the rent data, using the new latitude and longitude columns
rent_data <- st_as_sf(rent_data, coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)


# Process rent data and integrate with hex grid

# Keep CRS aligned
rent_pts <- rent_data |>
  st_transform(st_crs(hex_grid))

# Parse numeric CoStar fields
# Note that the CoStar data has some quirks in how it encodes missing values (e.g. "NA", "-", "—") that we need to account for 
# when parsing numbers. We will convert these to NA before parsing.
rent_pts <- rent_pts |>
  mutate(
    ask_rent_unit_num = parse_number(askRent_PerUnit, na = c("", "NA", "-", "—")),
    ask_rent_psf_num  = parse_number(askRent_PerSF,   na = c("", "NA", "-", "—")),
    vacancy_units_num = parse_number(vacancy_Units,   na = c("", "NA", "-", "—")),
    vacancy_pct_num   = parse_number(vacancy_Percent, na = c("", "NA", "-", "—")) / 100,
    inv_units_num     = as.numeric(inventory_Units)
  )

# Spatial join: point -> hex
rent_hex <- rent_pts |>
  st_join(hex_grid |> select(hex_id), join = st_within, left = FALSE)

# Hex-by-quarter summaries
hex_rent_summary <- rent_hex |>
  st_drop_geometry() |>
  summarise(
    n_records = n(),
    n_buildings = n_distinct(building_id),
    total_units = sum(inv_units_num, na.rm = TRUE),

    rent_unit_mean_w = weighted.mean(ask_rent_unit_num, inv_units_num, na.rm = TRUE),
    rent_psf_mean_w = weighted.mean(ask_rent_psf_num, inv_units_num, na.rm = TRUE),
    rent_unit_median = median(ask_rent_unit_num, na.rm = TRUE),
    rent_unit_p25 = quantile(ask_rent_unit_num, 0.25, na.rm = TRUE, names = FALSE),
    rent_unit_p75 = quantile(ask_rent_unit_num, 0.75, na.rm = TRUE, names = FALSE),
    rent_unit_sd = sd(ask_rent_unit_num, na.rm = TRUE),

    vacancy_pct_w = weighted.mean(vacancy_pct_num, inv_units_num, na.rm = TRUE),
    vacancy_units_total = sum(vacancy_units_num, na.rm = TRUE),
    .by = c(hex_id, Period)
  ) |>
  arrange(hex_id, Period) |>
  group_by(hex_id) |>
  mutate(
    rent_unit_qoq_pct = 100 * (rent_unit_mean_w / lag(rent_unit_mean_w, 1) - 1),
    rent_unit_yoy_pct = 100 * (rent_unit_mean_w / lag(rent_unit_mean_w, 4) - 1)
  ) |>
  ungroup()

# 5) Attach latest period back to hex geometry (for mapping)
hex_rent_latest <- hex_rent_summary |>
  filter(Period == max(Period, na.rm = TRUE), .by = hex_id)

hex_grid_rent <- hex_grid |>
  left_join(hex_rent_latest, by = "hex_id")

# Add a categorical variable based on rent price levels for visualization using jenks natural breaks classification
# We will classify the weighted mean asking rent per unit into 5 categories: "Very Low", "Low", "Medium", "High", "Very High"
# using the jenks natural breaks method to determine the breakpoints. This will help us visualize the spatial distribution of rent prices across the hexagonal grid.
# Include the rent price range in the labels for each category (e.g. "Low ($X - $Y)", "Medium ($Y - $Z)", etc.) to provide more context in the legend.

rent_breaks <- classInt::classIntervals(hex_grid_rent$rent_unit_mean_w, n = 5, style = "jenks")
hex_grid_rent <- hex_grid_rent %>%
  mutate(
    rent_price_category = cut(rent_unit_mean_w, breaks = rent_breaks$brks, labels = c("Very Low (< $1260)", "Low ($1260 - $1770)", "Medium ($1770 - $2740)", "High ($2741 - $4666)", "Very High (> $4666)"), include.lowest = TRUE)
  )

# Visualize the categorical variable on a map, drop NA values, and include the price range in the legend
# for each category
ggplot(hex_grid_rent %>% filter(!is.na(rent_price_category))) +
  geom_sf(data = hex_grid, fill = NA, color = grey(0.8)) +
  geom_sf(aes(fill = rent_price_category), color = NA) +
  scale_fill_viridis_d(option = "plasma", na.value = "grey80") +
  ggthemes::theme_map() +
  labs(title = "Categorized Weighted Mean Asking Rent Per Unit by Hexagon",
       fill = "Rent Price Category")

# Join rent data to hex units while calculating weighted median rent per hexagon
# as well as the rate of change in rent prices over time, considering 1- and 5-year
# changes.




hex_rent <- hex_grid %>%
  st_join(rent_data, join = st_intersects) %>%
  st_drop_geometry() %>%
  group_by(hex_id, Period) %>%
  summarise(
    median_rent_period = median(`Asking Rent Per Unit`, na.rm = TRUE),
    n_buildings = n_distinct(building_id, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(hex_id, Period) %>%
  group_by(hex_id) %>%
  mutate(
    rent_change_1yr = (median_rent_period - lag(median_rent_period, 4)) /
      lag(median_rent_period, 4) * 100,
    rent_change_5yr = (median_rent_period - lag(median_rent_period, 20)) /
      lag(median_rent_period, 20) * 100
  ) %>%
  ungroup()

# Summarise to most recent period for hex-level join
hex_rent_latest <- hex_rent %>%
  group_by(hex_id) %>%
  slice_max(Period, n = 1) %>%
  ungroup() %>%
  select(hex_id, median_rent_period, rent_change_1yr, rent_change_5yr, n_buildings)

hex_data <- hex_data %>%
  left_join(hex_rent_latest, by = "hex_id")


################################################################################
# Step 7: Process 311 data #####################################################
################################################################################

print_progress("Processing 311 data...")

hex_311_summary_file <- file.path(OUTPUT_DIR, "311_requests_by_hex_summary.rds")

if (Sys.getenv("AUSTIN_DATA_API_KEY") != "" && Sys.getenv("AUSTIN_DATA_API_SECRET") != "") {
  source(project_path("02g_process_311_requests.R"))
} else {
  print_progress("Austin data API credentials not set; using existing 311 output if available.")
}

if (file.exists(hex_311_summary_file)) {
  hex_311_summary <- load_output(hex_311_summary_file, "311 request hex summary")

  hex_data <- hex_data %>%
    left_join(
      hex_311_summary %>%
        st_drop_geometry() %>%
        select(
          hex_id,
          starts_with("sr_311_")
        ),
      by = "hex_id"
    ) %>%
    mutate(
      across(
        c(
          sr_311_total,
          sr_311_code_related_total,
          sr_311_housing_condition_total,
          sr_311_tenant_distress_total,
          sr_311_smoke_signal_total,
          sr_311_nuisance_or_disorder_total,
          sr_311_latest_12mo,
          sr_311_previous_12mo,
          sr_311_smoke_signal_latest_12mo,
          sr_311_smoke_signal_previous_12mo
        ),
        ~replace_na(.x, 0)
      )
    )
} else {
  print_progress("WARNING: 311 hex summary not found; skipping 311 join.")
}


################################################################################
# Step 8: Process eviction filing data #########################################
################################################################################

print_progress("Processing eviction filing data via 02b_process_evictions.R...")

source(project_path("02b_process_evictions.R"))

hex_eviction_summary_file <- file.path(OUTPUT_DIR, "eviction_filings_by_hex_summary.rds")
if (file.exists(hex_eviction_summary_file)) {
  hex_eviction_summary <- load_output(hex_eviction_summary_file, "eviction filing hex summary")

  hex_data <- hex_data %>%
    left_join(hex_eviction_summary, by = "hex_id") %>%
    mutate(
      across(
        c(
          eviction_defendant_rows_total,
          eviction_cases_total,
          starts_with("eviction_cases_20"),
          eviction_cases_latest_12mo,
          eviction_cases_previous_12mo,
          eviction_final_status_cases_total,
          eviction_dismissed_cases_total
        ),
        ~replace_na(., 0)
      )
    )
} else {
  print_progress("WARNING: Eviction hex summary not found after running 02b; skipping eviction join.")
}

################################################################################
# Step 9: Process corporate ownership data #####################################
################################################################################

print_progress("Calibrating residential parcel unit counts via 02d_calibrate_parcel_units.R...")

source(project_path("02d_calibrate_parcel_units.R"))

print_progress("Validating calibrated parcel unit counts against ACS tracts via 02e_validate_unit_counts.R...")

source(project_path("02e_validate_unit_counts.R"))

print_progress("Processing corporate ownership data via 02c_process_corporate_parcels.R...")

source(project_path("02c_process_corporate_parcels.R"))

hex_corporate_file <- file.path(OUTPUT_DIR, "corporate_ownership_by_hex.rds")
if (file.exists(hex_corporate_file)) {
  hex_corporate <- load_output(hex_corporate_file, "corporate ownership hex summary")

  hex_data <- hex_data %>%
    left_join(
      hex_corporate %>%
        st_drop_geometry() %>%
        select(
          hex_id,
          residential_parcels,
          residential_units,
          residential_improvement_sqft,
          residential_land_sqft,
          corporate_owned_parcels,
          corporate_owned_units,
          corporate_owned_imprv_sqft,
          corporate_owner_count,
          financialized_owner_parcels,
          geocoded_parcels,
          pct_corporate_parcels,
          pct_corporate_units,
          pct_corporate_improvement_sqft,
          pct_financialized_owner_parcels,
          corporate_unit_share_city,
          corporate_parcel_share_city,
          corporate_owned_units_per_km2,
          corporate_owned_parcels_per_km2,
          residential_units_per_km2,
          residential_parcels_per_km2,
          pct_corporate_owned,
          investor_owned_units
        ),
      by = "hex_id"
    )
} else {
  print_progress("WARNING: Corporate ownership hex summary not found after running 02c; skipping corporate join.")
}

################################################################################
# Step 10: Add land value placeholders ####################################
################################################################################

print_progress("Adding placeholder columns for land value data...")

hex_data <- hex_data %>%
  mutate(
    # Land value (to be added)
    land_value_per_sqft = NA_real_,
    land_value_change_pct = NA_real_
  )

################################################################################
# Step 11: Data quality checks and summary ################################
################################################################################

print_progress("Performing data quality checks...")

# Check for missing values
missing_summary <- hex_data %>%
  st_drop_geometry() %>%
  summarise(across(everything(), ~sum(is.na(.)) / n() * 100)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "pct_missing") %>%
  arrange(desc(pct_missing))

cat("\nVariables with missing data:\n")
print(filter(missing_summary, pct_missing > 0), n = 20)

# Summary statistics #######################################################
cat("\nData summary:\n")
cat(paste0("  - Total hexagons: ", nrow(hex_data), "\n"))
cat(paste0("  - Hexagons with demographic data: ", 
          sum(!is.na(hex_data$median_income)), "\n"))
cat(paste0("  - Hexagons with demolitions: ", 
          sum(hex_data$demo_count_total > 0, na.rm = TRUE), "\n"))
cat(paste0("  - Total demolitions: ", 
          sum(hex_data$demo_count_total, na.rm = TRUE), "\n"))
if ("eviction_cases_total" %in% names(hex_data)) {
  cat(paste0("  - Hexagons with eviction filings: ",
            sum(hex_data$eviction_cases_total > 0, na.rm = TRUE), "\n"))
  cat(paste0("  - Total geocoded eviction cases assigned to hexagons: ",
            sum(hex_data$eviction_cases_total, na.rm = TRUE), "\n"))
}

################################################################################
# Step 12: Save processed data ############################################
################################################################################

output_file <- file.path(OUTPUT_DIR, "hex_data_processed.rds")
save_output(hex_data, output_file, "processed hexagonal data")

# Also save as CSV (without geometry) for easy inspection
csv_file <- file.path(OUTPUT_DIR, "hex_data_processed.csv")
hex_data %>%
  st_drop_geometry() %>%
  write_csv(csv_file)

print_progress(paste0("Also saved CSV version to: ", csv_file))

################################################################################
# Summary #################################################################
################################################################################

print_header("STEP 02 COMPLETE")
cat("✓ Census/ACS demographic data processed and joined\n")
cat("✓ Building demolitions aggregated to hexagons\n")
cat("✓ Rent price time series added\n")
cat("✓ Full eviction filing records geocoded and aggregated to hexagons\n")
cat("✓ Corporate ownership aggregated to hexagons\n")
cat("✓ Land value placeholder columns added\n")
cat(paste0("✓ Processed data saved to: ", output_file, "\n"))
