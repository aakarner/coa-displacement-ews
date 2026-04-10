################################################################################
# 02 - Process and Aggregate Data to Hexagonal Grid
################################################################################
#
# This script processes various data sources and aggregates them to the
# hexagonal grid cells. It handles:
# - Census/ACS demographic and socioeconomic data (fetched via tidycensus API)
# - Building demolitions (from CSV if available, else creates empty object)
# - Rent prices over time (synthetic time series for demonstration)
# - Placeholder structure for future data (evictions, land value, ownership)
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

# Source utilities (enables standalone execution; also sourced by run_analysis.R)
source(here::here("R/utils.R"))

print_header("02 - PROCESSING AND AGGREGATING DATA")

# Configuration
OUTPUT_DIR <- here::here("output")
DATA_DIR <- here::here("data")
ACS_YEAR <- 2024  # Most recent complete ACS 5-year estimates as of 4/26

# Create data directory if it doesn't exist
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)

################################################################################
# Step 1: Load hexagonal grid
################################################################################

print_progress("Loading hexagonal grid...")
hex_grid <- load_output(
  file.path(OUTPUT_DIR, "hex_grid.rds"),
  "hexagonal grid"
)

################################################################################
# Step 2: Process Census/ACS Data
################################################################################

  print_progress("Fetching Census ACS data for Travis, Hays, and Williamson Counties, TX...")

# Note: You'll need to set up a Census API key first
# Get one free at: https://api.census.gov/data/key_signup.html
# Then run: census_api_key("YOUR_KEY_HERE", install = TRUE)

# Define variables to retrieve from ACS
# Using 5-year estimates for more reliable data at tract level
acs_vars <- c(
  # Income
  median_income = "B19013_001",
  
  # Race/Ethnicity
  total_pop = "B03002_001",
  white_nh = "B03002_003",
  black_nh = "B03002_004",
  asian_nh = "B03002_006",
  hispanic = "B03002_012",
  
  # Housing tenure
  total_tenure = "B25003_001",
  owner_occupied = "B25003_002",
  renter_occupied = "B25003_003",
  
  # Educational attainment (25+ years)
  total_edu = "B15003_001",
  less_than_hs = "B15003_002",
  hs_grad = "B15003_017",
  some_college = "B15003_019",
  bachelors = "B15003_022",
  graduate = "B15003_023",
  
  # Poverty
  total_poverty_det = "B17001_001",
  below_poverty = "B17001_002",
  
  # Median rent
  median_rent = "B25064_001",
  
  # Median home value
  median_home_value = "B25077_001"
)

# Fetch ACS data for Travis, Hays, and Williamson Counties
# Austin's city boundary spans all three counties, so we need tracts from each
# to ensure complete population coverage when spatially joining to the hex grid.
# We use tracts as the base geography.
acs_data <- tryCatch({
  # Try to fetch from Census API
  result <- get_acs(
    geography = "tract",
    variables = acs_vars,
    state = "TX",
    county = c("Travis", "Hays", "Williamson"),
    year = ACS_YEAR,
    survey = "acs5",
    geometry = TRUE,
    output = "tidy"
  ) %>%
    st_transform(4326) %>%
    mutate(orig_area = st_area(geometry))
  
  print_progress(paste0("Retrieved ACS data for ", nrow(result), " census tracts"))
  
  # Return the successfully fetched data
  result
  
}, error = function(e) {
  print_progress("WARNING: Could not fetch Census data. You may need to set up a Census API key.")
  print_progress("Get a free key at: https://api.census.gov/data/key_signup.html")
  print_progress("Then run: tidycensus::census_api_key('YOUR_KEY_HERE', install = TRUE)")
})

################################################################################
# Step 3: Spatially join Census data counts to hexagonal grid
################################################################################

print_progress("Spatially joining Census data to hexagonal grid...")

# Perform areal interpolation from census tracts to the hexagonal grid cells
hex_with_census <- hex_grid %>%
  st_intersection(acs_data) %>%
  mutate(
    intersection_area = st_area(geometry),
    weight = as.numeric(intersection_area / orig_area)
  ) %>%
  st_drop_geometry() %>%
  group_by(hex_id, variable) %>%
  summarize(interpE = sum(estimate * weight), .groups = "drop") %>%
  pivot_wider(id_cols = hex_id, names_from = variable, values_from = interpE) %>%
  left_join(hex_grid %>% select(hex_id, geometry), by = "hex_id")
  
print_progress(paste0("Census data joined to ", nrow(hex_with_census), " hexagons"))

################################################################################
# Step 4: Process census data, calculate shares, and produce visualizations
################################################################################

print_progress("Calculating derived demographic variables...")

acs_processed <- hex_with_census %>%
  st_sf() %>%
  mutate(
    # Race/ethnicity percentages
    pct_white = (white_nh / total_pop) * 100,
    pct_black = (black_nh / total_pop) * 100,
    pct_asian = (asian_nh / total_pop) * 100,
    pct_hispanic = (hispanic / total_pop) * 100,
    pct_poc = ((total_pop - white_nh) / total_pop) * 100,

    # Housing tenure
    pct_renter = (renter_occupied / total_tenure) * 100,
    
    # Education (bachelor's degree or higher)
    pct_college = ((bachelors + graduate) / total_edu) * 100,
    
    # Poverty rate
    poverty_rate = (below_poverty / total_poverty_det) * 100
  ) 
  # select(
  #   GEOID,
  #   median_income = median_incomeE,
  #   total_pop = total_popE,
  #   pct_white, pct_black, pct_asian, pct_hispanic, pct_poc,
  #   pct_renter, pct_college, poverty_rate,
  #   median_rent = median_rentE,
  #   median_home_value = median_home_valueE,
  #   orig_area,
  #   geometry
  # )

# Pull roads data for Austin for visualization
atx_roads <- 
  roads(state = "TX", county = "Travis County") %>%
  filter(RTTYP %in% c("I", "S")) %>%
  st_transform(4326)

# Convert to long format to faciliate mapping
acs_toMap <- acs_processed %>%
  select(hex_id, pct_white:poverty_rate, geometry) %>%
  pivot_longer(cols = pct_white:poverty_rate)

ggplot(acs_toMap) + 
  geom_sf(data = acs_toMap, aes(col = value, fill = value)) + 
  geom_sf(data = atx_roads[acs_toMap, ], color = "black") + 
  facet_wrap(~name) + 
  scale_fill_viridis_c(direction = -1) + 
  scale_color_viridis_c(direction = -1) +
  ggthemes::theme_map()

################################################################################
# Step 5: Process building demolitions (placeholder/synthetic)
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
# Step 6: Process rent price data (placeholder/synthetic)
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
  }

# Combine successful results from both attempts
buildings <- buildings %>%
  filter(!is.na(latitude) & !is.na(longitude)) %>%
  bind_rows(fallback)

# Join coordinates back to the full dataset
foo <- rent_data %>%
  left_join(
    buildings %>% select(`Building Address`, latitude, longitude),
    join_by(building_id)
  )

# Create a spatial data frame for the rent data, using the new latitude and longitude columns
rent_data <- st_as_sf(rent_data, coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)

# Check on missing coordinate data in rent_data
missing_coords <- rent_data %>%
  filter(is.na(latitude) | is.na(longitude))


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
# Step 7: Add placeholders for future data sources
################################################################################

print_progress("Adding placeholder columns for future data sources...")

hex_data <- hex_data %>%
  mutate(
    # Eviction filings (to be added)
    eviction_count = NA_real_,
    eviction_rate = NA_real_,
    
    # Land value (to be added)
    land_value_per_sqft = NA_real_,
    land_value_change_pct = NA_real_,
    
    # Corporate ownership (to be added)
    pct_corporate_owned = NA_real_,
    investor_owned_units = NA_real_
  )

################################################################################
# Step 8: Data quality checks and summary
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

# Summary statistics
cat("\nData summary:\n")
cat(paste0("  - Total hexagons: ", nrow(hex_data), "\n"))
cat(paste0("  - Hexagons with demographic data: ", 
          sum(!is.na(hex_data$median_income)), "\n"))
cat(paste0("  - Hexagons with demolitions: ", 
          sum(hex_data$demo_count_total > 0, na.rm = TRUE), "\n"))
cat(paste0("  - Total demolitions: ", 
          sum(hex_data$demo_count_total, na.rm = TRUE), "\n"))

################################################################################
# Step 9: Save processed data
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
# Summary
################################################################################

print_header("STEP 02 COMPLETE")
cat("✓ Census/ACS demographic data processed and joined\n")
cat("✓ Building demolitions aggregated to hexagons\n")
cat("✓ Rent price time series added\n")
cat("✓ Placeholder structure created for future data sources\n")
cat(paste0("✓ Processed data saved to: ", output_file, "\n"))
