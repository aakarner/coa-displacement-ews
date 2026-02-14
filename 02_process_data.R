################################################################################
# 02 - Process and Aggregate Data to Hexagonal Grid
################################################################################
#
# This script processes various data sources and aggregates them to the
# hexagonal grid cells. It handles:
# - Building demolitions
# - Rent prices over time
# - Census/ACS demographic and socioeconomic data
# - Placeholder structure for future data (evictions, land value, ownership)
#
# All data is spatially joined to hexagonal grid for analysis.
#
################################################################################

print_header("02 - PROCESSING AND AGGREGATING DATA")

# Source utilities
source(here::here("R/utils.R"))

# Configuration
OUTPUT_DIR <- here::here("output")
DATA_DIR <- here::here("data")
ACS_YEAR <- 2021  # Most recent complete ACS 5-year estimates

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

print_progress("Fetching Census ACS data for Travis County, TX...")

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

# Fetch ACS data for Travis County (where Austin is located)
# We use tracts as the base geography
acs_data <- tryCatch({
  # Try to fetch from Census API
  result <- get_acs(
    geography = "tract",
    variables = acs_vars,
    state = "TX",
    county = "Travis",
    year = ACS_YEAR,
    survey = "acs5",
    geometry = TRUE,
    output = "wide"
  ) %>%
    st_transform(4326)
  
  print_progress(paste0("Retrieved ACS data for ", nrow(result), " census tracts"))
  
  # Return the successfully fetched data
  result
  
}, error = function(e) {
  print_progress("WARNING: Could not fetch Census data. You may need to set up a Census API key.")
  print_progress("Get a free key at: https://api.census.gov/data/key_signup.html")
  print_progress("Then run: tidycensus::census_api_key('YOUR_KEY_HERE', install = TRUE)")
  print_progress("Creating synthetic ACS data for demonstration purposes...")
  
  # Create synthetic data for demonstration and return it
  hex_grid %>%
    st_transform(4326) %>%
    mutate(
      median_incomeE = rnorm(n(), 65000, 25000),
      total_popE = rnorm(n(), 4000, 2000),
      white_nhE = rnorm(n(), 2000, 1000),
      black_nhE = rnorm(n(), 400, 300),
      asian_nhE = rnorm(n(), 400, 300),
      hispanicE = rnorm(n(), 1200, 600),
      total_tenureE = rnorm(n(), 1500, 500),
      owner_occupiedE = rnorm(n(), 750, 400),
      renter_occupiedE = rnorm(n(), 750, 400),
      total_eduE = rnorm(n(), 2500, 800),
      less_than_hsE = rnorm(n(), 250, 150),
      hs_gradE = rnorm(n(), 500, 250),
      some_collegeE = rnorm(n(), 750, 300),
      bachelorsE = rnorm(n(), 600, 300),
      graduateE = rnorm(n(), 400, 200),
      total_poverty_detE = rnorm(n(), 3800, 1900),
      below_povertyE = rnorm(n(), 500, 300),
      median_rentE = rnorm(n(), 1300, 400),
      median_home_valueE = rnorm(n(), 450000, 150000)
    ) %>%
    # Ensure positive values
    mutate(across(ends_with("E"), ~pmax(., 0)))
})

################################################################################
# Step 3: Calculate derived Census variables
################################################################################

print_progress("Calculating derived demographic variables...")

acs_processed <- acs_data %>%
  st_sf() %>%
  mutate(
    # Race/ethnicity percentages
    pct_white = (white_nhE / total_popE) * 100,
    pct_black = (black_nhE / total_popE) * 100,
    pct_asian = (asian_nhE / total_popE) * 100,
    pct_hispanic = (hispanicE / total_popE) * 100,
    pct_poc = ((total_popE - white_nhE) / total_popE) * 100,
    orig_area = st_area(geometry),

    # Housing tenure
    pct_renter = (renter_occupiedE / total_tenureE) * 100,
    
    # Education (bachelor's degree or higher)
    pct_college = ((bachelorsE + graduateE) / total_eduE) * 100,
    
    # Poverty rate
    poverty_rate = (below_povertyE / total_poverty_detE) * 100
  ) %>%
  select(
    GEOID,
    median_income = median_incomeE,
    total_pop = total_popE,
    pct_white, pct_black, pct_asian, pct_hispanic, pct_poc,
    pct_renter, pct_college, poverty_rate,
    median_rent = median_rentE,
    median_home_value = median_home_valueE,
    orig_area,
    geometry
  )

################################################################################
# Step 4: Spatial join Census data to hexagonal grid
################################################################################

print_progress("Spatially joining Census data to hexagonal grid...")

# For each hexagon, calculate area-weighted average of overlapping tracts
hex_with_census <- hex_grid %>%
  st_intersection(acs_processed) %>%
  mutate(
    intersection_area = st_area(geometry),
    weight = as.numeric(intersection_area / orig_area)
  ) %>%
  st_drop_geometry() %>%
  group_by(hex_id, h3_index, longitude, latitude, area_km2) %>%
  summarise(
    # Population-weighted variables
    across(c(pct_white, pct_black, pct_asian, pct_hispanic, pct_poc, 
             pct_renter, pct_college, poverty_rate),
           ~weighted.mean(., w = weight * total_pop, na.rm = TRUE)),
    
    # Simple weighted means for medians and totals
    across(c(median_income, median_rent, median_home_value),
           ~weighted.mean(., w = weight, na.rm = TRUE)),
    
    # Sum total population
    total_pop = sum(total_pop * weight, na.rm = TRUE),
    
    .groups = "drop"
  ) %>%
  left_join(hex_grid %>% select(hex_id, geometry), by = "hex_id")

print_progress(paste0("Census data joined to ", nrow(hex_with_census), " hexagons"))

################################################################################
# Step 5: Process building demolitions (placeholder/synthetic)
################################################################################

print_progress("Processing building demolitions data...")

# NOTE: In a real implementation, you would load actual demolition permit data
# For now, we create synthetic data to demonstrate the structure

# Check if actual data exists
demo_file <- file.path(DATA_DIR, "Residential_Demolitions_dataset_20260202.csv")

if(file.exists(demo_file)) {
  print_progress("Loading demolitions data from file...")
  demolitions <- read_csv(demo_file) %>%
    st_as_sf(wkt = c("location"), crs = 4326, remove = FALSE)
} else {
  print_progress("Demolitions file not found - creating empty fallback object...")
  # Create empty sf object with expected schema to prevent downstream errors
  demolitions <- st_sf(
    calendar_year_issued = integer(),
    geometry = st_sfc(crs = 4326)
  )
}

# Aggregate demolitions to hex grid
hex_with_demos <- hex_grid %>%
  st_join(demolitions, join = st_intersects) %>%
  group_by(hex_id) %>%
  summarise(
    demo_count_total = sum(!is.na(calendar_year_issued)),
    demo_count_2020 = sum(calendar_year_issued == 2020, na.rm = TRUE),
    demo_count_2021 = sum(calendar_year_issued == 2021, na.rm = TRUE),
    demo_count_2022 = sum(calendar_year_issued == 2022, na.rm = TRUE),
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

# NOTE: In a real implementation, you would load actual rent data
# For now, we create synthetic time series to demonstrate the structure

# Check if actual data exists
rent_file <- file.path(DATA_DIR, "rent_prices.csv")

if(file.exists(rent_file)) {
  print_progress("Loading rent price data from file...")
  rent_data <- read_csv(rent_file)
} else {
  print_progress("Creating synthetic rent price data for demonstration...")
  
  # Create time series of rent prices for each hex
  set.seed(42)
  
  rent_data <- hex_data %>%
    st_drop_geometry() %>%
    select(hex_id, median_rent) %>%
    mutate(
      # Create quarterly rent observations from 2019-2022
      rent_2019_q1 = median_rent * runif(n(), 0.85, 0.95),
      rent_2019_q4 = median_rent * runif(n(), 0.88, 0.98),
      rent_2020_q1 = median_rent * runif(n(), 0.90, 1.00),
      rent_2020_q4 = median_rent * runif(n(), 0.92, 1.02),
      rent_2021_q1 = median_rent * runif(n(), 0.95, 1.05),
      rent_2021_q4 = median_rent * runif(n(), 1.00, 1.10),
      rent_2022_q1 = median_rent * runif(n(), 1.05, 1.15),
      rent_2022_q4 = median_rent * runif(n(), 1.10, 1.25)
    )
}

# Join rent time series to hex data
hex_data <- hex_data %>%
  left_join(st_drop_geometry(rent_data), by = "hex_id")

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
