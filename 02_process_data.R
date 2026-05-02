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

# Source utilities (enables standalone execution; also sourced by run_analysis.R)
source(here::here("R/utils.R"))

print_header("02 - PROCESSING AND AGGREGATING DATA")

# Configuration
OUTPUT_DIR <- here::here("output")
DATA_DIR <- here::here("data")
FIGURES_DIR <- here::here("figures")
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
# Step 3: Spatially join Census data counts to hexagonal grid #############
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
# Step 4: Process census data and visualize demographic shares ############
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

atx_311_file <- file.path(DATA_DIR, "Austin_311_Public_Data_20260202.csv")

if (file.exists(atx_311_file)) {
  atx_311 <- read.csv(atx_311_file)
  
  # Generate list of unique 311 service request calls organized from greatest to least
  SR_calls <- atx_311 %>%
      group_by(SR.Description) %>%
      summarise(Count = n(), .groups = "drop") %>%
      arrange(desc(Count))
  
  write.csv(SR_calls, file.path(OUTPUT_DIR, "311_service_request_counts.csv"), row.names = FALSE)
  
  # Identify service request descriptions that may relate to tenant harassment or lockouts
  harassment_calls <- atx_311 %>%
    filter(grepl("harassment|lockout|eviction|tenant", SR.Description, ignore.case = TRUE))
} else {
  print_progress("WARNING: 311 data file not found; skipping 311 processing.")
}


################################################################################
# Step 8: Process eviction filing data #########################################
################################################################################

print_progress("Processing eviction filing data...")

eviction_file <- file.path(DATA_DIR, "Eviction Case Data 01.01.2020 - 04.29.2026.csv")

clean_eviction_address <- function(x) {
  x %>%
    str_replace_all("\\\\r\\\\n|\\\\n|\\\\r", ", ") %>%
    str_replace_all("[\r\n]+", ", ") %>%
    str_replace_all("\\s*,\\s*", ", ") %>%
    str_squish() %>%
    na_if("")
}

collapse_unique_values <- function(x) {
  values <- unique(x[!is.na(x)])
  
  if (length(values) == 0) {
    NA_character_
  } else {
    str_c(values, collapse = " | ")
  }
}

clean_evictor_name_key <- function(x) {
  x %>%
    str_to_upper() %>%
    str_replace_all("&", " AND ") %>%
    str_replace_all("\\bD\\s*/\\s*B\\s*/\\s*A\\b|\\bD\\s+B\\s+A\\b", " DBA ") %>%
    str_replace_all("\\bDBA\\b", " DBA ") %>%
    str_replace_all("\\bBEXELY\\b", "BEXLEY") %>%
    str_replace_all("\\bAPARTMETS\\b", "APARTMENTS") %>%
    str_replace_all("[^A-Z0-9]+", " ") %>%
    str_squish() %>%
    na_if("")
}

clean_evictor_legal_name_key <- function(x) {
  clean_evictor_name_key(x) %>%
    str_replace("\\s+AS\\s+AGENT\\s+FOR\\s+.*$", "") %>%
    str_replace("\\s+BY\\s+AND\\s+THROUGH\\s+.*$", "") %>%
    str_replace("\\s+BY\\s+ITS\\s+AGENT\\s+.*$", "") %>%
    str_replace("\\s+SUCCESSOR\\s+IN\\s+INTEREST\\s+TO\\s+.*$", "") %>%
    str_replace("\\s+DBA\\s+.*$", "") %>%
    str_squish() %>%
    na_if("")
}

clean_evictor_property_name_key <- function(x) {
  key <- clean_evictor_name_key(x)
  property <- str_match(key, "\\s+DBA\\s+(.+)$")[, 2]
  
  coalesce(property, key) %>%
    str_remove("^THE\\s+") %>%
    str_squish() %>%
    na_if("")
}

if (file.exists(eviction_file)) {
  eviction_case_rows <- read_csv(
    eviction_file,
    skip = 4,
    col_names = c(
      "file_date",
      "case_number",
      "defendant_name",
      "defendant_address_raw",
      "plaintiff_name",
      "plaintiff_address_raw"
    ),
    show_col_types = FALSE
  )
  
  if (ncol(eviction_case_rows) != 6) {
    stop(
      "Unexpected eviction file structure. Expected 6 columns after skipping report header, found ",
      ncol(eviction_case_rows),
      "."
    )
  }

  eviction_case_rows <- eviction_case_rows %>%
    mutate(
      file_date = mdy(file_date),
      across(
        c(case_number, defendant_name, plaintiff_name),
        ~str_squish(na_if(., ""))
      ),
      defendant_address = clean_eviction_address(defendant_address_raw),
      plaintiff_address = clean_eviction_address(plaintiff_address_raw)
    )
  
  bad_eviction_rows <- eviction_case_rows %>%
    filter(is.na(case_number) | is.na(file_date))
  
  if (nrow(bad_eviction_rows) > 0) {
    stop(
      "Eviction file contains ",
      nrow(bad_eviction_rows),
      " row(s) missing case number or filing date."
    )
  }
  
  eviction_cases_all <- eviction_case_rows %>%
    group_by(case_number) %>%
    summarise(
      file_date = first(file_date),
      plaintiff_address = collapse_unique_values(plaintiff_address),
      defendant_address = collapse_unique_values(defendant_address),
      plaintiff_name = collapse_unique_values(plaintiff_name),
      defendant_name = collapse_unique_values(defendant_name),
      plaintiff_address_count = n_distinct(plaintiff_address, na.rm = TRUE),
      defendant_address_count = n_distinct(defendant_address, na.rm = TRUE),
      raw_party_row_count = n(),
      .groups = "drop"
    ) %>%
    select(
      case_number,
      file_date,
      plaintiff_address,
      defendant_address,
      plaintiff_name,
      defendant_name,
      plaintiff_address_count,
      defendant_address_count,
      raw_party_row_count
    )
  
  incomplete_eviction_cases <- eviction_cases_all %>%
    filter(is.na(plaintiff_address) | is.na(defendant_address))
  
  if (nrow(incomplete_eviction_cases) > 0) {
    write_csv(
      incomplete_eviction_cases,
      file.path(OUTPUT_DIR, "eviction_cases_address_issues.csv")
    )
    
    print_progress(
      paste0(
        "WARNING: ",
        nrow(incomplete_eviction_cases),
        " eviction case(s) are missing a plaintiff or defendant address and were saved to output/eviction_cases_address_issues.csv."
      )
    )
  }
  
  eviction_cases <- eviction_cases_all %>%
    filter(!is.na(plaintiff_address), !is.na(defendant_address))
  
  duplicate_eviction_cases <- eviction_cases %>%
    count(case_number, name = "case_row_count") %>%
    filter(case_row_count > 1)
  
  if (nrow(duplicate_eviction_cases) > 0) {
    stop("Eviction case cleaning did not produce a single row per case.")
  }
  
  evictor_name_lookup <- eviction_cases %>%
    mutate(
      plaintiff_name_key = clean_evictor_name_key(plaintiff_name),
      plaintiff_legal_name_key = clean_evictor_legal_name_key(plaintiff_name),
      plaintiff_property_name_key = clean_evictor_property_name_key(plaintiff_name)
    ) %>%
    count(
      plaintiff_name_key,
      plaintiff_legal_name_key,
      plaintiff_property_name_key,
      plaintiff_name,
      name = "variant_filings",
      sort = TRUE
    ) %>%
    arrange(plaintiff_legal_name_key, desc(variant_filings), plaintiff_name) %>%
    group_by(plaintiff_legal_name_key) %>%
    summarise(
      plaintiff_legal_name_clean = first(plaintiff_name),
      plaintiff_name_variant_count = n_distinct(plaintiff_name),
      .groups = "drop"
    )
  
  eviction_cases <- eviction_cases %>%
    mutate(
      plaintiff_name_key = clean_evictor_name_key(plaintiff_name),
      plaintiff_legal_name_key = clean_evictor_legal_name_key(plaintiff_name),
      plaintiff_property_name_key = clean_evictor_property_name_key(plaintiff_name)
    ) %>%
    left_join(evictor_name_lookup, by = "plaintiff_legal_name_key") %>%
    relocate(
      plaintiff_name_key,
      plaintiff_legal_name_clean,
      plaintiff_legal_name_key,
      plaintiff_property_name_key,
      plaintiff_name_variant_count,
      .after = plaintiff_name
    )
  
  save_output(
    eviction_cases,
    file.path(OUTPUT_DIR, "eviction_cases_clean.rds"),
    "cleaned eviction cases"
  )
  
  write_csv(eviction_cases, file.path(OUTPUT_DIR, "eviction_cases_clean.csv"))
  
  evictor_name_variants <- eviction_cases %>%
    count(
      plaintiff_legal_name_key,
      plaintiff_legal_name_clean,
      plaintiff_name,
      plaintiff_name_key,
      plaintiff_property_name_key,
      name = "filings",
      sort = TRUE
    ) %>%
    arrange(plaintiff_legal_name_key, desc(filings), plaintiff_name)
  
  write_csv(evictor_name_variants, file.path(OUTPUT_DIR, "evictor_name_variants.csv"))
  
  top_evictors <- eviction_cases %>%
    count(plaintiff_legal_name_clean, name = "eviction_filings", sort = TRUE) %>%
    slice_head(n = 25) %>%
    mutate(plaintiff_legal_name_clean = forcats::fct_reorder(plaintiff_legal_name_clean, eviction_filings))
  
  write_csv(top_evictors, file.path(OUTPUT_DIR, "top_evictors_clean.csv"))
  
  p_top_evictors <- ggplot(top_evictors, aes(x = eviction_filings, y = plaintiff_legal_name_clean)) +
    geom_col(fill = "#5b8e7d") +
    scale_x_continuous(labels = comma) +
    labs(
      title = "Top Eviction Filers in Travis County JP Precinct 3",
      subtitle = "Complete eviction cases, January 1, 2020 through April 29, 2026",
      x = "Eviction filings",
      y = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      plot.title = element_text(face = "bold")
    )
  
  ggsave(
    file.path(FIGURES_DIR, "02_top_evictors.png"),
    p_top_evictors,
    width = 10,
    height = 8,
    dpi = 300,
    bg = "white"
  )
  
  print_progress(paste0("Cleaned ", comma(nrow(eviction_cases)), " complete eviction cases"))
} else {
  print_progress("WARNING: Eviction case data file not found; skipping eviction processing.")
}


################################################################################
# Step 9: Process corporate ownership data #####################################
################################################################################

print_progress("Processing corporate ownership data...")

residential_parcel_files <- c(
  Travis = file.path(DATA_DIR, "residential_parcels_for_hex.csv"),
  Williamson = file.path(DATA_DIR, "williamson_residential_parcels_for_hex.csv"),
  Hays = file.path(DATA_DIR, "hays_residential_parcels_for_hex.csv")
)
jurisdictions_file <- file.path(DATA_DIR, "BOUNDARIES_jurisdictions_20260429.geojson")

missing_residential_files <- residential_parcel_files[!file.exists(residential_parcel_files)]

if (length(missing_residential_files) > 0) {
  stop(
    "Missing required residential parcel universe file(s): ",
    paste(missing_residential_files, collapse = ", ")
  )
}

residential_parcel_schemas <- map(
  residential_parcel_files,
  ~names(read_csv(.x, n_max = 0, col_types = cols(.default = col_character()), show_col_types = FALSE))
)

if (!all(map_lgl(residential_parcel_schemas, identical, residential_parcel_schemas[[1]]))) {
  stop("Residential parcel universe files do not have identical schemas.")
}

residential_parcels_raw <- imap_dfr(
  residential_parcel_files,
  ~read_csv(.x, col_types = cols(.default = col_character()), show_col_types = FALSE) %>%
    mutate(source_county = .y)
)

duplicate_parcel_ids <- residential_parcels_raw %>%
  count(parcel_id, name = "row_count") %>%
  filter(row_count > 1)

if (nrow(duplicate_parcel_ids) > 0) {
  stop(
    "Duplicate parcel_id values found after binding county parcel files. ",
    "Example duplicate(s): ",
    paste(head(duplicate_parcel_ids$parcel_id, 10), collapse = ", ")
  )
}

residential_parcels_clean <- residential_parcels_raw %>%
    mutate(
      lat = as.numeric(lat),
      lon = as.numeric(lon),
      parcel_count = replace_na(as.numeric(parcel_count), 0),
      property_units = replace_na(as.numeric(property_units), 0),
      improvement_sqft = replace_na(as.numeric(improvement_sqft), 0),
      land_sqft = replace_na(as.numeric(land_sqft), 0),
      corporate_parcel_count = replace_na(as.numeric(corporate_parcel_count), 0),
      corporate_units = replace_na(as.numeric(corporate_units), 0),
      corporate_improvement_sqft = replace_na(as.numeric(corporate_improvement_sqft), 0),
      is_residential = replace_na(as.logical(is_residential), FALSE),
      is_owner_occupied = replace_na(as.logical(is_owner_occupied), FALSE),
      is_corporate_owned = replace_na(as.logical(is_corporate_owned), FALSE),
      has_financialized_owner = replace_na(as.logical(has_financialized_owner), FALSE)
    )

missing_coords <- residential_parcels_clean %>%
  filter(is.na(lat) | is.na(lon))

if (nrow(missing_coords) > 0) {
  stop(
    "Residential parcel universe contains ",
    nrow(missing_coords),
    " row(s) with missing or non-numeric lat/lon coordinates."
  )
}

residential_parcel_county_totals <- residential_parcels_clean %>%
  group_by(source_county) %>%
  summarise(
    row_count = n(),
    parcel_count = sum(parcel_count, na.rm = TRUE),
    residential_units = sum(property_units, na.rm = TRUE),
    corporate_owned_parcels = sum(corporate_parcel_count, na.rm = TRUE),
    corporate_owned_units = sum(corporate_units, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  residential_parcel_county_totals,
  file.path(OUTPUT_DIR, "residential_parcel_universe_by_county.csv")
)

print_progress("Residential parcel universe by county:")
print(residential_parcel_county_totals)

if (nrow(residential_parcels_clean) > 0) {
  residential_parcels <- residential_parcels_clean %>%
    st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
    st_transform(st_crs(hex_grid))
  
  austin_boundaries <- NULL
  austin_full_purpose <- NULL
  
  if (file.exists(jurisdictions_file)) {
    austin_boundaries <- st_read(jurisdictions_file, quiet = TRUE) %>%
      st_transform(st_crs(hex_grid))
    
    austin_full_purpose <- austin_boundaries %>%
      filter(jurisdiction_type == "FULL")
  } else {
    print_progress("WARNING: Austin jurisdiction boundaries file not found; corporate maps will omit boundary overlays.")
  }
  
  residential_hex_joined <- residential_parcels %>%
    st_join(hex_grid %>% select(hex_id), join = st_within, left = FALSE)
  
  corporate_hex_summary <- residential_hex_joined %>%
    st_drop_geometry() %>%
    group_by(hex_id) %>%
    summarise(
      residential_parcels = sum(parcel_count, na.rm = TRUE),
      residential_units = sum(property_units, na.rm = TRUE),
      residential_improvement_sqft = sum(improvement_sqft, na.rm = TRUE),
      residential_land_sqft = sum(land_sqft, na.rm = TRUE),
      corporate_owned_parcels = sum(corporate_parcel_count, na.rm = TRUE),
      corporate_owned_units = sum(corporate_units, na.rm = TRUE),
      corporate_owned_imprv_sqft = sum(corporate_improvement_sqft, na.rm = TRUE),
      corporate_owner_count = n_distinct(owner_names[is_corporate_owned], na.rm = TRUE),
      financialized_owner_parcels = sum(parcel_count[has_financialized_owner], na.rm = TRUE),
      geocoded_parcels = sum(coord_source != "existing_coord", na.rm = TRUE),
      .groups = "drop"
    )
  
  citywide_corporate_units <- sum(corporate_hex_summary$corporate_owned_units, na.rm = TRUE)
  citywide_corporate_parcels <- sum(corporate_hex_summary$corporate_owned_parcels, na.rm = TRUE)
  citywide_residential_units <- sum(corporate_hex_summary$residential_units, na.rm = TRUE)
  citywide_residential_parcels <- sum(corporate_hex_summary$residential_parcels, na.rm = TRUE)
  
  corporate_hex_summary <- corporate_hex_summary %>%
    mutate(
      pct_corporate_parcels = if_else(
        residential_parcels > 0,
        corporate_owned_parcels / residential_parcels * 100,
        NA_real_
      ),
      pct_corporate_units = if_else(
        residential_units > 0,
        corporate_owned_units / residential_units * 100,
        NA_real_
      ),
      pct_corporate_improvement_sqft = if_else(
        residential_improvement_sqft > 0,
        corporate_owned_imprv_sqft / residential_improvement_sqft * 100,
        NA_real_
      ),
      pct_financialized_owner_parcels = if_else(
        residential_parcels > 0,
        financialized_owner_parcels / residential_parcels * 100,
        NA_real_
      ),
      corporate_unit_share_city = if (citywide_corporate_units > 0) {
        corporate_owned_units / citywide_corporate_units * 100
      } else {
        NA_real_
      },
      corporate_parcel_share_city = if (citywide_corporate_parcels > 0) {
        corporate_owned_parcels / citywide_corporate_parcels * 100
      } else {
        NA_real_
      }
    )
  
  hex_corporate <- hex_grid %>%
    left_join(corporate_hex_summary, by = "hex_id") %>%
    mutate(
      across(
        c(
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
          corporate_unit_share_city,
          corporate_parcel_share_city
        ),
        ~replace_na(., 0)
      ),
      corporate_owned_units_per_km2 = corporate_owned_units / area_km2,
      corporate_owned_parcels_per_km2 = corporate_owned_parcels / area_km2,
      residential_units_per_km2 = residential_units / area_km2,
      residential_parcels_per_km2 = residential_parcels / area_km2,
      investor_owned_units = corporate_owned_units,
      pct_corporate_owned = pct_corporate_units
    )
  
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
  
  save_output(
    residential_parcels,
    file.path(OUTPUT_DIR, "residential_parcels_for_hex_sf.rds"),
    "residential parcel universe points"
  )
  save_output(
    residential_parcels %>% filter(is_corporate_owned),
    file.path(OUTPUT_DIR, "corporate_owned_parcels_sf.rds"),
    "corporate-owned parcel points"
  )
  save_output(
    hex_corporate,
    file.path(OUTPUT_DIR, "corporate_ownership_by_hex.rds"),
    "corporate ownership hex summary"
  )
  
  hex_corporate %>%
    st_drop_geometry() %>%
    write_csv(file.path(OUTPUT_DIR, "corporate_ownership_by_hex.csv"))
  
  print_progress("Creating corporate ownership visualizations...")
  
  boundary_layers <- list()
  if (!is.null(austin_boundaries)) {
    boundary_layers <- c(
      boundary_layers,
      list(geom_sf(data = austin_boundaries, fill = NA, color = "grey65", linewidth = 0.15))
    )
  }
  if (!is.null(austin_full_purpose) && nrow(austin_full_purpose) > 0) {
    boundary_layers <- c(
      boundary_layers,
      list(geom_sf(data = austin_full_purpose, fill = NA, color = "black", linewidth = 0.45))
    )
  }
  
  p_corp_parcels <- ggplot() +
    geom_sf(data = hex_corporate, aes(fill = pct_corporate_parcels), color = NA) +
    boundary_layers +
    scale_fill_viridis_c(option = "magma", labels = label_percent(scale = 1), name = "Corporate parcels") +
    ggthemes::theme_map() +
    labs(
      title = "Share of Residential Parcels with Corporate Ownership",
      subtitle = "Austin full-purpose residential parcel universe aggregated to H3 hexagons"
    ) +
    theme(plot.title = element_text(face = "bold"))
  
  p_corp_units <- ggplot() +
    geom_sf(data = hex_corporate, aes(fill = pct_corporate_units), color = NA) +
    boundary_layers +
    scale_fill_viridis_c(option = "plasma", labels = label_percent(scale = 1), name = "Corporate units") +
    ggthemes::theme_map() +
    labs(
      title = "Share of Residential Units with Corporate Ownership",
      subtitle = "Corporate units divided by total residential units in each hex"
    ) +
    theme(plot.title = element_text(face = "bold"))
  
  p_corp_density <- ggplot() +
    geom_sf(data = hex_corporate, aes(fill = corporate_owned_units_per_km2), color = NA) +
    boundary_layers +
    scale_fill_viridis_c(option = "inferno", trans = "sqrt", labels = comma, name = "Units/km²") +
    ggthemes::theme_map() +
    labs(title = "Density of Corporate-Owned Residential Units") +
    theme(plot.title = element_text(face = "bold"))
  
  p_corp_points <- ggplot() +
    geom_sf(data = hex_grid, fill = NA, color = "grey85", linewidth = 0.1) +
    boundary_layers +
    geom_sf(
      data = residential_parcels %>% filter(is_corporate_owned),
      aes(size = corporate_units),
      color = "#1f78b4",
      alpha = 0.35
    ) +
    scale_size_continuous(range = c(0.2, 4), labels = comma, name = "Units") +
    ggthemes::theme_map() +
    labs(title = "Corporate-Owned Parcel Locations") +
    theme(plot.title = element_text(face = "bold"))

  p_corp_improvement <- ggplot() +
    geom_sf(data = hex_corporate, aes(fill = pct_corporate_improvement_sqft), color = NA) +
    boundary_layers +
    scale_fill_viridis_c(option = "cividis", labels = label_percent(scale = 1), name = "Corporate sqft") +
    ggthemes::theme_map() +
    labs(
      title = "Share of Residential Improvement Square Footage with Corporate Ownership",
      subtitle = "Corporate-owned improvement square feet divided by total residential improvement square feet"
    ) +
    theme(plot.title = element_text(face = "bold"))
  
  top_corporate_owners <- residential_parcels %>%
    filter(is_corporate_owned) %>%
    st_drop_geometry() %>%
    group_by(owner_names) %>%
    summarise(
      parcels = n_distinct(parcel_id),
      units = sum(corporate_units, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(units), desc(parcels)) %>%
    slice_head(n = 20) %>%
    mutate(owner_names = forcats::fct_reorder(owner_names, units))
  
  p_top_owners <- ggplot(top_corporate_owners, aes(x = units, y = owner_names)) +
    geom_col(fill = "#2a9d8f") +
    scale_x_continuous(labels = comma) +
    labs(
      title = "Top Corporate Owners by Estimated Residential Units",
      x = "Estimated units",
      y = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))
  
  ggsave(file.path(FIGURES_DIR, "02_corporate_owned_parcels_by_hex.png"), p_corp_parcels, width = 10, height = 8, dpi = 300, bg = "white")
  ggsave(file.path(FIGURES_DIR, "02_corporate_owned_units_by_hex.png"), p_corp_units, width = 10, height = 8, dpi = 300, bg = "white")
  ggsave(file.path(FIGURES_DIR, "02_corporate_owned_unit_density_by_hex.png"), p_corp_density, width = 10, height = 8, dpi = 300, bg = "white")
  ggsave(file.path(FIGURES_DIR, "02_corporate_owned_parcel_points.png"), p_corp_points, width = 10, height = 8, dpi = 300, bg = "white")
  ggsave(file.path(FIGURES_DIR, "02_corporate_owned_improvement_sqft_share_by_hex.png"), p_corp_improvement, width = 10, height = 8, dpi = 300, bg = "white")
  ggsave(file.path(FIGURES_DIR, "02_top_corporate_owners.png"), p_top_owners, width = 10, height = 8, dpi = 300, bg = "white")
  
  print_progress(paste0("Residential parcels joined to ", n_distinct(residential_hex_joined$hex_id), " hexagons"))
  print_progress(paste0("Total residential parcels in joined hexes: ", comma(citywide_residential_parcels)))
  print_progress(paste0("Total estimated residential units in joined hexes: ", comma(round(citywide_residential_units, 0))))
  print_progress(paste0("Total corporate-owned parcels: ", comma(citywide_corporate_parcels)))
  print_progress(paste0("Total estimated corporate-owned units: ", comma(round(citywide_corporate_units, 0))))
} else {
  stop("Residential parcel universe is empty after binding county files.")
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
cat("✓ Eviction filing records cleaned\n")
cat("✓ Corporate ownership aggregated to hexagons\n")
cat("✓ Land value placeholder columns added\n")
cat(paste0("✓ Processed data saved to: ", output_file, "\n"))
