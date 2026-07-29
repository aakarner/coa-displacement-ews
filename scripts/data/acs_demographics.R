################################################################################
# Process ACS Demographics to Hexagonal Grid
################################################################################
#
# This script creates a durable ACS demographic backbone for the displacement
# analysis. It pulls block-group ACS 5-year data for Austin's three counties,
# allocates additive counts with 2020 Census block population/housing weights,
# assigns medians from each hex's dominant residential block group, derives
# vulnerability fields, and saves a reusable hex-level artifact.
#
# Outputs:
#   - output/acs_demographics_by_hex.rds
#   - output/acs_demographics_by_hex.csv
#   - output/acs_dasymetric_block_hex_allocation.rds/.csv
#   - output/acs_dasymetric_hex_bg_crosswalk.rds/.csv
#   - output/acs_dasymetric_allocation_qa.csv
#
################################################################################

project_path <- function(...) {
  if (requireNamespace("here", quietly = TRUE)) {
    here::here(...)
  } else {
    file.path(getwd(), ...)
  }
}

source(project_path("R", "utils.R"))
source(project_path("R", "analysis_config.R"))

suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(tidycensus)
})

source(project_path("R", "acs_dasymetric.R"))

print_header("02f - ACS DEMOGRAPHICS TO HEX GRID")

OUTPUT_DIR <- project_path("output")
ACS_CACHE_DIR <- project_path("data", "raw_acs")
ACS_YEAR <- EWS_CONFIG$acs_current_year
ACS_SURVEY <- EWS_CONFIG$acs_survey
ACS_COUNTIES <- EWS_CONFIG$acs_counties
ANALYSIS_CRS <- 3857

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
sf::sf_use_s2(FALSE)

hex_grid <- load_output(file.path(OUTPUT_DIR, "hex_grid.rds"), "hexagonal grid") %>%
  st_transform(4326)

residential_parcel_support_file <- file.path(
  OUTPUT_DIR,
  "residential_parcels_for_hex_sf.rds"
)
if (!file.exists(residential_parcel_support_file)) {
  stop(
    "Missing residential parcel support: ",
    residential_parcel_support_file,
    ". Run 02d, 02e, and 02c before 02f.",
    call. = FALSE
  )
}
residential_parcels <- load_output(
  residential_parcel_support_file,
  "residential parcel dasymetric support points"
)

acs_vars <- c(
  median_income = "B19013_001",
  total_pop = "B03002_001",
  white_nh = "B03002_003",
  black_nh = "B03002_004",
  asian_nh = "B03002_006",
  hispanic = "B03002_012",
  total_housing_units = "B25001_001",
  population_in_occupied_housing = "B25008_001",
  total_tenure = "B25003_001",
  owner_occupied = "B25003_002",
  renter_occupied = "B25003_003",
  total_edu = "B15003_001",
  less_than_hs = "B15003_002",
  hs_grad = "B15003_017",
  some_college = "B15003_019",
  bachelors = "B15003_022",
  graduate = "B15003_023",
  total_poverty_det = "C17002_001",
  poverty_under_050 = "C17002_002",
  poverty_050_099 = "C17002_003",
  median_rent = "B25064_001",
  median_home_value = "B25077_001",
  gross_rent_occupied = "B25070_001",
  rent_burden_30_34 = "B25070_007",
  rent_burden_35_39 = "B25070_008",
  rent_burden_40_49 = "B25070_009",
  rent_burden_50_plus = "B25070_010"
)

population_count_vars <- c(
  "total_pop", "white_nh", "black_nh", "asian_nh", "hispanic",
  "total_edu", "less_than_hs", "hs_grad", "some_college", "bachelors",
  "graduate", "total_poverty_det", "poverty_under_050", "poverty_050_099"
)

housing_count_vars <- c(
  "total_housing_units", "population_in_occupied_housing", "total_tenure",
  "owner_occupied", "renter_occupied",
  "gross_rent_occupied",
  "rent_burden_30_34", "rent_burden_35_39", "rent_burden_40_49",
  "rent_burden_50_plus"
)

count_vars <- c(population_count_vars, housing_count_vars)
median_vars <- c("median_income", "median_rent", "median_home_value")

dir.create(ACS_CACHE_DIR, showWarnings = FALSE, recursive = TRUE)

load_acs_extract <- function(geography, variables, cache_label) {
  print_progress(
    paste0(
      "Fetching ACS ", ACS_YEAR, " ", ACS_SURVEY, " ", geography,
      " data for ", paste(ACS_COUNTIES, collapse = ", "), " Counties..."
    )
  )

  acs_cache_file <- file.path(
    ACS_CACHE_DIR,
    paste0("acs_", ACS_YEAR, "_", ACS_SURVEY, "_", cache_label, ".rds")
  )
  acs_extract <- NULL

  if (file.exists(acs_cache_file)) {
    print_progress(paste0("Loading cached ACS extract: ", acs_cache_file))
    acs_extract <- readRDS(acs_cache_file)
    missing_cached_variables <- setdiff(
      names(variables),
      unique(acs_extract$variable)
    )
    if (length(missing_cached_variables) > 0) {
      print_progress(
        "Cached ACS extract does not match the current variable specification; refreshing it."
      )
      acs_extract <- NULL
    }
  }

  if (is.null(acs_extract)) {
    acs_extract <- tidycensus::get_acs(
      geography = geography,
      variables = variables,
      state = "TX",
      county = ACS_COUNTIES,
      year = ACS_YEAR,
      survey = ACS_SURVEY,
      geometry = TRUE,
      output = "tidy",
      cache_table = TRUE
    )
    saveRDS(acs_extract, acs_cache_file)
  }

  acs_extract
}

acs_long <- load_acs_extract(
  geography = "block group",
  variables = acs_vars,
  cache_label = "block_group_demographics"
)
acs_median_tract <- load_acs_extract(
  geography = "tract",
  variables = acs_vars[median_vars],
  cache_label = "tract_medians"
)

print_progress(
  paste0("Retrieved ", nrow(acs_long), " block-group-variable rows.")
)

acs_source_geographies <- acs_long %>%
  transmute(source_geoid = GEOID, source_name = NAME, geometry) %>%
  distinct(source_geoid, .keep_all = TRUE)
acs_tract_geographies <- acs_median_tract %>%
  transmute(source_geoid = GEOID, source_name = NAME, geometry) %>%
  distinct(source_geoid, .keep_all = TRUE)

census_blocks <- load_census_block_ancillary(
  cache_dir = ACS_CACHE_DIR,
  counties = ACS_COUNTIES
)

print_progress("Allocating Census blocks to hexes with residential parcel support...")
block_hex_results <- build_census_block_hex_allocation(
  hex_grid = hex_grid,
  census_blocks = census_blocks,
  residential_parcels = residential_parcels,
  analysis_crs = ANALYSIS_CRS
)

print_progress("Building Census-block-informed ACS-to-hex crosswalk...")
crosswalk_results <- build_acs_hex_crosswalk(
  hex_grid = hex_grid,
  source_geographies = acs_source_geographies,
  census_blocks = census_blocks,
  block_hex_allocation = block_hex_results$allocation,
  analysis_crs = ANALYSIS_CRS
)
tract_crosswalk_results <- build_acs_hex_crosswalk(
  hex_grid = hex_grid,
  source_geographies = acs_tract_geographies,
  census_blocks = census_blocks,
  block_hex_allocation = block_hex_results$allocation,
  analysis_crs = ANALYSIS_CRS
)

print_progress("Allocating additive ACS counts with dasymetric weights...")
count_results <- allocate_acs_count_variables(
  acs_long = acs_long,
  crosswalk = crosswalk_results$crosswalk,
  population_variables = population_count_vars,
  housing_variables = housing_count_vars
)

print_progress("Assigning medians from dominant residential block groups with tract fallback...")
median_bg_data <- assign_acs_median_variables(
  acs_long = acs_long,
  dominant_source = crosswalk_results$dominant_source,
  median_variables = median_vars,
  source_geography = "block_group"
)
median_tract_data <- assign_acs_median_variables(
  acs_long = acs_median_tract,
  dominant_source = tract_crosswalk_results$dominant_source,
  median_variables = median_vars,
  source_geography = "tract"
)
median_data <- combine_acs_median_sources(
  primary = median_bg_data,
  fallback = median_tract_data,
  median_variables = median_vars
)

acs_hex <- hex_grid %>%
  select(hex_id, geometry) %>%
  left_join(count_results$values, by = "hex_id") %>%
  left_join(median_data, by = "hex_id") %>%
  mutate(
    acs_year = ACS_YEAR,
    acs_survey = ACS_SURVEY,
    acs_count_source_geography = "block_group",
    acs_count_allocation_method =
      paste(
        "2020 Census block population/housing totals allocated within blocks",
        "by residential parcel floor-area support"
      ),
    across(all_of(count_vars), ~replace_na(.x, 0)),
    below_poverty = poverty_under_050 + poverty_050_099,
    below_poverty_moe = sqrt(
      poverty_under_050_moe^2 + poverty_050_099_moe^2
    ),
    pct_white = if_else(total_pop > 0, white_nh / total_pop * 100, NA_real_),
    pct_black = if_else(total_pop > 0, black_nh / total_pop * 100, NA_real_),
    pct_asian = if_else(total_pop > 0, asian_nh / total_pop * 100, NA_real_),
    pct_hispanic = if_else(total_pop > 0, hispanic / total_pop * 100, NA_real_),
    pct_poc = if_else(total_pop > 0, (total_pop - white_nh) / total_pop * 100, NA_real_),
    pct_renter = if_else(total_tenure > 0, renter_occupied / total_tenure * 100, NA_real_),
    pct_owner = if_else(total_tenure > 0, owner_occupied / total_tenure * 100, NA_real_),
    pct_college = if_else(total_edu > 0, (bachelors + graduate) / total_edu * 100, NA_real_),
    poverty_rate = if_else(total_poverty_det > 0, below_poverty / total_poverty_det * 100, NA_real_),
    rent_burdened_30plus = rent_burden_30_34 + rent_burden_35_39 +
      rent_burden_40_49 + rent_burden_50_plus,
    pct_rent_burden_30plus = if_else(
      gross_rent_occupied > 0,
      rent_burdened_30plus / gross_rent_occupied * 100,
      NA_real_
    ),
    rent_burden_proxy = if_else(
      !is.na(median_income) & median_income > 0,
      median_rent * 12 / median_income,
      NA_real_
    ),
    median_income_relative_moe = if_else(
      median_income > 0,
      median_income_moe / median_income,
      NA_real_
    ),
    median_rent_relative_moe = if_else(
      median_rent > 0,
      median_rent_moe / median_rent,
      NA_real_
    ),
    median_home_value_relative_moe = if_else(
      median_home_value > 0,
      median_home_value_moe / median_home_value,
      NA_real_
    ),
    median_income_reliable = !is.na(median_income_relative_moe) &
      median_income_relative_moe <= EWS_CONFIG$acs_median_relative_moe_limit,
    median_rent_reliable = !is.na(median_rent_relative_moe) &
      median_rent_relative_moe <= EWS_CONFIG$acs_median_relative_moe_limit,
    median_home_value_reliable = !is.na(median_home_value_relative_moe) &
      median_home_value_relative_moe <= EWS_CONFIG$acs_median_relative_moe_limit,
    vuln_low_income = normalize_to_100(-median_income),
    vuln_high_renters = normalize_to_100(pct_renter),
    vuln_poverty = normalize_to_100(poverty_rate),
    vuln_rent_burden = normalize_to_100(pct_rent_burden_30plus),
    vulnerability_index = rowMeans(
      cbind(vuln_low_income, vuln_high_renters, vuln_poverty, vuln_rent_burden),
      na.rm = TRUE
    ),
    vulnerability_index = if_else(is.nan(vulnerability_index), NA_real_, vulnerability_index)
  ) %>%
  st_transform(4326)

output_rds <- file.path(OUTPUT_DIR, "acs_demographics_by_hex.rds")
output_csv <- file.path(OUTPUT_DIR, "acs_demographics_by_hex.csv")

save_output(
  block_hex_results$allocation,
  file.path(OUTPUT_DIR, "acs_dasymetric_block_hex_allocation.rds"),
  "ACS Census-block-to-hex dasymetric allocation"
)
write_csv(
  block_hex_results$allocation,
  file.path(OUTPUT_DIR, "acs_dasymetric_block_hex_allocation.csv")
)

save_output(
  crosswalk_results$crosswalk,
  file.path(OUTPUT_DIR, "acs_dasymetric_hex_bg_crosswalk.rds"),
  "ACS dasymetric hex/block-group crosswalk"
)
crosswalk_results$crosswalk %>%
  write_csv(file.path(OUTPUT_DIR, "acs_dasymetric_hex_bg_crosswalk.csv"))

allocation_qa <- bind_rows(
  block_hex_results$qa %>%
    mutate(qa_group = "block_to_hex_allocation", .before = 1),
  crosswalk_results$qa %>%
    mutate(qa_group = "block_group_count_crosswalk", .before = 1),
  tract_crosswalk_results$qa %>%
    mutate(qa_group = "tract_median_fallback_crosswalk", .before = 1),
  count_results$conservation_qa %>%
    transmute(
      qa_group = "count_conservation",
      metric = variable,
      value = conservation_difference,
      source_zone_estimate_total,
      expected_project_estimate,
      allocated_project_estimate
    )
)
write_csv(
  allocation_qa,
  file.path(OUTPUT_DIR, "acs_dasymetric_allocation_qa.csv")
)

save_output(acs_hex, output_rds, "ACS demographic hex summary")

acs_hex %>%
  st_drop_geometry() %>%
  write_csv(output_csv)

print_progress(paste0("Saved CSV version to: ", output_csv))

print_header("STEP 02f COMPLETE")
cat(paste0("Hexagons with ACS population data: ", sum(!is.na(acs_hex$total_pop)), "\n"))
cat(paste0("Hexagons with ACS median income: ", sum(!is.na(acs_hex$median_income)), "\n"))
cat(
  paste0(
    "Median-rent tract fallbacks: ",
    sum(acs_hex$median_rent_source_geography == "tract", na.rm = TRUE),
    "\n"
  )
)
cat(
  paste0(
    "Allocated project population: ",
    round(sum(acs_hex$total_pop, na.rm = TRUE)),
    "\n"
  )
)
cat(
  paste0(
    "Maximum count conservation difference: ",
    signif(max(abs(count_results$conservation_qa$conservation_difference)), 4),
    "\n"
  )
)
cat(paste0("Output: ", output_rds, "\n"))
