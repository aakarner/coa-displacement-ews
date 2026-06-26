################################################################################
# 02f - Process ACS Demographics to Hexagonal Grid
################################################################################
#
# This script creates a durable ACS demographic backbone for the displacement
# analysis. It pulls tract-level ACS 5-year data for Austin's three counties,
# interpolates tract data to the project hex grid, derives vulnerability fields,
# and saves a reusable hex-level artifact.
#
# Outputs:
#   - output/acs_demographics_by_hex.rds
#   - output/acs_demographics_by_hex.csv
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

suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(tidycensus)
})

print_header("02f - ACS DEMOGRAPHICS TO HEX GRID")

OUTPUT_DIR <- project_path("output")
ACS_YEAR <- 2024
ACS_SURVEY <- "acs5"
ACS_COUNTIES <- c("Travis", "Hays", "Williamson")
ANALYSIS_CRS <- 3857

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
sf::sf_use_s2(FALSE)

hex_grid <- load_output(file.path(OUTPUT_DIR, "hex_grid.rds"), "hexagonal grid") %>%
  st_transform(4326)

acs_vars <- c(
  median_income = "B19013_001",
  total_pop = "B03002_001",
  white_nh = "B03002_003",
  black_nh = "B03002_004",
  asian_nh = "B03002_006",
  hispanic = "B03002_012",
  total_housing_units = "B25001_001",
  total_tenure = "B25003_001",
  owner_occupied = "B25003_002",
  renter_occupied = "B25003_003",
  total_edu = "B15003_001",
  less_than_hs = "B15003_002",
  hs_grad = "B15003_017",
  some_college = "B15003_019",
  bachelors = "B15003_022",
  graduate = "B15003_023",
  total_poverty_det = "B17001_001",
  below_poverty = "B17001_002",
  median_rent = "B25064_001",
  median_home_value = "B25077_001",
  gross_rent_occupied = "B25070_001",
  rent_burden_30_34 = "B25070_007",
  rent_burden_35_39 = "B25070_008",
  rent_burden_40_49 = "B25070_009",
  rent_burden_50_plus = "B25070_010"
)

count_vars <- c(
  "total_pop", "white_nh", "black_nh", "asian_nh", "hispanic",
  "total_housing_units", "total_tenure", "owner_occupied", "renter_occupied",
  "total_edu", "less_than_hs", "hs_grad", "some_college", "bachelors",
  "graduate", "total_poverty_det", "below_poverty", "gross_rent_occupied",
  "rent_burden_30_34", "rent_burden_35_39", "rent_burden_40_49",
  "rent_burden_50_plus"
)

median_vars <- c("median_income", "median_rent", "median_home_value")

print_progress(
  paste0(
    "Fetching ACS ", ACS_YEAR, " ", ACS_SURVEY,
    " tract data for ", paste(ACS_COUNTIES, collapse = ", "), " Counties..."
  )
)

acs_long <- tidycensus::get_acs(
  geography = "tract",
  variables = acs_vars,
  state = "TX",
  county = ACS_COUNTIES,
  year = ACS_YEAR,
  survey = ACS_SURVEY,
  geometry = TRUE,
  output = "tidy"
)

print_progress(paste0("Retrieved ", nrow(acs_long), " tract-variable rows."))

acs_projected <- acs_long %>%
  st_transform(ANALYSIS_CRS) %>%
  mutate(tract_area_sqm = as.numeric(st_area(geometry)))

hex_projected <- hex_grid %>%
  st_transform(ANALYSIS_CRS) %>%
  select(hex_id, geometry)

print_progress("Intersecting ACS tracts with hex grid...")

hex_tract_long <- suppressWarnings(
  st_intersection(hex_projected, acs_projected)
) %>%
  mutate(intersection_area_sqm = as.numeric(st_area(geometry)))

print_progress(paste0("Created ", nrow(hex_tract_long), " hex-tract-variable intersections."))

count_data <- hex_tract_long %>%
  filter(variable %in% count_vars) %>%
  st_drop_geometry() %>%
  mutate(tract_weight = intersection_area_sqm / tract_area_sqm) %>%
  group_by(hex_id, variable) %>%
  summarise(value = sum(estimate * tract_weight, na.rm = TRUE), .groups = "drop")

# Medians are not counts, so do not allocate them by tract area. Instead, use an
# overlap-area weighted mean of tract medians for hexes that cross tract lines.
median_data <- hex_tract_long %>%
  filter(variable %in% median_vars) %>%
  st_drop_geometry() %>%
  group_by(hex_id, variable) %>%
  summarise(
    value = if (all(is.na(estimate))) {
      NA_real_
    } else {
      weighted.mean(estimate, intersection_area_sqm, na.rm = TRUE)
    },
    .groups = "drop"
  )

acs_wide <- bind_rows(count_data, median_data) %>%
  pivot_wider(names_from = variable, values_from = value)

acs_hex <- hex_grid %>%
  select(hex_id, geometry) %>%
  left_join(acs_wide, by = "hex_id") %>%
  mutate(
    acs_year = ACS_YEAR,
    acs_survey = ACS_SURVEY,
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

save_output(acs_hex, output_rds, "ACS demographic hex summary")

acs_hex %>%
  st_drop_geometry() %>%
  write_csv(output_csv)

print_progress(paste0("Saved CSV version to: ", output_csv))

print_header("STEP 02f COMPLETE")
cat(paste0("Hexagons with ACS population data: ", sum(!is.na(acs_hex$total_pop)), "\n"))
cat(paste0("Hexagons with ACS median income: ", sum(!is.na(acs_hex$median_income)), "\n"))
cat(paste0("Output: ", output_rds, "\n"))
