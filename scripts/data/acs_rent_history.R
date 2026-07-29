################################################################################
# Process Historical ACS Rent to Hexagonal Grid
################################################################################
#
# Creates a citywide rent-pressure backbone from non-overlapping ACS 5-year
# vintages. CoStar remains an optional enrichment source because it does not
# cover most project hexagons. ACS medians are assigned from the dominant
# residential block group in each hex using 2020 Census block housing counts.
# Suppressed block-group medians fall back to the corresponding dominant tract;
# medians are never averaged across source geographies.
#
# Outputs:
#   - output/acs_rent_by_hex_vintage.rds/.csv
#   - output/acs_rent_trends_by_hex.rds/.csv
#   - output/acs_rent_dominant_sources_by_hex_vintage.csv
#   - output/acs_rent_dasymetric_crosswalk_qa.csv
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
  library(dplyr)
  library(readr)
  library(sf)
  library(tidycensus)
  library(tidyr)
})

source(project_path("R", "acs_dasymetric.R"))

print_header("02h - HISTORICAL ACS RENT TO HEX GRID")

OUTPUT_DIR <- project_path("output")
ACS_CACHE_DIR <- project_path("data", "raw_acs")
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
    ". Run 02d, 02e, and 02c before 02h.",
    call. = FALSE
  )
}
residential_parcels <- load_output(
  residential_parcel_support_file,
  "residential parcel dasymetric support points"
)

census_blocks <- load_census_block_ancillary(
  cache_dir = ACS_CACHE_DIR,
  counties = EWS_CONFIG$acs_counties
)
block_hex_results <- build_census_block_hex_allocation(
  hex_grid = hex_grid,
  census_blocks = census_blocks,
  residential_parcels = residential_parcels,
  analysis_crs = ANALYSIS_CRS
)

annualized_log_change <- function(current, previous, current_year, previous_year) {
  if (
    length(current) == 0 || length(previous) == 0 ||
      is.na(current) || is.na(previous) || current <= 0 || previous <= 0 ||
      is.na(current_year) || is.na(previous_year) || current_year <= previous_year
  ) {
    return(NA_real_)
  }

  100 * (log(current) - log(previous)) / (current_year - previous_year)
}

load_acs_rent_extract <- function(acs_year, geography) {
  geography_slug <- gsub(" ", "_", geography)
  acs_cache_file <- file.path(
    ACS_CACHE_DIR,
    paste0(
      "acs_", acs_year, "_", EWS_CONFIG$acs_survey, "_",
      geography_slug, "_median_rent.rds"
    )
  )

  if (file.exists(acs_cache_file)) {
    print_progress(paste0("Loading cached ACS extract: ", acs_cache_file))
    return(readRDS(acs_cache_file))
  }

  acs_rent <- tidycensus::get_acs(
    geography = geography,
    variables = c(median_rent = "B25064_001"),
    state = "TX",
    county = EWS_CONFIG$acs_counties,
    year = acs_year,
    survey = EWS_CONFIG$acs_survey,
    geometry = TRUE,
    output = "tidy",
    cache_table = TRUE
  )
  saveRDS(acs_rent, acs_cache_file)
  acs_rent
}

fetch_acs_rent_vintage <- function(acs_year) {
  print_progress(
    paste0(
      "Fetching ", acs_year, " ", EWS_CONFIG$acs_survey,
      " median gross rent for ", paste(EWS_CONFIG$acs_counties, collapse = ", "),
      " Counties..."
    )
  )

  acs_rent <- load_acs_rent_extract(acs_year, "block group")
  acs_rent_tract <- load_acs_rent_extract(acs_year, "tract")

  source_geographies <- acs_rent %>%
    transmute(source_geoid = GEOID, source_name = NAME, geometry) %>%
    distinct(source_geoid, .keep_all = TRUE)
  tract_geographies <- acs_rent_tract %>%
    transmute(source_geoid = GEOID, source_name = NAME, geometry) %>%
    distinct(source_geoid, .keep_all = TRUE)

  crosswalk_results <- build_acs_hex_crosswalk(
    hex_grid = hex_grid,
    source_geographies = source_geographies,
    census_blocks = census_blocks,
    block_hex_allocation = block_hex_results$allocation,
    analysis_crs = ANALYSIS_CRS
  )
  tract_crosswalk_results <- build_acs_hex_crosswalk(
    hex_grid = hex_grid,
    source_geographies = tract_geographies,
    census_blocks = census_blocks,
    block_hex_allocation = block_hex_results$allocation,
    analysis_crs = ANALYSIS_CRS
  )

  rent_bg_hex <- assign_acs_median_variables(
    acs_long = acs_rent,
    dominant_source = crosswalk_results$dominant_source,
    median_variables = "median_rent",
    source_geography = "block_group"
  )
  rent_tract_hex <- assign_acs_median_variables(
    acs_long = acs_rent_tract,
    dominant_source = tract_crosswalk_results$dominant_source,
    median_variables = "median_rent",
    source_geography = "tract"
  )
  rent_hex <- combine_acs_median_sources(
    primary = rent_bg_hex,
    fallback = rent_tract_hex,
    median_variables = "median_rent"
  )

  current_cpi <- unname(EWS_CONFIG$acs_cpi_u[[as.character(EWS_CONFIG$acs_current_year)]])
  vintage_cpi <- unname(EWS_CONFIG$acs_cpi_u[[as.character(acs_year)]])

  vintage <- hex_grid %>%
    select(hex_id, geometry) %>%
    left_join(rent_hex, by = "hex_id") %>%
    mutate(
      acs_year = acs_year,
      acs_survey = EWS_CONFIG$acs_survey,
      acs_median_primary_geography = "block_group",
      acs_median_fallback_geography = "tract",
      cpi_u = vintage_cpi,
      median_rent_real = median_rent * current_cpi / vintage_cpi,
      median_rent_moe_real = median_rent_moe * current_cpi / vintage_cpi,
      median_rent_relative_moe = if_else(
        median_rent > 0,
        median_rent_moe / median_rent,
        NA_real_
      )
    )

  list(
    vintage = vintage,
    dominant_source = bind_rows(
      crosswalk_results$dominant_source %>%
        mutate(source_geography = "block_group", .before = 1),
      tract_crosswalk_results$dominant_source %>%
        mutate(source_geography = "tract", .before = 1)
    ) %>%
      mutate(acs_year = acs_year, .before = 1),
    qa = bind_rows(
      crosswalk_results$qa %>%
        mutate(source_geography = "block_group", .before = 1),
      tract_crosswalk_results$qa %>%
        mutate(source_geography = "tract", .before = 1)
    ) %>%
      mutate(acs_year = acs_year, .before = 1)
  )
}

vintage_results <- lapply(
  EWS_CONFIG$acs_years,
  fetch_acs_rent_vintage
)

acs_rent_vintages <- lapply(vintage_results, `[[`, "vintage") %>%
  bind_rows() %>%
  arrange(hex_id, acs_year)

rent_dominant_sources <- lapply(
  vintage_results,
  `[[`,
  "dominant_source"
) %>%
  bind_rows()

rent_crosswalk_qa <- lapply(vintage_results, `[[`, "qa") %>%
  bind_rows()

trend_from_vintages <- function(data) {
  data <- data %>% arrange(acs_year)
  current <- data %>% filter(acs_year == EWS_CONFIG$acs_current_year)
  prior <- data %>% filter(acs_year < EWS_CONFIG$acs_current_year)

  previous <- if (nrow(prior) > 0) prior %>% slice_tail(n = 1) else prior
  earliest <- if (nrow(prior) > 0) prior %>% slice_head(n = 1) else prior
  pre_previous <- if (nrow(prior) > 1) prior %>% slice_tail(n = 2) %>% slice_head(n = 1) else prior[0, ]

  current_value <- if (nrow(current) == 1) current$median_rent_real[[1]] else NA_real_
  current_nominal <- if (nrow(current) == 1) current$median_rent[[1]] else NA_real_
  current_moe <- if (nrow(current) == 1) current$median_rent_relative_moe[[1]] else NA_real_
  current_year <- if (nrow(current) == 1) current$acs_year[[1]] else NA_integer_
  current_source_geoid <- if (nrow(current) == 1) {
    current$median_rent_source_geoid[[1]]
  } else {
    NA_character_
  }
  current_source_geography <- if (nrow(current) == 1) {
    current$median_rent_source_geography[[1]]
  } else {
    NA_character_
  }
  current_source_share <- if (nrow(current) == 1) {
    current$median_rent_source_residential_share[[1]]
  } else {
    NA_real_
  }
  current_source_method <- if (nrow(current) == 1) {
    current$median_rent_source_assignment_method[[1]]
  } else {
    NA_character_
  }

  previous_value <- if (nrow(previous) == 1) previous$median_rent_real[[1]] else NA_real_
  previous_year <- if (nrow(previous) == 1) previous$acs_year[[1]] else NA_integer_
  earliest_value <- if (nrow(earliest) == 1) earliest$median_rent_real[[1]] else NA_real_
  earliest_year <- if (nrow(earliest) == 1) earliest$acs_year[[1]] else NA_integer_
  pre_previous_value <- if (nrow(pre_previous) == 1) pre_previous$median_rent_real[[1]] else NA_real_
  pre_previous_year <- if (nrow(pre_previous) == 1) pre_previous$acs_year[[1]] else NA_integer_

  recent_growth <- annualized_log_change(
    current_value, previous_value, current_year, previous_year
  )
  prior_growth <- annualized_log_change(
    previous_value, pre_previous_value, previous_year, pre_previous_year
  )
  long_growth <- annualized_log_change(
    current_value, earliest_value, current_year, earliest_year
  )

  relative_moes <- data$median_rent_relative_moe[
    is.finite(data$median_rent_relative_moe)
  ]
  vintage_count <- sum(!is.na(data$median_rent))
  max_relative_moe <- if (length(relative_moes) > 0) max(relative_moes, na.rm = TRUE) else NA_real_

  tibble(
    acs_rent_current_year = current_year,
    acs_rent_source_geoid = current_source_geoid,
    acs_rent_source_geography = current_source_geography,
    acs_rent_source_residential_share = current_source_share,
    acs_rent_source_assignment_method = current_source_method,
    acs_rent_current = current_nominal,
    acs_rent_current_real = current_value,
    acs_rent_growth_recent_annualized_pct = recent_growth,
    acs_rent_growth_prior_annualized_pct = prior_growth,
    acs_rent_growth_long_annualized_pct = long_growth,
    acs_rent_acceleration_pp = recent_growth - prior_growth,
    acs_rent_relative_moe_current = current_moe,
    acs_rent_relative_moe_max = max_relative_moe,
    acs_rent_vintages_available = vintage_count,
    acs_rent_trend_reliable = vintage_count == length(EWS_CONFIG$acs_years) &
      !is.na(max_relative_moe) &
      max_relative_moe <= EWS_CONFIG$acs_rent_relative_moe_limit
  )
}

acs_rent_trend_values <- acs_rent_vintages %>%
  st_drop_geometry() %>%
  group_by(hex_id) %>%
  group_modify(~trend_from_vintages(.x)) %>%
  ungroup()

acs_rent_trends <- hex_grid %>%
  select(hex_id, geometry) %>%
  left_join(acs_rent_trend_values, by = "hex_id") %>%
  mutate(analysis_as_of_date = EWS_CONFIG$analysis_as_of_date)

save_output(
  acs_rent_vintages,
  file.path(OUTPUT_DIR, "acs_rent_by_hex_vintage.rds"),
  "ACS rent vintage hex table"
)

acs_rent_vintages %>%
  st_drop_geometry() %>%
  write_csv(file.path(OUTPUT_DIR, "acs_rent_by_hex_vintage.csv"))

write_csv(
  rent_dominant_sources,
  file.path(OUTPUT_DIR, "acs_rent_dominant_sources_by_hex_vintage.csv")
)
write_csv(
  rent_crosswalk_qa,
  file.path(OUTPUT_DIR, "acs_rent_dasymetric_crosswalk_qa.csv")
)

save_output(
  acs_rent_trends,
  file.path(OUTPUT_DIR, "acs_rent_trends_by_hex.rds"),
  "ACS rent trend hex table"
)

acs_rent_trends %>%
  st_drop_geometry() %>%
  write_csv(file.path(OUTPUT_DIR, "acs_rent_trends_by_hex.csv"))

print_header("STEP 02h COMPLETE")
cat(paste0("ACS vintages: ", paste(EWS_CONFIG$acs_years, collapse = ", "), "\n"))
cat(paste0("Current ACS year: ", EWS_CONFIG$acs_current_year, "\n"))
cat(
  paste0(
    "Hexagons with reliable three-vintage rent trends: ",
    sum(acs_rent_trends$acs_rent_trend_reliable, na.rm = TRUE),
    "\n"
  )
)
cat(
  paste0(
    "Current-rent tract fallbacks: ",
    sum(
      acs_rent_trends$acs_rent_source_geography == "tract",
      na.rm = TRUE
    ),
    "\n"
  )
)
