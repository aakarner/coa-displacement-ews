################################################################################
# 02h - Process Historical ACS Rent to Hexagonal Grid
################################################################################
#
# Creates a citywide rent-pressure backbone from non-overlapping ACS 5-year
# vintages. CoStar remains an optional enrichment source because it does not
# cover most project hexagons.
#
# Outputs:
#   - output/acs_rent_by_hex_vintage.rds/.csv
#   - output/acs_rent_trends_by_hex.rds/.csv
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

print_header("02h - HISTORICAL ACS RENT TO HEX GRID")

OUTPUT_DIR <- project_path("output")
ANALYSIS_CRS <- 3857

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
sf::sf_use_s2(FALSE)

hex_grid <- load_output(file.path(OUTPUT_DIR, "hex_grid.rds"), "hexagonal grid") %>%
  st_transform(4326)

hex_projected <- hex_grid %>%
  st_transform(ANALYSIS_CRS) %>%
  select(hex_id, geometry)

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

fetch_acs_rent_vintage <- function(acs_year) {
  print_progress(
    paste0(
      "Fetching ", acs_year, " ", EWS_CONFIG$acs_survey,
      " median gross rent for ", paste(EWS_CONFIG$acs_counties, collapse = ", "),
      " Counties..."
    )
  )

  acs_rent <- tidycensus::get_acs(
    geography = "tract",
    variables = c(median_rent = "B25064_001"),
    state = "TX",
    county = EWS_CONFIG$acs_counties,
    year = acs_year,
    survey = EWS_CONFIG$acs_survey,
    geometry = TRUE,
    output = "tidy",
    cache_table = TRUE
  ) %>%
    st_transform(ANALYSIS_CRS) %>%
    mutate(tract_area_sqm = as.numeric(st_area(geometry)))

  intersections <- suppressWarnings(
    st_intersection(hex_projected, acs_rent)
  ) %>%
    mutate(intersection_area_sqm = as.numeric(st_area(geometry))) %>%
    st_drop_geometry()

  rent_hex <- intersections %>%
    group_by(hex_id) %>%
    mutate(overlap_weight = intersection_area_sqm / sum(intersection_area_sqm)) %>%
    summarise(
      median_rent = if (all(is.na(estimate))) {
        NA_real_
      } else {
        weighted.mean(estimate, intersection_area_sqm, na.rm = TRUE)
      },
      median_rent_moe = if (all(is.na(moe))) {
        NA_real_
      } else {
        sqrt(sum((moe * overlap_weight)^2, na.rm = TRUE))
      },
      .groups = "drop"
    )

  current_cpi <- unname(EWS_CONFIG$acs_cpi_u[[as.character(EWS_CONFIG$acs_current_year)]])
  vintage_cpi <- unname(EWS_CONFIG$acs_cpi_u[[as.character(acs_year)]])

  hex_grid %>%
    select(hex_id, geometry) %>%
    left_join(rent_hex, by = "hex_id") %>%
    mutate(
      acs_year = acs_year,
      acs_survey = EWS_CONFIG$acs_survey,
      cpi_u = vintage_cpi,
      median_rent_real = median_rent * current_cpi / vintage_cpi,
      median_rent_moe_real = median_rent_moe * current_cpi / vintage_cpi,
      median_rent_relative_moe = if_else(
        median_rent > 0,
        median_rent_moe / median_rent,
        NA_real_
      )
    )
}

acs_rent_vintages <- lapply(
  EWS_CONFIG$acs_years,
  fetch_acs_rent_vintage
) %>%
  bind_rows() %>%
  arrange(hex_id, acs_year)

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

  relative_moes <- data$median_rent_relative_moe[!is.na(data$median_rent)]
  vintage_count <- sum(!is.na(data$median_rent))
  max_relative_moe <- if (length(relative_moes) > 0) max(relative_moes, na.rm = TRUE) else NA_real_

  tibble(
    acs_rent_current_year = current_year,
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
