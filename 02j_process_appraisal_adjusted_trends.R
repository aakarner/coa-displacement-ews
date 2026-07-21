################################################################################
# 02j - County-Adjusted Appraisal Value Trends
################################################################################
#
# Removes county-wide appraisal-year shocks from parcel land-value changes
# before aggregating trends to hexagons. The primary baseline is the median
# inflation-adjusted annual log change among stable, improved real-property
# accounts with positive land and improvement values in both years. An all-real
# baseline is retained for sensitivity review.
#
# Outputs:
#   - output/appraisal_county_year_baselines.csv
#   - output/appraisal_county_adjustment_clip_thresholds.csv
#   - output/appraisal_adjusted_parcel_trends.rds
#   - output/appraisal_adjusted_trends_by_hex.rds/.csv
#   - output/appraisal_adjustment_qa.csv
#   - output/appraisal_land_area_fallback_qa.csv
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
  library(data.table)
  library(dplyr)
  library(sf)
})

print_header("02j - COUNTY-ADJUSTED APPRAISAL TRENDS")

OUTPUT_DIR <- project_path("output")
COUNTY_VALUES_FILE <- file.path(
  OUTPUT_DIR,
  "appraisal_county_land_values_by_account_year.rds"
)
PARCEL_PANEL_FILE <- file.path(OUTPUT_DIR, "appraisal_values_by_parcel_year.rds")
HEX_GRID_FILE <- file.path(OUTPUT_DIR, "hex_grid.rds")

years <- sort(as.integer(EWS_CONFIG$appraisal_years))
if (length(years) < 3L || any(diff(years) != 1L)) {
  stop(
    "County-adjusted appraisal trends require at least three consecutive years.",
    call. = FALSE
  )
}

interval_years <- years[-1L]
midpoint_year <- years[[ceiling(length(years) / 2)]]
prior_interval_years <- interval_years[interval_years <= midpoint_year]
recent_interval_years <- interval_years[interval_years > midpoint_year]
clip_probs <- EWS_CONFIG$appraisal_adjustment_clip_quantiles

if (
  length(clip_probs) != 2L || any(!is.finite(clip_probs)) ||
    clip_probs[[1]] < 0 || clip_probs[[2]] > 1 || clip_probs[[1]] >= clip_probs[[2]]
) {
  stop("Invalid appraisal adjustment clip quantiles.", call. = FALSE)
}

cpi <- data.table(
  tax_year = as.integer(names(EWS_CONFIG$appraisal_cpi_u)),
  cpi_u = as.numeric(EWS_CONFIG$appraisal_cpi_u)
)[tax_year %in% years]
current_cpi <- cpi[tax_year == EWS_CONFIG$appraisal_current_year, cpi_u]

county_values <- as.data.table(load_output(
  COUNTY_VALUES_FILE,
  "county appraisal values by account-year"
))
required_county_columns <- c(
  "source_county", "source_account_id", "tax_year",
  "land_market_value", "improvement_market_value"
)
missing_county_columns <- setdiff(required_county_columns, names(county_values))
if (length(missing_county_columns) > 0L) {
  stop(
    "County appraisal panel is missing: ",
    paste(missing_county_columns, collapse = ", "),
    call. = FALSE
  )
}

county_values <- county_values[tax_year %in% years]
county_values <- merge(county_values, cpi, by = "tax_year", all.x = TRUE)
county_values[, land_market_value_real := land_market_value * current_cpi / cpi_u]
setorder(county_values, source_county, source_account_id, tax_year)
county_values[, `:=`(
  previous_tax_year = shift(tax_year),
  previous_land_market_value_real = shift(land_market_value_real),
  previous_improvement_market_value = shift(improvement_market_value)
), by = .(source_county, source_account_id)]

county_pairs <- county_values[
  tax_year %in% interval_years &
    previous_tax_year == tax_year - 1L &
    is.finite(land_market_value_real) & land_market_value_real > 0 &
    is.finite(previous_land_market_value_real) & previous_land_market_value_real > 0
]
county_pairs[, annual_real_log_change_pct := 100 * (
  log(land_market_value_real) - log(previous_land_market_value_real)
)]
county_pairs[, improved_real_pair :=
  is.finite(improvement_market_value) & improvement_market_value > 0 &
    is.finite(previous_improvement_market_value) &
    previous_improvement_market_value > 0]

county_baselines <- county_pairs[, .(
  all_real_stable_accounts = .N,
  all_real_median_change_pct = median(annual_real_log_change_pct, na.rm = TRUE),
  improved_real_stable_accounts = sum(improved_real_pair),
  improved_real_median_change_pct = median(
    annual_real_log_change_pct[improved_real_pair],
    na.rm = TRUE
  )
), by = .(source_county, tax_year)]
county_baselines[, `:=`(
  previous_tax_year = tax_year - 1L,
  primary_baseline = "improved_real_median",
  primary_county_change_pct = improved_real_median_change_pct,
  improved_minus_all_real_pp =
    improved_real_median_change_pct - all_real_median_change_pct,
  analysis_as_of_date = EWS_CONFIG$analysis_as_of_date
)]
setcolorder(
  county_baselines,
  c(
    "source_county", "previous_tax_year", "tax_year",
    "all_real_stable_accounts", "all_real_median_change_pct",
    "improved_real_stable_accounts", "improved_real_median_change_pct",
    "improved_minus_all_real_pp", "primary_baseline",
    "primary_county_change_pct", "analysis_as_of_date"
  )
)
setorder(county_baselines, source_county, tax_year)

if (
  nrow(county_baselines) !=
    length(unique(county_values$source_county)) * length(interval_years) ||
    any(county_baselines$improved_real_stable_accounts < 100L) ||
    any(!is.finite(county_baselines$primary_county_change_pct))
) {
  stop("County appraisal baselines are incomplete or unstable.", call. = FALSE)
}

fwrite(
  county_baselines,
  file.path(OUTPUT_DIR, "appraisal_county_year_baselines.csv"),
  na = ""
)

county_pairs <- merge(
  county_pairs,
  county_baselines[, .(
    source_county, tax_year, primary_county_change_pct
  )],
  by = c("source_county", "tax_year"),
  all.x = TRUE
)
county_pairs[, county_adjusted_change_pct :=
  annual_real_log_change_pct - primary_county_change_pct]

clip_thresholds <- county_pairs[improved_real_pair == TRUE, .(
  clip_lower_pct = as.numeric(quantile(
    county_adjusted_change_pct,
    probs = clip_probs[[1]],
    na.rm = TRUE,
    names = FALSE,
    type = 8
  )),
  clip_upper_pct = as.numeric(quantile(
    county_adjusted_change_pct,
    probs = clip_probs[[2]],
    na.rm = TRUE,
    names = FALSE,
    type = 8
  )),
  improved_real_pairs = .N
), by = .(source_county, tax_year)]
clip_thresholds[, `:=`(
  lower_quantile = clip_probs[[1]],
  upper_quantile = clip_probs[[2]],
  analysis_as_of_date = EWS_CONFIG$analysis_as_of_date
)]
setorder(clip_thresholds, source_county, tax_year)
fwrite(
  clip_thresholds,
  file.path(OUTPUT_DIR, "appraisal_county_adjustment_clip_thresholds.csv"),
  na = ""
)

rm(county_values, county_pairs)
invisible(gc())

parcel_panel <- as.data.table(load_output(
  PARCEL_PANEL_FILE,
  "appraisal parcel-year panel"
))
required_parcel_columns <- c(
  "parcel_id", "source_county", "hex_id", "tax_year", "current_units",
  "current_land_sqft", "land_sqft_year", "land_market_value_real"
)
missing_parcel_columns <- setdiff(required_parcel_columns, names(parcel_panel))
if (length(missing_parcel_columns) > 0L) {
  stop(
    "Appraisal parcel panel is missing: ",
    paste(missing_parcel_columns, collapse = ", "),
    call. = FALSE
  )
}

parcel_panel <- parcel_panel[tax_year %in% years, ..required_parcel_columns]
setorder(parcel_panel, parcel_id, tax_year)
parcel_panel[, `:=`(
  previous_tax_year = shift(tax_year),
  previous_land_market_value_real = shift(land_market_value_real)
), by = parcel_id]
parcel_panel[, annual_raw_real_log_change_pct := fifelse(
  tax_year %in% interval_years &
    previous_tax_year == tax_year - 1L &
    is.finite(land_market_value_real) & land_market_value_real > 0 &
    is.finite(previous_land_market_value_real) &
    previous_land_market_value_real > 0,
  100 * (log(land_market_value_real) - log(previous_land_market_value_real)),
  NA_real_
)]

parcel_panel <- merge(
  parcel_panel,
  county_baselines[, .(
    source_county, tax_year, primary_county_change_pct
  )],
  by = c("source_county", "tax_year"),
  all.x = TRUE,
  sort = FALSE
)
parcel_panel <- merge(
  parcel_panel,
  clip_thresholds[, .(
    source_county, tax_year, clip_lower_pct, clip_upper_pct
  )],
  by = c("source_county", "tax_year"),
  all.x = TRUE,
  sort = FALSE
)
parcel_panel[, annual_county_adjusted_change_pct :=
  annual_raw_real_log_change_pct - primary_county_change_pct]
parcel_panel[, annual_county_adjusted_change_clipped_pct := pmin(
  pmax(annual_county_adjusted_change_pct, clip_lower_pct),
  clip_upper_pct
)]
parcel_panel[, annual_change_was_clipped :=
  is.finite(annual_county_adjusted_change_pct) &
    is.finite(annual_county_adjusted_change_clipped_pct) &
    annual_county_adjusted_change_pct !=
      annual_county_adjusted_change_clipped_pct]

williamson_geometry_file <- project_path(
  "data", "raw_parcels", "williamson", "wcad_parcels.rds"
)
williamson_area_cache <- file.path(
  OUTPUT_DIR,
  "appraisal_williamson_geometry_land_sqft.rds"
)
williamson_target_ids <- sub(
  "^WILLIAMSON:",
  "",
  unique(parcel_panel[source_county == "Williamson", parcel_id])
)

cache_is_current <- file.exists(williamson_area_cache) &&
  file.exists(williamson_geometry_file) &&
  file.mtime(williamson_area_cache) >= file.mtime(williamson_geometry_file)

if (cache_is_current) {
  williamson_land_area <- as.data.table(readRDS(williamson_area_cache))
} else if (file.exists(williamson_geometry_file)) {
  print_progress("Deriving Williamson land area from parcel polygons...")
  max_numeric_or_na <- function(x) {
    x <- suppressWarnings(as.numeric(x))
    if (all(!is.finite(x))) NA_real_ else max(x, na.rm = TRUE)
  }
  williamson_polygons <- readRDS(williamson_geometry_file) %>%
    filter(parcelid %in% williamson_target_ids) %>%
    select(parcelid, assessedacres, geometry)
  williamson_polygons <- suppressWarnings(st_make_valid(williamson_polygons)) %>%
    group_by(parcelid) %>%
    summarise(
      reported_acres = max_numeric_or_na(assessedacres),
      .groups = "drop"
    )
  williamson_polygons$geometry_land_sqft <- as.numeric(units::set_units(
    st_area(st_transform(williamson_polygons, 2277)),
    "ft^2"
  ))
  williamson_land_area <- williamson_polygons %>%
    st_drop_geometry() %>%
    transmute(
      parcel_id = paste0("WILLIAMSON:", parcelid),
      reported_land_sqft = reported_acres * 43560,
      geometry_land_sqft
    ) %>%
    as.data.table()
  saveRDS(williamson_land_area, williamson_area_cache)
} else {
  warning(
    "Williamson parcel polygons are unavailable; land-value level coverage will be limited.",
    call. = FALSE
  )
  williamson_land_area <- data.table(
    parcel_id = character(),
    reported_land_sqft = double(),
    geometry_land_sqft = double()
  )
}

if (nrow(williamson_land_area) > 0L) {
  geometry_reconciliation <- williamson_land_area[
    is.finite(reported_land_sqft) & reported_land_sqft > 0 &
      is.finite(geometry_land_sqft) & geometry_land_sqft > 0
  ]
  land_area_qa <- data.table(
    source_county = "Williamson",
    target_parcels = length(williamson_target_ids),
    parcels_with_geometry_area = williamson_land_area[
      is.finite(geometry_land_sqft) & geometry_land_sqft > 0,
      .N
    ],
    parcels_with_reported_and_geometry_area = nrow(geometry_reconciliation),
    median_absolute_pct_difference = 100 * median(
      abs(
        geometry_reconciliation$geometry_land_sqft -
          geometry_reconciliation$reported_land_sqft
      ) / geometry_reconciliation$reported_land_sqft,
      na.rm = TRUE
    ),
    pct_within_10_percent = 100 * mean(
      abs(
        geometry_reconciliation$geometry_land_sqft -
          geometry_reconciliation$reported_land_sqft
      ) / geometry_reconciliation$reported_land_sqft <= 0.10,
      na.rm = TRUE
    ),
    analysis_as_of_date = EWS_CONFIG$analysis_as_of_date
  )
} else {
  land_area_qa <- data.table(
    source_county = "Williamson",
    target_parcels = length(williamson_target_ids),
    parcels_with_geometry_area = 0L,
    parcels_with_reported_and_geometry_area = 0L,
    median_absolute_pct_difference = NA_real_,
    pct_within_10_percent = NA_real_,
    analysis_as_of_date = EWS_CONFIG$analysis_as_of_date
  )
}
fwrite(
  land_area_qa,
  file.path(OUTPUT_DIR, "appraisal_land_area_fallback_qa.csv"),
  na = ""
)

parcel_panel <- merge(
  parcel_panel,
  williamson_land_area[, .(parcel_id, geometry_land_sqft)],
  by = "parcel_id",
  all.x = TRUE,
  sort = FALSE
)

current_levels <- parcel_panel[
  tax_year == EWS_CONFIG$appraisal_current_year &
    is.finite(land_market_value_real) & land_market_value_real > 0 &
    (
      (is.finite(current_land_sqft) & current_land_sqft > 0) |
        (is.finite(land_sqft_year) & land_sqft_year > 0) |
        (is.finite(geometry_land_sqft) & geometry_land_sqft > 0)
    ),
  .(
    parcel_id,
    source_county,
    land_sqft_for_level = fcoalesce(
      fifelse(current_land_sqft > 0, current_land_sqft, NA_real_),
      fifelse(land_sqft_year > 0, land_sqft_year, NA_real_),
      fifelse(geometry_land_sqft > 0, geometry_land_sqft, NA_real_)
    ),
    land_area_source = fcase(
      is.finite(current_land_sqft) & current_land_sqft > 0,
      "calibrated_parcel",
      is.finite(land_sqft_year) & land_sqft_year > 0,
      "appraisal_reported_acres",
      is.finite(geometry_land_sqft) & geometry_land_sqft > 0,
      "williamson_parcel_geometry",
      default = NA_character_
    ),
    land_value_real_per_current_land_sqft =
      land_market_value_real / fcoalesce(
        fifelse(current_land_sqft > 0, current_land_sqft, NA_real_),
        fifelse(land_sqft_year > 0, land_sqft_year, NA_real_),
        fifelse(geometry_land_sqft > 0, geometry_land_sqft, NA_real_)
      )
  )
]
current_levels[, county_project_land_value_percentile := if (.N == 1L) {
  50
} else {
  100 * (frank(
    land_value_real_per_current_land_sqft,
    ties.method = "average"
  ) - 1) / (.N - 1)
}, by = source_county]

parcel_info <- unique(
  parcel_panel[, .(
    parcel_id, source_county, hex_id, current_units, current_land_sqft
  )],
  by = "parcel_id"
)
change_wide <- dcast(
  parcel_panel[tax_year %in% interval_years],
  parcel_id ~ tax_year,
  value.var = c(
    "annual_raw_real_log_change_pct",
    "annual_county_adjusted_change_clipped_pct",
    "annual_change_was_clipped"
  )
)
parcel_trends <- merge(parcel_info, change_wide, by = "parcel_id", all.x = TRUE)
parcel_trends <- merge(
  parcel_trends,
  current_levels[, .(
    parcel_id,
    land_sqft_for_level,
    land_area_source,
    land_value_real_per_current_land_sqft,
    county_project_land_value_percentile
  )],
  by = "parcel_id",
  all.x = TRUE
)

raw_columns <- paste0("annual_raw_real_log_change_pct_", interval_years)
adjusted_columns <- paste0(
  "annual_county_adjusted_change_clipped_pct_",
  interval_years
)
clip_columns <- paste0("annual_change_was_clipped_", interval_years)
prior_raw_columns <- paste0("annual_raw_real_log_change_pct_", prior_interval_years)
recent_raw_columns <- paste0("annual_raw_real_log_change_pct_", recent_interval_years)
prior_adjusted_columns <- paste0(
  "annual_county_adjusted_change_clipped_pct_",
  prior_interval_years
)
recent_adjusted_columns <- paste0(
  "annual_county_adjusted_change_clipped_pct_",
  recent_interval_years
)

complete_row_mean <- function(data, columns) {
  values <- as.matrix(data[, ..columns])
  complete <- rowSums(is.finite(values)) == length(columns)
  output <- rep(NA_real_, nrow(data))
  output[complete] <- rowMeans(values[complete, , drop = FALSE])
  output
}

parcel_trends[, appraisal_intervals_available := rowSums(
  is.finite(as.matrix(.SD))
), .SDcols = adjusted_columns]
parcel_trends[, appraisal_complete_adjusted_trend :=
  appraisal_intervals_available == length(interval_years)]
parcel_trends[, land_value_growth_long_raw_annualized_pct :=
  complete_row_mean(parcel_trends, raw_columns)]
parcel_trends[, land_value_growth_long_county_adjusted_pct :=
  complete_row_mean(parcel_trends, adjusted_columns)]
parcel_trends[, land_value_growth_prior_raw_annualized_pct :=
  complete_row_mean(parcel_trends, prior_raw_columns)]
parcel_trends[, land_value_growth_recent_raw_annualized_pct :=
  complete_row_mean(parcel_trends, recent_raw_columns)]
parcel_trends[, land_value_growth_prior_county_adjusted_pct :=
  complete_row_mean(parcel_trends, prior_adjusted_columns)]
parcel_trends[, land_value_growth_recent_county_adjusted_pct :=
  complete_row_mean(parcel_trends, recent_adjusted_columns)]
parcel_trends[, land_value_acceleration_raw_pp :=
  land_value_growth_recent_raw_annualized_pct -
    land_value_growth_prior_raw_annualized_pct]
parcel_trends[, land_value_acceleration_county_adjusted_pp :=
  land_value_growth_recent_county_adjusted_pct -
    land_value_growth_prior_county_adjusted_pct]
parcel_trends[, appraisal_annual_changes_clipped := rowSums(
  as.matrix(.SD) == TRUE,
  na.rm = TRUE
), .SDcols = clip_columns]
parcel_trends[, analysis_as_of_date := EWS_CONFIG$analysis_as_of_date]

save_output(
  parcel_trends,
  file.path(OUTPUT_DIR, "appraisal_adjusted_parcel_trends.rds"),
  "county-adjusted appraisal parcel trends"
)

median_or_na <- function(x) {
  if (length(x) == 0L || all(!is.finite(x))) NA_real_ else median(x, na.rm = TRUE)
}

hex_values <- parcel_trends[!is.na(hex_id), .(
  appraisal_current_parcels = .N,
  appraisal_current_units = sum(current_units, na.rm = TRUE),
  appraisal_complete_trend_parcels = sum(appraisal_complete_adjusted_trend),
  appraisal_complete_trend_units = sum(
    current_units[appraisal_complete_adjusted_trend],
    na.rm = TRUE
  ),
  appraisal_adjusted_trend_parcel_coverage_pct =
    100 * mean(appraisal_complete_adjusted_trend),
  appraisal_current_level_parcels = sum(
    is.finite(county_project_land_value_percentile)
  ),
  appraisal_current_level_parcel_coverage_pct =
    100 * mean(is.finite(county_project_land_value_percentile)),
  appraisal_annual_changes_clipped = sum(appraisal_annual_changes_clipped),
  land_value_real_per_current_land_sqft = median_or_na(
    land_value_real_per_current_land_sqft
  ),
  land_value_county_project_percentile_current = median_or_na(
    county_project_land_value_percentile
  ),
  land_value_growth_long_raw_annualized_pct = median_or_na(
    land_value_growth_long_raw_annualized_pct
  ),
  land_value_growth_long_county_adjusted_pct = median_or_na(
    land_value_growth_long_county_adjusted_pct
  ),
  land_value_growth_recent_raw_annualized_pct = median_or_na(
    land_value_growth_recent_raw_annualized_pct
  ),
  land_value_growth_recent_county_adjusted_pct = median_or_na(
    land_value_growth_recent_county_adjusted_pct
  ),
  land_value_growth_prior_raw_annualized_pct = median_or_na(
    land_value_growth_prior_raw_annualized_pct
  ),
  land_value_growth_prior_county_adjusted_pct = median_or_na(
    land_value_growth_prior_county_adjusted_pct
  ),
  land_value_acceleration_raw_pp = median_or_na(
    land_value_acceleration_raw_pp
  ),
  land_value_acceleration_county_adjusted_pp = median_or_na(
    land_value_acceleration_county_adjusted_pp
  )
), by = hex_id]
hex_values[, appraisal_adjusted_trend_reliable :=
  appraisal_current_units >= EWS_CONFIG$minimum_residential_units_for_rates &
    appraisal_adjusted_trend_parcel_coverage_pct >=
      EWS_CONFIG$appraisal_min_parcel_coverage * 100]

hex_grid <- load_output(HEX_GRID_FILE, "hexagonal grid")
hex_trends <- hex_grid %>%
  select(hex_id, geometry) %>%
  left_join(as_tibble(hex_values), by = "hex_id") %>%
  mutate(analysis_as_of_date = EWS_CONFIG$analysis_as_of_date)

save_output(
  hex_trends,
  file.path(OUTPUT_DIR, "appraisal_adjusted_trends_by_hex.rds"),
  "county-adjusted appraisal trends by hex"
)
fwrite(
  st_drop_geometry(hex_trends),
  file.path(OUTPUT_DIR, "appraisal_adjusted_trends_by_hex.csv"),
  na = ""
)

adjustment_qa <- parcel_trends[, .(
  current_parcels = .N,
  complete_adjusted_trend_parcels = sum(appraisal_complete_adjusted_trend),
  complete_adjusted_trend_coverage_pct =
    100 * mean(appraisal_complete_adjusted_trend),
  parcels_with_current_level = sum(
    is.finite(county_project_land_value_percentile)
  ),
  annual_changes_clipped = sum(appraisal_annual_changes_clipped),
  median_long_raw_change_pct = median_or_na(
    land_value_growth_long_raw_annualized_pct
  ),
  median_long_county_adjusted_change_pct = median_or_na(
    land_value_growth_long_county_adjusted_pct
  ),
  median_recent_raw_change_pct = median_or_na(
    land_value_growth_recent_raw_annualized_pct
  ),
  median_recent_county_adjusted_change_pct = median_or_na(
    land_value_growth_recent_county_adjusted_pct
  ),
  median_county_adjusted_acceleration_pp = median_or_na(
    land_value_acceleration_county_adjusted_pp
  )
), by = source_county]
adjustment_qa[, analysis_as_of_date := EWS_CONFIG$analysis_as_of_date]
setorder(adjustment_qa, source_county)
fwrite(
  adjustment_qa,
  file.path(OUTPUT_DIR, "appraisal_adjustment_qa.csv"),
  na = ""
)

print_header("STEP 02j COMPLETE")
cat(paste0(
  "County account-year rows: ",
  format(sum(county_baselines$all_real_stable_accounts), big.mark = ","),
  " stable interval records\n"
))
cat(paste0(
  "Complete adjusted parcel trends: ",
  format(sum(parcel_trends$appraisal_complete_adjusted_trend), big.mark = ","),
  " / ", format(nrow(parcel_trends), big.mark = ","), "\n"
))
cat(paste0(
  "Reliable adjusted hex trends: ",
  format(sum(hex_values$appraisal_adjusted_trend_reliable, na.rm = TRUE), big.mark = ","),
  " hexagons\n"
))
