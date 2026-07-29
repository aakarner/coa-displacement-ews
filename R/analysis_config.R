################################################################################
# Shared Analysis Configuration
################################################################################
#
# The Part 1 baseline represents information available as of a fixed date. Data
# sources may lag that date, but no source should include observations after it.
# Environment variables allow reproducible historical reruns without editing
# scripts.

parse_year_list <- function(value, default) {
  if (!nzchar(value)) return(default)
  years <- suppressWarnings(as.integer(strsplit(value, ",", fixed = TRUE)[[1]]))
  years <- sort(unique(years[!is.na(years)]))
  if (length(years) == 0) default else years
}

analysis_as_of_date <- as.Date(
  Sys.getenv("EWS_ANALYSIS_AS_OF_DATE", unset = "2026-04-01")
)

if (is.na(analysis_as_of_date)) {
  stop("EWS_ANALYSIS_AS_OF_DATE must use YYYY-MM-DD format.", call. = FALSE)
}

acs_years <- parse_year_list(
  Sys.getenv("EWS_ACS_YEARS", unset = ""),
  c(2014L, 2019L, 2024L)
)

acs_current_year <- as.integer(
  Sys.getenv("EWS_ACS_CURRENT_YEAR", unset = as.character(max(acs_years)))
)

if (!acs_current_year %in% acs_years) {
  stop("EWS_ACS_CURRENT_YEAR must be included in EWS_ACS_YEARS.", call. = FALSE)
}

# BLS CPI-U annual averages, used only to express ACS dollar measures in the
# latest configured ACS year's dollars. Add a value here before adding a new ACS
# vintage. Source: BLS CPI for All Urban Consumers, U.S. city average.
acs_cpi_u <- c(
  `2014` = 236.736,
  `2019` = 255.657,
  `2024` = 313.689
)

missing_cpi_years <- setdiff(as.character(acs_years), names(acs_cpi_u))
if (length(missing_cpi_years) > 0) {
  stop(
    "Missing CPI-U annual average(s) for ACS year(s): ",
    paste(missing_cpi_years, collapse = ", "),
    call. = FALSE
  )
}

appraisal_years <- parse_year_list(
  Sys.getenv("EWS_APPRAISAL_YEARS", unset = ""),
  2021:2025
)

appraisal_current_year <- as.integer(
  Sys.getenv(
    "EWS_APPRAISAL_CURRENT_YEAR",
    unset = as.character(max(appraisal_years))
  )
)

if (!appraisal_current_year %in% appraisal_years) {
  stop(
    "EWS_APPRAISAL_CURRENT_YEAR must be included in EWS_APPRAISAL_YEARS.",
    call. = FALSE
  )
}

# BLS CPI-U annual averages. Appraisal values are converted to the latest
# configured appraisal year's dollars before trends are calculated.
appraisal_cpi_u <- c(
  `2021` = 270.970,
  `2022` = 292.655,
  `2023` = 304.702,
  `2024` = 313.689,
  `2025` = 321.943
)

missing_appraisal_cpi_years <- setdiff(
  as.character(appraisal_years),
  names(appraisal_cpi_u)
)
if (length(missing_appraisal_cpi_years) > 0) {
  stop(
    "Missing CPI-U annual average(s) for appraisal year(s): ",
    paste(missing_appraisal_cpi_years, collapse = ", "),
    call. = FALSE
  )
}

amenity_cluster_k <- as.integer(
  Sys.getenv("EWS_AMENITY_CLUSTER_K", unset = "6")
)
if (is.na(amenity_cluster_k) || amenity_cluster_k < 2L) {
  stop("EWS_AMENITY_CLUSTER_K must be an integer of at least 2.", call. = FALSE)
}

baseline_cluster_specification <- Sys.getenv(
  "EWS_BASELINE_CLUSTER_SPECIFICATION",
  unset = "amenity_augmented"
)
if (!baseline_cluster_specification %in% c("baseline", "amenity_augmented")) {
  stop(
    "EWS_BASELINE_CLUSTER_SPECIFICATION must be 'baseline' or ",
    "'amenity_augmented'.",
    call. = FALSE
  )
}

EWS_CONFIG <- list(
  analysis_as_of_date = analysis_as_of_date,
  h3_resolution = 9L,
  acs_years = acs_years,
  acs_current_year = acs_current_year,
  acs_survey = "acs5",
  acs_counties = c("Travis", "Hays", "Williamson"),
  acs_cpi_u = acs_cpi_u,
  acs_rent_relative_moe_limit = 0.30,
  acs_median_relative_moe_limit = 0.30,
  appraisal_years = appraisal_years,
  appraisal_current_year = appraisal_current_year,
  appraisal_cpi_u = appraisal_cpi_u,
  appraisal_min_parcel_coverage = 0.80,
  appraisal_adjustment_clip_quantiles = c(0.01, 0.99),
  costar_include_qtd = FALSE,
  demolition_recent_years = 2L,
  transaction_recent_years = 2L,
  transaction_analysis_as_of_date = as.Date("2025-04-30"),
  amenity_window_months = 18L,
  amenity_access_radius_m = 800,
  minimum_residential_units_for_rates = 20L,
  amenity_cluster_k = amenity_cluster_k,
  baseline_cluster_specification = baseline_cluster_specification,
  cluster_assignment_distance_quantile = 0.95,
  cluster_assignment_margin_quantile = 0.10,
  forecast_horizons_years = c(1L, 3L, 5L)
)
