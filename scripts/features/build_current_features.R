################################################################################
# Build Current Features for Displacement Risk Pattern Discovery
################################################################################
#
# This script creates the hex-level feature table used by cluster analysis. It
# supports two paths:
#   1. By default, assemble a current local feature table from the saved stream
#      outputs: CoStar rents, demolition permits, eviction summaries, corporate
#      ownership, calibrated parcel units, and the hex grid.
#   2. Set EWS_USE_PROCESSED_HEX=true to use output/hex_data_processed.rds as a
#      legacy integrated starting point.
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
  library(lubridate)
  library(stringr)
})

print_header("03 - FEATURE ENGINEERING")

OUTPUT_DIR <- project_path("output")
DATA_DIR <- project_path("data")
CORPORATE_HEX_OVERRIDE_FILE <- Sys.getenv(
  "EWS_CORPORATE_HEX_FILE",
  unset = ""
)
HEX_FEATURE_OUTPUT_FILE <- Sys.getenv(
  "EWS_HEX_FEATURE_OUTPUT_FILE",
  unset = file.path(OUTPUT_DIR, "hex_features.rds")
)
FEATURE_LIST_OUTPUT_FILE <- Sys.getenv(
  "EWS_FEATURE_LIST_OUTPUT_FILE",
  unset = file.path(OUTPUT_DIR, "feature_list.csv")
)
USE_PROCESSED_HEX <- tolower(
  Sys.getenv("EWS_USE_PROCESSED_HEX", unset = "false")
) %in% c("true", "t", "1", "yes", "y")

safe_pct_change <- function(current, baseline) {
  ifelse(!is.na(current) & !is.na(baseline) & baseline > 0,
         100 * (current / baseline - 1),
         NA_real_)
}

safe_annualized_log_change <- function(current, baseline, years) {
  ifelse(
    !is.na(current) & !is.na(baseline) & current > 0 & baseline > 0 & years > 0,
    100 * (log(current) - log(baseline)) / years,
    NA_real_
  )
}

weighted_mean_or_na <- function(x, w) {
  ok <- !is.na(x) & !is.na(w) & w > 0
  if (!any(ok)) return(NA_real_)
  weighted.mean(x[ok], w[ok])
}

parse_number_flexible <- function(x) {
  if (is.numeric(x)) return(as.numeric(x))
  readr::parse_number(as.character(x), na = c("", "NA", "-", "\u2014"))
}

first_non_missing <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) NA_real_ else x[[1]]
}

last_non_missing <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) NA_real_ else x[[length(x)]]
}

recent_cv <- function(x, n = 8) {
  x <- tail(x[!is.na(x)], n)
  if (length(x) < 3 || mean(x) <= 0) return(NA_real_)
  sd(x) / mean(x)
}

recent_mean_delta <- function(x, n = 4) {
  x <- x[!is.na(x)]
  if (length(x) >= n * 2) {
    mean(tail(x, n)) - mean(tail(head(x, -n), n))
  } else if (length(x) >= 2) {
    x[[length(x)]] - x[[length(x) - 1]]
  } else {
    NA_real_
  }
}

cap_upper_quantile <- function(x, q = 0.99) {
  if (all(is.na(x))) return(x)
  cap <- as.numeric(quantile(x, q, na.rm = TRUE, names = FALSE))
  pmin(x, cap)
}

load_sf_output <- function(path, description) {
  load_output(path, description) %>%
    st_transform(4326)
}

################################################################################
# Current local stream assembly
################################################################################

build_current_stream_features <- function() {
  print_progress("Assembling current feature table from saved local stream outputs...")

  hex_grid <- load_sf_output(file.path(OUTPUT_DIR, "hex_grid.rds"), "hexagonal grid")

  hex_features <- hex_grid %>%
    select(hex_id, h3_index, longitude, latitude, area_km2, geometry)

  ##############################################################################
  # Rent features from CoStar time series
  ##############################################################################

  rent_file <- file.path(DATA_DIR, "CoStarHistoric-clean.csv")
  geocoded_buildings_file <- file.path(DATA_DIR, "geocoded_buildings.csv")

  if (file.exists(rent_file) && file.exists(geocoded_buildings_file)) {
    print_progress("Creating current rent features from CoStar time series...")

    rent_data <- read_csv(rent_file, show_col_types = FALSE) %>%
      mutate(
        period_label = Period,
        costar_period_is_qtd = str_detect(Period, " QTD$"),
        Period = yq(str_remove(Period, " QTD$")),
        ask_rent_unit_num = parse_number_flexible(askRent_PerUnit),
        ask_rent_psf_num = parse_number_flexible(askRent_PerSF),
        vacancy_units_num = parse_number_flexible(vacancy_Units),
        vacancy_pct_num = parse_number_flexible(vacancy_Percent) / 100,
        inventory_units_num = parse_number_flexible(inventory_Units)
      ) %>%
      filter(
        !is.na(Period),
        Period <= floor_date(EWS_CONFIG$analysis_as_of_date, "quarter"),
        EWS_CONFIG$costar_include_qtd | !costar_period_is_qtd
      )

    building_coords <- read_csv(geocoded_buildings_file, show_col_types = FALSE) %>%
      distinct(`Building Address`, `Building Name`, `Zip Code`, .keep_all = TRUE) %>%
      select(`Building Address`, `Building Name`, `Zip Code`, latitude, longitude)

    rent_pts <- rent_data %>%
      left_join(building_coords, by = c("Building Address", "Building Name", "Zip Code")) %>%
      filter(!is.na(latitude), !is.na(longitude), !is.na(Period)) %>%
      st_as_sf(coords = c("longitude", "latitude"), crs = 4326, remove = FALSE) %>%
      st_transform(st_crs(hex_grid))

    rent_hex <- rent_pts %>%
      st_join(hex_grid %>% select(hex_id), join = st_within, left = FALSE)

    rent_hex_quarter <- rent_hex %>%
      st_drop_geometry() %>%
      group_by(hex_id, Period) %>%
      summarise(
        n_rent_records = n(),
        n_buildings_current = n_distinct(`Building Address`, na.rm = TRUE),
        rent_units_current = sum(inventory_units_num, na.rm = TRUE),
        rent_unit_mean_w = weighted_mean_or_na(ask_rent_unit_num, inventory_units_num),
        rent_psf_mean_w = weighted_mean_or_na(ask_rent_psf_num, inventory_units_num),
        vacancy_pct_w = weighted_mean_or_na(vacancy_pct_num, inventory_units_num),
        vacancy_units_total = sum(vacancy_units_num, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      arrange(hex_id, Period) %>%
      group_by(hex_id) %>%
      mutate(
        rent_change_qoq = safe_pct_change(rent_unit_mean_w, lag(rent_unit_mean_w, 1)),
        rent_change_recent = safe_pct_change(rent_unit_mean_w, lag(rent_unit_mean_w, 4)),
        rent_change_5yr = safe_pct_change(rent_unit_mean_w, lag(rent_unit_mean_w, 20)),
        rent_growth_1yr_annualized_pct = safe_annualized_log_change(
          rent_unit_mean_w,
          lag(rent_unit_mean_w, 4),
          1
        ),
        rent_growth_5yr_annualized_pct = safe_annualized_log_change(
          rent_unit_mean_w,
          lag(rent_unit_mean_w, 20),
          5
        ),
        rent_growth_previous_1yr_annualized_pct = lag(
          rent_growth_1yr_annualized_pct,
          4
        )
      ) %>%
      ungroup()

    rent_features <- rent_hex_quarter %>%
      group_by(hex_id) %>%
      arrange(Period, .by_group = TRUE) %>%
      summarise(
        costar_present = 1,
        rent_first_period = min(Period, na.rm = TRUE),
        rent_latest_period = max(Period, na.rm = TRUE),
        rent_current = last_non_missing(rent_unit_mean_w),
        rent_psf_current = last_non_missing(rent_psf_mean_w),
        vacancy_pct_current = last_non_missing(vacancy_pct_w),
        rent_units_current = last_non_missing(rent_units_current),
        n_buildings_current = last_non_missing(n_buildings_current),
        rent_acceleration = recent_mean_delta(rent_change_recent),
        rent_change_recent = last_non_missing(rent_change_recent),
        rent_change_total = {
          five_year <- last_non_missing(rent_change_5yr)
          ifelse(is.na(five_year),
                 safe_pct_change(last_non_missing(rent_unit_mean_w),
                                 first_non_missing(rent_unit_mean_w)),
                 five_year)
        },
        costar_rent_growth_recent_annualized_pct = last_non_missing(
          rent_growth_1yr_annualized_pct
        ),
        costar_rent_growth_long_annualized_pct = last_non_missing(
          rent_growth_5yr_annualized_pct
        ),
        costar_rent_acceleration_pp = last_non_missing(
          rent_growth_1yr_annualized_pct - rent_growth_previous_1yr_annualized_pct
        ),
        rent_volatility = recent_cv(rent_unit_mean_w),
        costar_data_end = max(Period, na.rm = TRUE),
        .groups = "drop"
      )

    hex_features <- hex_features %>%
      left_join(rent_features, by = "hex_id")
  } else {
    print_progress("WARNING: CoStar rent file or geocoded building file missing; skipping rent features.")
  }

  ##############################################################################
  # Demolition permit features
  ##############################################################################

  demolitions_file <- file.path(DATA_DIR, "Issued_Construction_Permits_20260401.csv")
  if (file.exists(demolitions_file)) {
    print_progress("Creating demolition features from residential demolition permits...")

    latest_demo_start <- EWS_CONFIG$analysis_as_of_date %m-%
      years(EWS_CONFIG$demolition_recent_years) + days(1)
    previous_demo_start <- latest_demo_start %m-%
      years(EWS_CONFIG$demolition_recent_years)

    demolitions <- read_csv(demolitions_file, show_col_types = FALSE) %>%
      mutate(
        issue_date_parsed = ymd(`Issued Date`),
        issued_year = year(issue_date_parsed),
        is_demolition_work_class = str_detect(
          `Work Class`,
          regex("^demolition$", ignore_case = TRUE)
        ),
        is_residential_demo = str_detect(
          `Permit Class Mapped`,
          regex("residential", ignore_case = TRUE)
        ),
        is_total_demolition = str_detect(
          Description,
          regex("total\\s+demo", ignore_case = TRUE)
        )
      ) %>%
      filter(
        !is.na(Latitude),
        !is.na(Longitude),
        !is.na(issue_date_parsed),
        issue_date_parsed <= EWS_CONFIG$analysis_as_of_date,
        is_demolition_work_class,
        is_residential_demo
      ) %>%
      st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326, remove = FALSE) %>%
      st_transform(st_crs(hex_grid))

    demo_hex <- demolitions %>%
      st_join(hex_grid %>% select(hex_id), join = st_within, left = FALSE) %>%
      st_drop_geometry() %>%
      group_by(hex_id) %>%
      summarise(
        demo_count_total = n(),
        demo_count_2020 = sum(issued_year == 2020, na.rm = TRUE),
        demo_count_2021 = sum(issued_year == 2021, na.rm = TRUE),
        demo_count_2022 = sum(issued_year == 2022, na.rm = TRUE),
        demo_count_2023 = sum(issued_year == 2023, na.rm = TRUE),
        demo_count_2024 = sum(issued_year == 2024, na.rm = TRUE),
        demo_count_2025 = sum(issued_year == 2025, na.rm = TRUE),
        demo_count_2026 = sum(issued_year == 2026, na.rm = TRUE),
        demo_residential_total = n(),
        demo_total_demolition_count = sum(is_total_demolition, na.rm = TRUE),
        demo_latest_24mo = sum(issue_date_parsed >= latest_demo_start, na.rm = TRUE),
        demo_previous_24mo = sum(
          issue_date_parsed >= previous_demo_start &
            issue_date_parsed < latest_demo_start,
          na.rm = TRUE
        ),
        demo_total_latest_24mo = sum(
          is_total_demolition & issue_date_parsed >= latest_demo_start,
          na.rm = TRUE
        ),
        demo_total_previous_24mo = sum(
          is_total_demolition &
            issue_date_parsed >= previous_demo_start &
            issue_date_parsed < latest_demo_start,
          na.rm = TRUE
        ),
        demo_first_date = min(issue_date_parsed, na.rm = TRUE),
        demo_last_date = max(issue_date_parsed, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        demo_analysis_as_of = EWS_CONFIG$analysis_as_of_date,
        demo_latest_window_start = latest_demo_start,
        demo_previous_window_start = previous_demo_start
      )

    hex_features <- hex_features %>%
      left_join(demo_hex, by = "hex_id")
  } else {
    print_progress("WARNING: Demolition permit file missing; skipping demolition features.")
  }

  ##############################################################################
  # Eviction and corporate ownership summaries
  ##############################################################################

  eviction_file <- file.path(OUTPUT_DIR, "eviction_filings_by_hex_summary.rds")
  if (file.exists(eviction_file)) {
    print_progress("Joining eviction filing summary features...")

    eviction_features <- load_output(eviction_file, "eviction filing hex summary")

    hex_features <- hex_features %>%
      left_join(eviction_features, by = "hex_id")
  } else {
    print_progress("WARNING: Eviction hex summary missing; skipping eviction features.")
  }

  requests_311_file <- file.path(OUTPUT_DIR, "311_requests_by_hex_summary.rds")
  if (file.exists(requests_311_file)) {
    print_progress("Joining 311 request summary features...")

    requests_311_features <- load_output(requests_311_file, "311 request hex summary") %>%
      st_drop_geometry() %>%
      select(hex_id, starts_with("sr_311_"))

    hex_features <- hex_features %>%
      left_join(requests_311_features, by = "hex_id")
  } else {
    print_progress("WARNING: 311 request hex summary missing; skipping 311 features.")
  }

  corporate_file <- file.path(OUTPUT_DIR, "corporate_ownership_by_hex.rds")
  if (file.exists(corporate_file)) {
    print_progress("Joining corporate ownership summary features...")

    corporate_features <- load_output(corporate_file, "corporate ownership hex summary") %>%
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
        investor_owned_units,
        pct_corporate_owned
      )

    hex_features <- hex_features %>%
      left_join(corporate_features, by = "hex_id")
  } else {
    print_progress("WARNING: Corporate ownership summary missing; skipping ownership features.")
  }

  hex_features
}

################################################################################
# Load processed pipeline data if present, otherwise assemble current streams
################################################################################

processed_file <- file.path(OUTPUT_DIR, "hex_data_processed.rds")

if (USE_PROCESSED_HEX && file.exists(processed_file)) {
  print_progress("Loading existing integrated processed hex data...")
  hex_features <- load_output(processed_file, "processed hexagonal data")
} else {
  if (USE_PROCESSED_HEX && !file.exists(processed_file)) {
    print_progress(
      "WARNING: EWS_USE_PROCESSED_HEX=true but integrated data are missing."
    )
  }
  hex_features <- build_current_stream_features()
}

if (nzchar(CORPORATE_HEX_OVERRIDE_FILE)) {
  if (!file.exists(CORPORATE_HEX_OVERRIDE_FILE)) {
    stop(
      "EWS_CORPORATE_HEX_FILE does not exist: ",
      CORPORATE_HEX_OVERRIDE_FILE,
      call. = FALSE
    )
  }

  print_progress(
    paste0(
      "Replacing parcel-unit and ownership denominators from ",
      basename(CORPORATE_HEX_OVERRIDE_FILE),
      "..."
    )
  )
  corporate_override_columns <- c(
    "residential_parcels",
    "residential_units",
    "residential_improvement_sqft",
    "residential_land_sqft",
    "corporate_owned_parcels",
    "corporate_owned_units",
    "corporate_owned_imprv_sqft",
    "corporate_owner_count",
    "financialized_owner_parcels",
    "geocoded_parcels",
    "pct_corporate_parcels",
    "pct_corporate_units",
    "pct_corporate_improvement_sqft",
    "pct_financialized_owner_parcels",
    "corporate_unit_share_city",
    "corporate_parcel_share_city",
    "corporate_owned_units_per_km2",
    "corporate_owned_parcels_per_km2",
    "residential_units_per_km2",
    "residential_parcels_per_km2",
    "investor_owned_units",
    "pct_corporate_owned"
  )
  corporate_override <- load_output(
    CORPORATE_HEX_OVERRIDE_FILE,
    "override corporate ownership hex summary"
  ) %>%
    st_drop_geometry() %>%
    select(hex_id, any_of(corporate_override_columns))

  missing_override_columns <- setdiff(
    corporate_override_columns,
    names(corporate_override)
  )
  if (length(missing_override_columns) > 0L) {
    stop(
      "Override corporate summary is missing: ",
      paste(missing_override_columns, collapse = ", "),
      call. = FALSE
    )
  }

  hex_features <- hex_features %>%
    select(-any_of(corporate_override_columns)) %>%
    left_join(
      corporate_override,
      by = "hex_id",
      relationship = "one-to-one"
    )
}

acs_demographics_file <- file.path(OUTPUT_DIR, "acs_demographics_by_hex.rds")
if (file.exists(acs_demographics_file)) {
  print_progress("Joining ACS demographic backbone features...")

  acs_demographics <- load_output(
    acs_demographics_file,
    "ACS demographic hex summary"
  ) %>%
    st_drop_geometry()

  demographic_cols <- setdiff(names(acs_demographics), "hex_id")
  demographic_cols_to_join <- setdiff(demographic_cols, names(st_drop_geometry(hex_features)))

  if (length(demographic_cols_to_join) > 0) {
    hex_features <- hex_features %>%
      left_join(
        acs_demographics %>%
          select(hex_id, all_of(demographic_cols_to_join)),
        by = "hex_id"
      )
  } else {
    print_progress("ACS demographic fields already present; skipping duplicate join.")
  }
} else {
  print_progress("WARNING: ACS demographic hex summary missing; run the acs_demographics target.")
}

acs_rent_trends_file <- file.path(OUTPUT_DIR, "acs_rent_trends_by_hex.rds")
if (file.exists(acs_rent_trends_file)) {
  print_progress("Joining historical ACS rent trend features...")

  acs_rent_trends <- load_output(
    acs_rent_trends_file,
    "ACS rent trend hex summary"
  ) %>%
    st_drop_geometry()

  rent_trend_cols <- setdiff(names(acs_rent_trends), names(st_drop_geometry(hex_features)))

  hex_features <- hex_features %>%
    left_join(
      acs_rent_trends %>% select(hex_id, all_of(setdiff(rent_trend_cols, "hex_id"))),
      by = "hex_id"
    )
} else {
  print_progress("WARNING: ACS rent trends missing; run the acs_rent_history target.")
}

appraisal_trends_file <- file.path(
  OUTPUT_DIR,
  "appraisal_adjusted_trends_by_hex.rds"
)
if (file.exists(appraisal_trends_file)) {
  print_progress("Joining county-adjusted appraisal trend features...")

  appraisal_trends <- load_output(
    appraisal_trends_file,
    "county-adjusted appraisal trend hex summary"
  ) %>%
    st_drop_geometry() %>%
    select(-any_of("analysis_as_of_date"))

  appraisal_cols_to_join <- setdiff(
    names(appraisal_trends),
    names(st_drop_geometry(hex_features))
  )
  hex_features <- hex_features %>%
    left_join(
      appraisal_trends %>%
        select(hex_id, all_of(setdiff(appraisal_cols_to_join, "hex_id"))),
      by = "hex_id"
    )
} else {
  print_progress(
    "WARNING: Adjusted appraisal trends missing; run the appraisal_adjusted_features target."
  )
}

ownership_transaction_file <- file.path(
  OUTPUT_DIR,
  "ownership_transaction_features_by_hex.rds"
)
if (file.exists(ownership_transaction_file)) {
  print_progress("Joining ownership-change and transaction features...")

  ownership_transaction_features <- load_output(
    ownership_transaction_file,
    "ownership-change and transaction hex summary"
  ) %>%
    st_drop_geometry() %>%
    mutate(hex_id_join = as.character(hex_id)) %>%
    select(-hex_id)

  ownership_transaction_cols_to_join <- setdiff(
    names(ownership_transaction_features),
    names(st_drop_geometry(hex_features))
  )
  hex_features <- hex_features %>%
    mutate(hex_id_join = as.character(hex_id)) %>%
    left_join(
      ownership_transaction_features %>%
        select(hex_id_join, all_of(setdiff(
          ownership_transaction_cols_to_join,
          "hex_id_join"
        ))),
      by = "hex_id_join"
    ) %>%
    select(-hex_id_join)
} else {
  print_progress(
    "WARNING: Ownership/transaction features missing; run the ownership_transaction_features target."
  )
}

amenity_change_file <- file.path(
  OUTPUT_DIR,
  "amenity_change_features_by_hex.rds"
)
if (file.exists(amenity_change_file)) {
  print_progress("Joining amenity-change features...")

  amenity_change_features <- load_output(
    amenity_change_file,
    "amenity change hex summary"
  ) %>%
    st_drop_geometry() %>%
    mutate(hex_id_join = as.character(hex_id)) %>%
    select(-hex_id)

  amenity_cols_to_join <- setdiff(
    names(amenity_change_features),
    names(st_drop_geometry(hex_features))
  )
  hex_features <- hex_features %>%
    mutate(hex_id_join = as.character(hex_id)) %>%
    left_join(
      amenity_change_features %>%
        select(hex_id_join, all_of(setdiff(
          amenity_cols_to_join,
          "hex_id_join"
        ))),
      by = "hex_id_join"
    ) %>%
    select(-hex_id_join)
} else {
  print_progress(
    "WARNING: Amenity-change features missing; run 02m and 02n before clustering."
  )
}

required_demographic_cols <- c(
  "median_income", "median_rent", "median_home_value", "total_pop",
  "pct_renter", "poverty_rate", "pct_college", "pct_rent_burden_30plus",
  "rent_burden_proxy", "vulnerability_index", "pct_poc", "pct_black",
  "pct_hispanic"
)

for (col in setdiff(required_demographic_cols, names(hex_features))) {
  hex_features[[col]] <- NA_real_
}

required_appraisal_cols <- c(
  "land_value_county_project_percentile_current",
  "land_value_growth_long_county_adjusted_pct",
  "land_value_growth_recent_county_adjusted_pct",
  "land_value_acceleration_county_adjusted_pp"
)
for (col in setdiff(required_appraisal_cols, names(hex_features))) {
  hex_features[[col]] <- NA_real_
}
if (!"appraisal_adjusted_trend_reliable" %in% names(hex_features)) {
  hex_features$appraisal_adjusted_trend_reliable <- FALSE
}

required_ownership_transaction_cols <- c(
  "transaction_pressure_index", "transaction_window_coverage_pct",
  "transaction_recent_per_100_parcels",
  "transaction_previous_per_100_parcels",
  "transaction_recent_per_100_units",
  "transaction_recent_unit_exposure_pct",
  "transaction_rate_change_per_100_parcels",
  "transaction_log_count_change", "ownership_change_index",
  "ownership_history_coverage_pct",
  "ownership_change_recent_per_100_parcels",
  "corporate_acquisition_recent_per_100_parcels",
  "corporate_net_acquisition_recent_per_100_parcels",
  "corporate_acquisition_recent_share",
  "corporate_acquisition_recent_unit_exposure_pct"
)
for (col in setdiff(
  required_ownership_transaction_cols,
  names(hex_features)
)) {
  hex_features[[col]] <- NA_real_
}
if (!"transaction_window_complete" %in% names(hex_features)) {
  hex_features$transaction_window_complete <- FALSE
}

required_amenity_cols <- c(
  "amenity_change_index", "amenity_recent_weighted_openings",
  "amenity_previous_weighted_openings", "amenity_weighted_opening_change",
  "amenity_recent_opening_events", "amenity_previous_opening_events",
  "amenity_cafe_score", "amenity_full_service_restaurant_score",
  "amenity_drinking_place_score", "amenity_geocode_match_pct"
)
for (col in setdiff(required_amenity_cols, names(hex_features))) {
  hex_features[[col]] <- NA_real_
}
if (!"amenity_window_complete" %in% names(hex_features)) {
  hex_features$amenity_window_complete <- FALSE
}

required_acs_rent_cols <- c(
  "acs_rent_current", "acs_rent_current_real",
  "acs_rent_growth_recent_annualized_pct",
  "acs_rent_growth_prior_annualized_pct",
  "acs_rent_growth_long_annualized_pct", "acs_rent_acceleration_pp",
  "acs_rent_relative_moe_current", "acs_rent_relative_moe_max",
  "acs_rent_vintages_available"
)

for (col in setdiff(required_acs_rent_cols, names(hex_features))) {
  hex_features[[col]] <- NA_real_
}

if (!"acs_rent_trend_reliable" %in% names(hex_features)) {
  hex_features$acs_rent_trend_reliable <- FALSE
}

################################################################################
# Derived features
################################################################################

print_progress("Creating derived pressure, ownership, and rate features...")

numeric_zero_cols <- c(
  "demo_count_total", "demo_count_2020", "demo_count_2021", "demo_count_2022",
  "demo_count_2023", "demo_count_2024", "demo_count_2025", "demo_count_2026",
  "demo_residential_total", "demo_total_demolition_count",
  "demo_latest_24mo", "demo_previous_24mo",
  "demo_total_latest_24mo", "demo_total_previous_24mo",
  "eviction_defendant_rows_total", "eviction_cases_total", "eviction_cases_2020",
  "eviction_cases_2021", "eviction_cases_2022", "eviction_cases_2023",
  "eviction_cases_2024", "eviction_cases_2025", "eviction_cases_2026",
  "eviction_cases_latest_12mo", "eviction_cases_previous_12mo",
  "eviction_final_status_cases_total", "eviction_dismissed_cases_total",
  "sr_311_total", "sr_311_code_related_total", "sr_311_housing_condition_total",
  "sr_311_tenant_distress_total", "sr_311_smoke_signal_total",
  "sr_311_nuisance_or_disorder_total", "sr_311_latest_12mo",
  "sr_311_previous_12mo", "sr_311_smoke_signal_latest_12mo",
  "sr_311_smoke_signal_previous_12mo",
  "residential_parcels", "residential_units", "corporate_owned_parcels",
  "corporate_owned_units", "corporate_owner_count", "financialized_owner_parcels",
  "investor_owned_units"
)

for (col in intersect(numeric_zero_cols, names(hex_features))) {
  hex_features[[col]] <- replace_na(hex_features[[col]], 0)
}

if (!"costar_present" %in% names(hex_features)) hex_features$costar_present <- 0
if (!"rent_units_current" %in% names(hex_features)) hex_features$rent_units_current <- 0

required_311_cols <- c(
  "sr_311_total", "sr_311_code_related_total", "sr_311_housing_condition_total",
  "sr_311_tenant_distress_total", "sr_311_smoke_signal_total",
  "sr_311_nuisance_or_disorder_total", "sr_311_latest_12mo",
  "sr_311_previous_12mo", "sr_311_smoke_signal_latest_12mo",
  "sr_311_smoke_signal_previous_12mo",
  "sr_311_latest_12mo_change_pct",
  "sr_311_smoke_signal_latest_12mo_change_pct"
)

for (col in setdiff(required_311_cols, names(hex_features))) {
  hex_features[[col]] <- NA_real_
}

hex_features <- hex_features %>%
  mutate(
    costar_present = replace_na(costar_present, 0),
    costar_units_current = replace_na(rent_units_current, 0),
    demo_recent = demo_latest_24mo,
    demo_previous = demo_previous_24mo,
    demo_density = if_else(area_km2 > 0, demo_count_total / area_km2, NA_real_),
    demo_recent_density = if_else(area_km2 > 0, demo_recent / area_km2, NA_real_),
    demo_total_recent_density = if_else(
      area_km2 > 0,
      demo_total_latest_24mo / area_km2,
      NA_real_
    ),
    demo_trend = log1p(demo_recent) - log1p(demo_previous),
    demo_total_trend = log1p(demo_total_latest_24mo) -
      log1p(demo_total_previous_24mo),
    demo_trend_positive = pmax(demo_trend, 0),
    has_recent_demos = if_else(demo_recent > 0, 1, 0),

    # Very small unit denominators create unstable rates, so rates are only
    # calculated where the parcel-derived unit count is large enough to support
    # interpretation.
    eviction_rate_units_denominator = if_else(
      residential_units >= EWS_CONFIG$minimum_residential_units_for_rates,
      residential_units,
      NA_real_
    ),
    eviction_cases_per_100_units = if_else(!is.na(eviction_rate_units_denominator),
                                           100 * eviction_cases_total / eviction_rate_units_denominator,
                                           NA_real_),
    eviction_latest_12mo_per_100_units = if_else(!is.na(eviction_rate_units_denominator),
                                                 100 * eviction_cases_latest_12mo / eviction_rate_units_denominator,
                                                 NA_real_),
    eviction_cases_total_density = if_else(area_km2 > 0, eviction_cases_total / area_km2, NA_real_),
    eviction_cases_latest_12mo_density = if_else(area_km2 > 0, eviction_cases_latest_12mo / area_km2, NA_real_),
    eviction_recent_share = if_else(eviction_cases_total > 0,
                                    eviction_cases_latest_12mo / eviction_cases_total,
                                    NA_real_),
    sr_311_per_100_units = if_else(
      !is.na(eviction_rate_units_denominator),
      100 * sr_311_total / eviction_rate_units_denominator,
      NA_real_
    ),
    sr_311_smoke_signal_per_100_units = if_else(
      !is.na(eviction_rate_units_denominator),
      100 * sr_311_smoke_signal_total / eviction_rate_units_denominator,
      NA_real_
    ),
    sr_311_latest_12mo_per_100_units = if_else(
      !is.na(eviction_rate_units_denominator),
      100 * sr_311_latest_12mo / eviction_rate_units_denominator,
      NA_real_
    ),
    sr_311_smoke_signal_latest_12mo_per_100_units = if_else(
      !is.na(eviction_rate_units_denominator),
      100 * sr_311_smoke_signal_latest_12mo / eviction_rate_units_denominator,
      NA_real_
    ),
    sr_311_latest_12mo_density = if_else(area_km2 > 0, sr_311_latest_12mo / area_km2, NA_real_),
    sr_311_smoke_signal_latest_12mo_density = if_else(
      area_km2 > 0,
      sr_311_smoke_signal_latest_12mo / area_km2,
      NA_real_
    ),
    sr_311_smoke_signal_share = if_else(
      sr_311_total > 0,
      sr_311_smoke_signal_total / sr_311_total,
      NA_real_
    ),

    pct_corporate_units = coalesce(pct_corporate_units, pct_corporate_owned, 0),
    pct_corporate_parcels = replace_na(pct_corporate_parcels, 0),
    pct_financialized_owner_parcels = replace_na(pct_financialized_owner_parcels, 0),
    corporate_owned_units_per_km2 = replace_na(corporate_owned_units_per_km2, 0),
    residential_units_per_km2 = replace_na(residential_units_per_km2, 0),

    rent_level_ratio = NA_real_,
    acs_rent_growth_recent_for_clustering = if_else(
      acs_rent_trend_reliable,
      acs_rent_growth_recent_annualized_pct,
      NA_real_
    ),
    acs_rent_acceleration_for_clustering = if_else(
      acs_rent_trend_reliable,
      acs_rent_acceleration_pp,
      NA_real_
    ),
    rent_pressure_index = rowMeans(
      cbind(
        normalize_robust_to_100(rent_current),
        normalize_robust_to_100(rent_change_recent),
        normalize_robust_to_100(rent_change_total),
        normalize_robust_to_100(rent_acceleration)
      ),
      na.rm = TRUE
    ),
    rent_pressure_citywide_index = rowMeans(
      cbind(
        normalize_robust_to_100(acs_rent_current_real),
        normalize_robust_to_100(acs_rent_growth_recent_for_clustering),
        normalize_robust_to_100(acs_rent_acceleration_for_clustering)
      ),
      na.rm = TRUE
    ),
    land_value_pressure_index = if_else(
      appraisal_adjusted_trend_reliable,
      rowMeans(
        cbind(
          land_value_county_project_percentile_current,
          normalize_robust_to_100(
            land_value_growth_long_county_adjusted_pct
          ),
          normalize_robust_to_100(
            land_value_growth_recent_county_adjusted_pct
          ),
          normalize_robust_to_100(
            land_value_acceleration_county_adjusted_pp
          )
        ),
        na.rm = FALSE
      ),
      NA_real_
    ),
    costar_rent_pressure_index = rowMeans(
      cbind(
        normalize_robust_to_100(rent_current),
        normalize_robust_to_100(costar_rent_growth_recent_annualized_pct),
        normalize_robust_to_100(costar_rent_acceleration_pp),
        normalize_robust_to_100(rent_volatility)
      ),
      na.rm = TRUE
    ),
    eviction_pressure_index = rowMeans(
      cbind(
        normalize_robust_to_100(eviction_latest_12mo_per_100_units),
        normalize_robust_to_100(eviction_cases_latest_12mo_change_pct),
        normalize_robust_to_100(eviction_recent_share)
      ),
      na.rm = TRUE
    ),
    ownership_pressure_index = rowMeans(
      cbind(
        normalize_robust_to_100(pct_corporate_units),
        normalize_robust_to_100(corporate_owned_units_per_km2),
        normalize_robust_to_100(pct_financialized_owner_parcels)
      ),
      na.rm = TRUE
    ),
    demolition_pressure_index = rowMeans(
      cbind(
        normalize_robust_to_100(demo_recent_density),
        normalize_robust_to_100(demo_trend_positive),
        normalize_robust_to_100(demo_total_recent_density)
      ),
      na.rm = TRUE
    ),
    demographic_vulnerability_index = rowMeans(
      cbind(
        normalize_robust_to_100(-median_income),
        normalize_robust_to_100(pct_renter),
        normalize_robust_to_100(poverty_rate),
        normalize_robust_to_100(pct_rent_burden_30plus),
        normalize_robust_to_100(-pct_college)
      ),
      na.rm = TRUE
    ),
    demographic_vulnerability_equity_index = rowMeans(
      cbind(
        normalize_robust_to_100(-median_income),
        normalize_robust_to_100(pct_renter),
        normalize_robust_to_100(poverty_rate),
        normalize_robust_to_100(pct_rent_burden_30plus),
        normalize_robust_to_100(-pct_college),
        normalize_robust_to_100(pct_poc)
      ),
      na.rm = TRUE
    ),
    sr_311_pressure_index = rowMeans(
      cbind(
        normalize_robust_to_100(sr_311_latest_12mo_per_100_units),
        normalize_robust_to_100(sr_311_smoke_signal_latest_12mo_per_100_units),
        normalize_robust_to_100(sr_311_latest_12mo_density),
        normalize_robust_to_100(sr_311_smoke_signal_latest_12mo_density),
        normalize_robust_to_100(sr_311_smoke_signal_latest_12mo_change_pct)
      ),
      na.rm = TRUE
    )
  ) %>%
  mutate(
    eviction_cases_per_100_units = cap_upper_quantile(eviction_cases_per_100_units, 0.99),
    eviction_latest_12mo_per_100_units = cap_upper_quantile(eviction_latest_12mo_per_100_units, 0.99),
    sr_311_per_100_units = cap_upper_quantile(sr_311_per_100_units, 0.99),
    sr_311_smoke_signal_per_100_units = cap_upper_quantile(sr_311_smoke_signal_per_100_units, 0.99),
    sr_311_latest_12mo_per_100_units = cap_upper_quantile(sr_311_latest_12mo_per_100_units, 0.99),
    sr_311_smoke_signal_latest_12mo_per_100_units = cap_upper_quantile(sr_311_smoke_signal_latest_12mo_per_100_units, 0.99)
  ) %>%
  mutate(
    across(
      c(rent_pressure_index, rent_pressure_citywide_index,
        costar_rent_pressure_index, land_value_pressure_index,
        eviction_pressure_index,
        ownership_pressure_index, ownership_change_index,
        transaction_pressure_index, amenity_change_index,
        demolition_pressure_index,
        demographic_vulnerability_index,
        demographic_vulnerability_equity_index, sr_311_pressure_index),
      ~if_else(is.nan(.x), NA_real_, .x)
    )
  )

################################################################################
# Spatial lag features
################################################################################

safe_spatial_lag <- function(data, var_name, k = 6) {
  if (!var_name %in% names(data)) return(rep(NA_real_, nrow(data)))

  values <- data[[var_name]]
  if (all(is.na(values))) return(rep(NA_real_, nrow(data)))

  if (!requireNamespace("spdep", quietly = TRUE)) {
    print_progress("Package 'spdep' not installed; skipping spatial lag features for this run.")
    return(rep(NA_real_, nrow(data)))
  }

  tryCatch({
    centroids <- suppressWarnings(st_point_on_surface(data))
    coords <- st_coordinates(centroids)
    knn <- spdep::knearneigh(coords, k = k)
    nb <- spdep::knn2nb(knn)
    vapply(
      nb,
      function(neighbor_ids) {
        neighbor_values <- values[neighbor_ids]
        if (all(is.na(neighbor_values))) NA_real_ else mean(neighbor_values, na.rm = TRUE)
      },
      numeric(1)
    )
  }, error = function(e) {
    warning(paste("Error calculating spatial lag for", var_name, ":", e$message))
    rep(NA_real_, nrow(data))
  })
}

print_progress("Calculating spatial lag features...")

hex_features <- hex_features %>%
  mutate(
    rent_change_total_lag = safe_spatial_lag(., "rent_change_total"),
    rent_pressure_index_lag = safe_spatial_lag(., "rent_pressure_index"),
    rent_pressure_citywide_index_lag = safe_spatial_lag(
      .,
      "rent_pressure_citywide_index"
    ),
    demo_density_lag = safe_spatial_lag(., "demo_density"),
    eviction_latest_12mo_per_100_units_lag = safe_spatial_lag(., "eviction_latest_12mo_per_100_units"),
    ownership_pressure_index_lag = safe_spatial_lag(., "ownership_pressure_index")
  )

################################################################################
# Interactions and data sufficiency
################################################################################

hex_features <- hex_features %>%
  mutate(
    neighborhood_rent_pressure = rowMeans(
      cbind(rent_pressure_citywide_index, rent_pressure_citywide_index_lag),
      na.rm = TRUE
    ),
    rent_eviction_interaction = rent_pressure_citywide_index * eviction_pressure_index,
    rent_ownership_interaction = rent_pressure_citywide_index * ownership_pressure_index,
    demo_ownership_interaction = demolition_pressure_index * ownership_pressure_index
  ) %>%
  mutate(
    neighborhood_rent_pressure = if_else(is.nan(neighborhood_rent_pressure), NA_real_, neighborhood_rent_pressure)
  )

feature_cols <- hex_features %>%
  st_drop_geometry() %>%
  select(
    any_of(c(
      "rent_current", "rent_psf_current", "vacancy_pct_current",
      "rent_change_recent", "rent_change_total", "rent_acceleration",
      "rent_volatility", "rent_pressure_index", "costar_rent_pressure_index",
      "rent_pressure_citywide_index", "costar_present",
      "acs_rent_current", "acs_rent_current_real",
      "acs_rent_growth_recent_annualized_pct",
      "acs_rent_growth_prior_annualized_pct",
      "acs_rent_growth_long_annualized_pct", "acs_rent_acceleration_pp",
      "acs_rent_growth_recent_for_clustering",
      "acs_rent_acceleration_for_clustering",
      "acs_rent_relative_moe_current", "acs_rent_relative_moe_max",
      "acs_rent_vintages_available", "acs_rent_trend_reliable",
      "land_value_pressure_index",
      "land_value_county_project_percentile_current",
      "land_value_growth_long_county_adjusted_pct",
      "land_value_growth_recent_county_adjusted_pct",
      "land_value_acceleration_county_adjusted_pp",
      "appraisal_adjusted_trend_reliable",
      "transaction_pressure_index", "transaction_window_coverage_pct",
      "transaction_window_complete",
      "transaction_recent_per_100_parcels",
      "transaction_previous_per_100_parcels",
      "transaction_recent_per_100_units",
      "transaction_recent_unit_exposure_pct",
      "transaction_rate_change_per_100_parcels",
      "transaction_log_count_change", "ownership_change_index",
      "ownership_history_coverage_pct",
      "ownership_change_recent_per_100_parcels",
      "corporate_acquisition_recent_per_100_parcels",
      "corporate_net_acquisition_recent_per_100_parcels",
      "corporate_acquisition_recent_share",
      "corporate_acquisition_recent_unit_exposure_pct",
      "amenity_change_index", "amenity_window_complete",
      "amenity_geocode_match_pct",
      "amenity_recent_weighted_openings",
      "amenity_previous_weighted_openings",
      "amenity_weighted_opening_change",
      "amenity_recent_opening_events", "amenity_previous_opening_events",
      "amenity_cafe_score", "amenity_full_service_restaurant_score",
      "amenity_drinking_place_score",
      "median_income", "median_rent", "median_home_value", "total_pop",
      "pct_renter", "poverty_rate", "pct_college", "pct_rent_burden_30plus",
      "rent_burden_proxy", "vulnerability_index", "demographic_vulnerability_index",
      "demographic_vulnerability_equity_index",
      "pct_poc", "pct_black", "pct_hispanic",
      "demo_density", "demo_recent", "demo_recent_density",
      "demo_total_recent_density", "demo_trend", "demo_total_trend",
      "demo_trend_positive",
      "demolition_pressure_index",
      "eviction_cases_per_100_units", "eviction_latest_12mo_per_100_units",
      "eviction_cases_total_density", "eviction_cases_latest_12mo_density",
      "eviction_cases_latest_12mo_change_pct", "eviction_pressure_index",
      "sr_311_total", "sr_311_smoke_signal_total", "sr_311_latest_12mo",
      "sr_311_smoke_signal_latest_12mo", "sr_311_latest_12mo_change_pct",
      "sr_311_smoke_signal_latest_12mo_change_pct", "sr_311_per_100_units",
      "sr_311_smoke_signal_per_100_units", "sr_311_latest_12mo_per_100_units",
      "sr_311_smoke_signal_latest_12mo_per_100_units",
      "sr_311_latest_12mo_density", "sr_311_smoke_signal_latest_12mo_density",
      "sr_311_smoke_signal_share", "sr_311_pressure_index",
      "pct_corporate_units", "pct_corporate_parcels",
      "pct_financialized_owner_parcels", "corporate_owned_units_per_km2",
      "residential_units_per_km2", "ownership_pressure_index",
      "rent_change_total_lag", "rent_pressure_index_lag",
      "rent_pressure_citywide_index_lag", "demo_density_lag",
      "eviction_latest_12mo_per_100_units_lag", "ownership_pressure_index_lag",
      "neighborhood_rent_pressure", "rent_eviction_interaction",
      "rent_ownership_interaction", "demo_ownership_interaction"
    ))
  ) %>%
  names()

hex_features <- hex_features %>%
  mutate(
    missing_feature_count = rowSums(is.na(st_drop_geometry(select(., all_of(feature_cols))))),
    missing_feature_pct = 100 * missing_feature_count / length(feature_cols),
    sufficient_data = missing_feature_pct < 50,
    primary_cluster_eligible = residential_units >=
      EWS_CONFIG$minimum_residential_units_for_rates,
    analysis_as_of_date = EWS_CONFIG$analysis_as_of_date
  )

print_header("FEATURE SUMMARY")
cat(paste0("Feature columns: ", length(feature_cols), "\n"))
cat(paste0("Hexagons with sufficient data: ", sum(hex_features$sufficient_data, na.rm = TRUE), " / ", nrow(hex_features), "\n\n"))

missing_summary <- hex_features %>%
  st_drop_geometry() %>%
  select(all_of(feature_cols)) %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "missing_count") %>%
  mutate(missing_pct = 100 * missing_count / nrow(hex_features)) %>%
  arrange(desc(missing_pct))

cat("Top feature missingness:\n")
print(head(missing_summary, 15))

output_file <- HEX_FEATURE_OUTPUT_FILE
save_output(hex_features, output_file, "engineered current-stream features")

feature_list <- tibble(
  feature_name = feature_cols,
  category = case_when(
    str_detect(feature_name, "rent") | str_detect(feature_name, "vacancy") ~ "Rent",
    str_detect(feature_name, "land_value|appraisal") ~ "Appraisal",
    str_detect(feature_name, "demo") ~ "Demolitions",
    str_detect(feature_name, "eviction") ~ "Evictions",
    str_detect(feature_name, "sr_311") ~ "311",
    str_detect(feature_name, "transaction") ~ "Transactions",
    str_detect(feature_name, "amenity") ~ "Amenities",
    str_detect(
      feature_name,
      "ownership_change|corporate_acquisition|corporate_net_acquisition"
    ) ~ "Ownership Change",
    str_detect(feature_name, "corporate|financialized|ownership|residential") ~ "Ownership",
    str_detect(feature_name, "_lag") ~ "Spatial Lag",
    str_detect(feature_name, "interaction") ~ "Interactions",
    TRUE ~ "Other"
  )
)

write_csv(feature_list, FEATURE_LIST_OUTPUT_FILE)
print_progress(
  paste0("Saved feature list to: ", FEATURE_LIST_OUTPUT_FILE)
)

print_header("STEP 03 COMPLETE")
cat(paste0("Features saved to: ", output_file, "\n"))
