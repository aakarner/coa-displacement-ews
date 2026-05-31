################################################################################
# 02d - Calibrate Residential Parcel Unit Counts
################################################################################
#
# Builds a calibrated residential unit count for the parcel universe. The script
# keeps credible parcel unit counts, uses CoStar-matched large multifamily
# properties to estimate a local sqft-per-unit factor, and writes diagnostics for
# review before corporate ownership is aggregated to the hex grid.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(purrr)
  library(readr)
  library(sf)
  library(stringr)
  library(tidyr)
})

source(here::here("R/utils.R"))

print_header("02d - CALIBRATE PARCEL UNIT COUNTS")

OUTPUT_DIR <- here::here("output")
DATA_DIR <- here::here("data")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

residential_parcel_files <- c(
  Travis = file.path(DATA_DIR, "residential_parcels_for_hex.csv"),
  Williamson = file.path(DATA_DIR, "williamson_residential_parcels_for_hex.csv"),
  Hays = file.path(DATA_DIR, "hays_residential_parcels_for_hex.csv")
)

costar_file <- file.path(DATA_DIR, "CoStarHistoric-clean.csv")
costar_geocode_file <- file.path(DATA_DIR, "geocoded_buildings.csv")

required_files <- c(residential_parcel_files, costar_file, costar_geocode_file)
missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop(
    "Missing required parcel unit calibration file(s): ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

normalize_address_key <- function(x) {
  x %>%
    str_to_upper() %>%
    str_replace_all("\\bAUSTIN\\b\\s*\\bTX\\b\\s*\\d{5}(-\\d{4})?\\s*$", "") %>%
    str_replace_all("\\bTEXAS\\b\\s*\\d{5}(-\\d{4})?\\s*$", "") %>%
    str_replace_all("\\bTX\\b\\s*\\d{5}(-\\d{4})?\\s*$", "") %>%
    str_replace_all("\\bROAD\\b", "RD") %>%
    str_replace_all("\\bSTREET\\b", "ST") %>%
    str_replace_all("\\bAVENUE\\b", "AVE") %>%
    str_replace_all("\\bBOULEVARD\\b", "BLVD") %>%
    str_replace_all("\\bDRIVE\\b", "DR") %>%
    str_replace_all("\\bLANE\\b", "LN") %>%
    str_replace_all("\\bCOURT\\b", "CT") %>%
    str_replace_all("\\bPLACE\\b", "PL") %>%
    str_replace_all("\\bPARKWAY\\b", "PKWY") %>%
    str_replace_all("\\bHIGHWAY\\b", "HWY") %>%
    str_replace_all("\\bINTERSTATE\\b", "IH") %>%
    str_replace_all("\\bNORTH\\b", "N") %>%
    str_replace_all("\\bSOUTH\\b", "S") %>%
    str_replace_all("\\bEAST\\b", "E") %>%
    str_replace_all("\\bWEST\\b", "W") %>%
    str_replace_all("[^A-Z0-9]+", " ") %>%
    str_squish() %>%
    na_if("")
}

zip5 <- function(x) {
  str_extract(as.character(x), "\\d{5}")
}

parse_logical <- function(x) {
  case_when(
    str_to_upper(as.character(x)) %in% c("TRUE", "T", "1", "YES", "Y") ~ TRUE,
    str_to_upper(as.character(x)) %in% c("FALSE", "F", "0", "NO", "N") ~ FALSE,
    TRUE ~ NA
  )
}

safe_ratio <- function(num, den) {
  if_else(!is.na(num) & !is.na(den) & den > 0, num / den, NA_real_)
}

parcel_schemas <- map(
  residential_parcel_files,
  ~names(read_csv(.x, n_max = 0, col_types = cols(.default = col_character()), show_col_types = FALSE))
)

if (!all(map_lgl(parcel_schemas, identical, parcel_schemas[[1]]))) {
  stop("Residential parcel universe files do not have identical schemas.", call. = FALSE)
}

parcels_raw <- imap_dfr(
  residential_parcel_files,
  ~read_csv(.x, col_types = cols(.default = col_character()), show_col_types = FALSE) %>%
    mutate(source_county = .y)
)

duplicate_parcel_ids <- parcels_raw %>%
  count(parcel_id, name = "row_count") %>%
  filter(row_count > 1)

if (nrow(duplicate_parcel_ids) > 0) {
  stop(
    "Duplicate parcel_id values found after binding county parcel files. ",
    "Example duplicate(s): ",
    paste(head(duplicate_parcel_ids$parcel_id, 10), collapse = ", "),
    call. = FALSE
  )
}

parcels <- parcels_raw %>%
  mutate(
    parcel_id = as.character(parcel_id),
    lat = as.numeric(lat),
    lon = as.numeric(lon),
    parcel_count = replace_na(as.numeric(parcel_count), 0),
    units_raw = as.numeric(property_units),
    improvement_sqft = replace_na(as.numeric(improvement_sqft), 0),
    land_sqft = replace_na(as.numeric(land_sqft), 0),
    corporate_parcel_count = replace_na(as.numeric(corporate_parcel_count), 0),
    corporate_units_raw = as.numeric(corporate_units),
    corporate_improvement_sqft = replace_na(as.numeric(corporate_improvement_sqft), 0),
    is_residential = replace_na(parse_logical(is_residential), FALSE),
    is_owner_occupied = replace_na(parse_logical(is_owner_occupied), FALSE),
    is_corporate_owned = replace_na(parse_logical(is_corporate_owned), FALSE),
    has_financialized_owner = replace_na(parse_logical(has_financialized_owner), FALSE),
    parcel_zip5 = coalesce(zip5(situs_zip), zip5(situs_address)),
    parcel_address_key = normalize_address_key(situs_address),
    sqft_per_raw_unit = safe_ratio(improvement_sqft, units_raw),
    zoning_key = str_to_upper(coalesce(propertyChar_zoning, "")),
    imprv_state_key = str_to_upper(coalesce(propertyProf_imprvStateCd, "")),
    has_mf_zoning = str_detect(zoning_key, "\\bMF|MULTI|APART"),
    has_commercial_mixed_zoning = str_detect(zoning_key, "\\b(CS|GR|LI|LO|LR|GO|CBD)(\\b|[^A-Z])"),
    is_multifamily_like = has_mf_zoning |
      str_detect(zoning_key, "CONDO|PUD|MU") |
      imprv_state_key %in% c("B1", "B2", "B3", "B4"),
    is_single_family_like = !is_multifamily_like &
      (str_detect(zoning_key, "\\bSF|SINGLE") | imprv_state_key %in% c("A1", "A2", "A3", "A4")),
    likely_derived_single_family_units = is_single_family_like &
      !is.na(units_raw) &
      units_raw > 1 &
      units_raw <= 20 &
      improvement_sqft > 0 &
      abs(units_raw - round(units_raw)) > 0.001 &
      abs(units_raw - improvement_sqft / 900) <= pmax(0.25, units_raw * 0.02),
    likely_derived_large_mf_units = is_multifamily_like &
      !is.na(units_raw) &
      units_raw >= 20 &
      improvement_sqft >= 18000 &
      abs(units_raw - improvement_sqft / 900) <= pmax(1, units_raw * 0.02)
  )

missing_coords <- parcels %>%
  filter(is.na(lat) | is.na(lon))

if (nrow(missing_coords) > 0) {
  stop(
    "Residential parcel universe contains ",
    nrow(missing_coords),
    " row(s) with missing or non-numeric lat/lon coordinates.",
    call. = FALSE
  )
}

costar <- read_csv(costar_file, show_col_types = FALSE) %>%
  mutate(
    Period = yq(str_replace(Period, "\\s+QTD$", "")),
    costar_zip5 = zip5(`Zip Code`),
    costar_address_key = normalize_address_key(`Building Address`),
    costar_units = as.numeric(inventory_Units),
    costar_avg_sf = parse_number(as.character(inventory_AvgSF), na = c("", "NA", "-", "—"))
  )

latest_period <- max(costar$Period, na.rm = TRUE)

costar_geocodes <- read_csv(costar_geocode_file, show_col_types = FALSE) %>%
  mutate(
    costar_zip5 = zip5(`Zip Code`),
    costar_address_key = normalize_address_key(`Building Address`)
  ) %>%
  distinct(
    `Building Address`,
    `Building Name`,
    `Zip Code`,
    costar_address_key,
    costar_zip5,
    .keep_all = TRUE
  )

costar_latest <- costar %>%
  filter(Period == latest_period) %>%
  distinct(
    `Building Address`,
    `Building Name`,
    `Zip Code`,
    .keep_all = TRUE
  ) %>%
  left_join(
    costar_geocodes %>%
      select(`Building Address`, `Building Name`, `Zip Code`, building_id, latitude, longitude),
    by = c("Building Address", "Building Name", "Zip Code")
  ) %>%
  mutate(
    costar_id = row_number(),
    building_id = coalesce(as.character(building_id), as.character(costar_id))
  ) %>%
  select(
    costar_id,
    building_id,
    building_address = `Building Address`,
    building_name = `Building Name`,
    building_zip = `Zip Code`,
    costar_zip5,
    costar_address_key,
    latitude,
    longitude,
    costar_units,
    costar_avg_sf
  )

parcel_match_fields <- parcels %>%
  select(
    parcel_id,
    source_county,
    situs_address,
    parcel_zip5,
    parcel_address_key,
    lat,
    lon,
    improvement_sqft,
    units_raw,
    is_multifamily_like
  )

exact_address_zip <- costar_latest %>%
  filter(!is.na(costar_address_key), !is.na(costar_zip5)) %>%
  inner_join(
    parcel_match_fields %>% filter(!is.na(parcel_address_key), !is.na(parcel_zip5)),
    by = c("costar_address_key" = "parcel_address_key", "costar_zip5" = "parcel_zip5"),
    relationship = "many-to-many"
  ) %>%
  mutate(match_type = "exact_address_zip", match_distance_m = NA_real_)

matched_costar_ids <- unique(exact_address_zip$costar_id)

exact_address <- costar_latest %>%
  filter(!costar_id %in% matched_costar_ids, !is.na(costar_address_key)) %>%
  inner_join(
    parcel_match_fields %>% filter(!is.na(parcel_address_key)),
    by = c("costar_address_key" = "parcel_address_key"),
    relationship = "many-to-many"
  ) %>%
  mutate(match_type = "exact_address", match_distance_m = NA_real_)

matched_costar_ids <- unique(c(matched_costar_ids, exact_address$costar_id))

costar_unmatched <- costar_latest %>%
  filter(
    !costar_id %in% matched_costar_ids,
    !is.na(latitude),
    !is.na(longitude)
  )

spatial_matches <- tibble()

if (nrow(costar_unmatched) > 0) {
  costar_pts <- costar_unmatched %>%
    st_as_sf(coords = c("longitude", "latitude"), crs = 4326, remove = FALSE) %>%
    st_transform(3857)

  parcel_pts <- parcel_match_fields %>%
    st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
    st_transform(3857)

  nearest_idx <- st_nearest_feature(costar_pts, parcel_pts)
  nearest_parcels <- parcel_match_fields[nearest_idx, ] %>%
    rename(
      nearest_parcel_id = parcel_id,
      nearest_source_county = source_county,
      nearest_situs_address = situs_address,
      nearest_parcel_zip5 = parcel_zip5,
      nearest_parcel_address_key = parcel_address_key,
      nearest_lat = lat,
      nearest_lon = lon,
      nearest_improvement_sqft = improvement_sqft,
      nearest_units_raw = units_raw,
      nearest_is_multifamily_like = is_multifamily_like
    )

  spatial_matches <- bind_cols(
    costar_unmatched,
    nearest_parcels
  ) %>%
    mutate(
      match_distance_m = as.numeric(st_distance(costar_pts, parcel_pts[nearest_idx, ], by_element = TRUE)),
      match_type = case_when(
        match_distance_m <= 50 ~ "nearest_spatial_strong",
        match_distance_m <= 100 ~ "nearest_spatial_review",
        TRUE ~ "unmatched"
      )
    ) %>%
    filter(match_type != "unmatched") %>%
    transmute(
      costar_id,
      building_id,
      building_address,
      building_name,
      building_zip,
      costar_zip5,
      costar_address_key,
      latitude,
      longitude,
      costar_units,
      costar_avg_sf,
      parcel_id = nearest_parcel_id,
      source_county = nearest_source_county,
      situs_address = nearest_situs_address,
      parcel_zip5 = nearest_parcel_zip5,
      lat = nearest_lat,
      lon = nearest_lon,
      improvement_sqft = nearest_improvement_sqft,
      units_raw = nearest_units_raw,
      is_multifamily_like = nearest_is_multifamily_like,
      match_type,
      match_distance_m
    )
}

costar_matches <- bind_rows(
  exact_address_zip,
  exact_address,
  spatial_matches
) %>%
  mutate(
    match_confidence = case_when(
      match_type == "exact_address_zip" ~ "high",
      match_type == "exact_address" ~ "medium",
      match_type == "nearest_spatial_strong" ~ "high",
      match_type == "nearest_spatial_review" ~ "review",
      TRUE ~ "low"
    ),
    address_similarity = if_else(match_type %in% c("exact_address_zip", "exact_address"), 1, NA_real_),
    use_for_calibration = match_type %in% c("exact_address_zip", "nearest_spatial_strong") |
      (match_type == "exact_address" & is.na(match_distance_m))
  ) %>%
  select(
    costar_id,
    building_id,
    building_name,
    building_address,
    building_zip,
    costar_units,
    costar_avg_sf,
    parcel_id,
    source_county,
    situs_address,
    units_raw,
    improvement_sqft,
    match_type,
    match_distance_m,
    address_similarity,
    match_confidence,
    use_for_calibration
  )

calibration_groups <- costar_matches %>%
  filter(use_for_calibration) %>%
  group_by(costar_id, building_id, building_name, building_address, building_zip, costar_units) %>%
  summarise(
    matched_parcel_count = n_distinct(parcel_id),
    matched_improvement_sqft = sum(improvement_sqft, na.rm = TRUE),
    match_types = str_c(sort(unique(match_type)), collapse = " | "),
    .groups = "drop"
  ) %>%
  mutate(
    sqft_per_unit = safe_ratio(matched_improvement_sqft, costar_units),
    plausible_for_calibration = !is.na(sqft_per_unit) &
      costar_units > 0 &
      matched_improvement_sqft > 0 &
      sqft_per_unit >= 250 &
      sqft_per_unit <= 2500
  )

calibration_sample <- calibration_groups %>%
  filter(plausible_for_calibration)

if (nrow(calibration_sample) < 25) {
  stop(
    "Fewer than 25 plausible CoStar-parcel calibration matches were found. ",
    "Review output/costar_parcel_unit_calibration_matches.csv before applying calibration.",
    call. = FALSE
  )
}

costar_median_sqft_per_unit <- median(calibration_sample$sqft_per_unit, na.rm = TRUE)
costar_p25_sqft_per_unit <- quantile(calibration_sample$sqft_per_unit, 0.25, na.rm = TRUE, names = FALSE)
costar_p75_sqft_per_unit <- quantile(calibration_sample$sqft_per_unit, 0.75, na.rm = TRUE, names = FALSE)
costar_p90_sqft_per_unit <- quantile(calibration_sample$sqft_per_unit, 0.90, na.rm = TRUE, names = FALSE)
costar_p95_sqft_per_unit <- quantile(calibration_sample$sqft_per_unit, 0.95, na.rm = TRUE, names = FALSE)
costar_weak_evidence_sqft_per_unit <- max(costar_p75_sqft_per_unit, 2000)
costar_unmatched_strong_mf_sqft_per_unit <- max(costar_p90_sqft_per_unit, 2200)
costar_unmatched_weak_mf_sqft_per_unit <- max(costar_p95_sqft_per_unit, 2500)
unmatched_strong_mf_unit_cap <- 750
unmatched_weak_mf_unit_cap <- 500

costar_direct_calibration_parcel_ids <- costar_matches %>%
  inner_join(
    calibration_groups %>%
      filter(plausible_for_calibration) %>%
      select(costar_id),
    by = "costar_id"
  ) %>%
  filter(use_for_calibration) %>%
  distinct(parcel_id) %>%
  pull(parcel_id) %>%
  as.character()

costar_direct_units_by_parcel <- costar_matches %>%
  inner_join(
    calibration_groups %>%
      filter(plausible_for_calibration) %>%
      select(costar_id),
    by = "costar_id"
  ) %>%
  filter(
    use_for_calibration,
    match_type %in% c("exact_address_zip", "nearest_spatial_strong"),
    !is.na(costar_units),
    costar_units > 0
  ) %>%
  group_by(costar_id) %>%
  mutate(
    positive_improvement_sqft_total = sum(improvement_sqft[improvement_sqft > 0], na.rm = TRUE),
    direct_unit_weight = case_when(
      positive_improvement_sqft_total > 0 ~ improvement_sqft / positive_improvement_sqft_total,
      TRUE ~ 1 / n()
    ),
    direct_costar_units_allocated = costar_units * direct_unit_weight
  ) %>%
  ungroup() %>%
  group_by(parcel_id) %>%
  summarise(
    direct_costar_units = sum(direct_costar_units_allocated, na.rm = TRUE),
    direct_costar_property_count = n_distinct(costar_id),
    direct_costar_match_types = str_c(sort(unique(match_type)), collapse = " | "),
    .groups = "drop"
  ) %>%
  mutate(
    direct_costar_units = round(direct_costar_units)
  )

parcels_calibrated <- parcels %>%
  left_join(costar_direct_units_by_parcel, by = "parcel_id") %>%
  mutate(
    direct_costar_units = if_else(!is.na(direct_costar_units) & direct_costar_units > 0, direct_costar_units, NA_real_),
    direct_costar_property_count = replace_na(direct_costar_property_count, 0L),
    has_direct_costar_calibration_match = parcel_id %in% costar_direct_calibration_parcel_ids,
    needs_multifamily_estimate = is_multifamily_like &
      improvement_sqft > 0 &
      (is.na(units_raw) | units_raw <= 0 | likely_derived_large_mf_units),
    units_calibrated_pre_conservative = case_when(
      likely_derived_single_family_units ~ 1,
      !is.na(units_raw) &
        units_raw > 0 &
        !likely_derived_large_mf_units &
        units_raw <= 5000 ~ units_raw,
      is_single_family_like & (is.na(units_raw) | units_raw <= 0) ~ 1,
      needs_multifamily_estimate &
        (has_direct_costar_calibration_match | has_mf_zoning) ~
        pmax(1, round(improvement_sqft / costar_median_sqft_per_unit)),
      needs_multifamily_estimate &
        !has_direct_costar_calibration_match &
        !has_mf_zoning &
        has_commercial_mixed_zoning ~ NA_real_,
      needs_multifamily_estimate ~
        pmax(1, round(improvement_sqft / costar_weak_evidence_sqft_per_unit)),
      !is.na(units_raw) & units_raw > 0 ~ units_raw,
      TRUE ~ NA_real_
    ),
    unit_estimation_method_pre_conservative = case_when(
      likely_derived_single_family_units ~ "single_family_fractional_fallback_1",
      !is.na(units_raw) &
        units_raw > 0 &
        !likely_derived_large_mf_units &
        units_raw <= 5000 ~ "parcel_units_retained",
      is_single_family_like & (is.na(units_raw) | units_raw <= 0) ~ "single_family_default_1",
      needs_multifamily_estimate &
        (has_direct_costar_calibration_match | has_mf_zoning) ~
        "costar_sqft_per_unit_estimate_strong_mf",
      needs_multifamily_estimate &
        !has_direct_costar_calibration_match &
        !has_mf_zoning &
        has_commercial_mixed_zoning ~ "commercial_mixed_use_mf_estimate_excluded",
      needs_multifamily_estimate ~ "costar_sqft_per_unit_estimate_weak_mf",
      !is.na(units_raw) & units_raw > 0 ~ "parcel_units_retained_high_value",
      TRUE ~ "unknown_missing_units"
    ),
    units_calibrated = case_when(
      !is.na(direct_costar_units) & is_multifamily_like ~ direct_costar_units,
      likely_derived_single_family_units ~ 1,
      !is.na(units_raw) &
        units_raw > 0 &
        !likely_derived_large_mf_units &
        units_raw <= 5000 ~ units_raw,
      is_single_family_like & (is.na(units_raw) | units_raw <= 0) ~ 1,
      needs_multifamily_estimate &
        (has_direct_costar_calibration_match | has_mf_zoning) ~
        pmin(
          unmatched_strong_mf_unit_cap,
          pmax(1, round(improvement_sqft / costar_unmatched_strong_mf_sqft_per_unit))
        ),
      needs_multifamily_estimate &
        !has_direct_costar_calibration_match &
        !has_mf_zoning &
        has_commercial_mixed_zoning ~ NA_real_,
      needs_multifamily_estimate ~
        pmin(
          unmatched_weak_mf_unit_cap,
          pmax(1, round(improvement_sqft / costar_unmatched_weak_mf_sqft_per_unit))
        ),
      !is.na(units_raw) & units_raw > 0 ~ units_raw,
      TRUE ~ NA_real_
    ),
    unit_estimation_method = case_when(
      !is.na(direct_costar_units) & is_multifamily_like ~ "direct_costar_units",
      likely_derived_single_family_units ~ "single_family_fractional_fallback_1",
      !is.na(units_raw) &
        units_raw > 0 &
        !likely_derived_large_mf_units &
        units_raw <= 5000 ~ "parcel_units_retained",
      is_single_family_like & (is.na(units_raw) | units_raw <= 0) ~ "single_family_default_1",
      needs_multifamily_estimate &
        (has_direct_costar_calibration_match | has_mf_zoning) ~
        "conservative_costar_sqft_per_unit_estimate_strong_mf",
      needs_multifamily_estimate &
        !has_direct_costar_calibration_match &
        !has_mf_zoning &
        has_commercial_mixed_zoning ~ "commercial_mixed_use_mf_estimate_excluded",
      needs_multifamily_estimate ~ "conservative_costar_sqft_per_unit_estimate_weak_mf",
      !is.na(units_raw) & units_raw > 0 ~ "parcel_units_retained_high_value",
      TRUE ~ "unknown_missing_units"
    ),
    unit_estimation_confidence = case_when(
      unit_estimation_method == "direct_costar_units" ~ "high",
      unit_estimation_method == "parcel_units_retained" ~ "medium",
      unit_estimation_method == "single_family_default_1" ~ "medium",
      unit_estimation_method == "single_family_fractional_fallback_1" ~ "medium",
      unit_estimation_method == "conservative_costar_sqft_per_unit_estimate_strong_mf" ~ "low",
      unit_estimation_method == "conservative_costar_sqft_per_unit_estimate_weak_mf" ~ "low",
      unit_estimation_method == "commercial_mixed_use_mf_estimate_excluded" ~ "low",
      unit_estimation_method == "parcel_units_retained_high_value" ~ "low",
      TRUE ~ "low"
    ),
    unit_estimation_notes = case_when(
      unit_estimation_method == "direct_costar_units" ~
        paste0(
          "Assigned direct CoStar units from high-confidence match; matched CoStar properties: ",
          direct_costar_property_count
        ),
      unit_estimation_method == "conservative_costar_sqft_per_unit_estimate_strong_mf" ~
        paste0(
          "Estimated using conservative unmatched MF sqft/unit: ",
          round(costar_unmatched_strong_mf_sqft_per_unit, 1),
          "; cap: ",
          unmatched_strong_mf_unit_cap
        ),
      unit_estimation_method == "conservative_costar_sqft_per_unit_estimate_weak_mf" ~
        paste0(
          "Estimated using conservative weak-evidence sqft/unit: ",
          round(costar_unmatched_weak_mf_sqft_per_unit, 1),
          "; cap: ",
          unmatched_weak_mf_unit_cap
        ),
      unit_estimation_method == "commercial_mixed_use_mf_estimate_excluded" ~
        "Large multifamily fallback excluded because parcel has commercial/mixed-use zoning and no direct CoStar/MF-zoning evidence",
      likely_derived_large_mf_units ~ "Raw units looked like prior improvement_sqft / 900 fallback",
      unit_estimation_method == "single_family_fractional_fallback_1" ~
        "Raw fractional units looked like prior improvement_sqft / 900 fallback; assigned 1",
      unit_estimation_method == "single_family_default_1" ~ "Assigned 1 unit based on single-family-like parcel attributes",
      unit_estimation_method == "unknown_missing_units" ~ "No reliable unit count or calibration rule available",
      TRUE ~ "Raw parcel unit count retained"
    ),
    units_calibrated_conservative = if_else(units_calibrated < 0, NA_real_, units_calibrated),
    unit_estimation_method_conservative = unit_estimation_method,
    unit_estimation_confidence_conservative = unit_estimation_confidence,
    unit_estimation_notes_conservative = unit_estimation_notes,
    conservative_unit_delta = units_calibrated_conservative - units_calibrated_pre_conservative,
    units_calibrated = if_else(units_calibrated_pre_conservative < 0, NA_real_, units_calibrated_pre_conservative),
    unit_estimation_method = unit_estimation_method_pre_conservative,
    unit_estimation_confidence = case_when(
      unit_estimation_method == "parcel_units_retained" ~ "medium",
      unit_estimation_method == "single_family_default_1" ~ "medium",
      unit_estimation_method == "single_family_fractional_fallback_1" ~ "medium",
      unit_estimation_method == "costar_sqft_per_unit_estimate_strong_mf" ~ "medium",
      unit_estimation_method == "costar_sqft_per_unit_estimate_weak_mf" ~ "low",
      unit_estimation_method == "commercial_mixed_use_mf_estimate_excluded" ~ "low",
      unit_estimation_method == "parcel_units_retained_high_value" ~ "low",
      TRUE ~ "low"
    ),
    unit_estimation_notes = case_when(
      unit_estimation_method == "costar_sqft_per_unit_estimate_strong_mf" ~
        paste0("Estimated using CoStar median sqft/unit: ", round(costar_median_sqft_per_unit, 1)),
      unit_estimation_method == "costar_sqft_per_unit_estimate_weak_mf" ~
        paste0("Estimated using conservative weak-evidence sqft/unit: ", round(costar_weak_evidence_sqft_per_unit, 1)),
      unit_estimation_method == "commercial_mixed_use_mf_estimate_excluded" ~
        "Large multifamily fallback excluded because parcel has commercial/mixed-use zoning and no direct CoStar/MF-zoning evidence",
      likely_derived_large_mf_units ~ "Raw units looked like prior improvement_sqft / 900 fallback",
      unit_estimation_method == "single_family_fractional_fallback_1" ~
        "Raw fractional units looked like prior improvement_sqft / 900 fallback; assigned 1",
      unit_estimation_method == "single_family_default_1" ~ "Assigned 1 unit based on single-family-like parcel attributes",
      unit_estimation_method == "unknown_missing_units" ~ "No reliable unit count or calibration rule available",
      TRUE ~ "Raw parcel unit count retained"
    ),
    corporate_units = if_else(is_corporate_owned, replace_na(units_calibrated, 0), 0),
    corporate_units_conservative = if_else(is_corporate_owned, replace_na(units_calibrated_conservative, 0), 0),
    property_units = units_calibrated,
    property_units_conservative = units_calibrated_conservative
  ) %>%
  select(
    -zoning_key,
    -imprv_state_key,
    -sqft_per_raw_unit
  )

if (nrow(parcels_calibrated) != nrow(parcels_raw)) {
  stop("Calibrated parcel row count does not match the original parcel row count.", call. = FALSE)
}

if (any(duplicated(parcels_calibrated$parcel_id))) {
  stop("Calibrated parcel output contains duplicate parcel_id values.", call. = FALSE)
}

if (any(parcels_calibrated$units_calibrated < 0, na.rm = TRUE)) {
  stop("Calibrated parcel output contains negative unit counts.", call. = FALSE)
}

costar_match_output <- costar_matches %>%
  left_join(
    calibration_groups %>%
      select(costar_id, matched_parcel_count, matched_improvement_sqft, sqft_per_unit, plausible_for_calibration),
    by = "costar_id"
  )

write_csv(
  costar_match_output,
  file.path(OUTPUT_DIR, "costar_parcel_unit_calibration_matches.csv")
)

save_output(
  parcels_calibrated,
  file.path(OUTPUT_DIR, "residential_parcels_unit_calibrated.rds"),
  "calibrated residential parcel unit counts"
)

write_csv(
  parcels_calibrated,
  file.path(OUTPUT_DIR, "residential_parcels_unit_calibrated.csv")
)

project_grid_diagnostics <- tibble()
hex_grid_file <- file.path(OUTPUT_DIR, "hex_grid.rds")

if (file.exists(hex_grid_file)) {
  hex_grid <- load_output(hex_grid_file, "hexagonal grid")

  parcels_in_grid <- parcels_calibrated %>%
    st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
    st_transform(st_crs(hex_grid)) %>%
    st_join(hex_grid %>% select(hex_id), join = st_within, left = FALSE) %>%
    st_drop_geometry()

  project_grid_diagnostics <- tibble(
    metric_group = "project_grid_unit_totals",
    metric = c(
      "raw_units_in_project_grid",
      "calibrated_units_in_project_grid",
      "conservative_units_in_project_grid",
      "parcels_in_project_grid"
    ),
    value = c(
      sum(parcels_in_grid$units_raw, na.rm = TRUE),
      sum(parcels_in_grid$units_calibrated, na.rm = TRUE),
      sum(parcels_in_grid$units_calibrated_conservative, na.rm = TRUE),
      nrow(parcels_in_grid)
    )
  )
}

diagnostics <- bind_rows(
  tibble(
    metric_group = "unit_totals",
    metric = "raw_units_total_all_parcels",
    value = sum(parcels$units_raw, na.rm = TRUE)
  ),
  tibble(
    metric_group = "unit_totals",
    metric = c(
      "calibrated_units_total_all_parcels",
      "conservative_units_total_all_parcels",
      "conservative_minus_calibrated_units_total_all_parcels"
    ),
    value = c(
      sum(parcels_calibrated$units_calibrated, na.rm = TRUE),
      sum(parcels_calibrated$units_calibrated_conservative, na.rm = TRUE),
      sum(parcels_calibrated$conservative_unit_delta, na.rm = TRUE)
    )
  ),
  parcels_calibrated %>%
    group_by(source_county) %>%
    summarise(value = sum(units_calibrated, na.rm = TRUE), .groups = "drop") %>%
    transmute(metric_group = "calibrated_units_by_county", metric = source_county, value),
  parcels_calibrated %>%
    count(unit_estimation_method, wt = units_calibrated, name = "value") %>%
    transmute(metric_group = "calibrated_units_by_method", metric = unit_estimation_method, value),
  costar_matches %>%
    count(match_type, name = "value") %>%
    transmute(metric_group = "costar_match_rows_by_type", metric = match_type, value),
  tibble(
    metric_group = "costar_calibration",
    metric = c(
      "latest_period",
      "latest_costar_properties",
      "calibration_property_count",
      "median_sqft_per_unit",
      "p25_sqft_per_unit",
      "p75_sqft_per_unit",
      "p90_sqft_per_unit",
      "p95_sqft_per_unit",
      "weak_evidence_sqft_per_unit",
      "unmatched_strong_mf_sqft_per_unit",
      "unmatched_weak_mf_sqft_per_unit",
      "unmatched_strong_mf_unit_cap",
      "unmatched_weak_mf_unit_cap"
    ),
    value = c(
      as.numeric(format(latest_period, "%Y%m%d")),
      nrow(costar_latest),
      nrow(calibration_sample),
      costar_median_sqft_per_unit,
      costar_p25_sqft_per_unit,
      costar_p75_sqft_per_unit,
      costar_p90_sqft_per_unit,
      costar_p95_sqft_per_unit,
      costar_weak_evidence_sqft_per_unit,
      costar_unmatched_strong_mf_sqft_per_unit,
      costar_unmatched_weak_mf_sqft_per_unit,
      unmatched_strong_mf_unit_cap,
      unmatched_weak_mf_unit_cap
    )
  ),
  tibble(
    metric_group = "flags",
    metric = c(
      "unmatched_costar_properties",
      "review_spatial_match_rows",
      "implausible_calibration_groups",
      "parcels_with_1000_plus_calibrated_units",
      "single_family_fractional_fallback_parcels",
      "commercial_mixed_use_mf_estimate_excluded_parcels",
      "weak_mf_estimate_parcels",
      "direct_costar_unit_parcels",
      "unknown_missing_unit_parcels"
    ),
    value = c(
      nrow(costar_latest) - n_distinct(costar_matches$costar_id),
      sum(costar_matches$match_type == "nearest_spatial_review", na.rm = TRUE),
      sum(!calibration_groups$plausible_for_calibration, na.rm = TRUE),
      sum(parcels_calibrated$units_calibrated >= 1000, na.rm = TRUE),
      sum(parcels_calibrated$unit_estimation_method == "single_family_fractional_fallback_1", na.rm = TRUE),
      sum(parcels_calibrated$unit_estimation_method == "commercial_mixed_use_mf_estimate_excluded", na.rm = TRUE),
      sum(parcels_calibrated$unit_estimation_method_conservative == "conservative_costar_sqft_per_unit_estimate_weak_mf", na.rm = TRUE),
      sum(parcels_calibrated$unit_estimation_method_conservative == "direct_costar_units", na.rm = TRUE),
      sum(parcels_calibrated$unit_estimation_method == "unknown_missing_units", na.rm = TRUE)
    )
  ),
  project_grid_diagnostics
)

write_csv(
  diagnostics,
  file.path(OUTPUT_DIR, "unit_calibration_diagnostics.csv")
)

large_changes <- parcels_calibrated %>%
  mutate(
    unit_change = units_calibrated - units_raw,
    pct_unit_change = if_else(units_raw > 0, unit_change / units_raw * 100, NA_real_)
  ) %>%
  filter(abs(unit_change) >= 50 | abs(pct_unit_change) >= 50) %>%
  arrange(desc(abs(unit_change))) %>%
  select(
    parcel_id,
    source_county,
    situs_address,
    units_raw,
    units_calibrated,
    unit_change,
    pct_unit_change,
    improvement_sqft,
    unit_estimation_method,
    unit_estimation_confidence,
    unit_estimation_notes
  )

write_csv(
  large_changes,
  file.path(OUTPUT_DIR, "unit_calibration_large_changes.csv")
)

print_progress(paste0("Latest CoStar period: ", latest_period))
print_progress(paste0("CoStar properties in latest period: ", nrow(costar_latest)))
print_progress(paste0("CoStar match rows: ", nrow(costar_matches)))
print_progress(paste0("Plausible calibration properties: ", nrow(calibration_sample)))
print_progress(paste0("Median matched sqft/unit: ", round(costar_median_sqft_per_unit, 1)))
print_progress(
  paste0(
    "Raw units: ",
    scales::comma(round(sum(parcels$units_raw, na.rm = TRUE), 0)),
    "; calibrated units: ",
    scales::comma(round(sum(parcels_calibrated$units_calibrated, na.rm = TRUE), 0))
  )
)

print_header("02d COMPLETE")
