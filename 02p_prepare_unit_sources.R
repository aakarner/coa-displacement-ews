################################################################################
# 02p - Prepare Residential Unit Count Sources
################################################################################
#
# Creates a source-record table and a separate parcel-link table for residential
# unit evidence. Project totals are stored once per source record and are never
# copied into parcel counts. This is a shadow-stage input to 02q; it does not
# alter the production calibration in 02d or the ACS validation in 02e.
#
# Optional environment variables:
#   TCAD_PROPERTY_PROFILE   Existing landlord-mapper property_profile.csv
#   AUSTIN_URO_UNIT_FILE    Existing City URO CSV
#   AUSTIN_AHI_UNIT_FILE    Existing City affordable-housing inventory CSV
#   REFRESH_UNIT_SOURCES    Set to "true" to rebuild cached extracts
################################################################################

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(readr)
  library(sf)
  library(stringr)
  library(tidyr)
})

source(here::here("R", "utils.R"))
source(here::here("R", "unit_count_helpers.R"))

print_header("02p - PREPARE RESIDENTIAL UNIT COUNT SOURCES")

OUTPUT_DIR <- here::here("output")
UNIT_SOURCE_DIR <- here::here("data", "raw_parcels", "unit_sources")
PARCEL_FILE <- file.path(OUTPUT_DIR, "residential_parcels_unit_calibrated.rds")
COSTAR_MATCH_FILE <- file.path(
  OUTPUT_DIR,
  "costar_parcel_unit_calibration_matches.csv"
)
REFRESH <- str_to_lower(Sys.getenv("REFRESH_UNIT_SOURCES", unset = "false")) %in%
  c("true", "t", "1", "yes", "y")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(UNIT_SOURCE_DIR, showWarnings = FALSE, recursive = TRUE)

required_files <- c(PARCEL_FILE, COSTAR_MATCH_FILE)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Run 02d_calibrate_parcel_units.R before 02p. Missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

project_parent <- dirname(here::here())
tcad_profile_default <- file.path(
  project_parent,
  "landlord-mapper",
  "output",
  "property_profile.csv"
)
tcad_profile_file <- Sys.getenv(
  "TCAD_PROPERTY_PROFILE",
  unset = tcad_profile_default
)
tcad_compact_file <- file.path(
  UNIT_SOURCE_DIR,
  "tcad_property_profile_unit_fields.csv"
)
wcad_property_file <- here::here(
  "data",
  "raw_parcels",
  "williamson",
  "wcad_property_certified.csv"
)
wcad_parcel_file <- here::here(
  "data",
  "raw_parcels",
  "williamson",
  "wcad_parcels_flat.csv"
)
wcad_compact_file <- file.path(
  UNIT_SOURCE_DIR,
  "wcad_property_unit_fields.csv"
)

missing_county_files <- c(
  wcad_property_file,
  wcad_parcel_file
)[!file.exists(c(wcad_property_file, wcad_parcel_file))]
if (length(missing_county_files) > 0L) {
  stop(
    "Missing Williamson appraisal source file(s): ",
    paste(missing_county_files, collapse = ", "),
    call. = FALSE
  )
}

uro_cache_file <- file.path(UNIT_SOURCE_DIR, "austin_uro_units.csv")
ahi_cache_file <- file.path(
  UNIT_SOURCE_DIR,
  "austin_affordable_housing_inventory.csv"
)
uro_input_file <- Sys.getenv("AUSTIN_URO_UNIT_FILE", unset = "")
ahi_input_file <- Sys.getenv("AUSTIN_AHI_UNIT_FILE", unset = "")

cache_public_source <- function(cache_file, explicit_file, temporary_file, url, label) {
  if (file.exists(cache_file) && !REFRESH) {
    return(cache_file)
  }

  candidates <- c(explicit_file, temporary_file)
  candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
  if (length(candidates) > 0L) {
    copied <- file.copy(candidates[[1]], cache_file, overwrite = TRUE)
    if (!copied) {
      stop("Could not cache ", label, " from ", candidates[[1]], call. = FALSE)
    }
    return(cache_file)
  }

  print_progress(paste0("Downloading ", label, "..."))
  download_error <- tryCatch(
    {
      utils::download.file(url, cache_file, mode = "wb", quiet = TRUE)
      NULL
    },
    error = function(e) e
  )
  if (!is.null(download_error) || !file.exists(cache_file)) {
    stop(
      "Could not obtain ",
      label,
      ". Set its environment-variable path or place a cached CSV at ",
      cache_file,
      ".",
      call. = FALSE
    )
  }
  cache_file
}

uro_file <- cache_public_source(
  uro_cache_file,
  uro_input_file,
  "/tmp/austin_uro_units.csv",
  paste0(
    "https://data.austintexas.gov/resource/uic2-id33.csv",
    "?$limit=50000"
  ),
  "Austin URO multifamily-unit inventory"
)

ahi_file <- cache_public_source(
  ahi_cache_file,
  ahi_input_file,
  "/tmp/austin_affordable_housing_inventory.csv",
  paste0(
    "https://data.austintexas.gov/resource/ifzc-3xz8.csv",
    "?$limit=50000"
  ),
  "Austin Affordable Housing Inventory"
)

parcels <- readRDS(PARCEL_FILE) %>%
  mutate(
    parcel_id = as.character(parcel_id),
    source_county = as.character(source_county),
    parcel_address_key = normalize_unit_address(
      strip_unit_address_locality(situs_address, situs_city)
    ),
    parcel_zip5 = coalesce(
      unit_zip5(parcel_zip5),
      unit_zip5(situs_zip),
      unit_zip5(situs_address)
    ),
    improvement_sqft = unit_numeric(improvement_sqft),
    land_sqft = unit_numeric(land_sqft),
    lat = unit_numeric(lat),
    lon = unit_numeric(lon),
    propertyProf_imprvActualYearBuilt = unit_numeric(
      propertyProf_imprvActualYearBuilt
    )
  )

if (anyDuplicated(parcels$parcel_id)) {
  stop("Parcel universe has duplicate parcel_id values.", call. = FALSE)
}

tcad_fields <- c(
  "propertyProf_pID",
  "propertyProf_imprvUnits",
  "propertyProf_imprvMainArea",
  "propertyProf_imprvTotalArea",
  "propertyProf_imprvStories",
  "propertyProf_imprvType",
  "propertyProf_imprvClass",
  "propertyProf_imprvQuality",
  "propertyProf_imprvCondition",
  "propertyProf_imprvActualYearBuilt",
  "propertyProf_imprvEffYearBuilt",
  "propertyProf_imprvStateCd"
)

if (!file.exists(tcad_compact_file) || REFRESH) {
  if (!file.exists(tcad_profile_file)) {
    stop(
      "TCAD property profile not found. Set TCAD_PROPERTY_PROFILE. Expected: ",
      tcad_profile_file,
      call. = FALSE
    )
  }
  print_progress("Extracting compact TCAD unit and improvement fields...")
  tcad_header <- names(
    data.table::fread(tcad_profile_file, nrows = 0L, showProgress = FALSE)
  )
  missing_tcad_fields <- setdiff(tcad_fields, tcad_header)
  if (length(missing_tcad_fields) > 0L) {
    stop(
      "TCAD profile is missing fields: ",
      paste(missing_tcad_fields, collapse = ", "),
      call. = FALSE
    )
  }
  tcad_compact <- data.table::fread(
    tcad_profile_file,
    select = tcad_fields,
    colClasses = "character",
    showProgress = FALSE
  )
  readr::write_csv(as_tibble(tcad_compact), tcad_compact_file)
  rm(tcad_compact)
  invisible(gc())
}

tcad <- read_csv(
  tcad_compact_file,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE
) %>%
  transmute(
    parcel_id = as.character(propertyProf_pID),
    tcad_imprv_units = unit_numeric(propertyProf_imprvUnits),
    tcad_main_area = unit_numeric(propertyProf_imprvMainArea),
    tcad_total_area = unit_numeric(propertyProf_imprvTotalArea),
    tcad_stories = unit_numeric(propertyProf_imprvStories),
    tcad_imprv_type = na_if(propertyProf_imprvType, ""),
    tcad_imprv_class = na_if(propertyProf_imprvClass, ""),
    tcad_imprv_quality = na_if(propertyProf_imprvQuality, ""),
    tcad_imprv_condition = na_if(propertyProf_imprvCondition, ""),
    tcad_actual_year_built = unit_numeric(propertyProf_imprvActualYearBuilt),
    tcad_effective_year_built = unit_numeric(propertyProf_imprvEffYearBuilt),
    tcad_imprv_state_code = na_if(propertyProf_imprvStateCd, "")
  ) %>%
  distinct(parcel_id, .keep_all = TRUE)

wcad_property_fields <- c(
  "PropertyID",
  "QuickRefID",
  "PropertyTypeDesc",
  "TotalSqFtLivingArea",
  "LegalDescription",
  "PropertyComment",
  "DBA",
  "SubUnit",
  "PropertyLegalType",
  "UnitTypeKey",
  "UnitNumber",
  "CondoBuilding",
  "CondoPercentage",
  "CondoUnit"
)
wcad_parcel_fields <- c(
  "propertyid",
  "parcelid",
  "unit",
  "building",
  "usedscrp"
)

if (!file.exists(wcad_compact_file) || REFRESH) {
  print_progress("Extracting compact Williamson unit and legal fields...")
  wcad_property_header <- names(
    data.table::fread(wcad_property_file, nrows = 0L, showProgress = FALSE)
  )
  wcad_parcel_header <- names(
    data.table::fread(wcad_parcel_file, nrows = 0L, showProgress = FALSE)
  )
  missing_wcad_fields <- c(
    setdiff(wcad_property_fields, wcad_property_header),
    setdiff(wcad_parcel_fields, wcad_parcel_header)
  )
  if (length(missing_wcad_fields) > 0L) {
    stop(
      "Williamson sources are missing fields: ",
      paste(unique(missing_wcad_fields), collapse = ", "),
      call. = FALSE
    )
  }

  wcad_property_extract <- data.table::fread(
    wcad_property_file,
    select = wcad_property_fields,
    colClasses = "character",
    showProgress = FALSE
  ) %>%
    as_tibble() %>%
    distinct(PropertyID, .keep_all = TRUE)

  wcad_parcel_extract <- data.table::fread(
    wcad_parcel_file,
    select = wcad_parcel_fields,
    colClasses = "character",
    showProgress = FALSE
  ) %>%
    as_tibble() %>%
    group_by(propertyid) %>%
    summarise(
      wcad_parcel_id = as.character(
        first_non_missing_unit_value(parcelid)
      ),
      wcad_parcel_unit = as.character(
        first_non_missing_unit_value(unit)
      ),
      wcad_parcel_building = as.character(
        first_non_missing_unit_value(building)
      ),
      wcad_use_description = as.character(
        first_non_missing_unit_value(usedscrp)
      ),
      .groups = "drop"
    )

  wcad_compact <- wcad_property_extract %>%
    left_join(
      wcad_parcel_extract,
      by = c("PropertyID" = "propertyid"),
      relationship = "one-to-one"
    ) %>%
    mutate(
      wcad_match_id = coalesce(wcad_parcel_id, QuickRefID),
      parcel_id = paste0("WILLIAMSON:", wcad_match_id)
    ) %>%
    filter(parcel_id %in% parcels$parcel_id) %>%
    select(parcel_id, everything(), -wcad_match_id)

  if (n_distinct(wcad_compact$parcel_id) != sum(
    parcels$source_county == "Williamson"
  )) {
    stop(
      "Williamson compact extract does not cover the full EWS parcel subset.",
      call. = FALSE
    )
  }

  write_csv(wcad_compact, wcad_compact_file)
  rm(
    wcad_property_extract,
    wcad_parcel_extract,
    wcad_compact
  )
  invisible(gc())
}

wcad <- read_csv(
  wcad_compact_file,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE
) %>%
  transmute(
    parcel_id = as.character(parcel_id),
    wcad_property_id = as.character(PropertyID),
    wcad_property_type = na_if(PropertyTypeDesc, ""),
    wcad_living_area = unit_numeric(TotalSqFtLivingArea),
    wcad_legal_description = na_if(LegalDescription, ""),
    wcad_property_comment = na_if(PropertyComment, ""),
    wcad_dba = na_if(DBA, ""),
    wcad_subunit = na_if(SubUnit, ""),
    wcad_legal_type = na_if(PropertyLegalType, ""),
    wcad_unit_type = na_if(UnitTypeKey, ""),
    wcad_unit_number = na_if(UnitNumber, ""),
    wcad_condo_building = na_if(CondoBuilding, ""),
    wcad_condo_percentage = unit_numeric(CondoPercentage),
    wcad_condo_unit = na_if(CondoUnit, ""),
    wcad_parcel_unit = na_if(wcad_parcel_unit, ""),
    wcad_parcel_building = na_if(wcad_parcel_building, ""),
    wcad_use_description = na_if(wcad_use_description, "")
  ) %>%
  distinct(parcel_id, .keep_all = TRUE)

parcels_enhanced <- parcels %>%
  left_join(tcad, by = "parcel_id") %>%
  left_join(wcad, by = "parcel_id") %>%
  mutate(
    appraisal_state_code = coalesce(
      tcad_imprv_state_code,
      as.character(propertyProf_imprvStateCd)
    ),
    model_improvement_sqft = coalesce(
      na_if(tcad_total_area, 0),
      na_if(improvement_sqft, 0)
    ),
    model_main_area = coalesce(
      na_if(tcad_main_area, 0),
      na_if(improvement_sqft, 0)
    ),
    model_year_built = coalesce(
      tcad_actual_year_built,
      propertyProf_imprvActualYearBuilt
    ),
    tcad_sqft_per_reported_unit = if_else(
      tcad_imprv_units > 0,
      model_improvement_sqft / tcad_imprv_units,
      NA_real_
    ),
    tcad_explicit_units_plausible = source_county == "Travis" &
      appraisal_state_code == "B1" &
      tcad_imprv_units > 1 &
      tcad_imprv_units <= 5000 &
      tcad_sqft_per_reported_unit >= 250 &
      tcad_sqft_per_reported_unit <= 2500,
    deterministic_account_units = case_when(
      source_county == "Travis" &
        appraisal_state_code %in% c("A1", "A2", "A3", "A4") ~ 1,
      source_county == "Travis" & appraisal_state_code == "B2" ~ 2,
      source_county == "Travis" & appraisal_state_code == "B3" ~ 3,
      source_county == "Travis" & appraisal_state_code == "B4" ~ 4,
      TRUE ~ NA_real_
    ),
    hays_improvement_code = if_else(
      source_county == "Hays",
      as.character(propertyProf_imprvStateCd),
      NA_character_
    ),
    hays_land_code = if_else(
      source_county == "Hays",
      as.character(propertyProf_landStateCd),
      NA_character_
    ),
    hays_deterministic_units = case_when(
      source_county == "Hays" &
        (
          str_detect(coalesce(hays_improvement_code, ""), "^A") |
            str_detect(coalesce(hays_land_code, ""), "^A")
        ) ~ 1,
      source_county == "Hays" &
        (
          hays_improvement_code == "B2" |
            hays_land_code == "B2"
        ) ~ 2,
      source_county == "Hays" &
        (
          hays_improvement_code == "B3" |
            hays_land_code == "B3"
        ) ~ 3,
      source_county == "Hays" &
        (
          hays_improvement_code == "B4" |
            hays_land_code == "B4"
        ) ~ 4,
      TRUE ~ NA_real_
    ),
    hays_multifamily_model_candidate = source_county == "Hays" &
      (
        hays_improvement_code == "B1" |
          hays_land_code == "B1"
      ) &
      model_improvement_sqft > 0,
    wcad_evidence_text = str_to_upper(
      str_c(
        coalesce(wcad_legal_description, ""),
        coalesce(wcad_property_comment, ""),
        coalesce(wcad_dba, ""),
        coalesce(wcad_use_description, ""),
        sep = " | "
      )
    ),
    wcad_primary_evidence_text = str_to_upper(
      str_c(
        coalesce(wcad_legal_description, ""),
        coalesce(wcad_dba, ""),
        coalesce(wcad_use_description, ""),
        sep = " | "
      )
    ),
    wcad_apartment_primary_signal = source_county == "Williamson" &
      str_detect(
        wcad_primary_evidence_text,
        paste0(
          "APARTMENT|(^|[^A-Z])APTS?([^A-Z]|$)|",
          "MULTI[- ]?FAMILY"
        )
      ),
    wcad_apartment_comment_signal = source_county == "Williamson" &
      str_detect(
        str_to_upper(coalesce(wcad_property_comment, "")),
        paste0(
          "APARTMENT|(^|[^A-Z])APTS?([^A-Z]|$)|",
          "MULTI[- ]?FAMILY"
        )
      ),
    wcad_apartment_signal = wcad_apartment_primary_signal |
      (
        wcad_property_type %in% c("C3", "C5") &
          wcad_apartment_comment_signal
      ),
    wcad_small_multifamily_units = case_when(
      source_county == "Williamson" &
        str_detect(wcad_evidence_text, "FOURPLEX|4-PLEX") ~ 4,
      source_county == "Williamson" &
        str_detect(wcad_evidence_text, "TRIPLEX") ~ 3,
      source_county == "Williamson" &
        str_detect(wcad_evidence_text, "DUPLEX") ~ 2,
      TRUE ~ NA_real_
    ),
    wcad_residential_type = source_county == "Williamson" &
      wcad_property_type %in% c(
        "Residential",
        "Manufactured Home",
        "LTRR-Land Transitional Residential"
      ),
    wcad_condo_signal = source_county == "Williamson" &
      (
        wcad_legal_type == "C" |
          !is.na(wcad_condo_unit) |
          !is.na(wcad_condo_building) |
          str_detect(
            wcad_evidence_text,
            "CONDOMINIUM|(^|[^A-Z])CONDO([^A-Z]|$)"
          )
      ),
    wcad_non_unit_reference_account = source_county == "Williamson" &
      wcad_condo_signal &
      str_detect(wcad_evidence_text, "REFERENCE ONLY"),
    wcad_explicit_residential_unit_account = wcad_residential_type &
      wcad_condo_signal &
      !wcad_non_unit_reference_account &
      model_improvement_sqft > 0,
    wcad_apartment_model_candidate = source_county == "Williamson" &
      wcad_property_type %in% c("C3", "C5") &
      wcad_apartment_signal &
      !wcad_non_unit_reference_account &
      model_improvement_sqft > 0,
    wcad_nonresidential_condo_account = source_county == "Williamson" &
      !wcad_residential_type &
      wcad_condo_signal &
      !wcad_non_unit_reference_account &
      !wcad_apartment_model_candidate,
    wcad_non_unit_amenity_parcel = source_county == "Williamson" &
      wcad_residential_type &
      (is.na(model_improvement_sqft) | model_improvement_sqft <= 0) &
      (
        str_detect(wcad_evidence_text, fixed("(PARK")) |
          str_detect(wcad_evidence_text, "AMENIT")
      ),
    wcad_non_unit_transitional_land = source_county == "Williamson" &
      wcad_property_type == "LTRC-Land Transitional Commercial" &
      (is.na(model_improvement_sqft) | model_improvement_sqft <= 0),
    wcad_nonresidential_account = source_county == "Williamson" &
      wcad_property_type == "C6" &
      wcad_use_description == "C6" &
      !wcad_residential_type &
      !wcad_condo_signal &
      !wcad_apartment_model_candidate &
      !wcad_non_unit_transitional_land,
    wcad_single_unit_rule_units = if_else(
      wcad_residential_type &
        model_improvement_sqft > 0 &
        !wcad_explicit_residential_unit_account &
        is.na(wcad_small_multifamily_units) &
        !wcad_apartment_signal,
      1,
      NA_real_
    ),
    county_model_candidate_signal = hays_multifamily_model_candidate |
      wcad_apartment_model_candidate,
    county_unit_exclusion_reason = case_when(
      wcad_non_unit_reference_account ~
        "williamson_reference_only_common_interest_account",
      wcad_nonresidential_condo_account ~
        "williamson_nonresidential_condominium_account",
      wcad_non_unit_amenity_parcel ~
        "williamson_park_or_amenity_parcel_without_units",
      wcad_non_unit_transitional_land ~
        "williamson_transitional_commercial_land_without_units",
      wcad_nonresidential_account ~
        "williamson_other_nonresidential_account",
      TRUE ~ NA_character_
    ),
    county_unit_exclude_from_unit_universe =
      !is.na(county_unit_exclusion_reason),
    county_unit_review_reason = case_when(
      source_county == "Williamson" &
        !county_unit_exclude_from_unit_universe &
        (is.na(model_improvement_sqft) | model_improvement_sqft <= 0) ~
        "williamson_zero_or_missing_residential_floor_area",
      source_county == "Williamson" &
        !county_unit_exclude_from_unit_universe &
        !wcad_residential_type &
        wcad_condo_signal &
        !wcad_apartment_model_candidate ~
        "williamson_commercial_condominium_in_residential_extract",
      source_county == "Williamson" &
        !county_unit_exclude_from_unit_universe &
        !wcad_residential_type &
        !wcad_apartment_model_candidate ~
        "williamson_nonresidential_type_in_residential_extract",
      source_county == "Williamson" &
        !county_unit_exclude_from_unit_universe &
        wcad_residential_type &
        wcad_apartment_signal &
        is.na(wcad_small_multifamily_units) &
        !wcad_explicit_residential_unit_account ~
        "williamson_ambiguous_residential_apartment_text",
      TRUE ~ NA_character_
    ),
    county_unit_evidence_class = case_when(
      !is.na(hays_deterministic_units) ~
        "hays_deterministic_appraisal_account",
      hays_multifamily_model_candidate ~
        "hays_multifamily_model_candidate",
      wcad_explicit_residential_unit_account ~
        "wcad_explicit_residential_unit_account",
      !is.na(wcad_small_multifamily_units) ~
        "wcad_small_multifamily_legal_description",
      wcad_apartment_model_candidate ~
        "wcad_apartment_model_candidate",
      !is.na(wcad_single_unit_rule_units) ~
        "williamson_single_unit_rule",
      wcad_non_unit_reference_account ~
        "wcad_non_unit_reference_account",
      wcad_nonresidential_condo_account ~
        "wcad_nonresidential_condominium_account",
      wcad_non_unit_amenity_parcel ~
        "wcad_non_unit_park_or_amenity_parcel",
      wcad_non_unit_transitional_land ~
        "wcad_non_unit_transitional_commercial_land",
      wcad_nonresidential_account ~
        "wcad_other_nonresidential_account",
      !is.na(county_unit_review_reason) ~
        "county_source_review",
      TRUE ~ NA_character_
    )
  )

record_template <- tibble(
  source_record_id = character(),
  source_name = character(),
  source_priority = integer(),
  source_tier = character(),
  unit_count_kind = character(),
  source_unit_count = double(),
  source_status = character(),
  source_project_name = character(),
  source_address = character(),
  source_address_key = character(),
  source_zip5 = character(),
  source_lat = double(),
  source_lon = double(),
  source_consistent = logical(),
  use_as_strict_model_label = logical(),
  use_as_deterministic_count = logical(),
  use_as_rule_based_count = logical(),
  use_as_sensitivity_label = logical()
)

link_template <- tibble(
  source_record_id = character(),
  parcel_id = character(),
  match_method = character(),
  match_confidence = character(),
  match_distance_m = double(),
  address_similarity = double()
)

tcad_explicit_records <- parcels_enhanced %>%
  filter(tcad_explicit_units_plausible) %>%
  transmute(
    source_record_id = paste0("tcad_explicit:", parcel_id),
    source_name = "tcad_explicit_units",
    source_priority = 30L,
    source_tier = "direct_appraisal_property_total",
    unit_count_kind = "reported_property_total",
    source_unit_count = tcad_imprv_units,
    source_status = "current_appraisal",
    source_project_name = NA_character_,
    source_address = situs_address,
    source_address_key = parcel_address_key,
    source_zip5 = parcel_zip5,
    source_lat = lat,
    source_lon = lon,
    source_consistent = TRUE,
    use_as_strict_model_label = TRUE,
    use_as_deterministic_count = FALSE,
    use_as_sensitivity_label = FALSE
  )

tcad_explicit_links <- parcels_enhanced %>%
  filter(tcad_explicit_units_plausible) %>%
  transmute(
    source_record_id = paste0("tcad_explicit:", parcel_id),
    parcel_id,
    match_method = "appraisal_parcel_id",
    match_confidence = "high",
    match_distance_m = 0,
    address_similarity = 1
  )

tcad_account_records <- parcels_enhanced %>%
  filter(!is.na(deterministic_account_units)) %>%
  transmute(
    source_record_id = paste0("tcad_account:", parcel_id),
    source_name = "tcad_account_rule",
    source_priority = 40L,
    source_tier = "deterministic_appraisal_account",
    unit_count_kind = "account_based_count",
    source_unit_count = deterministic_account_units,
    source_status = appraisal_state_code,
    source_project_name = NA_character_,
    source_address = situs_address,
    source_address_key = parcel_address_key,
    source_zip5 = parcel_zip5,
    source_lat = lat,
    source_lon = lon,
    source_consistent = TRUE,
    use_as_strict_model_label = FALSE,
    use_as_deterministic_count = TRUE,
    use_as_sensitivity_label = FALSE
  )

tcad_account_links <- parcels_enhanced %>%
  filter(!is.na(deterministic_account_units)) %>%
  transmute(
    source_record_id = paste0("tcad_account:", parcel_id),
    parcel_id,
    match_method = "appraisal_parcel_id",
    match_confidence = "high",
    match_distance_m = 0,
    address_similarity = 1
  )

hays_account_records <- parcels_enhanced %>%
  filter(!is.na(hays_deterministic_units)) %>%
  transmute(
    source_record_id = paste0("hays_account:", parcel_id),
    source_name = "hays_appraisal_account_rule",
    source_priority = 40L,
    source_tier = "deterministic_appraisal_account",
    unit_count_kind = "appraisal_state_code_count",
    source_unit_count = hays_deterministic_units,
    source_status = str_c(
      coalesce(hays_improvement_code, ""),
      coalesce(hays_land_code, ""),
      sep = " | "
    ),
    source_project_name = NA_character_,
    source_address = situs_address,
    source_address_key = parcel_address_key,
    source_zip5 = parcel_zip5,
    source_lat = lat,
    source_lon = lon,
    source_consistent = TRUE,
    use_as_strict_model_label = FALSE,
    use_as_deterministic_count = TRUE,
    use_as_rule_based_count = FALSE,
    use_as_sensitivity_label = FALSE
  )

hays_account_links <- parcels_enhanced %>%
  filter(!is.na(hays_deterministic_units)) %>%
  transmute(
    source_record_id = paste0("hays_account:", parcel_id),
    parcel_id,
    match_method = "appraisal_parcel_id",
    match_confidence = "high",
    match_distance_m = 0,
    address_similarity = 1
  )

wcad_condo_records <- parcels_enhanced %>%
  filter(wcad_explicit_residential_unit_account) %>%
  transmute(
    source_record_id = paste0("wcad_condo_account:", parcel_id),
    source_name = "wcad_explicit_residential_unit_account",
    source_priority = 40L,
    source_tier = "deterministic_appraisal_account",
    unit_count_kind = "explicit_condominium_unit_account",
    source_unit_count = 1,
    source_status = str_c(
      coalesce(wcad_legal_type, ""),
      coalesce(wcad_unit_type, ""),
      sep = " | "
    ),
    source_project_name = NA_character_,
    source_address = situs_address,
    source_address_key = parcel_address_key,
    source_zip5 = parcel_zip5,
    source_lat = lat,
    source_lon = lon,
    source_consistent = TRUE,
    use_as_strict_model_label = FALSE,
    use_as_deterministic_count = TRUE,
    use_as_rule_based_count = FALSE,
    use_as_sensitivity_label = FALSE
  )

wcad_condo_links <- parcels_enhanced %>%
  filter(wcad_explicit_residential_unit_account) %>%
  transmute(
    source_record_id = paste0("wcad_condo_account:", parcel_id),
    parcel_id,
    match_method = "wcad_property_and_unit_identifier",
    match_confidence = "high",
    match_distance_m = 0,
    address_similarity = 1
  )

wcad_small_multifamily_records <- parcels_enhanced %>%
  filter(
    !wcad_explicit_residential_unit_account,
    !is.na(wcad_small_multifamily_units)
  ) %>%
  transmute(
    source_record_id = paste0("wcad_small_multifamily:", parcel_id),
    source_name = "wcad_small_multifamily_legal_rule",
    source_priority = 42L,
    source_tier = "deterministic_appraisal_account",
    unit_count_kind = "legal_description_unit_count",
    source_unit_count = wcad_small_multifamily_units,
    source_status = case_when(
      wcad_small_multifamily_units == 2 ~ "duplex",
      wcad_small_multifamily_units == 3 ~ "triplex",
      wcad_small_multifamily_units == 4 ~ "fourplex",
      TRUE ~ "small_multifamily"
    ),
    source_project_name = wcad_dba,
    source_address = situs_address,
    source_address_key = parcel_address_key,
    source_zip5 = parcel_zip5,
    source_lat = lat,
    source_lon = lon,
    source_consistent = TRUE,
    use_as_strict_model_label = FALSE,
    use_as_deterministic_count = TRUE,
    use_as_rule_based_count = FALSE,
    use_as_sensitivity_label = FALSE
  )

wcad_small_multifamily_links <- parcels_enhanced %>%
  filter(
    !wcad_explicit_residential_unit_account,
    !is.na(wcad_small_multifamily_units)
  ) %>%
  transmute(
    source_record_id = paste0("wcad_small_multifamily:", parcel_id),
    parcel_id,
    match_method = "wcad_legal_description",
    match_confidence = "high",
    match_distance_m = 0,
    address_similarity = 1
  )

williamson_single_unit_records <- parcels_enhanced %>%
  filter(!is.na(wcad_single_unit_rule_units)) %>%
  transmute(
    source_record_id = paste0("williamson_single_unit:", parcel_id),
    source_name = "williamson_single_unit_rule",
    source_priority = 60L,
    source_tier = "rule_based_single_unit_assumption",
    unit_count_kind = "residential_type_and_floor_area_rule",
    source_unit_count = wcad_single_unit_rule_units,
    source_status = wcad_property_type,
    source_project_name = NA_character_,
    source_address = situs_address,
    source_address_key = parcel_address_key,
    source_zip5 = parcel_zip5,
    source_lat = lat,
    source_lon = lon,
    source_consistent = TRUE,
    use_as_strict_model_label = FALSE,
    use_as_deterministic_count = FALSE,
    use_as_rule_based_count = TRUE,
    use_as_sensitivity_label = FALSE
  )

williamson_single_unit_links <- parcels_enhanced %>%
  filter(!is.na(wcad_single_unit_rule_units)) %>%
  transmute(
    source_record_id = paste0("williamson_single_unit:", parcel_id),
    parcel_id,
    match_method = "wcad_residential_single_unit_rule",
    match_confidence = "high",
    match_distance_m = 0,
    address_similarity = 1
  )

costar_matches <- read_csv(COSTAR_MATCH_FILE, show_col_types = FALSE) %>%
  mutate(
    parcel_id = as.character(parcel_id),
    costar_id = as.character(costar_id),
    source_record_id = paste0("costar:", costar_id),
    costar_units = unit_numeric(costar_units),
    match_distance_m = unit_numeric(match_distance_m),
    address_similarity = unit_numeric(address_similarity)
  ) %>%
  filter(
    use_for_calibration %in% TRUE,
    plausible_for_calibration %in% TRUE,
    match_type %in% c("exact_address_zip", "nearest_spatial_strong"),
    costar_units > 0,
    parcel_id %in% parcels_enhanced$parcel_id
  )

costar_records <- costar_matches %>%
  group_by(source_record_id, costar_id) %>%
  summarise(
    source_name = "costar_current",
    source_priority = 20L,
    source_tier = "direct_commercial_property_total",
    unit_count_kind = "reported_project_total",
    source_unit_count = first(costar_units),
    source_status = "latest_available_period",
    source_project_name = first_non_missing_unit_value(building_name),
    source_address = first_non_missing_unit_value(building_address),
    source_address_key = normalize_unit_address(source_address),
    source_zip5 = unit_zip5(first_non_missing_unit_value(building_zip)),
    source_lat = NA_real_,
    source_lon = NA_real_,
    source_consistent = n_distinct(costar_units) == 1L,
    use_as_strict_model_label = source_consistent,
    use_as_deterministic_count = FALSE,
    use_as_sensitivity_label = FALSE,
    .groups = "drop"
  ) %>%
  select(-costar_id)

costar_links <- costar_matches %>%
  transmute(
    source_record_id,
    parcel_id,
    match_method = as.character(match_type),
    match_confidence = "high",
    match_distance_m,
    address_similarity
  ) %>%
  distinct()

ahi_raw <- read_csv(ahi_file, show_col_types = FALSE) %>%
  mutate(
    Project_ID = as.character(Project_ID),
    Total_Units = unit_numeric(Total_Units),
    Longitude = unit_numeric(Longitude),
    Latitude = unit_numeric(Latitude),
    source_address_key = normalize_unit_address(Address),
    source_zip5 = unit_zip5(ZIP)
  ) %>%
  filter(
    Development_Status == "Project Completed",
    str_detect(coalesce(Unit_Type, ""), "Multifamily"),
    Total_Units > 0,
    !is.na(Project_ID)
  )

ahi_records_base <- ahi_raw %>%
  group_by(Project_ID) %>%
  summarise(
    source_record_id = paste0("ahi:", Project_ID),
    source_name = "austin_affordable_housing_inventory",
    source_priority = 10L,
    source_tier = "direct_city_program_property_total",
    unit_count_kind = "reported_completed_project_total",
    source_unit_count = first(Total_Units),
    source_status = "project_completed",
    source_project_name = first_non_missing_unit_value(Project_Name),
    source_address = first_non_missing_unit_value(Address),
    source_address_key = first_non_missing_unit_value(source_address_key),
    source_zip5 = first_non_missing_unit_value(source_zip5),
    source_lat = unit_numeric(first_non_missing_unit_value(Latitude)),
    source_lon = unit_numeric(first_non_missing_unit_value(Longitude)),
    source_consistent = n_distinct(Total_Units) == 1L,
    raw_parcel_ids = str_c(na.omit(unique(Parcel_ID)), collapse = " | "),
    use_as_strict_model_label = source_consistent,
    use_as_deterministic_count = FALSE,
    use_as_sensitivity_label = FALSE,
    .groups = "drop"
  )

ahi_id_links <- ahi_records_base %>%
  select(source_record_id, raw_parcel_ids) %>%
  separate_longer_delim(raw_parcel_ids, delim = regex("[^A-Za-z0-9]+")) %>%
  transmute(
    source_record_id,
    parcel_id = raw_parcel_ids
  ) %>%
  filter(
    !is.na(parcel_id),
    parcel_id != "",
    parcel_id != "0",
    parcel_id %in% parcels_enhanced$parcel_id
  ) %>%
  mutate(
    match_method = "reported_parcel_id",
    match_confidence = "high",
    match_distance_m = 0,
    address_similarity = NA_real_
  )

parcel_address_lookup <- parcels_enhanced %>%
  filter(!is.na(parcel_address_key), !is.na(parcel_zip5)) %>%
  select(parcel_id, parcel_address_key, parcel_zip5, lat, lon)

ahi_address_links <- ahi_records_base %>%
  filter(!is.na(source_address_key), !is.na(source_zip5)) %>%
  inner_join(
    parcel_address_lookup,
    by = c(
      "source_address_key" = "parcel_address_key",
      "source_zip5" = "parcel_zip5"
    ),
    relationship = "many-to-many"
  ) %>%
  transmute(
    source_record_id,
    parcel_id,
    match_method = "exact_address_zip",
    match_confidence = "high",
    match_distance_m = NA_real_,
    address_similarity = 1
  )

ahi_matched_ids <- unique(c(
  ahi_id_links$source_record_id,
  ahi_address_links$source_record_id
))

ahi_fuzzy_candidates <- ahi_records_base %>%
  filter(
    !source_record_id %in% ahi_matched_ids,
    !is.na(source_address_key),
    !is.na(source_zip5)
  ) %>%
  mutate(street_number = unit_street_number(source_address_key)) %>%
  filter(!is.na(street_number)) %>%
  select(
    source_record_id,
    source_address_key,
    source_zip5,
    street_number
  ) %>%
  inner_join(
    parcel_address_lookup %>%
      mutate(street_number = unit_street_number(parcel_address_key)),
    by = c("source_zip5" = "parcel_zip5", "street_number"),
    relationship = "many-to-many"
  ) %>%
  mutate(
    address_similarity = unit_address_similarity(
      source_address_key,
      parcel_address_key
    )
  ) %>%
  group_by(source_record_id) %>%
  arrange(desc(address_similarity), parcel_id, .by_group = TRUE) %>%
  mutate(
    best_similarity = first(address_similarity),
    second_similarity = nth(address_similarity, 2, default = 0)
  ) %>%
  filter(
    row_number() == 1L,
    best_similarity >= 0.85,
    best_similarity - second_similarity >= 0.05
  ) %>%
  ungroup()

ahi_fuzzy_links <- ahi_fuzzy_candidates %>%
  transmute(
    source_record_id,
    parcel_id,
    match_method = "unique_fuzzy_address_zip",
    match_confidence = "high",
    match_distance_m = NA_real_,
    address_similarity
  )

ahi_high_links <- bind_rows(
  ahi_id_links,
  ahi_address_links,
  ahi_fuzzy_links
) %>%
  arrange(source_record_id, match_method) %>%
  distinct(source_record_id, parcel_id, .keep_all = TRUE)

ahi_still_unmatched <- ahi_records_base %>%
  filter(
    !source_record_id %in% ahi_high_links$source_record_id,
    is.finite(source_lat),
    is.finite(source_lon)
  )

ahi_spatial_links <- link_template
if (nrow(ahi_still_unmatched) > 0L) {
  source_points <- st_as_sf(
    ahi_still_unmatched,
    coords = c("source_lon", "source_lat"),
    crs = 4326,
    remove = FALSE
  ) %>%
    st_transform(32614)
  parcel_points <- parcels_enhanced %>%
    filter(is.finite(lon), is.finite(lat)) %>%
    st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
    st_transform(32614)
  nearest_index <- st_nearest_feature(source_points, parcel_points)
  nearest_distance <- as.numeric(
    st_distance(
      source_points,
      parcel_points[nearest_index, ],
      by_element = TRUE
    )
  )
  ahi_spatial_links <- tibble(
    source_record_id = source_points$source_record_id,
    parcel_id = parcel_points$parcel_id[nearest_index],
    match_method = "nearest_spatial_review",
    match_confidence = "review",
    match_distance_m = nearest_distance,
    address_similarity = unit_address_similarity(
      source_points$source_address_key,
      parcel_points$parcel_address_key[nearest_index]
    )
  ) %>%
    filter(match_distance_m <= 150)
}

ahi_records <- ahi_records_base %>%
  select(-raw_parcel_ids) %>%
  mutate(
    use_as_strict_model_label = use_as_strict_model_label &
      source_record_id %in% ahi_high_links$source_record_id
  )

ahi_links <- bind_rows(ahi_high_links, ahi_spatial_links) %>%
  arrange(
    source_record_id,
    factor(match_confidence, levels = c("high", "review"))
  ) %>%
  distinct(source_record_id, parcel_id, .keep_all = TRUE)

uro_raw <- read_csv(
  uro_file,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE
) %>%
  transmute(
    property_id = as.character(`Property ID`),
    project_name = as.character(`Property Name`),
    source_address = as.character(`Physical Address`),
    source_address_key = normalize_unit_address(`Physical Address`),
    source_zip5 = unit_zip5(`Zip Code`),
    source_status = as.character(Status),
    is_multifamily = `Multifamily Property?` == "Yes",
    source_unit_count = unit_numeric(`Multifamily Unit Count`)
  ) %>%
  filter(
    is_multifamily,
    source_unit_count > 0,
    !is.na(source_address_key),
    !is.na(source_zip5)
  )

uro_records <- uro_raw %>%
  group_by(source_address_key, source_zip5) %>%
  summarise(
    source_record_id = paste0(
      "uro:",
      str_replace_all(first(source_address_key), " ", "_"),
      ":",
      first(source_zip5)
    ),
    source_name = "austin_universal_recycling_inventory",
    source_priority = 50L,
    source_tier = "official_administrative_estimate",
    unit_count_kind = "estimated_multifamily_unit_count",
    source_unit_count = first(source_unit_count),
    source_status = str_c(sort(unique(source_status)), collapse = " | "),
    source_project_name = first_non_missing_unit_value(project_name),
    source_address = first_non_missing_unit_value(source_address),
    source_lat = NA_real_,
    source_lon = NA_real_,
    source_consistent = n_distinct(source_unit_count) == 1L,
    use_as_strict_model_label = FALSE,
    use_as_deterministic_count = FALSE,
    use_as_sensitivity_label = source_consistent,
    .groups = "drop"
  )

uro_links <- uro_records %>%
  inner_join(
    parcel_address_lookup,
    by = c(
      "source_address_key" = "parcel_address_key",
      "source_zip5" = "parcel_zip5"
    ),
    relationship = "many-to-many"
  ) %>%
  transmute(
    source_record_id,
    parcel_id,
    match_method = "exact_address_zip",
    match_confidence = "high",
    match_distance_m = NA_real_,
    address_similarity = 1
  )

uro_records <- uro_records %>%
  mutate(
    use_as_sensitivity_label = use_as_sensitivity_label &
      source_record_id %in% uro_links$source_record_id
  )

source_records <- bind_rows(
  record_template,
  tcad_explicit_records,
  tcad_account_records,
  hays_account_records,
  wcad_condo_records,
  wcad_small_multifamily_records,
  williamson_single_unit_records,
  costar_records,
  ahi_records,
  uro_records
) %>%
  mutate(
    use_as_strict_model_label = replace_na(
      use_as_strict_model_label,
      FALSE
    ),
    use_as_deterministic_count = replace_na(
      use_as_deterministic_count,
      FALSE
    ),
    use_as_rule_based_count = replace_na(
      use_as_rule_based_count,
      FALSE
    ),
    use_as_sensitivity_label = replace_na(
      use_as_sensitivity_label,
      FALSE
    )
  ) %>%
  arrange(source_priority, source_name, source_record_id)

source_links <- bind_rows(
  link_template,
  tcad_explicit_links,
  tcad_account_links,
  hays_account_links,
  wcad_condo_links,
  wcad_small_multifamily_links,
  williamson_single_unit_links,
  costar_links,
  ahi_links,
  uro_links
) %>%
  filter(
    source_record_id %in% source_records$source_record_id,
    parcel_id %in% parcels_enhanced$parcel_id
  ) %>%
  distinct(source_record_id, parcel_id, .keep_all = TRUE)

if (anyDuplicated(source_records$source_record_id)) {
  stop("Unit-source records are not unique by source_record_id.", call. = FALSE)
}
if (anyDuplicated(source_links[c("source_record_id", "parcel_id")])) {
  stop("Unit-source links contain duplicate source/parcel pairs.", call. = FALSE)
}

source_link_counts <- source_links %>%
  group_by(source_record_id) %>%
  summarise(
    linked_parcel_count = n_distinct(parcel_id),
    high_confidence_link_count = n_distinct(
      parcel_id[match_confidence == "high"]
    ),
    review_link_count = n_distinct(
      parcel_id[match_confidence == "review"]
    ),
    .groups = "drop"
  )

source_records <- source_records %>%
  left_join(source_link_counts, by = "source_record_id") %>%
  mutate(
    linked_parcel_count = replace_na(linked_parcel_count, 0L),
    high_confidence_link_count = replace_na(
      high_confidence_link_count,
      0L
    ),
    review_link_count = replace_na(review_link_count, 0L),
    source_is_linked = linked_parcel_count > 0L
  )

unit_source_qa <- bind_rows(
  source_records %>%
    group_by(source_name, source_tier, unit_count_kind) %>%
    summarise(
      records = n(),
      linked_records = sum(source_is_linked),
      linked_parcels = n_distinct(
        source_links$parcel_id[
          source_links$source_record_id %in% source_record_id
        ]
      ),
      source_units = sum(
        if_else(source_is_linked, source_unit_count, 0),
        na.rm = TRUE
      ),
      strict_label_records = sum(
        use_as_strict_model_label & source_is_linked
      ),
      deterministic_records = sum(
        use_as_deterministic_count & source_is_linked
      ),
      rule_based_records = sum(
        use_as_rule_based_count & source_is_linked
      ),
      sensitivity_records = sum(
        use_as_sensitivity_label & source_is_linked
      ),
      .groups = "drop"
    ) %>%
    mutate(qa_section = "source_coverage"),
  source_links %>%
    count(match_method, match_confidence, name = "records") %>%
    transmute(
      source_name = NA_character_,
      source_tier = NA_character_,
      unit_count_kind = NA_character_,
      records,
      linked_records = NA_integer_,
      linked_parcels = NA_integer_,
      source_units = NA_real_,
      strict_label_records = NA_integer_,
      deterministic_records = NA_integer_,
      rule_based_records = NA_integer_,
      sensitivity_records = NA_integer_,
      qa_section = paste0(
        "link_method:",
        match_method,
        ":",
        match_confidence
      )
    )
)

source_manifest <- tibble(
  source_name = c(
    "production_parcel_universe",
    "tcad_property_profile",
    "tcad_compact_extract",
    "wcad_property_certified",
    "wcad_parcel_geometry_attributes",
    "wcad_compact_extract",
    "costar_parcel_matches",
    "austin_universal_recycling_inventory",
    "austin_affordable_housing_inventory"
  ),
  source_path = c(
    PARCEL_FILE,
    tcad_profile_file,
    tcad_compact_file,
    wcad_property_file,
    wcad_parcel_file,
    wcad_compact_file,
    COSTAR_MATCH_FILE,
    uro_file,
    ahi_file
  )
) %>%
  mutate(
    source_exists = file.exists(source_path),
    source_bytes = if_else(
      source_exists,
      as.numeric(file.info(source_path)$size),
      NA_real_
    ),
    source_modified = if_else(
      source_exists,
      format(file.info(source_path)$mtime, tz = "UTC", usetz = TRUE),
      NA_character_
    )
  )

unmatched_source_records <- source_records %>%
  filter(
    high_confidence_link_count == 0L,
    use_as_strict_model_label | use_as_sensitivity_label
  )

county_unit_classification_qa <- parcels_enhanced %>%
  filter(source_county %in% c("Hays", "Williamson")) %>%
  mutate(
    county_unit_evidence_class = coalesce(
      county_unit_evidence_class,
      "unclassified_county_source"
    )
  ) %>%
  group_by(source_county, county_unit_evidence_class) %>%
  summarise(
    parcels = n(),
    current_units = sum(units_raw, na.rm = TRUE),
    improvement_sqft = sum(improvement_sqft, na.rm = TRUE),
    model_candidate_parcels = sum(county_model_candidate_signal, na.rm = TRUE),
    excluded_non_unit_parcels = sum(
      county_unit_exclude_from_unit_universe,
      na.rm = TRUE
    ),
    review_parcels = sum(!is.na(county_unit_review_reason)),
    .groups = "drop"
  )

county_unit_exclusion_audit <- parcels_enhanced %>%
  filter(
    county_unit_exclude_from_unit_universe |
      !is.na(county_unit_review_reason)
  ) %>%
  select(
    parcel_id,
    source_county,
    situs_address,
    lat,
    lon,
    county_unit_evidence_class,
    county_unit_exclusion_reason,
    county_unit_review_reason,
    units_raw,
    units_calibrated,
    model_improvement_sqft,
    appraisal_state_code,
    wcad_property_id,
    wcad_property_type,
    wcad_use_description,
    wcad_legal_type,
    wcad_legal_description,
    wcad_property_comment,
    wcad_dba,
    wcad_apartment_primary_signal,
    wcad_apartment_comment_signal,
    wcad_condo_signal
  )

if (
  any(
    parcels_enhanced$county_unit_exclude_from_unit_universe &
      !is.na(parcels_enhanced$county_unit_review_reason)
  )
) {
  stop("County unit exclusions and review flags must not overlap.", call. = FALSE)
}
if (
  any(
    parcels_enhanced$county_unit_exclude_from_unit_universe !=
      !is.na(parcels_enhanced$county_unit_exclusion_reason)
  )
) {
  stop("County unit exclusion flags and reasons are inconsistent.", call. = FALSE)
}

save_output(
  parcels_enhanced,
  file.path(OUTPUT_DIR, "residential_parcels_unit_source_attributes.rds"),
  "parcel attributes for residential unit source modeling"
)
save_output(
  source_records,
  file.path(OUTPUT_DIR, "residential_unit_source_records.rds"),
  "residential unit source records"
)
save_output(
  source_links,
  file.path(OUTPUT_DIR, "residential_unit_source_parcel_links.rds"),
  "residential unit source-to-parcel links"
)

write_csv(
  source_records,
  file.path(OUTPUT_DIR, "residential_unit_source_records.csv")
)
write_csv(
  source_links,
  file.path(OUTPUT_DIR, "residential_unit_source_parcel_links.csv")
)
write_csv(
  unit_source_qa,
  file.path(OUTPUT_DIR, "residential_unit_source_qa.csv")
)
write_csv(
  source_manifest,
  file.path(OUTPUT_DIR, "residential_unit_source_manifest.csv")
)
write_csv(
  unmatched_source_records,
  file.path(OUTPUT_DIR, "residential_unit_unmatched_source_records.csv")
)
write_csv(
  county_unit_classification_qa,
  file.path(OUTPUT_DIR, "residential_unit_county_classification_qa.csv")
)
write_csv(
  county_unit_exclusion_audit,
  file.path(OUTPUT_DIR, "residential_unit_county_exclusion_audit.csv")
)

print_progress(
  paste0(
    "Prepared ",
    scales::comma(nrow(source_records)),
    " source records and ",
    scales::comma(nrow(source_links)),
    " parcel links."
  )
)
print_progress(
  paste0(
    "Strict linked project records: ",
    scales::comma(
      sum(
        source_records$use_as_strict_model_label &
          source_records$source_is_linked
      )
    ),
    "; linked URO sensitivity records: ",
    scales::comma(
      sum(
        source_records$use_as_sensitivity_label &
          source_records$source_is_linked
      )
    )
  )
)
