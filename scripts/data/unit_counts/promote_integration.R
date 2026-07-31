################################################################################
# Promote the Validated Residential Unit Hierarchy
################################################################################
#
# Converts the reviewed `unit_integration` shadow parcel surface into the
# canonical input used by `corporate_features`. The `unit_validation` targeted
# table remains unchanged as a reproducible baseline. Promotion is rejected if
# the shadow table is not linked exactly to that baseline by parcel ID and
# pre-promotion unit count.
#
# Outputs:
#   output/residential_parcels_unit_promoted.rds
#   output/residential_unit_promotion_manifest.csv
#   output/residential_unit_land_use_exclusions.csv
#   output/pre_unit_model_promotion/...
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
  library(stringr)
  library(tibble)
})

source(here::here("R", "utils.R"))

print_header("PROMOTE VALIDATED RESIDENTIAL UNIT HIERARCHY")

OUTPUT_DIR <- here::here("output")
BASELINE_FILE <- file.path(
  OUTPUT_DIR,
  "residential_parcels_unit_targeted.rds"
)
SHADOW_FILE <- file.path(
  OUTPUT_DIR,
  "residential_parcels_unit_shadow_integrated.rds"
)
PROMOTED_FILE <- file.path(
  OUTPUT_DIR,
  "residential_parcels_unit_promoted.rds"
)
MANIFEST_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_promotion_manifest.csv"
)
LAND_USE_EXCLUSION_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_land_use_exclusions.csv"
)
LAND_USE_FILE <- here::here("data", "austin_land_use_inventory_202607.csv")
LAND_USE_CODE_FILE <- here::here("config", "austin_land_use_codes.csv")
BOUNDARY_FILE <- here::here(
  "data",
  "BOUNDARIES_jurisdictions_20260429.geojson"
)
ARCHIVE_DIR <- file.path(OUTPUT_DIR, "pre_unit_model_promotion")
PROMOTION_VERSION <- "v2_2026-07-31_land_use_validated"

required_files <- c(
  BASELINE_FILE,
  SHADOW_FILE,
  LAND_USE_FILE,
  LAND_USE_CODE_FILE,
  BOUNDARY_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Run targets unit_validation and unit_integration before promotion. ",
    "Missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

baseline <- readRDS(BASELINE_FILE) %>%
  mutate(parcel_id = as.character(parcel_id))
shadow <- readRDS(SHADOW_FILE) %>%
  transmute(
    parcel_id = as.character(parcel_id),
    promotion_baseline_targeted_units = as.numeric(current_targeted_units),
    promoted_units = as.numeric(shadow_units),
    unit_model_selection_method = as.character(shadow_selection_method),
    unit_model_allocation_method = as.character(allocation_method),
    unit_model_used = as.logical(shadow_model_used),
    unit_model_units_changed = as.logical(shadow_units_changed),
    unit_model_unit_delta = as.numeric(shadow_unit_delta),
    unit_model_project_id = as.character(project_id)
  )

if (
  anyDuplicated(baseline$parcel_id) ||
    anyDuplicated(shadow$parcel_id) ||
    !setequal(baseline$parcel_id, shadow$parcel_id) ||
    nrow(baseline) != nrow(shadow)
) {
  stop("Baseline and shadow parcel universes do not match exactly.", call. = FALSE)
}

baseline_link <- baseline %>%
  transmute(
    parcel_id,
    effective_baseline_targeted_units = coalesce(
      as.numeric(units_calibrated_targeted),
      as.numeric(units_calibrated),
      0
    )
  ) %>%
  inner_join(
    shadow %>%
      select(parcel_id, promotion_baseline_targeted_units),
    by = "parcel_id",
    relationship = "one-to-one"
  )

if (
  any(!is.finite(shadow$promoted_units)) ||
    any(shadow$promoted_units < 0) ||
    any(
      abs(
        baseline_link$effective_baseline_targeted_units -
          baseline_link$promotion_baseline_targeted_units
      ) > 1e-9
    )
) {
  stop(
    "Shadow units are invalid or were built from a different targeted baseline.",
    call. = FALSE
  )
}

promoted <- baseline %>%
  left_join(shadow, by = "parcel_id", relationship = "one-to-one") %>%
  mutate(
    unit_model_promotion_version = PROMOTION_VERSION,
    unit_model_promotion_applied = unit_model_selection_method %in% c(
      "strict_direct_project_total",
      "documented_project_count",
      "williamson_validated_main_area_stratified",
      "in_domain_main_area_stratified"
    ),
    unit_estimation_method_targeted = case_when(
      unit_model_promotion_applied ~ paste0(
        "promoted_",
        unit_model_selection_method
      ),
      TRUE ~ unit_estimation_method_targeted
    ),
    unit_estimation_confidence_targeted = case_when(
      unit_model_selection_method %in% c(
        "strict_direct_project_total",
        "documented_project_count"
      ) ~ "high",
      unit_model_used ~ "medium",
      TRUE ~ unit_estimation_confidence_targeted
    ),
    unit_estimation_notes_targeted = case_when(
      unit_model_promotion_applied &
        !is.na(unit_estimation_notes_targeted) &
        nzchar(unit_estimation_notes_targeted) ~ paste0(
          unit_estimation_notes_targeted,
          "; promoted through ",
          unit_model_selection_method
        ),
      unit_model_promotion_applied ~ paste0(
        "Promoted through ",
        unit_model_selection_method
      ),
      TRUE ~ unit_estimation_notes_targeted
    )
  )

land_use_codes <- read_csv(
  LAND_USE_CODE_FILE,
  show_col_types = FALSE,
  col_types = cols(
    land_use_code = col_integer(),
    detailed_land_use = col_character(),
    audit_group = col_character(),
    city_multiunit_signal = col_logical(),
    city_residential_signal = col_logical()
  )
)

land_use_by_identifier <- read_csv(
  LAND_USE_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_character())
) %>%
  transmute(
    land_use_code = as.integer(land_use),
    city_parcel_id = str_trim(parcel_id_10),
    city_property_id = str_trim(property_id),
    city_record_county = case_when(
      str_detect(city_parcel_id, "^[0-9]{10}$") ~ "Travis",
      str_detect(city_parcel_id, "^R[0-9]+$") &
        str_remove(city_parcel_id, "^R") == city_property_id ~ "Hays",
      str_detect(city_parcel_id, "^R[0-9]+$") ~ "Williamson",
      TRUE ~ NA_character_
    ),
    city_match_key = case_when(
      city_record_county == "Travis" ~ city_property_id,
      city_record_county %in% c("Hays", "Williamson") ~ city_parcel_id,
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(city_record_county), !is.na(city_match_key)) %>%
  left_join(land_use_codes, by = "land_use_code", relationship = "many-to-one") %>%
  group_by(city_record_county, city_match_key) %>%
  summarise(
    city_land_use_labels = str_c(
      sort(unique(na.omit(detailed_land_use))),
      collapse = " | "
    ),
    city_any_multiunit = any(city_multiunit_signal %in% TRUE),
    city_any_single_unit = any(audit_group == "single_unit", na.rm = TRUE),
    city_any_mixed_use = any(audit_group == "mixed_use", na.rm = TRUE),
    city_any_nonresidential = any(
      audit_group %in% c("nonresidential", "group_quarters"),
      na.rm = TRUE
    ),
    .groups = "drop"
  )

unknown_land_use_codes <- read_csv(
  LAND_USE_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_character())
) %>%
  transmute(land_use_code = as.integer(land_use)) %>%
  distinct() %>%
  anti_join(land_use_codes, by = "land_use_code")
if (nrow(unknown_land_use_codes) > 0L) {
  stop(
    "City land-use validation has unmapped codes: ",
    paste(unknown_land_use_codes$land_use_code, collapse = ", "),
    call. = FALSE
  )
}

full_purpose <- st_read(BOUNDARY_FILE, quiet = TRUE) %>%
  filter(jurisdiction_type == "FULL") %>%
  st_make_valid() %>%
  st_transform(4326) %>%
  summarise()

parcel_boundary_status <- promoted %>%
  filter(is.finite(as.numeric(lon)), is.finite(as.numeric(lat))) %>%
  transmute(
    parcel_id,
    lon = as.numeric(lon),
    lat = as.numeric(lat)
  ) %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
  transmute(
    parcel_id,
    in_austin_full_purpose = lengths(st_within(., full_purpose)) > 0
  ) %>%
  st_drop_geometry()

promoted <- promoted %>%
  mutate(
    parcel_match_key = case_when(
      source_county == "Travis" ~ as.character(parcel_id),
      source_county %in% c("Hays", "Williamson") ~
        str_remove(as.character(parcel_id), "^(HAYS|WILLIAMSON):"),
      TRUE ~ NA_character_
    )
  ) %>%
  left_join(
    land_use_by_identifier,
    by = c(
      "source_county" = "city_record_county",
      "parcel_match_key" = "city_match_key"
    ),
    relationship = "many-to-one"
  ) %>%
  left_join(parcel_boundary_status, by = "parcel_id", relationship = "one-to-one") %>%
  mutate(
    in_austin_full_purpose = coalesce(in_austin_full_purpose, FALSE),
    appraisal_multiunit_signal =
      as.character(propertyProf_imprvStateCd) %in%
        c("A4", "B1", "B2", "B3", "B4") |
        coalesce(as.logical(county_model_candidate_signal), FALSE)
  )

model_or_fallback_methods <- c(
  "in_domain_main_area_stratified",
  "williamson_validated_main_area_stratified",
  "current_targeted_fallback",
  "current_targeted_zero_fallback"
)

project_land_use_validation <- promoted %>%
  filter(!is.na(unit_model_project_id)) %>%
  group_by(unit_model_project_id) %>%
  summarise(
    project_in_full_purpose = any(in_austin_full_purpose),
    city_matched_full_purpose_rows = sum(
      in_austin_full_purpose & !is.na(city_any_nonresidential)
    ),
    city_any_multiunit = any(
      city_any_multiunit[in_austin_full_purpose] %in% TRUE
    ),
    city_any_single_unit = any(
      city_any_single_unit[in_austin_full_purpose] %in% TRUE
    ),
    city_any_mixed_use = any(
      city_any_mixed_use[in_austin_full_purpose] %in% TRUE
    ),
    city_any_nonresidential = any(
      city_any_nonresidential[in_austin_full_purpose] %in% TRUE
    ),
    project_appraisal_multiunit_signal = any(appraisal_multiunit_signal),
    project_selection_method = first(unit_model_selection_method),
    project_pre_validation_units = sum(promoted_units),
    city_land_use_labels = str_c(
      sort(unique(na.omit(city_land_use_labels[in_austin_full_purpose]))),
      collapse = " | "
    ),
    .groups = "drop"
  ) %>%
  mutate(
    unit_land_use_validation_excluded =
      project_in_full_purpose &
        city_matched_full_purpose_rows > 0 &
        city_any_nonresidential &
        !city_any_multiunit &
        !city_any_single_unit &
        !city_any_mixed_use &
        !project_appraisal_multiunit_signal &
        project_selection_method %in% model_or_fallback_methods,
    unit_land_use_exclusion_reason = if_else(
      unit_land_use_validation_excluded,
      paste(
        "Modeled or fallback project has only zoning/broad residential",
        "evidence and exact-ID City nonresidential land-use evidence"
      ),
      NA_character_
    )
  )

promoted <- promoted %>%
  left_join(
    project_land_use_validation %>%
      select(
        unit_model_project_id,
        unit_land_use_validation_excluded,
        unit_land_use_exclusion_reason
      ),
    by = "unit_model_project_id",
    relationship = "many-to-one"
  ) %>%
  mutate(
    unit_land_use_validation_excluded = coalesce(
      unit_land_use_validation_excluded,
      FALSE
    ),
    unit_land_use_pre_validation_units = promoted_units,
    unit_land_use_excluded_units = if_else(
      unit_land_use_validation_excluded,
      promoted_units,
      0
    ),
    promoted_units = if_else(
      unit_land_use_validation_excluded,
      0,
      promoted_units
    ),
    unit_model_units_changed = abs(
      promoted_units - promotion_baseline_targeted_units
    ) > 1e-9,
    unit_model_unit_delta = promoted_units - promotion_baseline_targeted_units,
    unit_model_promotion_applied = unit_model_promotion_applied |
      unit_land_use_validation_excluded,
    unit_estimation_method_targeted = if_else(
      unit_land_use_validation_excluded,
      "city_land_use_nonresidential_exclusion",
      unit_estimation_method_targeted
    ),
    unit_estimation_confidence_targeted = if_else(
      unit_land_use_validation_excluded,
      "high_exclusion_confidence",
      unit_estimation_confidence_targeted
    ),
    unit_estimation_notes_targeted = if_else(
      unit_land_use_validation_excluded,
      unit_land_use_exclusion_reason,
      unit_estimation_notes_targeted
    ),
    units_calibrated_targeted = promoted_units,
    targeted_unit_delta = units_calibrated_targeted - units_calibrated,
    property_units_targeted = units_calibrated_targeted,
    corporate_units_targeted = if_else(
      as.logical(is_corporate_owned),
      units_calibrated_targeted,
      0
    )
  ) %>%
  select(-parcel_match_key, -appraisal_multiunit_signal)

if (
  nrow(promoted) != nrow(baseline) ||
    anyDuplicated(promoted$parcel_id) ||
    any(!is.finite(promoted$units_calibrated_targeted)) ||
    any(promoted$units_calibrated_targeted < 0) ||
    abs(
      sum(promoted$units_calibrated_targeted) +
        sum(promoted$unit_land_use_excluded_units) -
        sum(shadow$promoted_units)
    ) > 1e-6
) {
  stop("Promoted parcel surface failed final validation.", call. = FALSE)
}

dir.create(ARCHIVE_DIR, recursive = TRUE, showWarnings = FALSE)
archive_files <- c(
  "corporate_ownership_by_hex.rds",
  "corporate_ownership_by_hex.csv",
  "hex_features.rds",
  "feature_list.csv",
  "cluster_analysis_results.rds",
  "hex_features_with_clusters.rds",
  "cluster_profiles.csv",
  "amenity_cluster_sensitivity.rds",
  "amenity_cluster_assignments.csv",
  "amenity_cluster_profiles.csv",
  "amenity_cluster_recommendations.csv"
)
archive_sources <- file.path(OUTPUT_DIR, archive_files)
archive_destinations <- file.path(ARCHIVE_DIR, archive_files)
archive_needed <- file.exists(archive_sources) & !file.exists(archive_destinations)
if (any(archive_needed)) {
  archived <- file.copy(
    archive_sources[archive_needed],
    archive_destinations[archive_needed],
    overwrite = FALSE
  )
  if (!all(archived)) {
    stop("Could not archive every pre-promotion canonical output.", call. = FALSE)
  }
}

save_output(
  promoted,
  PROMOTED_FILE,
  "promoted residential parcel unit hierarchy"
)

land_use_exclusions <- project_land_use_validation %>%
  filter(unit_land_use_validation_excluded) %>%
  arrange(desc(project_pre_validation_units), unit_model_project_id)
write_csv(land_use_exclusions, LAND_USE_EXCLUSION_FILE)

manifest <- tibble(
  promotion_version = PROMOTION_VERSION,
  promoted_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  baseline_file = basename(BASELINE_FILE),
  shadow_file = basename(SHADOW_FILE),
  promoted_file = basename(PROMOTED_FILE),
  parcels = nrow(promoted),
  baseline_units = sum(promoted$promotion_baseline_targeted_units),
  promoted_units = sum(promoted$units_calibrated_targeted),
  land_use_excluded_projects = nrow(land_use_exclusions),
  land_use_excluded_parcels = sum(promoted$unit_land_use_validation_excluded),
  land_use_excluded_units = sum(promoted$unit_land_use_excluded_units),
  unit_delta = promoted_units - baseline_units,
  changed_parcels = sum(promoted$unit_model_units_changed),
  hierarchy_applied_parcels = sum(promoted$unit_model_promotion_applied),
  model_prediction_parcels = sum(promoted$unit_model_used)
)
write_csv(manifest, MANIFEST_FILE)

print_progress("Promotion manifest:")
print(manifest)
print_progress(
  paste0(
    "Promoted parcel hierarchy saved. Run target corporate_features or a ",
    "downstream target to rebuild canonical outputs."
  )
)
