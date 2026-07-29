################################################################################
# Promote the Validated Residential Unit Hierarchy
################################################################################
#
# Converts the reviewed 02t shadow parcel surface into the canonical input used
# by 02c. The 02e targeted table remains unchanged as a reproducible baseline.
# Promotion is rejected if the shadow table is not linked exactly to that
# baseline by parcel ID and pre-promotion unit count.
#
# Outputs:
#   output/residential_parcels_unit_promoted.rds
#   output/residential_unit_promotion_manifest.csv
#   output/pre_unit_model_promotion/...
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tibble)
})

source(here::here("R", "utils.R"))

print_header("02v - PROMOTE VALIDATED RESIDENTIAL UNIT HIERARCHY")

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
ARCHIVE_DIR <- file.path(OUTPUT_DIR, "pre_unit_model_promotion")
PROMOTION_VERSION <- "v1_2026-07-28"

required_files <- c(BASELINE_FILE, SHADOW_FILE)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Run 02e and 02t before promotion. Missing: ",
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
    ),
    units_calibrated_targeted = promoted_units,
    targeted_unit_delta = units_calibrated_targeted - units_calibrated,
    property_units_targeted = units_calibrated_targeted,
    corporate_units_targeted = if_else(
      as.logical(is_corporate_owned),
      units_calibrated_targeted,
      0
    )
  )

if (
  nrow(promoted) != nrow(baseline) ||
    anyDuplicated(promoted$parcel_id) ||
    any(!is.finite(promoted$units_calibrated_targeted)) ||
    any(promoted$units_calibrated_targeted < 0) ||
    abs(
      sum(promoted$units_calibrated_targeted) -
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

manifest <- tibble(
  promotion_version = PROMOTION_VERSION,
  promoted_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  baseline_file = basename(BASELINE_FILE),
  shadow_file = basename(SHADOW_FILE),
  promoted_file = basename(PROMOTED_FILE),
  parcels = nrow(promoted),
  baseline_units = sum(promoted$promotion_baseline_targeted_units),
  promoted_units = sum(promoted$units_calibrated_targeted),
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
    "Promoted parcel hierarchy saved. Re-run 02c and downstream stages ",
    "without a shadow override to rebuild canonical outputs."
  )
)
