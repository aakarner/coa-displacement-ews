################################################################################
# Build Candidate Residential Unit Integration
################################################################################
#
# Applies the validated unit hierarchy to residential projects:
#   1. selected strict direct project totals;
#   2. documented counts for unresolved model candidates;
#   3. main/living-area stratified predictions for validated Williamson rows;
#   4. in-domain main/living-area predictions elsewhere; and
#   5. current targeted parcel totals as an explicit fallback.
#
# Project totals are allocated back to parcels once. Reviewed WCAD companion
# accounts use their configured allocation parcel, and overlapping cross-county
# projects use the Travis parcel representation. All outputs are shadow files;
# canonical parcel, feature, and cluster files are not overwritten.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
  library(stringr)
  library(tibble)
  library(tidyr)
})

source(here::here("R", "utils.R"))
source(here::here("R", "analysis_config.R"))

print_header("BUILD SHADOW RESIDENTIAL UNIT INTEGRATION")

OUTPUT_DIR <- here::here("output")
SOURCE_PARCEL_FILE <- file.path(
  OUTPUT_DIR,
  "residential_parcels_unit_source_attributes.rds"
)
TARGETED_PARCEL_FILE <- file.path(
  OUTPUT_DIR,
  "residential_parcels_unit_targeted.rds"
)
MEMBERSHIP_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_project_membership.rds"
)
PROJECT_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_projects.rds"
)
PREDICTION_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_model_predictions.rds"
)
WILLIAMSON_VALIDATION_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_williamson_validation.rds"
)
REVIEWED_GROUP_FILE <- here::here(
  "config",
  "williamson_project_groups.csv"
)
HEX_GRID_FILE <- file.path(OUTPUT_DIR, "hex_grid.rds")

required_files <- c(
  SOURCE_PARCEL_FILE,
  TARGETED_PARCEL_FILE,
  MEMBERSHIP_FILE,
  PROJECT_FILE,
  PREDICTION_FILE,
  WILLIAMSON_VALIDATION_FILE,
  REVIEWED_GROUP_FILE,
  HEX_GRID_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Run targets unit_validation through williamson_validation before ",
    "unit_integration. Missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

collapse_values <- function(x) {
  observed <- sort(unique(as.character(x[!is.na(x) & nzchar(x)])))
  if (length(observed) == 0L) {
    return(NA_character_)
  }
  paste(observed, collapse = " | ")
}

source_parcels <- readRDS(SOURCE_PARCEL_FILE) %>%
  mutate(parcel_id = as.character(parcel_id))
targeted_parcels <- readRDS(TARGETED_PARCEL_FILE) %>%
  transmute(
    parcel_id = as.character(parcel_id),
    current_targeted_units = as.numeric(units_calibrated_targeted),
    current_targeted_method = as.character(
      unit_estimation_method_targeted
    ),
    current_targeted_confidence = as.character(
      unit_estimation_confidence_targeted
    ),
    targeted_unit_adjustment_applied = as.logical(
      targeted_unit_adjustment_applied
    )
  )
parcels <- source_parcels %>%
  left_join(
    targeted_parcels,
    by = "parcel_id",
    relationship = "one-to-one"
  ) %>%
  mutate(
    current_targeted_units = coalesce(
      current_targeted_units,
      as.numeric(units_calibrated),
      0
    )
  )

if (
  nrow(parcels) != nrow(targeted_parcels) ||
    anyDuplicated(parcels$parcel_id) ||
    any(!is.finite(parcels$current_targeted_units)) ||
    any(parcels$current_targeted_units < 0)
) {
  stop("Targeted parcel-unit join failed validation.", call. = FALSE)
}

membership <- readRDS(MEMBERSHIP_FILE) %>%
  mutate(
    parcel_id = as.character(parcel_id),
    project_id = as.character(project_id)
  )
projects <- readRDS(PROJECT_FILE) %>%
  mutate(project_id = as.character(project_id))
predictions <- readRDS(PREDICTION_FILE) %>%
  mutate(project_id = as.character(project_id))
williamson_validation <- readRDS(WILLIAMSON_VALIDATION_FILE) %>%
  mutate(validation_development_id = as.character(
    validation_development_id
  ))
reviewed_groups <- read_csv(
  REVIEWED_GROUP_FILE,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE
)

if (
  nrow(membership) != nrow(parcels) ||
    anyDuplicated(membership$parcel_id) ||
    !setequal(membership$parcel_id, parcels$parcel_id)
) {
  stop("Project membership does not cover the parcel universe once.", call. = FALSE)
}
if (anyDuplicated(predictions$project_id)) {
  stop("Model predictions must be project-unique.", call. = FALSE)
}

project_current_targeted <- membership %>%
  inner_join(
    parcels %>%
      select(parcel_id, current_targeted_units),
    by = "parcel_id",
    relationship = "one-to-one"
  ) %>%
  group_by(project_id) %>%
  summarise(
    current_targeted_project_units = sum(current_targeted_units),
    .groups = "drop"
  )

direct_project_selection <- projects %>%
  filter(
    selected_observed_tier == "strict_direct_project_total",
    is.finite(selected_observed_units),
    selected_observed_units > 0
  ) %>%
  inner_join(
    project_current_targeted,
    by = "project_id",
    relationship = "one-to-one"
  ) %>%
  mutate(
    strict_direct_project_count = TRUE,
    documented_project_count = FALSE,
    shadow_project_units = selected_observed_units,
    shadow_selection_method = "strict_direct_project_total",
    shadow_model_used = FALSE,
    shadow_project_delta = shadow_project_units -
      current_targeted_project_units
  )

expected_direct_projects <- projects %>%
  filter(
    selected_observed_tier == "strict_direct_project_total",
    is.finite(selected_observed_units),
    selected_observed_units > 0
  ) %>%
  nrow()
if (nrow(direct_project_selection) != expected_direct_projects) {
  stop(
    "Not every selected direct project has an allocatable parcel project.",
    call. = FALSE
  )
}

candidate_selection <- predictions %>%
  left_join(
    project_current_targeted,
    by = "project_id",
    relationship = "one-to-one"
  ) %>%
  left_join(
    williamson_validation %>%
      select(
        validation_development_id,
        manual_reported_units,
        validation_confidence,
        validation_status,
        main_area_measure_anomaly,
        companion_account_group,
        cross_county_project
      ),
    by = c("project_id" = "validation_development_id"),
    relationship = "one-to-one"
  ) %>%
  mutate(
    strict_direct_project_count = FALSE,
    documented_project_count = is.finite(manual_reported_units) &
      manual_reported_units > 0,
    williamson_model_eligible =
      str_detect(project_counties, fixed("Williamson")) &
        validation_status ==
          "usable_for_williamson_sensitivity_validation" &
        !coalesce(main_area_measure_anomaly, FALSE) &
        !coalesce(cross_county_project, FALSE),
    shadow_project_units = case_when(
      documented_project_count ~ manual_reported_units,
      williamson_model_eligible ~ recommended_prediction,
      production_prediction_eligible ~ recommended_prediction,
      TRUE ~ current_targeted_project_units
    ),
    shadow_selection_method = case_when(
      documented_project_count ~ "documented_project_count",
      williamson_model_eligible ~
        "williamson_validated_main_area_stratified",
      production_prediction_eligible ~
        "in_domain_main_area_stratified",
      current_targeted_project_units == 0 ~
        "current_targeted_zero_fallback",
      TRUE ~ "current_targeted_fallback"
    ),
    shadow_model_used = shadow_selection_method %in% c(
      "williamson_validated_main_area_stratified",
      "in_domain_main_area_stratified"
    ),
    shadow_project_delta = shadow_project_units -
      current_targeted_project_units
  )

direct_candidate_overlap <- intersect(
  direct_project_selection$project_id,
  candidate_selection$project_id
)
if (length(direct_candidate_overlap) > 0L) {
  stop(
    "Direct-label projects and unresolved model candidates overlap.",
    call. = FALSE
  )
}

project_selection <- bind_rows(
  direct_project_selection,
  candidate_selection
) %>%
  arrange(project_id)

if (
  anyDuplicated(project_selection$project_id) ||
    any(!is.finite(project_selection$shadow_project_units)) ||
    any(project_selection$shadow_project_units < 0)
) {
  stop("Shadow project hierarchy produced invalid counts.", call. = FALSE)
}

reviewed_allocation_overrides <- reviewed_groups %>%
  select(reviewed_group_id, parcel_id, allocation_parcel_id) %>%
  left_join(
    membership %>% select(parcel_id, project_id),
    by = "parcel_id",
    relationship = "one-to-one"
  ) %>%
  group_by(reviewed_group_id) %>%
  summarise(
    project_id = first(project_id),
    project_count = n_distinct(project_id),
    allocation_parcel_id = first(allocation_parcel_id),
    allocation_parcel_count = n_distinct(allocation_parcel_id),
    .groups = "drop"
  ) %>%
  filter(project_id %in% project_selection$project_id) %>%
  transmute(
    project_id,
    allocation_parcel_id,
    allocation_override_reason = "reviewed_wcad_companion_group"
  )

cross_county_allocation_overrides <- projects %>%
  filter(
    project_id %in% project_selection$project_id,
    project_cross_county_address_overlap
  ) %>%
  select(project_id) %>%
  inner_join(
    membership %>% select(project_id, parcel_id),
    by = "project_id",
    relationship = "one-to-many"
  ) %>%
  inner_join(
    parcels %>%
      select(
        parcel_id,
        source_county,
        model_main_area,
        model_improvement_sqft
      ),
    by = "parcel_id",
    relationship = "many-to-one"
  ) %>%
  filter(source_county == "Travis") %>%
  arrange(
    project_id,
    desc(coalesce(model_main_area, model_improvement_sqft, 0))
  ) %>%
  group_by(project_id) %>%
  slice_head(n = 1L) %>%
  ungroup() %>%
  transmute(
    project_id,
    allocation_parcel_id = parcel_id,
    allocation_override_reason =
      "cross_county_overlap_use_travis_representation"
  )

allocation_overrides <- bind_rows(
  reviewed_allocation_overrides,
  cross_county_allocation_overrides
)
if (anyDuplicated(allocation_overrides$project_id)) {
  stop("A project has conflicting parcel-allocation overrides.", call. = FALSE)
}

project_allocation <- membership %>%
  filter(project_id %in% project_selection$project_id) %>%
  inner_join(
    parcels %>%
      select(
        parcel_id,
        current_targeted_units,
        model_main_area,
        model_improvement_sqft,
        county_unit_exclude_from_unit_universe
      ),
    by = "parcel_id",
    relationship = "many-to-one"
  ) %>%
  left_join(
    allocation_overrides,
    by = "project_id",
    relationship = "many-to-one"
  ) %>%
  group_by(project_id) %>%
  mutate(
    allocation_eligible = !coalesce(
      county_unit_exclude_from_unit_universe,
      FALSE
    ),
    has_allocation_override = any(!is.na(allocation_parcel_id)),
    eligible_main_area = if_else(
      allocation_eligible,
      coalesce(model_main_area, 0),
      0
    ),
    eligible_improvement_area = if_else(
      allocation_eligible,
      coalesce(model_improvement_sqft, 0),
      0
    ),
    eligible_current_units = if_else(
      allocation_eligible,
      current_targeted_units,
      0
    ),
    allocation_method = case_when(
      has_allocation_override ~ first(allocation_override_reason),
      sum(eligible_main_area) > 0 ~ "member_main_area_share",
      sum(eligible_improvement_area) > 0 ~
        "member_improvement_area_share",
      sum(eligible_current_units) > 0 ~ "current_targeted_unit_share",
      sum(allocation_eligible) > 0 ~ "equal_eligible_parcel_share",
      TRUE ~ "invalid_no_eligible_parcel"
    ),
    allocation_weight = case_when(
      has_allocation_override ~ as.numeric(
        parcel_id == first(allocation_parcel_id)
      ),
      sum(eligible_main_area) > 0 ~
        eligible_main_area / sum(eligible_main_area),
      sum(eligible_improvement_area) > 0 ~
        eligible_improvement_area / sum(eligible_improvement_area),
      sum(eligible_current_units) > 0 ~
        eligible_current_units / sum(eligible_current_units),
      sum(allocation_eligible) > 0 ~
        as.numeric(allocation_eligible) / sum(allocation_eligible),
      TRUE ~ NA_real_
    )
  ) %>%
  ungroup() %>%
  left_join(
    project_selection %>%
      select(
        project_id,
        shadow_project_units,
        shadow_selection_method
      ),
    by = "project_id",
    relationship = "many-to-one"
  ) %>%
  mutate(
    shadow_allocated_units = shadow_project_units * allocation_weight
  )

allocation_qa <- project_allocation %>%
  group_by(project_id) %>%
  summarise(
    member_parcels = n(),
    allocation_method = first(allocation_method),
    allocation_weight_sum = sum(allocation_weight),
    selected_project_units = first(shadow_project_units),
    allocated_project_units = sum(shadow_allocated_units),
    allocation_difference = allocated_project_units -
      selected_project_units,
    .groups = "drop"
  )
if (
  nrow(allocation_qa) != nrow(project_selection) ||
  any(!is.finite(allocation_qa$allocation_weight_sum)) ||
    any(abs(allocation_qa$allocation_weight_sum - 1) > 1e-9) ||
    any(abs(allocation_qa$allocation_difference) > 1e-6)
) {
  stop("Selected project-to-parcel allocation failed QA.", call. = FALSE)
}

shadow_parcels <- parcels %>%
  left_join(
    membership %>% select(parcel_id, project_id),
    by = "parcel_id",
    relationship = "one-to-one"
  ) %>%
  left_join(
    project_allocation %>%
      select(
        parcel_id,
        shadow_allocated_units,
        allocation_method
      ),
    by = "parcel_id",
    relationship = "one-to-one"
  ) %>%
  left_join(
    project_selection %>%
      select(
        project_id,
        shadow_selection_method,
        shadow_model_used
      ),
    by = "project_id",
    relationship = "many-to-one"
  ) %>%
  mutate(
    shadow_units = coalesce(
      shadow_allocated_units,
      current_targeted_units
    ),
    shadow_unit_delta = shadow_units - current_targeted_units,
    shadow_units_changed = abs(shadow_unit_delta) > 1e-9,
    shadow_selection_method = coalesce(
      shadow_selection_method,
      "unchanged_non_selected_project"
    ),
    shadow_model_used = coalesce(shadow_model_used, FALSE),
    property_units_shadow = shadow_units,
    corporate_units_shadow = if_else(
      as.logical(is_corporate_owned),
      shadow_units,
      0
    )
  )

if (
  nrow(shadow_parcels) != nrow(parcels) ||
    anyDuplicated(shadow_parcels$parcel_id) ||
    any(!is.finite(shadow_parcels$shadow_units)) ||
    any(shadow_parcels$shadow_units < 0)
) {
  stop("Shadow parcel table failed row/count validation.", call. = FALSE)
}

hex_grid <- readRDS(HEX_GRID_FILE)
shadow_points <- shadow_parcels %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
  st_transform(st_crs(hex_grid))
shadow_points_in_grid <- shadow_points %>%
  st_join(
    hex_grid %>% select(hex_id),
    join = st_within,
    left = FALSE
  )

shadow_hex_summary <- shadow_points_in_grid %>%
  st_drop_geometry() %>%
  group_by(hex_id) %>%
  summarise(
    residential_parcels = sum(parcel_count, na.rm = TRUE),
    residential_units = sum(shadow_units, na.rm = TRUE),
    residential_improvement_sqft = sum(improvement_sqft, na.rm = TRUE),
    residential_land_sqft = sum(land_sqft, na.rm = TRUE),
    corporate_owned_parcels = sum(corporate_parcel_count, na.rm = TRUE),
    corporate_owned_units = sum(corporate_units_shadow, na.rm = TRUE),
    corporate_owned_imprv_sqft = sum(
      corporate_improvement_sqft,
      na.rm = TRUE
    ),
    corporate_owner_count = n_distinct(
      owner_names[is_corporate_owned],
      na.rm = TRUE
    ),
    financialized_owner_parcels = sum(
      parcel_count[has_financialized_owner],
      na.rm = TRUE
    ),
    geocoded_parcels = sum(coord_source != "existing_coord", na.rm = TRUE),
    .groups = "drop"
  )

citywide_corporate_units <- sum(
  shadow_hex_summary$corporate_owned_units,
  na.rm = TRUE
)
citywide_corporate_parcels <- sum(
  shadow_hex_summary$corporate_owned_parcels,
  na.rm = TRUE
)

shadow_hex_summary <- shadow_hex_summary %>%
  mutate(
    pct_corporate_parcels = if_else(
      residential_parcels > 0,
      corporate_owned_parcels / residential_parcels * 100,
      NA_real_
    ),
    pct_corporate_units = if_else(
      residential_units > 0,
      corporate_owned_units / residential_units * 100,
      NA_real_
    ),
    pct_corporate_improvement_sqft = if_else(
      residential_improvement_sqft > 0,
      corporate_owned_imprv_sqft /
        residential_improvement_sqft * 100,
      NA_real_
    ),
    pct_financialized_owner_parcels = if_else(
      residential_parcels > 0,
      financialized_owner_parcels / residential_parcels * 100,
      NA_real_
    ),
    corporate_unit_share_city = if (citywide_corporate_units > 0) {
      corporate_owned_units / citywide_corporate_units * 100
    } else {
      rep(NA_real_, n())
    },
    corporate_parcel_share_city = if (citywide_corporate_parcels > 0) {
      corporate_owned_parcels / citywide_corporate_parcels * 100
    } else {
      rep(NA_real_, n())
    }
  )

shadow_hex <- hex_grid %>%
  left_join(shadow_hex_summary, by = "hex_id") %>%
  mutate(
    across(
      c(
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
        corporate_unit_share_city,
        corporate_parcel_share_city
      ),
      ~replace_na(.x, 0)
    ),
    corporate_owned_units_per_km2 = corporate_owned_units / area_km2,
    corporate_owned_parcels_per_km2 = corporate_owned_parcels / area_km2,
    residential_units_per_km2 = residential_units / area_km2,
    residential_parcels_per_km2 = residential_parcels / area_km2,
    investor_owned_units = corporate_owned_units,
    pct_corporate_owned = pct_corporate_units
  )

current_features <- shadow_points_in_grid %>%
  st_drop_geometry() %>%
  group_by(hex_id) %>%
  summarise(
    current_feature_units = sum(current_targeted_units, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  right_join(
    hex_grid %>% st_drop_geometry() %>% select(hex_id),
    by = "hex_id",
    relationship = "one-to-one"
  ) %>%
  mutate(
    current_feature_units = replace_na(current_feature_units, 0),
    current_primary_cluster_eligible =
      current_feature_units >= EWS_CONFIG$minimum_residential_units_for_rates
  )

hex_comparison <- shadow_hex %>%
  st_drop_geometry() %>%
  select(
    hex_id,
    shadow_residential_units = residential_units,
    shadow_corporate_owned_units = corporate_owned_units,
    shadow_pct_corporate_units = pct_corporate_units
  ) %>%
  left_join(
    current_features,
    by = "hex_id",
    relationship = "one-to-one"
  ) %>%
  mutate(
    current_primary_cluster_eligible = coalesce(
      current_primary_cluster_eligible,
      FALSE
    ),
    shadow_primary_cluster_eligible =
      shadow_residential_units >=
        EWS_CONFIG$minimum_residential_units_for_rates,
    residential_unit_delta = shadow_residential_units -
      current_feature_units,
    eligibility_transition = case_when(
      current_primary_cluster_eligible &
        shadow_primary_cluster_eligible ~ "eligible_both",
      !current_primary_cluster_eligible &
        shadow_primary_cluster_eligible ~ "eligible_shadow_only",
      current_primary_cluster_eligible &
        !shadow_primary_cluster_eligible ~ "eligible_current_only",
      TRUE ~ "ineligible_both"
    )
  )

eligibility_summary <- hex_comparison %>%
  group_by(eligibility_transition) %>%
  summarise(
    hexes = n(),
    current_units = sum(current_feature_units, na.rm = TRUE),
    shadow_units = sum(shadow_residential_units, na.rm = TRUE),
    unit_delta = sum(residential_unit_delta, na.rm = TRUE),
    .groups = "drop"
  )

county_comparison <- shadow_parcels %>%
  group_by(source_county) %>%
  summarise(
    parcels = n(),
    current_targeted_units = sum(current_targeted_units),
    shadow_units = sum(shadow_units),
    unit_delta = sum(shadow_unit_delta),
    changed_parcels = sum(shadow_units_changed),
    .groups = "drop"
  )

strategy_summary <- project_selection %>%
  group_by(shadow_selection_method) %>%
  summarise(
    projects = n(),
    current_targeted_units = sum(current_targeted_project_units),
    shadow_units = sum(shadow_project_units),
    unit_delta = sum(shadow_project_delta),
    .groups = "drop"
  )

save_output(
  shadow_parcels,
  file.path(
    OUTPUT_DIR,
    "residential_parcels_unit_shadow_integrated.rds"
  ),
  "shadow integrated residential parcel units"
)
save_output(
  shadow_hex,
  file.path(
    OUTPUT_DIR,
    "corporate_ownership_by_hex_unit_shadow.rds"
  ),
  "shadow unit corporate ownership hex summary"
)
write_csv(
  project_selection,
  file.path(OUTPUT_DIR, "residential_unit_shadow_project_selection.csv")
)
write_csv(
  project_allocation,
  file.path(OUTPUT_DIR, "residential_unit_shadow_parcel_allocation.csv")
)
write_csv(
  allocation_qa,
  file.path(OUTPUT_DIR, "residential_unit_shadow_allocation_qa.csv")
)
write_csv(
  strategy_summary,
  file.path(OUTPUT_DIR, "residential_unit_shadow_strategy_summary.csv")
)
write_csv(
  county_comparison,
  file.path(OUTPUT_DIR, "residential_unit_shadow_county_comparison.csv")
)
write_csv(
  hex_comparison,
  file.path(OUTPUT_DIR, "residential_unit_shadow_hex_comparison.csv")
)
write_csv(
  eligibility_summary,
  file.path(
    OUTPUT_DIR,
    "residential_unit_shadow_eligibility_summary.csv"
  )
)

print_progress("Shadow selection by method:")
print(strategy_summary)
print_progress("Hex eligibility transitions:")
print(eligibility_summary)
print_progress(
  paste0(
    "Shadow integration changes ",
    scales::comma(sum(shadow_parcels$shadow_units_changed)),
    " parcel rows and ",
    scales::comma(
      sum(
        abs(hex_comparison$residential_unit_delta) > 1e-9,
        na.rm = TRUE
      )
    ),
    " hex totals."
  )
)
print_progress(
  paste0(
    "Saved shadow corporate summary for feature regeneration. ",
    "Canonical parcel and hex outputs remain unchanged."
  )
)
