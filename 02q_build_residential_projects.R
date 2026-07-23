################################################################################
# 02q - Build Residential Projects and Unit-Model Training Tables
################################################################################
#
# Groups parcel/accounts conservatively using shared high-confidence source
# records and exact multifamily addresses. It selects strict labels only where
# direct sources agree, retains URO estimates for sensitivity testing, and
# writes unresolved multifamily projects as future model candidates.
#
# This script creates shadow outputs only. It does not modify 02d, 02e, or the
# parcel unit field used by feature engineering and clustering.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
})

source(here::here("R", "utils.R"))
source(here::here("R", "unit_count_helpers.R"))

print_header("02q - BUILD RESIDENTIAL PROJECTS")

OUTPUT_DIR <- here::here("output")
PARCEL_FILE <- file.path(
  OUTPUT_DIR,
  "residential_parcels_unit_source_attributes.rds"
)
SOURCE_RECORD_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_source_records.rds"
)
SOURCE_LINK_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_source_parcel_links.rds"
)

required_files <- c(PARCEL_FILE, SOURCE_RECORD_FILE, SOURCE_LINK_FILE)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Run 02p_prepare_unit_sources.R before 02q. Missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

parcels <- readRDS(PARCEL_FILE) %>%
  mutate(
    parcel_id = as.character(parcel_id),
    source_county = as.character(source_county),
    project_address_key = if_else(
      !is.na(parcel_address_key),
      paste(
        source_county,
        parcel_address_key,
        coalesce(parcel_zip5, ""),
        sep = ":"
      ),
      NA_character_
    ),
    mf_project_signal = is_multifamily_like |
      appraisal_state_code %in% c("A4", "B1", "B2", "B3", "B4") |
      has_mf_zoning |
      replace_na(county_model_candidate_signal, FALSE)
  )

source_records <- readRDS(SOURCE_RECORD_FILE)
source_links <- readRDS(SOURCE_LINK_FILE) %>%
  mutate(parcel_id = as.character(parcel_id))

source_group_links <- source_links %>%
  filter(match_confidence == "high") %>%
  add_count(source_record_id, name = "linked_parcel_count") %>%
  filter(linked_parcel_count > 1L) %>%
  transmute(
    link_group_id = paste0("source:", source_record_id),
    parcel_id,
    grouping_method = "shared_high_confidence_source"
  )

address_group_links <- parcels %>%
  filter(!is.na(project_address_key)) %>%
  group_by(project_address_key) %>%
  mutate(
    address_parcel_count = n(),
    address_has_mf_signal = any(mf_project_signal, na.rm = TRUE),
    address_lat_span_m = (max(lat, na.rm = TRUE) - min(lat, na.rm = TRUE)) *
      111320,
    address_lon_span_m = (max(lon, na.rm = TRUE) - min(lon, na.rm = TRUE)) *
      111320 * cos(mean(lat, na.rm = TRUE) * pi / 180)
  ) %>%
  ungroup() %>%
  filter(
    address_parcel_count > 1L,
    address_has_mf_signal,
    address_parcel_count <= 2000L,
    is.finite(address_lat_span_m),
    is.finite(address_lon_span_m),
    pmax(address_lat_span_m, address_lon_span_m) <= 250
  ) %>%
  transmute(
    link_group_id = paste0("address:", project_address_key),
    parcel_id,
    grouping_method = "exact_multifamily_address"
  )

component_links <- bind_rows(source_group_links, address_group_links) %>%
  distinct(link_group_id, parcel_id, .keep_all = TRUE)

components <- unit_connected_components(parcels$parcel_id, component_links) %>%
  mutate(project_id = paste0("project:", component_key))

membership_methods <- component_links %>%
  inner_join(
    components %>% select(parcel_id, project_id),
    by = "parcel_id"
  ) %>%
  group_by(project_id) %>%
  summarise(
    project_grouping_methods = str_c(
      sort(unique(grouping_method)),
      collapse = " | "
    ),
    project_link_group_count = n_distinct(link_group_id),
    .groups = "drop"
  )

project_membership <- components %>%
  select(parcel_id, project_id) %>%
  left_join(membership_methods, by = "project_id") %>%
  mutate(
    project_grouping_methods = coalesce(
      project_grouping_methods,
      "single_parcel"
    ),
    project_link_group_count = replace_na(project_link_group_count, 0L)
  )

project_parcels <- parcels %>%
  inner_join(project_membership, by = "parcel_id")

projects <- project_parcels %>%
  group_by(project_id) %>%
  summarise(
    project_counties = str_c(
      sort(unique(na.omit(source_county))),
      collapse = " | "
    ),
    project_county_count = n_distinct(source_county, na.rm = TRUE),
    project_cross_county = project_county_count > 1L,
    source_county = unit_mode(source_county),
    project_parcel_count = n_distinct(parcel_id),
    project_excluded_unit_parcel_count = sum(
      county_unit_exclude_from_unit_universe,
      na.rm = TRUE
    ),
    project_required_unit_parcel_count = sum(
      !county_unit_exclude_from_unit_universe,
      na.rm = TRUE
    ),
    project_address_count = n_distinct(parcel_address_key, na.rm = TRUE),
    project_grouping_methods = first(project_grouping_methods),
    project_link_group_count = first(project_link_group_count),
    project_improvement_sqft = sum(model_improvement_sqft, na.rm = TRUE),
    project_main_area = sum(model_main_area, na.rm = TRUE),
    project_land_sqft = sum(land_sqft, na.rm = TRUE),
    project_floor_area_ratio = if_else(
      project_land_sqft > 0,
      project_improvement_sqft / project_land_sqft,
      NA_real_
    ),
    project_year_built = unit_weighted_mean(
      model_year_built,
      pmax(model_improvement_sqft, 1)
    ),
    project_effective_year_built = unit_weighted_mean(
      tcad_effective_year_built,
      pmax(model_improvement_sqft, 1)
    ),
    project_stories = unit_weighted_mean(
      tcad_stories,
      pmax(model_improvement_sqft, 1)
    ),
    project_max_stories = {
      observed_stories <- tcad_stories[is.finite(tcad_stories)]
      if (length(observed_stories) > 0L) {
        max(observed_stories)
      } else {
        NA_real_
      }
    },
    project_imprv_type = unit_mode(tcad_imprv_type),
    project_imprv_class = unit_mode(tcad_imprv_class),
    project_imprv_quality = unit_mode(tcad_imprv_quality),
    project_imprv_condition = unit_mode(tcad_imprv_condition),
    project_state_codes = str_c(
      sort(unique(na.omit(appraisal_state_code))),
      collapse = " | "
    ),
    project_zoning = unit_mode(propertyChar_zoning),
    project_has_mf_zoning = any(has_mf_zoning, na.rm = TRUE),
    project_has_commercial_mixed_zoning = any(
      has_commercial_mixed_zoning,
      na.rm = TRUE
    ),
    project_is_multifamily_like = any(mf_project_signal, na.rm = TRUE),
    project_county_model_candidate = any(
      county_model_candidate_signal,
      na.rm = TRUE
    ),
    project_county_review = any(
      !is.na(county_unit_review_reason),
      na.rm = TRUE
    ),
    project_county_review_reasons = str_c(
      sort(unique(na.omit(county_unit_review_reason))),
      collapse = " | "
    ),
    project_county_evidence_classes = str_c(
      sort(unique(na.omit(county_unit_evidence_class))),
      collapse = " | "
    ),
    project_wcad_property_type = unit_mode(wcad_property_type),
    project_wcad_apartment_signal = any(
      wcad_apartment_signal,
      na.rm = TRUE
    ),
    project_condo_account_count = sum(
      appraisal_state_code == "A4",
      na.rm = TRUE
    ),
    project_b1_parcel_count = sum(
      appraisal_state_code == "B1",
      na.rm = TRUE
    ),
    project_lat = unit_weighted_mean(lat, pmax(model_improvement_sqft, 1)),
    project_lon = unit_weighted_mean(lon, pmax(model_improvement_sqft, 1)),
    current_primary_units = sum(units_calibrated, na.rm = TRUE),
    current_conservative_units = sum(
      units_calibrated_conservative,
      na.rm = TRUE
    ),
    current_floor_area_estimate = any(
      str_detect(unit_estimation_method, "costar_sqft"),
      na.rm = TRUE
    ),
    current_needs_multifamily_estimate = any(
      needs_multifamily_estimate,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  mutate(
    project_improvement_sqft = na_if(project_improvement_sqft, 0),
    project_main_area = na_if(project_main_area, 0),
    project_land_sqft = na_if(project_land_sqft, 0)
  )

project_county_membership <- project_parcels %>%
  distinct(project_id, source_county)

project_source_links <- source_links %>%
  inner_join(
    project_membership %>% select(parcel_id, project_id),
    by = "parcel_id"
  ) %>%
  inner_join(source_records, by = "source_record_id") %>%
  group_by(project_id, source_record_id) %>%
  summarise(
    source_name = first(source_name),
    source_priority = first(source_priority),
    source_tier = first(source_tier),
    unit_count_kind = first(unit_count_kind),
    source_unit_count = first(source_unit_count),
    source_consistent = first(source_consistent),
    use_as_strict_model_label = first(use_as_strict_model_label),
    use_as_deterministic_count = first(use_as_deterministic_count),
    use_as_rule_based_count = first(use_as_rule_based_count),
    use_as_sensitivity_label = first(use_as_sensitivity_label),
    linked_project_parcels = n_distinct(parcel_id),
    best_match_confidence = if_else(
      any(match_confidence == "high"),
      "high",
      "review"
    ),
    match_methods = str_c(sort(unique(match_method)), collapse = " | "),
    .groups = "drop"
  )

project_source_estimates <- project_source_links %>%
  group_by(project_id, source_name) %>%
  summarise(
    source_priority = min(source_priority),
    source_tier = first(source_tier),
    unit_count_kind = first(unit_count_kind),
    source_record_count = n_distinct(source_record_id),
    source_project_units = case_when(
      first(source_name) == "tcad_explicit_units" &
        source_record_count > 1L ~ NA_real_,
      TRUE ~ sum(source_unit_count, na.rm = TRUE)
    ),
    source_record_units_sum = sum(source_unit_count, na.rm = TRUE),
    source_record_units_min = min(source_unit_count, na.rm = TRUE),
    source_record_units_max = max(source_unit_count, na.rm = TRUE),
    source_records_consistent = all(source_consistent),
    source_matches_high_confidence = all(best_match_confidence == "high"),
    use_as_strict_model_label = all(use_as_strict_model_label) &
      source_matches_high_confidence &
      !is.na(source_project_units),
    use_as_deterministic_count = all(use_as_deterministic_count) &
      source_matches_high_confidence,
    use_as_rule_based_count = all(use_as_rule_based_count) &
      source_matches_high_confidence,
    use_as_sensitivity_label = all(use_as_sensitivity_label) &
      source_matches_high_confidence,
    .groups = "drop"
  )

deterministic_counts <- project_source_estimates %>%
  filter(use_as_deterministic_count, source_project_units > 0) %>%
  group_by(project_id) %>%
  summarise(
    deterministic_record_count = sum(source_record_count),
    deterministic_units = sum(source_project_units),
    deterministic_source_names = str_c(
      sort(unique(source_name)),
      collapse = " | "
    ),
    .groups = "drop"
  )

rule_based_counts <- project_source_estimates %>%
  filter(use_as_rule_based_count, source_project_units > 0) %>%
  group_by(project_id) %>%
  summarise(
    rule_based_record_count = sum(source_record_count),
    rule_based_units = sum(source_project_units),
    rule_based_source_names = str_c(
      sort(unique(source_name)),
      collapse = " | "
    ),
    .groups = "drop"
  )

strict_comparison <- project_source_estimates %>%
  filter(use_as_strict_model_label, source_project_units > 0) %>%
  group_by(project_id) %>%
  arrange(source_priority, .by_group = TRUE) %>%
  summarise(
    strict_source_count = n(),
    strict_source_names = str_c(source_name, collapse = " | "),
    strict_source_min_units = min(source_project_units),
    strict_source_max_units = max(source_project_units),
    strict_source_relative_spread = unit_relative_spread(
      source_project_units
    ),
    strict_direct_sources_agree = strict_source_count == 1L |
      strict_source_relative_spread <= 0.20,
    preferred_strict_source = first(source_name),
    preferred_strict_units = first(source_project_units),
    .groups = "drop"
  ) %>%
  left_join(deterministic_counts, by = "project_id") %>%
  left_join(
    projects %>%
      select(project_id, project_required_unit_parcel_count),
    by = "project_id"
  ) %>%
  mutate(
    deterministic_count_covers_project = !is.na(
      deterministic_record_count
    ) & deterministic_record_count == project_required_unit_parcel_count,
    strict_deterministic_relative_difference = if_else(
      deterministic_count_covers_project,
      abs(preferred_strict_units - deterministic_units) /
        pmax(
          (preferred_strict_units + deterministic_units) / 2,
          1
        ),
      NA_real_
    ),
    strict_sources_agree = strict_direct_sources_agree &
      (
        !deterministic_count_covers_project |
          strict_deterministic_relative_difference <= 0.20
      ),
    selected_strict_source = if_else(
      strict_sources_agree,
      preferred_strict_source,
      NA_character_
    ),
    selected_strict_units = if_else(
      strict_sources_agree,
      preferred_strict_units,
      NA_real_
    )
  ) %>%
  select(
    -deterministic_record_count,
    -deterministic_units,
    -deterministic_source_names,
    -project_required_unit_parcel_count,
    -deterministic_count_covers_project
  )

uro_sensitivity <- project_source_estimates %>%
  filter(use_as_sensitivity_label, source_project_units > 0) %>%
  group_by(project_id) %>%
  summarise(
    uro_source_record_count = sum(source_record_count),
    uro_sensitivity_units = sum(source_project_units),
    .groups = "drop"
  )

projects <- projects %>%
  left_join(strict_comparison, by = "project_id") %>%
  left_join(deterministic_counts, by = "project_id") %>%
  left_join(rule_based_counts, by = "project_id") %>%
  left_join(uro_sensitivity, by = "project_id") %>%
  mutate(
    strict_source_count = replace_na(strict_source_count, 0L),
    strict_sources_agree = replace_na(strict_sources_agree, FALSE),
    strict_direct_sources_agree = replace_na(
      strict_direct_sources_agree,
      FALSE
    ),
    deterministic_record_count = replace_na(
      deterministic_record_count,
      0L
    ),
    rule_based_record_count = replace_na(rule_based_record_count, 0L),
    uro_source_record_count = replace_na(uro_source_record_count, 0L),
    deterministic_count_covers_project = deterministic_record_count ==
      project_required_unit_parcel_count,
    rule_based_count_covers_project = rule_based_record_count ==
      project_required_unit_parcel_count,
    selected_deterministic_units = if_else(
      deterministic_count_covers_project &
        (
          strict_source_count == 0L |
            (
              strict_sources_agree &
                strict_deterministic_relative_difference <= 0.20
            )
        ),
      deterministic_units,
      NA_real_
    ),
    selected_rule_based_units = if_else(
      rule_based_count_covers_project &
        deterministic_record_count == 0L &
        strict_source_count == 0L,
      rule_based_units,
      NA_real_
    ),
    selected_observed_units = coalesce(
      selected_deterministic_units,
      selected_strict_units,
      selected_rule_based_units
    ),
    selected_observed_source = case_when(
      !is.na(selected_deterministic_units) ~ deterministic_source_names,
      !is.na(selected_strict_units) ~ selected_strict_source,
      !is.na(selected_rule_based_units) ~ rule_based_source_names,
      TRUE ~ NA_character_
    ),
    selected_observed_tier = case_when(
      !is.na(selected_deterministic_units) ~
        "deterministic_appraisal_accounts",
      !is.na(selected_strict_units) ~ "strict_direct_project_total",
      !is.na(selected_rule_based_units) ~
        "rule_based_single_unit_assumption",
      TRUE ~ NA_character_
    ),
    strict_label_sqft_per_unit = if_else(
      selected_strict_units > 0,
      project_improvement_sqft / selected_strict_units,
      NA_real_
    ),
    training_label_eligible = !is.na(selected_strict_units) &
      strict_sources_agree &
      !deterministic_count_covers_project &
      project_is_multifamily_like &
      project_improvement_sqft > 0 &
      selected_strict_units >= 5 &
      strict_label_sqft_per_unit >= 200 &
      strict_label_sqft_per_unit <= 3000,
    model_candidate = project_is_multifamily_like &
      project_improvement_sqft > 0 &
      is.na(selected_observed_units) &
      (
        current_needs_multifamily_estimate |
          current_floor_area_estimate |
          project_county_model_candidate
      ),
    source_conflict_requires_review = strict_source_count > 0L &
      !strict_sources_agree
  )

training_table <- projects %>%
  filter(training_label_eligible) %>%
  transmute(
    project_id,
    unit_count = selected_strict_units,
    label_source = selected_strict_source,
    label_source_count = strict_source_count,
    label_relative_spread = strict_source_relative_spread,
    source_county,
    project_counties,
    project_cross_county,
    project_improvement_sqft,
    project_main_area,
    project_land_sqft,
    project_floor_area_ratio,
    project_year_built,
    project_effective_year_built,
    project_stories,
    project_max_stories,
    project_imprv_type,
    project_imprv_class,
    project_imprv_quality,
    project_imprv_condition,
    project_zoning,
    project_has_mf_zoning,
    project_has_commercial_mixed_zoning,
    project_parcel_count,
    project_address_count,
    project_condo_account_count,
    project_b1_parcel_count,
    project_lat,
    project_lon,
    sqft_per_unit = strict_label_sqft_per_unit,
    uro_sensitivity_units
  )

model_candidates <- projects %>%
  filter(model_candidate)

source_conflicts <- projects %>%
  filter(source_conflict_requires_review) %>%
  select(
    project_id,
    source_county,
    project_counties,
    project_cross_county,
    project_parcel_count,
    project_required_unit_parcel_count,
    project_excluded_unit_parcel_count,
    project_grouping_methods,
    strict_source_names,
    strict_source_min_units,
    strict_source_max_units,
    strict_source_relative_spread,
    preferred_strict_source,
    preferred_strict_units,
    deterministic_source_names,
    deterministic_units,
    strict_deterministic_relative_difference,
    project_county_evidence_classes,
    project_county_review_reasons,
    project_improvement_sqft,
    current_primary_units
  )

training_county_qa <- parcels %>%
  distinct(source_county) %>%
  left_join(
    training_table %>%
      select(project_id) %>%
      inner_join(project_county_membership, by = "project_id") %>%
      count(source_county, name = "value"),
    by = "source_county"
  ) %>%
  transmute(
    qa_section = "training_labels_by_county",
    metric = source_county,
    value = replace_na(value, 0L),
    note = if_else(
      value == 0L,
      "No strict project-level labels; do not estimate a county effect.",
      NA_character_
    )
  )

project_qa <- bind_rows(
  tibble(
    qa_section = "project_totals",
    metric = c(
      "parcel_rows",
      "projects",
      "multi_parcel_projects",
      "cross_county_projects",
      "excluded_non_unit_reference_parcels",
      "multifamily_like_projects",
      "strict_labeled_projects",
      "training_eligible_projects",
      "source_conflict_projects",
      "deterministic_only_projects",
      "rule_based_only_projects",
      "uro_sensitivity_projects",
      "county_model_candidate_projects",
      "county_source_review_projects",
      "unresolved_model_candidates"
    ),
    value = c(
      nrow(project_membership),
      nrow(projects),
      sum(projects$project_parcel_count > 1L),
      sum(projects$project_cross_county),
      sum(projects$project_excluded_unit_parcel_count),
      sum(projects$project_is_multifamily_like),
      sum(!is.na(projects$selected_strict_units)),
      nrow(training_table),
      sum(projects$source_conflict_requires_review),
      sum(
        is.na(projects$selected_strict_units) &
          !is.na(projects$selected_deterministic_units)
      ),
      sum(!is.na(projects$selected_rule_based_units)),
      sum(!is.na(projects$uro_sensitivity_units)),
      sum(projects$project_county_model_candidate),
      sum(projects$project_county_review),
      nrow(model_candidates)
    ),
    note = NA_character_
  ),
  training_table %>%
    count(label_source, name = "value") %>%
    transmute(
      qa_section = "training_labels_by_source",
      metric = label_source,
      value,
      note = NA_character_
    ),
  training_county_qa,
  model_candidates %>%
    select(project_id) %>%
    inner_join(project_county_membership, by = "project_id") %>%
    count(source_county, name = "value") %>%
    transmute(
      qa_section = "model_candidates_by_county",
      metric = source_county,
      value,
      note = "Counts county involvement; cross-county projects appear once per county."
    ),
  projects %>%
    filter(source_conflict_requires_review) %>%
    transmute(
      qa_section = "source_conflict",
      metric = project_id,
      value = strict_source_relative_spread,
      note = paste(
        strict_source_names,
        paste0(
          "range=",
          round(strict_source_min_units),
          "-",
          round(strict_source_max_units)
        ),
        sep = "; "
      )
  )
)

if (
  nrow(project_membership) != nrow(parcels) ||
    n_distinct(project_membership$parcel_id) != nrow(parcels)
) {
  stop("Project membership must contain each parcel exactly once.", call. = FALSE)
}
if (
  any(
    projects$project_required_unit_parcel_count +
      projects$project_excluded_unit_parcel_count !=
      projects$project_parcel_count
  )
) {
  stop("Project unit-bearing and excluded parcel counts do not balance.", call. = FALSE)
}
if (
  any(
    is.na(projects$selected_observed_tier) !=
      is.na(projects$selected_observed_units)
  )
) {
  stop("Selected unit counts and hierarchy tiers are inconsistent.", call. = FALSE)
}

save_output(
  project_membership,
  file.path(OUTPUT_DIR, "residential_unit_project_membership.rds"),
  "residential unit project membership"
)
save_output(
  projects,
  file.path(OUTPUT_DIR, "residential_unit_projects.rds"),
  "residential unit projects"
)
save_output(
  training_table,
  file.path(OUTPUT_DIR, "residential_unit_training_table.rds"),
  "residential unit model training table"
)
save_output(
  model_candidates,
  file.path(OUTPUT_DIR, "residential_unit_model_candidates.rds"),
  "unresolved residential unit model candidates"
)

write_csv(
  project_membership,
  file.path(OUTPUT_DIR, "residential_unit_project_membership.csv")
)
write_csv(
  projects,
  file.path(OUTPUT_DIR, "residential_unit_projects.csv")
)
write_csv(
  training_table,
  file.path(OUTPUT_DIR, "residential_unit_training_table.csv")
)
write_csv(
  model_candidates,
  file.path(OUTPUT_DIR, "residential_unit_model_candidates.csv")
)
write_csv(
  project_source_estimates,
  file.path(OUTPUT_DIR, "residential_unit_project_source_comparison.csv")
)
write_csv(
  source_conflicts,
  file.path(OUTPUT_DIR, "residential_unit_source_conflicts.csv")
)
write_csv(
  projects %>% filter(project_cross_county),
  file.path(OUTPUT_DIR, "residential_unit_cross_county_projects.csv")
)
write_csv(
  project_qa,
  file.path(OUTPUT_DIR, "residential_unit_project_qa.csv")
)

print_progress(
  paste0(
    "Built ",
    scales::comma(nrow(projects)),
    " projects from ",
    scales::comma(nrow(project_membership)),
    " parcels."
  )
)
print_progress(
  paste0(
    "Training projects: ",
    scales::comma(nrow(training_table)),
    "; unresolved model candidates: ",
    scales::comma(nrow(model_candidates)),
    "; direct-source conflicts held out: ",
    scales::comma(sum(projects$source_conflict_requires_review))
  )
)
