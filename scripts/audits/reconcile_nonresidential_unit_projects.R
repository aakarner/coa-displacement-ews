################################################################################
# Reconcile Modeled Unit Projects on City Nonresidential Land
################################################################################
#
# The City land-use comparison identifies projects that entered the EWS
# multi-unit modeling pathway but occupy land the City classifies as
# nonresidential or group quarters. This audit separates projects supported by
# an appraisal multi-unit code from projects admitted only because zoning or a
# broad parcel classification allowed residential use. It verifies the
# production exclusions applied by the promoted unit-surface target.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(scales)
  library(stringr)
})

source(here::here("R", "utils.R"))

print_header("RECONCILE MODELED UNITS ON NONRESIDENTIAL LAND")

OUTPUT_DIR <- here::here("output")
FIGURES_DIR <- here::here("figures")
PROJECT_AUDIT_FILE <- file.path(
  OUTPUT_DIR,
  "land_use_unit_classification_project_audit.csv"
)
PARCEL_AUDIT_FILE <- file.path(
  OUTPUT_DIR,
  "land_use_unit_classification_parcel_audit.csv"
)
PROMOTED_PARCEL_FILE <- file.path(
  OUTPUT_DIR,
  "residential_parcels_unit_promoted.rds"
)
MEMBERSHIP_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_project_membership.rds"
)
PROJECT_SELECTION_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_shadow_project_selection.csv"
)

required_files <- c(
  PROJECT_AUDIT_FILE,
  PARCEL_AUDIT_FILE,
  PROMOTED_PARCEL_FILE,
  MEMBERSHIP_FILE,
  PROJECT_SELECTION_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Nonresidential unit reconciliation is missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

collapse_values <- function(x, limit = Inf) {
  values <- sort(unique(na.omit(str_trim(as.character(x)))))
  values <- values[values != ""]
  if (length(values) == 0L) {
    return(NA_character_)
  }
  if (is.finite(limit) && length(values) > limit) {
    values <- c(
      values[seq_len(limit)],
      paste0("+", length(values) - limit, " more")
    )
  }
  str_c(values, collapse = " | ")
}

project_audit <- read_csv(
  PROJECT_AUDIT_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_guess(), project_id = col_character())
)

candidate_project_ids <- project_audit %>%
  filter(unit_count_treatment %in% c("Modeled estimate", "Fallback estimate")) %>%
  pull(project_id)

parcel_audit <- read_csv(
  PARCEL_AUDIT_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_character())
)

membership <- readRDS(MEMBERSHIP_FILE) %>%
  transmute(
    parcel_id = as.character(parcel_id),
    project_id = as.character(project_id)
  )

parcels <- readRDS(PROMOTED_PARCEL_FILE) %>%
  mutate(parcel_id = as.character(parcel_id)) %>%
  left_join(membership, by = "parcel_id", relationship = "one-to-one") %>%
  filter(project_id %in% candidate_project_ids)

project_selection <- read_csv(
  PROJECT_SELECTION_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_guess(), project_id = col_character())
) %>%
  filter(project_id %in% candidate_project_ids) %>%
  select(
    project_id,
    project_model_floor_area,
    project_model_floor_area_source,
    project_year_built,
    project_stories,
    current_targeted_project_units,
    shadow_project_units,
    prediction_lower_80,
    prediction_upper_80,
    prediction_caution,
    model_out_of_domain,
    production_prediction_eligible
  )

if (
  anyDuplicated(membership$parcel_id) ||
    anyDuplicated(parcels$parcel_id) ||
    anyDuplicated(project_selection$project_id)
) {
  stop("Reconciliation inputs contain duplicate identifiers.", call. = FALSE)
}

project_parcel_evidence <- parcels %>%
  group_by(project_id) %>%
  summarise(
    appraisal_parcel_rows = n(),
    appraisal_addresses = collapse_values(situs_address, limit = 3),
    appraisal_state_codes = collapse_values(propertyProf_imprvStateCd),
    appraisal_land_codes = collapse_values(propertyProf_landStateCd),
    appraisal_zoning = collapse_values(propertyChar_zoning, limit = 5),
    county_evidence_classes = collapse_values(county_unit_evidence_class),
    appraisal_multiunit_code_signal = any(
      propertyProf_imprvStateCd %in% c("A4", "B1", "B2", "B3", "B4"),
      na.rm = TRUE
    ) | any(coalesce(county_model_candidate_signal, FALSE)),
    zoning_or_broad_residential_signal = any(
      coalesce(has_mf_zoning, FALSE) |
        str_detect(
          str_to_upper(coalesce(propertyChar_zoning, "")),
          "CONDO|PUD|MU"
        ),
      na.rm = TRUE
    ),
    appraisal_nonresidential_code_signal = any(
      str_detect(
        str_to_upper(coalesce(propertyProf_imprvStateCd, "")),
        "^F"
      ),
      na.rm = TRUE
    ),
    raw_parcel_unit_field_sum = sum(
      if_else(is.finite(as.numeric(units_raw)), as.numeric(units_raw), 0),
      na.rm = TRUE
    ),
    promoted_project_units_all_jurisdictions = sum(
      as.numeric(promoted_units),
      na.rm = TRUE
    ),
    .groups = "drop"
  )

city_match_evidence <- parcel_audit %>%
  filter(
    in_austin_full_purpose == "TRUE",
    project_id %in% candidate_project_ids
  ) %>%
  group_by(project_id) %>%
  summarise(
    city_match_methods = collapse_values(city_match_method),
    identifier_matched_parcels = sum(
      city_match_method == "identifier",
      na.rm = TRUE
    ),
    spatially_matched_parcels = sum(
      city_match_method == "spatial_point_within",
      na.rm = TRUE
    ),
    .groups = "drop"
  )

project_evidence <- project_audit %>%
  left_join(
    project_parcel_evidence,
    by = "project_id",
    relationship = "one-to-one"
  ) %>%
  left_join(city_match_evidence, by = "project_id", relationship = "one-to-one") %>%
  left_join(project_selection, by = "project_id", relationship = "one-to-one") %>%
  mutate(
    candidate_evidence_basis = case_when(
      appraisal_multiunit_code_signal ~
        "Current appraisal multi-unit code or county classifier",
      zoning_or_broad_residential_signal ~
        "Zoning or broad parcel classification only",
      TRUE ~ "Unresolved candidate signal"
    )
  )

focus_projects <- project_evidence %>%
  filter(
    comparison_status ==
      "Review: EWS core multi-unit / City nonresidential",
    unit_count_treatment %in% c("Modeled estimate", "Fallback estimate")
  ) %>%
  mutate(
    reconciliation_finding = case_when(
      candidate_evidence_basis ==
        "Zoning or broad parcel classification only" ~
        paste(
          "Probable false positive: zoning indicates what may be built,",
          "not current residential use"
        ),
      candidate_evidence_basis ==
        "Current appraisal multi-unit code or county classifier" ~
        "Conflicting current-use evidence: reconcile appraisal and City records",
      TRUE ~ "Unresolved candidate evidence: manual review required"
    ),
    recommended_next_action = case_when(
      candidate_evidence_basis ==
        "Zoning or broad parcel classification only" ~
        paste(
          "Excluded from the promoted surface unless another current",
          "residential-use source is later found"
        ),
      candidate_evidence_basis ==
        "Current appraisal multi-unit code or county classifier" ~
        paste(
          "Retain provisionally and verify City inventory vintage,",
          "mixed use, or redevelopment"
        ),
      TRUE ~ "Do not promote a revised count until manually resolved"
    ),
    provisional_exclusion = candidate_evidence_basis ==
      "Zoning or broad parcel classification only",
    review_priority = case_when(
      provisional_exclusion &
        project_pre_validation_units_inside_city >= 100 ~ "1 - High",
      provisional_exclusion ~ "2 - Medium",
      project_pre_validation_units_inside_city >= 100 ~ "2 - Medium",
      TRUE ~ "3 - Lower"
    )
  ) %>%
  arrange(
    review_priority,
    desc(project_pre_validation_units_inside_city),
    project_id
  ) %>%
  mutate(review_rank = row_number()) %>%
  select(
    review_rank,
    review_priority,
    project_id,
    sample_address,
    project_counties,
    project_parcel_count,
    project_grouping_methods,
    project_units_inside_city,
    project_pre_validation_units_inside_city,
    project_land_use_excluded_units_inside_city,
    unit_count_treatment,
    shadow_selection_method,
    candidate_evidence_basis,
    reconciliation_finding,
    recommended_next_action,
    provisional_exclusion,
    city_land_use_labels,
    city_match_methods,
    identifier_matched_parcels,
    spatially_matched_parcels,
    appraisal_addresses,
    appraisal_state_codes,
    appraisal_land_codes,
    appraisal_zoning,
    county_evidence_classes,
    appraisal_multiunit_code_signal,
    zoning_or_broad_residential_signal,
    appraisal_nonresidential_code_signal,
    selected_observed_units,
    selected_observed_source,
    raw_parcel_unit_field_sum,
    current_targeted_project_units,
    shadow_project_units,
    promoted_project_units_all_jurisdictions,
    project_model_floor_area,
    project_model_floor_area_source,
    project_year_built,
    project_stories,
    prediction_lower_80,
    prediction_upper_80,
    prediction_caution,
    model_out_of_domain,
    production_prediction_eligible
  )

if (
  nrow(focus_projects) == 0L ||
    any(is.na(focus_projects$candidate_evidence_basis)) ||
    any(
      !focus_projects$provisional_exclusion &
        !focus_projects$appraisal_multiunit_code_signal
    ) ||
    any(
      focus_projects$provisional_exclusion &
        focus_projects$project_land_use_excluded_units_inside_city <= 0
    ) ||
    any(!is.na(focus_projects$selected_observed_units))
) {
  stop(
    "Focused reconciliation did not preserve the expected project logic.",
    call. = FALSE
  )
}

reconciliation_summary <- focus_projects %>%
  group_by(
    candidate_evidence_basis,
    reconciliation_finding,
    recommended_next_action,
    provisional_exclusion,
    unit_count_treatment
  ) %>%
  summarise(
    projects = n(),
    units = sum(project_pre_validation_units_inside_city),
    share_of_review_projects_pct = 100 * n() / nrow(focus_projects),
    share_of_review_units_pct =
      100 * sum(project_pre_validation_units_inside_city) /
        sum(focus_projects$project_pre_validation_units_inside_city),
    .groups = "drop"
  ) %>%
  arrange(desc(provisional_exclusion), desc(units))

candidate_scope <- project_evidence %>%
  filter(unit_count_treatment %in% c("Modeled estimate", "Fallback estimate")) %>%
  group_by(candidate_evidence_basis, city_project_class) %>%
  summarise(
    projects = n(),
    units = sum(project_pre_validation_units_inside_city),
    .groups = "drop"
  ) %>%
  arrange(candidate_evidence_basis, desc(units))

city_promoted_units <- sum(project_audit$project_units_inside_city, na.rm = TRUE)
city_pre_validation_units <- sum(
  project_audit$project_pre_validation_units_inside_city,
  na.rm = TRUE
)
core_multiunit_units <- sum(
  project_audit$project_pre_validation_units_inside_city[
    project_audit$core_multiunit_project %in% TRUE
  ],
  na.rm = TRUE
)
focus_units <- sum(focus_projects$project_pre_validation_units_inside_city)
provisional_units <- sum(
  focus_projects$project_land_use_excluded_units_inside_city[
    focus_projects$provisional_exclusion
  ]
)

impact_summary <- tibble(
  metric = c(
    "focused_review_projects",
    "focused_review_units",
    "production_exclusion_projects",
    "production_exclusion_units",
    "production_exclusion_share_of_focused_units_pct",
    "production_exclusion_share_of_city_pre_validation_units_pct",
    "production_exclusion_share_of_core_multiunit_units_pct",
    "city_promoted_units_after_land_use_validation"
  ),
  value = c(
    nrow(focus_projects),
    focus_units,
    sum(focus_projects$provisional_exclusion),
    provisional_units,
    100 * provisional_units / focus_units,
    100 * provisional_units / city_pre_validation_units,
    100 * provisional_units / core_multiunit_units,
    city_promoted_units
  ),
  unit = c(
    "projects",
    "housing-unit estimates",
    "projects",
    "housing-unit estimates",
    "percent",
    "percent",
    "percent",
    "housing-unit estimates"
  ),
  interpretation = c(
    paste(
      "Modeled or fallback EWS projects on City nonresidential or",
      "group-quarters land."
    ),
    "Promoted units attached to the focused review projects.",
    paste(
      "Projects supported only by zoning or another broad classification,",
      "with no appraisal multi-unit code."
    ),
    "Units removed by the promoted-surface City land-use validation.",
    "Share of the focused contradiction explained by zoning-only candidates.",
    paste(
      "Effect of the production exclusion on the City",
      "promoted unit total."
    ),
    paste(
      "Effect of the production exclusion relative to the broad core multi-unit",
      "workflow total."
    ),
    "Current promoted City unit total after the production exclusion."
  )
)

write_csv(
  focus_projects,
  file.path(
    OUTPUT_DIR,
    "residential_unit_nonresidential_reconciliation_projects.csv"
  )
)
write_csv(
  reconciliation_summary,
  file.path(
    OUTPUT_DIR,
    "residential_unit_nonresidential_reconciliation_summary.csv"
  )
)
write_csv(
  candidate_scope,
  file.path(OUTPUT_DIR, "residential_unit_candidate_evidence_scope.csv")
)
write_csv(
  impact_summary,
  file.path(
    OUTPUT_DIR,
    "residential_unit_nonresidential_reconciliation_impact.csv"
  )
)

plot_data <- reconciliation_summary %>%
  mutate(
    evidence_label = if_else(
      provisional_exclusion,
      "Zoning/broad classification only",
      "Current appraisal multi-unit signal"
    ),
    evidence_label = factor(
      evidence_label,
      levels = c(
        "Current appraisal multi-unit signal",
        "Zoning/broad classification only"
      )
    )
  )

p <- ggplot(
  plot_data,
  aes(x = units, y = evidence_label, fill = unit_count_treatment)
) +
  geom_col(width = 0.62) +
  geom_text(
    aes(label = if_else(units >= 100, comma(round(units)), "")),
    position = position_stack(vjust = 0.5),
    color = "white",
    fontface = "bold",
    size = 3.6
  ) +
  scale_x_continuous(
    labels = label_number(big.mark = ","),
    expand = expansion(mult = c(0, 0.08))
  ) +
  scale_fill_manual(
    values = c(
      "Modeled estimate" = "#2D6A8A",
      "Fallback estimate" = "#B5533C"
    )
  ) +
  labs(
    title = "Why modeled units appear on City nonresidential land",
    subtitle = paste0(
      comma(nrow(focus_projects)),
      " unresolved projects; zoning-only candidates are production exclusions"
    ),
    x = "Promoted housing-unit estimates",
    y = NULL,
    fill = "Unit-count treatment",
    caption = "The promoted surface excludes the zoning-only contradiction group."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.y = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(color = "#222222")
  )

ggsave(
  file.path(
    FIGURES_DIR,
    "residential_unit_nonresidential_reconciliation.png"
  ),
  p,
  width = 10,
  height = 5.5,
  dpi = 300,
  bg = "white"
)

print_progress(
  paste0(
    "Focused review: ",
    comma(nrow(focus_projects)),
    " projects and ",
    comma(round(focus_units)),
    " units."
  )
)
print_progress(
  paste0(
    "Production zoning-only exclusions: ",
    comma(sum(focus_projects$provisional_exclusion)),
    " projects and ",
    comma(round(provisional_units)),
    " units."
  )
)
print_progress("Production exclusions were verified against the promoted surface.")
