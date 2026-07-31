################################################################################
# Audit Residential Unit Classification Against the Austin Land Use Inventory
################################################################################
#
# The City Land Use Inventory independently classifies the primary use of each
# parcel. It does not report a current unit count. This audit therefore tests
# whether parcels and projects classified as multi-unit by the EWS are located
# on City duplex, three/fourplex, apartment/condo, or retirement-housing land.
# It does not use the City classification to overwrite the production surface.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(sf)
  library(stringr)
  library(tidyr)
})

source(here::here("R", "utils.R"))

print_header("AUDIT UNIT CLASSIFICATION AGAINST CITY LAND USE")

OUTPUT_DIR <- here::here("output")
FIGURES_DIR <- here::here("figures")
LAND_USE_FILE <- here::here("data", "austin_land_use_inventory_202607.csv")
LAND_USE_GEOMETRY_FILE <- here::here(
  "data",
  "austin_land_use_inventory_202607.geojson"
)
LAND_USE_CODE_FILE <- here::here("config", "austin_land_use_codes.csv")
BOUNDARY_FILE <- here::here("data", "BOUNDARIES_jurisdictions_20260429.geojson")
ACS_STRUCTURE_FILE <- here::here(
  "data",
  "raw_acs",
  "acsdt1y2024-b25024.dat"
)
PARCEL_FILE <- file.path(OUTPUT_DIR, "residential_parcels_unit_promoted.rds")
PROJECT_FILE <- file.path(OUTPUT_DIR, "residential_unit_projects.rds")
MEMBERSHIP_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_project_membership.rds"
)
PROJECT_SELECTION_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_shadow_project_selection.csv"
)

required_files <- c(
  LAND_USE_FILE,
  LAND_USE_GEOMETRY_FILE,
  LAND_USE_CODE_FILE,
  BOUNDARY_FILE,
  ACS_STRUCTURE_FILE,
  PARCEL_FILE,
  PROJECT_FILE,
  MEMBERSHIP_FILE,
  PROJECT_SELECTION_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Land-use unit classification audit is missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)
sf_use_s2(FALSE)

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

collapse_land_use_records <- function(data, group_columns) {
  data %>%
    left_join(land_use_codes, by = "land_use_code", relationship = "many-to-one") %>%
    group_by(across(all_of(group_columns))) %>%
    summarise(
      city_record_count = n(),
      city_land_use_codes = str_c(
        sort(unique(na.omit(land_use_code))),
        collapse = " | "
      ),
      city_land_use_labels = str_c(
        sort(unique(na.omit(detailed_land_use))),
        collapse = " | "
      ),
      city_audit_groups = str_c(
        sort(unique(na.omit(audit_group))),
        collapse = " | "
      ),
      city_any_multiunit = any(city_multiunit_signal %in% TRUE),
      city_any_single_unit = any(audit_group == "single_unit", na.rm = TRUE),
      city_any_mixed_use = any(audit_group == "mixed_use", na.rm = TRUE),
      city_any_nonresidential = any(
        audit_group %in% c("nonresidential", "group_quarters"),
        na.rm = TRUE
      ),
      city_any_unknown = any(is.na(audit_group) | audit_group == "unknown"),
      .groups = "drop"
    ) %>%
    mutate(
      city_land_use_class = case_when(
        city_any_multiunit & city_any_single_unit ~
          "Multi-unit and single-unit records",
        city_any_multiunit ~ "Multi-unit residential",
        city_any_single_unit ~ "Single-unit residential",
        city_any_mixed_use ~ "Mixed use; residential form unspecified",
        city_any_nonresidential ~ "Nonresidential or group quarters",
        city_any_unknown ~ "Unknown City classification",
        TRUE ~ "Other City classification"
      )
    )
}

print_progress("Loading and classifying City Land Use Inventory records...")

land_use <- read_csv(
  LAND_USE_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_character())
) %>%
  transmute(
    city_object_id = objectid,
    city_land_use_id = land_use_id,
    land_use_code = as.integer(land_use),
    general_land_use_code = as.integer(general_land_use),
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
  )

unknown_codes <- setdiff(unique(na.omit(land_use$land_use_code)), land_use_codes$land_use_code)
if (length(unknown_codes) > 0L) {
  stop(
    "City Land Use Inventory contains unmapped codes: ",
    paste(sort(unknown_codes), collapse = ", "),
    call. = FALSE
  )
}

land_use_by_identifier <- land_use %>%
  filter(!is.na(city_record_county), !is.na(city_match_key)) %>%
  collapse_land_use_records(c("city_record_county", "city_match_key"))

print_progress("Loading the promoted parcel and project surfaces...")

membership <- readRDS(MEMBERSHIP_FILE) %>%
  transmute(
    parcel_id = as.character(parcel_id),
    project_id = as.character(project_id)
  )

project_selection <- read_csv(
  PROJECT_SELECTION_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_guess())
) %>%
  transmute(
    project_id = as.character(project_id),
    shadow_selection_method = as.character(shadow_selection_method)
  )

projects <- readRDS(PROJECT_FILE) %>%
  mutate(project_id = as.character(project_id)) %>%
  left_join(project_selection, by = "project_id", relationship = "one-to-one") %>%
  transmute(
    project_id,
    project_counties,
    project_parcel_count,
    project_grouping_methods,
    project_is_multifamily_like,
    selected_observed_units,
    selected_observed_source,
    selected_observed_tier,
    model_candidate,
    source_conflict_requires_review,
    shadow_selection_method,
    core_multiunit_project =
      coalesce(selected_observed_units >= 2, FALSE) |
        coalesce(model_candidate, FALSE),
    unit_count_treatment = case_when(
      selected_observed_tier == "deterministic_appraisal_accounts" ~
        "Deterministic appraisal count",
      selected_observed_tier == "strict_direct_project_total" ~
        "Direct reported total",
      shadow_selection_method %in% c(
        "in_domain_main_area_stratified",
        "williamson_validated_main_area_stratified"
      ) ~ "Modeled estimate",
      shadow_selection_method == "documented_project_count" ~
        "Reviewed documented total",
      shadow_selection_method %in% c(
        "current_targeted_fallback",
        "current_targeted_zero_fallback"
      ) ~ "Fallback estimate",
      selected_observed_tier == "rule_based_single_unit_assumption" ~
        "One-unit rule",
      source_conflict_requires_review ~ "Source conflict; baseline retained",
      TRUE ~ "Other baseline count"
    )
  )

parcels <- readRDS(PARCEL_FILE) %>%
  transmute(
    parcel_id = as.character(parcel_id),
    source_county,
    situs_address,
    appraisal_state_code = as.character(propertyProf_imprvStateCd),
    wcad_property_type,
    is_single_family_like,
    is_multifamily_like,
    county_unit_evidence_class,
    promoted_units = as.numeric(promoted_units),
    pre_validation_units = as.numeric(unit_land_use_pre_validation_units),
    land_use_excluded_units = as.numeric(unit_land_use_excluded_units),
    land_use_validation_excluded = as.logical(
      unit_land_use_validation_excluded
    ),
    lat = as.numeric(lat),
    lon = as.numeric(lon),
    parcel_match_key = case_when(
      source_county == "Travis" ~ parcel_id,
      source_county %in% c("Hays", "Williamson") ~
        str_remove(parcel_id, "^(HAYS|WILLIAMSON):"),
      TRUE ~ NA_character_
    )
  ) %>%
  left_join(membership, by = "parcel_id", relationship = "one-to-one")

if (anyDuplicated(parcels$parcel_id) || any(is.na(parcels$project_id))) {
  stop("Promoted parcels do not link one-to-one to project membership.", call. = FALSE)
}

parcel_points <- parcels %>%
  filter(is.finite(lon), is.finite(lat)) %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
  st_transform(3857)

austin_full_purpose <- st_read(BOUNDARY_FILE, quiet = TRUE) %>%
  filter(jurisdiction_type == "FULL") %>%
  st_make_valid() %>%
  st_transform(3857) %>%
  summarise()

parcel_boundary_status <- parcel_points %>%
  transmute(
    parcel_id,
    in_austin_full_purpose = lengths(st_within(., austin_full_purpose)) > 0
  ) %>%
  st_drop_geometry()

parcels <- parcels %>%
  left_join(parcel_boundary_status, by = "parcel_id", relationship = "one-to-one") %>%
  mutate(in_austin_full_purpose = replace_na(in_austin_full_purpose, FALSE))

identifier_matches <- parcels %>%
  left_join(
    land_use_by_identifier,
    by = c(
      "source_county" = "city_record_county",
      "parcel_match_key" = "city_match_key"
    ),
    relationship = "many-to-one"
  ) %>%
  mutate(city_match_method = if_else(!is.na(city_land_use_class), "identifier", NA_character_))

unmatched_points <- identifier_matches %>%
  filter(is.na(city_land_use_class), is.finite(lon), is.finite(lat)) %>%
  select(parcel_id, lon, lat) %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
  st_transform(3857)

print_progress(
  paste0(
    "Spatially matching ",
    scales::comma(nrow(unmatched_points)),
    " parcel rows not linked by identifier..."
  )
)

land_use_geometry <- st_read(LAND_USE_GEOMETRY_FILE, quiet = TRUE) %>%
  transmute(
    land_use_code = as.integer(land_use),
    geometry
  ) %>%
  st_make_valid() %>%
  st_transform(3857)

spatial_matches <- st_join(
  unmatched_points,
  land_use_geometry,
  join = st_within,
  left = FALSE
) %>%
  st_drop_geometry() %>%
  select(parcel_id, land_use_code) %>%
  collapse_land_use_records("parcel_id") %>%
  mutate(city_match_method = "spatial_point_within")

city_match_columns <- c(
  "city_record_count",
  "city_land_use_codes",
  "city_land_use_labels",
  "city_audit_groups",
  "city_any_multiunit",
  "city_any_single_unit",
  "city_any_mixed_use",
  "city_any_nonresidential",
  "city_any_unknown",
  "city_land_use_class",
  "city_match_method"
)

parcel_audit <- identifier_matches %>%
  left_join(
    spatial_matches %>%
      rename_with(~ paste0(.x, "_spatial"), all_of(city_match_columns)),
    by = "parcel_id",
    relationship = "one-to-one"
  )

for (column in city_match_columns) {
  spatial_column <- paste0(column, "_spatial")
  parcel_audit[[column]] <- coalesce(
    parcel_audit[[column]],
    parcel_audit[[spatial_column]]
  )
}

parcel_audit <- parcel_audit %>%
  select(-ends_with("_spatial")) %>%
  left_join(projects, by = "project_id", relationship = "many-to-one")

if (nrow(parcel_audit) != nrow(parcels) || anyDuplicated(parcel_audit$parcel_id)) {
  stop("Land-use audit changed the promoted parcel universe.", call. = FALSE)
}

project_unit_totals <- parcel_audit %>%
  filter(in_austin_full_purpose) %>%
  group_by(project_id) %>%
  summarise(
    project_units_inside_city = sum(promoted_units, na.rm = TRUE),
    project_pre_validation_units_inside_city = sum(
      pre_validation_units,
      na.rm = TRUE
    ),
    project_land_use_excluded_units_inside_city = sum(
      land_use_excluded_units,
      na.rm = TRUE
    ),
    city_parcel_rows = n(),
    city_matched_parcel_rows = sum(!is.na(city_land_use_class)),
    city_matched_units = sum(promoted_units[!is.na(city_land_use_class)], na.rm = TRUE),
    city_land_use_classes = str_c(
      sort(unique(na.omit(city_land_use_class))),
      collapse = " | "
    ),
    city_land_use_labels = str_c(
      sort(unique(na.omit(city_land_use_labels))),
      collapse = " | "
    ),
    city_any_multiunit = any(city_any_multiunit %in% TRUE),
    city_any_single_unit = any(city_any_single_unit %in% TRUE),
    city_any_mixed_use = any(city_any_mixed_use %in% TRUE),
    city_any_nonresidential = any(city_any_nonresidential %in% TRUE),
    sample_address = first(na.omit(situs_address), default = NA_character_),
    .groups = "drop"
  )

project_audit <- projects %>%
  inner_join(project_unit_totals, by = "project_id", relationship = "one-to-one") %>%
  mutate(
    city_project_class = case_when(
      city_matched_parcel_rows == 0L ~ "Unmatched to City inventory",
      city_any_multiunit ~ "City multi-unit signal",
      city_any_single_unit & !city_any_mixed_use & !city_any_nonresidential ~
        "City single-unit only",
      city_any_mixed_use ~ "City mixed use; residential form unspecified",
      city_any_nonresidential ~ "City nonresidential or group quarters only",
      TRUE ~ "Other City classification"
    ),
    comparison_status = case_when(
      core_multiunit_project & city_any_multiunit ~ "Agreement: multi-unit",
      !core_multiunit_project & city_project_class == "City single-unit only" ~
        "Agreement: not core multi-unit",
      core_multiunit_project & city_project_class == "City single-unit only" ~
        "Review: EWS core multi-unit / City single-unit",
      core_multiunit_project &
        city_project_class == "City nonresidential or group quarters only" ~
        "Review: EWS core multi-unit / City nonresidential",
      !core_multiunit_project & city_any_multiunit ~
        "Review: City multi-unit / outside EWS core",
      city_project_class == "Unmatched to City inventory" ~ "Unmatched",
      TRUE ~ "Not directly comparable"
    )
  )

coverage_summary <- bind_rows(
  parcel_audit %>% mutate(audit_scope = "Entire promoted parcel universe"),
  parcel_audit %>%
    filter(in_austin_full_purpose) %>%
    mutate(audit_scope = "Austin full-purpose boundary")
) %>%
  group_by(audit_scope, source_county) %>%
  summarise(
    parcel_rows = n(),
    identifier_matched_rows = sum(city_match_method == "identifier", na.rm = TRUE),
    spatial_matched_rows = sum(city_match_method == "spatial_point_within", na.rm = TRUE),
    matched_rows = sum(!is.na(city_land_use_class)),
    matched_units = sum(promoted_units[!is.na(city_land_use_class)], na.rm = TRUE),
    promoted_units = sum(promoted_units, na.rm = TRUE),
    row_match_pct = 100 * matched_rows / parcel_rows,
    unit_match_pct = 100 * matched_units / promoted_units,
    .groups = "drop"
  )

comparison_summary <- project_audit %>%
  group_by(core_multiunit_project, city_project_class, comparison_status) %>%
  summarise(
    projects = n(),
    units = sum(project_units_inside_city, na.rm = TRUE),
    matched_units = sum(city_matched_units, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(core_multiunit_project) %>%
  mutate(
    project_pct_within_ews_class = 100 * projects / sum(projects),
    unit_pct_within_ews_class = 100 * units / sum(units)
  ) %>%
  ungroup() %>%
  arrange(desc(core_multiunit_project), desc(units))

city_land_use_summary <- parcel_audit %>%
  filter(in_austin_full_purpose) %>%
  mutate(city_land_use_class = replace_na(city_land_use_class, "Unmatched")) %>%
  group_by(city_land_use_class, city_land_use_labels, core_multiunit_project) %>%
  summarise(
    parcel_rows = n(),
    projects = n_distinct(project_id),
    promoted_units = sum(promoted_units, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(city_land_use_class, desc(promoted_units))

acs_structure <- read_delim(
  ACS_STRUCTURE_FILE,
  delim = "|",
  show_col_types = FALSE,
  col_types = cols(.default = col_double(), GEO_ID = col_character())
) %>%
  filter(GEO_ID == "1600000US4805000")

if (nrow(acs_structure) != 1L) {
  stop("ACS B25024 file does not contain exactly one Austin city row.", call. = FALSE)
}

acs_2plus_units <- sum(
  unlist(acs_structure[paste0("B25024_E", sprintf("%03d", 4:9))])
)
acs_2plus_moe <- sqrt(sum(
  unlist(acs_structure[paste0("B25024_M", sprintf("%03d", 4:9))])^2
))
acs_5plus_units <- sum(
  unlist(acs_structure[paste0("B25024_E", sprintf("%03d", 6:9))])
)
acs_5plus_moe <- sqrt(sum(
  unlist(acs_structure[paste0("B25024_M", sprintf("%03d", 6:9))])^2
))

city_parcel_benchmarks <- parcel_audit %>%
  filter(in_austin_full_purpose) %>%
  summarise(
    explicit_multiunit_land = sum(promoted_units[city_any_multiunit], na.rm = TRUE),
    mixed_use_land = sum(promoted_units[city_any_mixed_use], na.rm = TRUE),
    apartment_retirement_land = sum(
      promoted_units[
        str_detect(
          coalesce(city_audit_groups, ""),
          "large_multiunit|special_multiunit"
        )
      ],
      na.rm = TRUE
    )
  )

benchmark_comparison <- tibble(
  comparison = c(
    "City explicit multi-unit land versus ACS 2+ structures",
    "City explicit multi-unit plus mixed-use land versus ACS 2+ structures",
    "City apartment/condo and retirement land versus ACS 5+ structures"
  ),
  ews_promoted_units = c(
    city_parcel_benchmarks$explicit_multiunit_land,
    city_parcel_benchmarks$explicit_multiunit_land +
      city_parcel_benchmarks$mixed_use_land,
    city_parcel_benchmarks$apartment_retirement_land
  ),
  acs_2024_units = c(acs_2plus_units, acs_2plus_units, acs_5plus_units),
  acs_approximate_moe_90 = c(acs_2plus_moe, acs_2plus_moe, acs_5plus_moe)
) %>%
  mutate(
    unit_difference = ews_promoted_units - acs_2024_units,
    pct_difference = 100 * unit_difference / acs_2024_units,
    interpretation = c(
      "Conservative lower comparison because mixed-use residential parcels are excluded.",
      "Upper comparison because all promoted units on mixed-use parcels are included.",
      paste0(
        "Approximate comparison only: City Apartment/Condo is a parcel use, ",
        "whereas ACS 5+ describes units in a structure."
      )
    )
  )

city_totals <- coverage_summary %>%
  filter(audit_scope == "Austin full-purpose boundary") %>%
  summarise(
    parcel_rows = sum(parcel_rows),
    promoted_units = sum(promoted_units),
    matched_rows = sum(matched_rows),
    matched_units = sum(matched_units)
  )

core_totals <- project_audit %>%
  filter(core_multiunit_project) %>%
  summarise(
    projects = n(),
    units = sum(project_units_inside_city),
    city_multiunit_units = sum(project_units_inside_city[city_any_multiunit]),
    city_single_unit_units = sum(
      project_units_inside_city[city_project_class == "City single-unit only"]
    ),
    city_mixed_use_units = sum(
      project_units_inside_city[
        city_project_class == "City mixed use; residential form unspecified"
      ]
    ),
    city_nonresidential_units = sum(
      project_units_inside_city[
        city_project_class == "City nonresidential or group quarters only"
      ]
    ),
    unmatched_units = sum(
      project_units_inside_city[
        city_project_class == "Unmatched to City inventory"
      ]
    )
  )

headline_metrics <- bind_rows(
  tibble(
    metric = c(
      "city_parcel_rows",
      "city_promoted_units",
      "city_lui_matched_rows",
      "city_lui_matched_units"
    ),
    value = c(
      city_totals$parcel_rows,
      city_totals$promoted_units,
      city_totals$matched_rows,
      city_totals$matched_units
    ),
    unit = c("parcel rows", "housing units", "parcel rows", "housing units")
  ),
  tibble(
    metric = c(
      "core_multiunit_projects",
      "core_multiunit_units",
      "core_units_city_multiunit_signal",
      "core_units_city_single_unit_only",
      "core_units_city_mixed_use",
      "core_units_city_nonresidential",
      "core_units_unmatched"
    ),
    value = c(
      core_totals$projects,
      core_totals$units,
      core_totals$city_multiunit_units,
      core_totals$city_single_unit_units,
      core_totals$city_mixed_use_units,
      core_totals$city_nonresidential_units,
      core_totals$unmatched_units
    ),
    unit = c("projects", rep("housing units", 6))
  )
)

disagreements <- project_audit %>%
  filter(str_starts(comparison_status, "Review:")) %>%
  arrange(desc(project_units_inside_city), comparison_status)

review_summary <- disagreements %>%
  group_by(comparison_status, unit_count_treatment) %>%
  summarise(
    projects = n(),
    units = sum(project_units_inside_city, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(comparison_status, desc(units))

write_csv(
  parcel_audit %>% select(-lat, -lon),
  file.path(OUTPUT_DIR, "land_use_unit_classification_parcel_audit.csv")
)
write_csv(
  project_audit,
  file.path(OUTPUT_DIR, "land_use_unit_classification_project_audit.csv")
)
write_csv(
  coverage_summary,
  file.path(OUTPUT_DIR, "land_use_unit_classification_coverage.csv")
)
write_csv(
  comparison_summary,
  file.path(OUTPUT_DIR, "land_use_unit_classification_comparison.csv")
)
write_csv(
  city_land_use_summary,
  file.path(OUTPUT_DIR, "land_use_unit_classification_city_classes.csv")
)
write_csv(
  headline_metrics,
  file.path(OUTPUT_DIR, "land_use_unit_classification_summary.csv")
)
write_csv(
  benchmark_comparison,
  file.path(OUTPUT_DIR, "land_use_unit_classification_benchmark.csv")
)
write_csv(
  disagreements,
  file.path(OUTPUT_DIR, "land_use_unit_classification_disagreements.csv")
)
write_csv(
  review_summary,
  file.path(OUTPUT_DIR, "land_use_unit_classification_review_summary.csv")
)

plot_data <- comparison_summary %>%
  filter(core_multiunit_project) %>%
  mutate(
    city_project_class = factor(
      city_project_class,
      levels = city_project_class[order(units)]
    )
  )

p <- ggplot(plot_data, aes(x = units, y = city_project_class)) +
  geom_col(aes(fill = city_project_class), width = 0.7) +
  geom_text(
    aes(label = scales::comma(round(units))),
    hjust = -0.08,
    size = 3.4
  ) +
  scale_x_continuous(
    labels = scales::label_number(scale_cut = scales::cut_short_scale()),
    expand = expansion(mult = c(0, 0.18))
  ) +
  scale_fill_manual(
    values = c(
      "City multi-unit signal" = "#237A8B",
      "City nonresidential or group quarters only" = "#B84A4A",
      "City single-unit only" = "#D18F2B",
      "City mixed use; residential form unspecified" = "#6E6AA8",
      "Unmatched to City inventory" = "#777777"
    ),
    guide = "none"
  ) +
  labs(
    title = "City Land-Use Classification of EWS Core Multi-Unit Projects",
    subtitle = "Final promoted units inside Austin's full-purpose boundary",
    x = "Promoted residential units",
    y = NULL,
    caption = paste0(
      "City Land Use Inventory snapshot downloaded July 2026; ",
      "classification audit only, not a unit-count source"
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(color = "grey35")
  )

ggsave(
  file.path(FIGURES_DIR, "land_use_multifamily_classification_audit.png"),
  p,
  width = 10,
  height = 6,
  dpi = 300,
  bg = "white"
)

print_progress("Land-use classification comparison:")
print(comparison_summary)
print_header("LAND-USE UNIT CLASSIFICATION AUDIT COMPLETE")
