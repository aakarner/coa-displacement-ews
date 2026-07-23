################################################################################
# 02o - Audit Parcel and ACS Housing-Unit Disagreement
################################################################################
#
# Compares parcel-derived housing units with ACS total housing units on the
# exact H3 grid. This diagnostic preserves all parcel calibration variants,
# uses ACS margins of error, and traces discordant hexes to parcel estimation
# methods and building vintages. It does not change features or clusters.
#
# Outputs:
#   output/parcel_acs_hex_unit_audit.csv
#   output/parcel_acs_discordant_hex_review.csv
#   output/parcel_acs_discordant_method_audit.csv
#   output/parcel_acs_county_unit_summary.csv
#   output/parcel_acs_block_group_unit_audit.csv
#   output/parcel_acs_unit_audit_summary.csv
#   figures/02o_parcel_acs_threshold_disagreement.png
#   figures/02o_parcel_acs_unit_scatter.png
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
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(sf)
  library(stringr)
  library(tidyr)
})

print_header("02o - PARCEL AND ACS HOUSING-UNIT AUDIT")

OUTPUT_DIR <- project_path("output")
FIGURES_DIR <- project_path("figures")
ACS_CACHE_DIR <- project_path("data", "raw_acs")
threshold <- as.numeric(EWS_CONFIG$minimum_residential_units_for_rates)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

input_files <- c(
  parcels = file.path(OUTPUT_DIR, "residential_parcels_for_hex_sf.rds"),
  hex_grid = file.path(OUTPUT_DIR, "hex_grid.rds"),
  hex_features = file.path(OUTPUT_DIR, "hex_features.rds"),
  block_hex = file.path(
    OUTPUT_DIR,
    "acs_dasymetric_block_hex_allocation.rds"
  ),
  hex_bg = file.path(OUTPUT_DIR, "acs_dasymetric_hex_bg_crosswalk.rds"),
  block_group_validation = file.path(
    OUTPUT_DIR,
    "unit_calibration_block_group_validation.csv"
  ),
  calibration_summary = file.path(
    OUTPUT_DIR,
    "unit_calibration_validation_summary.csv"
  ),
  acs_block_groups = file.path(
    ACS_CACHE_DIR,
    paste0(
      "acs_", EWS_CONFIG$acs_current_year,
      "_acs5_block_group_demographics.rds"
    )
  ),
  jurisdictions = project_path(
    "data",
    "BOUNDARIES_jurisdictions_20260429.geojson"
  )
)

missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0) {
  stop(
    "Parcel/ACS audit is missing required input(s):\n- ",
    paste(missing_files, collapse = "\n- "),
    call. = FALSE
  )
}

county_from_geoid <- function(x) {
  recode(
    substr(as.character(x), 1L, 5L),
    `48209` = "Hays",
    `48453` = "Travis",
    `48491` = "Williamson",
    .default = NA_character_
  )
}

safe_sum <- function(x) sum(as.numeric(x), na.rm = TRUE)

################################################################################
# Step 1: Assign parcel evidence to the exact H3 grid
################################################################################

print_progress("Loading parcel, ACS, and H3 artifacts...")
parcels <- readRDS(input_files[["parcels"]])
hex_grid <- readRDS(input_files[["hex_grid"]]) %>%
  select(hex_id, geometry)
hex_features <- readRDS(input_files[["hex_features"]])
block_hex <- readRDS(input_files[["block_hex"]])
hex_bg <- readRDS(input_files[["hex_bg"]])

required_parcel_columns <- c(
  "parcel_id",
  "source_county",
  "units_raw",
  "units_calibrated",
  "units_calibrated_conservative",
  "units_calibrated_targeted",
  "unit_estimation_method_targeted",
  "unit_estimation_confidence_targeted",
  "direct_costar_units",
  "has_direct_costar_calibration_match",
  "targeted_unit_delta",
  "targeted_unit_adjustment_applied",
  "targeted_adjustment_block_group_geoid",
  "propertyProf_imprvActualYearBuilt",
  "geometry"
)
missing_parcel_columns <- setdiff(required_parcel_columns, names(parcels))
if (length(missing_parcel_columns) > 0) {
  stop(
    "Parcel audit input is missing: ",
    paste(missing_parcel_columns, collapse = ", "),
    call. = FALSE
  )
}

required_hex_columns <- c(
  "hex_id",
  "residential_units",
  "total_housing_units",
  "total_housing_units_moe",
  "total_tenure",
  "total_tenure_moe",
  "total_pop",
  "population_in_occupied_housing"
)
missing_hex_columns <- setdiff(required_hex_columns, names(hex_features))
if (length(missing_hex_columns) > 0) {
  stop(
    "Hex feature input is missing: ",
    paste(missing_hex_columns, collapse = ", "),
    call. = FALSE
  )
}

analysis_crs <- 3857
hex_projected <- st_transform(hex_grid, analysis_crs)
parcel_hex <- suppressWarnings(
  st_join(
    st_transform(parcels, analysis_crs),
    hex_projected,
    join = st_within,
    left = FALSE
  )
)

if (anyDuplicated(parcel_hex$parcel_id)) {
  stop("One or more parcel points joined to multiple H3 cells.", call. = FALSE)
}

parcel_hex <- parcel_hex %>%
  mutate(
    year_built = suppressWarnings(
      as.integer(propertyProf_imprvActualYearBuilt)
    ),
    building_era = case_when(
      year_built >= 2024L ~ "2024_or_later",
      year_built >= 2020L ~ "2020_to_2023",
      year_built >= 2010L ~ "2010_to_2019",
      !is.na(year_built) ~ "before_2010",
      TRUE ~ "missing"
    ),
    method_category = case_when(
      unit_estimation_method_targeted == "parcel_units_retained" ~
        "retained_parcel_unit_value",
      str_detect(unit_estimation_method_targeted, "costar_sqft") ~
        "multifamily_floor_area_estimate",
      str_detect(unit_estimation_method_targeted, "single_family") ~
        "single_family_default_or_correction",
      unit_estimation_method_targeted ==
        "commercial_mixed_use_mf_estimate_excluded" ~
        "excluded_mixed_use_multifamily",
      TRUE ~ "unknown_or_excluded"
    )
  )

parcel_by_hex <- parcel_hex %>%
  st_drop_geometry() %>%
  group_by(hex_id) %>%
  summarise(
    parcel_records = n(),
    parcel_raw_units = safe_sum(units_raw),
    parcel_primary_units = safe_sum(units_calibrated),
    parcel_conservative_units = safe_sum(units_calibrated_conservative),
    parcel_targeted_units = safe_sum(units_calibrated_targeted),
    parcel_targeted_adjustment = safe_sum(targeted_unit_delta),
    direct_costar_match_parcels = sum(
      has_direct_costar_calibration_match,
      na.rm = TRUE
    ),
    direct_costar_units = safe_sum(direct_costar_units),
    targeted_units_on_direct_costar_matches = safe_sum(
      if_else(
        has_direct_costar_calibration_match,
        units_calibrated_targeted,
        0
      )
    ),
    multifamily_floor_area_estimated_units = safe_sum(
      if_else(
        method_category == "multifamily_floor_area_estimate",
        units_calibrated_targeted,
        0
      )
    ),
    excluded_mixed_use_raw_units = safe_sum(
      if_else(
        method_category == "excluded_mixed_use_multifamily",
        units_raw,
        0
      )
    ),
    units_built_2020_or_later = safe_sum(
      if_else(year_built >= 2020L, units_calibrated_targeted, 0)
    ),
    units_built_2024_or_later = safe_sum(
      if_else(year_built >= 2024L, units_calibrated_targeted, 0)
    ),
    .groups = "drop"
  )

dominant_parcel_county <- parcel_hex %>%
  st_drop_geometry() %>%
  group_by(hex_id, source_county) %>%
  summarise(
    county_targeted_units = safe_sum(units_calibrated_targeted),
    county_parcel_records = n(),
    .groups = "drop"
  ) %>%
  arrange(
    hex_id,
    desc(county_targeted_units),
    desc(county_parcel_records),
    source_county
  ) %>%
  group_by(hex_id) %>%
  slice_head(n = 1L) %>%
  ungroup() %>%
  transmute(hex_id, parcel_source_county = source_county)

################################################################################
# Step 2: Attach ACS uncertainty and allocation-quality metadata
################################################################################

dominant_acs_county <- hex_bg %>%
  arrange(hex_id, desc(project_block_housing_units), source_geoid) %>%
  group_by(hex_id) %>%
  slice_head(n = 1L) %>%
  ungroup() %>%
  transmute(
    hex_id,
    acs_source_county = county_from_geoid(source_geoid),
    dominant_acs_source_geoid = source_geoid
  )

allocation_quality <- block_hex %>%
  group_by(hex_id) %>%
  summarise(
    census_block_housing_control = safe_sum(
      block_housing_units_contribution
    ),
    parcel_supported_block_housing_control = safe_sum(
      if_else(
        block_hex_allocation_method ==
          "residential_parcel_floor_area_proxy",
        block_housing_units_contribution,
        0
      )
    ),
    point_fallback_block_housing_control = safe_sum(
      if_else(
        block_hex_allocation_method ==
          "block_point_no_residential_parcel_support",
        block_housing_units_contribution,
        0
      )
    ),
    .groups = "drop"
  ) %>%
  mutate(
    parcel_supported_block_housing_share = if_else(
      census_block_housing_control > 0,
      parcel_supported_block_housing_control /
        census_block_housing_control,
      NA_real_
    ),
    point_fallback_block_housing_share = if_else(
      census_block_housing_control > 0,
      point_fallback_block_housing_control /
        census_block_housing_control,
      NA_real_
    )
  )

hex_audit <- hex_grid %>%
  left_join(
    hex_features %>%
      st_drop_geometry() %>%
      select(all_of(required_hex_columns)),
    by = "hex_id"
  ) %>%
  left_join(parcel_by_hex, by = "hex_id") %>%
  left_join(dominant_parcel_county, by = "hex_id") %>%
  left_join(dominant_acs_county, by = "hex_id") %>%
  left_join(allocation_quality, by = "hex_id") %>%
  mutate(
    across(
      c(
        parcel_records,
        parcel_raw_units,
        parcel_primary_units,
        parcel_conservative_units,
        parcel_targeted_units,
        parcel_targeted_adjustment,
        direct_costar_match_parcels,
        direct_costar_units,
        targeted_units_on_direct_costar_matches,
        multifamily_floor_area_estimated_units,
        excluded_mixed_use_raw_units,
        units_built_2020_or_later,
        units_built_2024_or_later
      ),
      ~replace_na(as.numeric(.x), 0)
    ),
    source_county = coalesce(acs_source_county, parcel_source_county),
    county_assignment_source = case_when(
      !is.na(acs_source_county) ~ "dominant_acs_source",
      !is.na(parcel_source_county) ~ "dominant_parcel_source",
      TRUE ~ "unassigned"
    ),
    county_source_disagreement = !is.na(acs_source_county) &
      !is.na(parcel_source_county) &
      acs_source_county != parcel_source_county,
    acs_total_housing_units = replace_na(total_housing_units, 0),
    acs_total_housing_units_moe = replace_na(total_housing_units_moe, 0),
    acs_occupied_housing_units = replace_na(total_tenure, 0),
    acs_occupied_housing_units_moe = replace_na(total_tenure_moe, 0),
    acs_total_housing_lower_90 = pmax(
      0,
      acs_total_housing_units - acs_total_housing_units_moe
    ),
    acs_total_housing_upper_90 =
      acs_total_housing_units + acs_total_housing_units_moe,
    acs_total_housing_relative_moe = if_else(
      acs_total_housing_units > 0,
      acs_total_housing_units_moe / acs_total_housing_units,
      NA_real_
    ),
    parcel_unit_range_lower = pmin(
      parcel_primary_units,
      parcel_conservative_units,
      parcel_targeted_units
    ),
    parcel_unit_range_upper = pmax(
      parcel_primary_units,
      parcel_conservative_units,
      parcel_targeted_units
    ),
    parcel_qualifies = parcel_targeted_units >= threshold,
    acs_qualifies = acs_total_housing_units >= threshold,
    agreement_status = case_when(
      parcel_qualifies & acs_qualifies ~ "both_at_or_above_threshold",
      !parcel_qualifies & acs_qualifies ~
        "acs_only_at_or_above_threshold",
      parcel_qualifies & !acs_qualifies ~
        "parcel_only_at_or_above_threshold",
      TRUE ~ "both_below_threshold"
    ),
    review_class = case_when(
      agreement_status == "acs_only_at_or_above_threshold" &
        parcel_unit_range_upper < threshold &
        acs_total_housing_lower_90 >= threshold ~ "robust_acs_only",
      agreement_status == "acs_only_at_or_above_threshold" ~
        "uncertainty_sensitive_acs_only",
      agreement_status == "parcel_only_at_or_above_threshold" &
        parcel_unit_range_lower >= threshold &
        acs_total_housing_upper_90 < threshold ~ "robust_parcel_only",
      agreement_status == "parcel_only_at_or_above_threshold" ~
        "uncertainty_sensitive_parcel_only",
      TRUE ~ "not_discordant"
    ),
    robust_disagreement = review_class %in% c(
      "robust_acs_only",
      "robust_parcel_only"
    ),
    diagnostic_pattern = case_when(
      agreement_status == "acs_only_at_or_above_threshold" &
        parcel_records == 0 ~ "no_matched_residential_parcels",
      agreement_status == "acs_only_at_or_above_threshold" &
        excluded_mixed_use_raw_units > 0 ~
        "mixed_use_multifamily_excluded",
      agreement_status == "acs_only_at_or_above_threshold" &
        coalesce(point_fallback_block_housing_share, 0) >= 0.5 ~
        "acs_block_point_fallback_majority",
      agreement_status == "acs_only_at_or_above_threshold" ~
        "other_parcel_undercoverage",
      agreement_status == "parcel_only_at_or_above_threshold" &
        parcel_targeted_units > 0 &
        units_built_2020_or_later / parcel_targeted_units >= 0.5 ~
        "recent_construction_dominant",
      agreement_status == "parcel_only_at_or_above_threshold" &
        parcel_targeted_units > 0 &
        multifamily_floor_area_estimated_units /
          parcel_targeted_units >= 0.5 ~
        "multifamily_floor_area_estimate_dominant",
      agreement_status == "parcel_only_at_or_above_threshold" &
        acs_total_housing_units == 0 &
        total_pop == 0 ~ "no_allocated_acs_residential_evidence",
      agreement_status == "parcel_only_at_or_above_threshold" ~
        "other_parcel_only",
      TRUE ~ "not_discordant"
    ),
    parcel_minus_acs_units =
      parcel_targeted_units - acs_total_housing_units,
    parcel_to_acs_ratio = if_else(
      acs_total_housing_units > 0,
      parcel_targeted_units / acs_total_housing_units,
      NA_real_
    ),
    absolute_unit_difference = abs(parcel_minus_acs_units),
    log_unit_ratio = log1p(parcel_targeted_units) -
      log1p(acs_total_housing_units)
  )

max_current_difference <- max(
  abs(hex_audit$parcel_targeted_units - hex_audit$residential_units),
  na.rm = TRUE
)
if (max_current_difference > 1e-6) {
  stop(
    "Reconstructed parcel units do not match current hex features; ",
    "maximum difference is ", signif(max_current_difference, 5), ".",
    call. = FALSE
  )
}

discordant_statuses <- c(
  "acs_only_at_or_above_threshold",
  "parcel_only_at_or_above_threshold"
)

discordant_hexes <- hex_audit %>%
  filter(agreement_status %in% discordant_statuses) %>%
  st_drop_geometry() %>%
  arrange(
    desc(robust_disagreement),
    desc(absolute_unit_difference),
    desc(total_pop),
    hex_id
  ) %>%
  select(
    hex_id,
    source_county,
    agreement_status,
    review_class,
    robust_disagreement,
    diagnostic_pattern,
    parcel_records,
    parcel_raw_units,
    parcel_primary_units,
    parcel_conservative_units,
    parcel_targeted_units,
    parcel_unit_range_lower,
    parcel_unit_range_upper,
    acs_total_housing_units,
    acs_total_housing_units_moe,
    acs_total_housing_lower_90,
    acs_total_housing_upper_90,
    acs_occupied_housing_units,
    parcel_minus_acs_units,
    parcel_to_acs_ratio,
    total_pop,
    population_in_occupied_housing,
    direct_costar_match_parcels,
    direct_costar_units,
    targeted_units_on_direct_costar_matches,
    multifamily_floor_area_estimated_units,
    excluded_mixed_use_raw_units,
    units_built_2020_or_later,
    units_built_2024_or_later,
    parcel_supported_block_housing_share,
    point_fallback_block_housing_share,
    dominant_acs_source_geoid,
    county_assignment_source,
    county_source_disagreement
  )

################################################################################
# Step 3: Summarise methods and counties
################################################################################

hex_status_lookup <- hex_audit %>%
  st_drop_geometry() %>%
  transmute(
    hex_id,
    agreement_status,
    review_class,
    diagnostic_pattern,
    audit_source_county = source_county
  )

discordant_method_audit <- parcel_hex %>%
  st_drop_geometry() %>%
  left_join(hex_status_lookup, by = "hex_id") %>%
  filter(agreement_status %in% discordant_statuses) %>%
  mutate(source_county = audit_source_county) %>%
  group_by(
    agreement_status,
    review_class,
    diagnostic_pattern,
    source_county,
    method_category,
    unit_estimation_method_targeted,
    unit_estimation_confidence_targeted,
    building_era
  ) %>%
  summarise(
    parcel_records = n(),
    raw_units = safe_sum(units_raw),
    primary_units = safe_sum(units_calibrated),
    conservative_units = safe_sum(units_calibrated_conservative),
    targeted_units = safe_sum(units_calibrated_targeted),
    direct_costar_match_parcels = sum(
      has_direct_costar_calibration_match,
      na.rm = TRUE
    ),
    direct_costar_units = safe_sum(direct_costar_units),
    .groups = "drop"
  ) %>%
  arrange(
    agreement_status,
    review_class,
    source_county,
    desc(targeted_units)
  )

county_unit_summary <- bind_rows(
  hex_audit %>%
    st_drop_geometry() %>%
    mutate(summary_geography = coalesce(source_county, "Unassigned")) %>%
    group_by(summary_geography, agreement_status) %>%
    summarise(
      hexes = n(),
      parcel_records = safe_sum(parcel_records),
      parcel_targeted_units = safe_sum(parcel_targeted_units),
      acs_total_housing_units = safe_sum(acs_total_housing_units),
      acs_occupied_housing_units = safe_sum(acs_occupied_housing_units),
      total_population = safe_sum(total_pop),
      household_population = safe_sum(population_in_occupied_housing),
      .groups = "drop"
    ),
  hex_audit %>%
    st_drop_geometry() %>%
    mutate(summary_geography = "All") %>%
    group_by(summary_geography, agreement_status) %>%
    summarise(
      hexes = n(),
      parcel_records = safe_sum(parcel_records),
      parcel_targeted_units = safe_sum(parcel_targeted_units),
      acs_total_housing_units = safe_sum(acs_total_housing_units),
      acs_occupied_housing_units = safe_sum(acs_occupied_housing_units),
      total_population = safe_sum(total_pop),
      household_population = safe_sum(population_in_occupied_housing),
      .groups = "drop"
    )
) %>%
  arrange(summary_geography, agreement_status)

################################################################################
# Step 4: Update block-group validation with targeted adjustments
################################################################################

print_progress("Updating block-group validation with targeted adjustments...")
block_group_validation <- read_csv(
  input_files[["block_group_validation"]],
  col_types = cols(GEOID = col_character(), .default = col_guess()),
  show_col_types = FALSE
)

targeted_adjustments <- parcels %>%
  st_drop_geometry() %>%
  filter(targeted_unit_adjustment_applied) %>%
  group_by(GEOID = targeted_adjustment_block_group_geoid) %>%
  summarise(
    targeted_adjustment_parcels = n(),
    targeted_unit_delta = safe_sum(targeted_unit_delta),
    .groups = "drop"
  )

block_group_audit <- block_group_validation %>%
  left_join(targeted_adjustments, by = "GEOID") %>%
  mutate(
    targeted_adjustment_parcels = replace_na(
      targeted_adjustment_parcels,
      0L
    ),
    targeted_unit_delta = replace_na(targeted_unit_delta, 0),
    parcel_residential_units_targeted =
      parcel_residential_units + targeted_unit_delta,
    targeted_minus_acs_full_units =
      parcel_residential_units_targeted - acs_total_housing_units,
    targeted_to_acs_full_ratio = if_else(
      acs_total_housing_units > 0,
      parcel_residential_units_targeted / acs_total_housing_units,
      NA_real_
    ),
    targeted_full_count_moe_flag = case_when(
      parcel_residential_units_targeted >
        acs_total_housing_units + acs_total_housing_units_moe ~
        "parcel_above_acs_moe",
      parcel_residential_units_targeted <
        acs_total_housing_units - acs_total_housing_units_moe ~
        "parcel_below_acs_moe",
      TRUE ~ "within_acs_moe"
    ),
    source_county = county_from_geoid(GEOID),
    fully_contained_in_austin = austin_area_share >= 0.95
  )

fully_contained_bg <- block_group_audit %>%
  filter(fully_contained_in_austin, acs_total_housing_units > 0)

block_group_method_summary <- bind_rows(
  fully_contained_bg %>%
    transmute(
      variant = "primary",
      parcel_units = parcel_residential_units,
      acs_units = acs_total_housing_units,
      acs_moe = acs_total_housing_units_moe
    ),
  fully_contained_bg %>%
    transmute(
      variant = "conservative",
      parcel_units = parcel_residential_units_conservative,
      acs_units = acs_total_housing_units,
      acs_moe = acs_total_housing_units_moe
    ),
  fully_contained_bg %>%
    transmute(
      variant = "targeted",
      parcel_units = parcel_residential_units_targeted,
      acs_units = acs_total_housing_units,
      acs_moe = acs_total_housing_units_moe
    )
) %>%
  mutate(
    ratio = parcel_units / acs_units,
    moe_flag = case_when(
      parcel_units > acs_units + acs_moe ~ "above",
      parcel_units < acs_units - acs_moe ~ "below",
      TRUE ~ "within"
    )
  ) %>%
  group_by(variant) %>%
  summarise(
    block_groups = n(),
    parcel_units = safe_sum(parcel_units),
    acs_units = safe_sum(acs_units),
    aggregate_ratio = parcel_units / acs_units,
    median_ratio = median(ratio, na.rm = TRUE),
    p10_ratio = quantile(ratio, 0.10, na.rm = TRUE),
    p90_ratio = quantile(ratio, 0.90, na.rm = TRUE),
    spearman_correlation = cor(
      parcel_units,
      acs_units,
      method = "spearman"
    ),
    within_moe_share = mean(moe_flag == "within"),
    above_moe_share = mean(moe_flag == "above"),
    below_moe_share = mean(moe_flag == "below"),
    .groups = "drop"
  )

################################################################################
# Step 5: Calculate aggregate ACS MOE and citywide benchmark
################################################################################

acs_block_groups <- readRDS(input_files[["acs_block_groups"]]) %>%
  st_drop_geometry() %>%
  filter(variable == "total_housing_units") %>%
  transmute(
    source_geoid = GEOID,
    estimate = as.numeric(estimate),
    moe = as.numeric(moe)
  )

project_source_weights <- hex_bg %>%
  group_by(source_geoid) %>%
  summarise(
    project_housing_share = safe_sum(housing_allocation_weight),
    .groups = "drop"
  )

acs_project_total <- acs_block_groups %>%
  left_join(project_source_weights, by = "source_geoid") %>%
  mutate(
    project_housing_share = replace_na(project_housing_share, 0),
    allocated_estimate = estimate * project_housing_share,
    allocated_moe_component = moe * project_housing_share
  ) %>%
  summarise(
    estimate = safe_sum(allocated_estimate),
    moe_90 = sqrt(sum(allocated_moe_component^2, na.rm = TRUE))
  )

calibration_summary <- read_csv(
  input_files[["calibration_summary"]],
  show_col_types = FALSE
)
city_acs_benchmark <- calibration_summary %>%
  filter(
    metric_group == "external_city_benchmark",
    metric == "acs_2024_1yr_city_total"
  ) %>%
  pull(value)
if (length(city_acs_benchmark) != 1L) city_acs_benchmark <- NA_real_

jurisdictions <- st_read(
  input_files[["jurisdictions"]],
  quiet = TRUE
) %>%
  filter(jurisdiction_type == "FULL") %>%
  st_make_valid() %>%
  st_transform(analysis_crs) %>%
  summarise()

parcels_full_purpose <- suppressWarnings(
  st_join(
    st_transform(parcels, analysis_crs),
    jurisdictions,
    join = st_within,
    left = FALSE
  )
)

grid_totals <- hex_audit %>%
  st_drop_geometry() %>%
  summarise(
    parcel_raw_units = safe_sum(parcel_raw_units),
    parcel_primary_units = safe_sum(parcel_primary_units),
    parcel_conservative_units = safe_sum(parcel_conservative_units),
    parcel_targeted_units = safe_sum(parcel_targeted_units),
    acs_total_housing_units = safe_sum(acs_total_housing_units),
    acs_occupied_housing_units = safe_sum(acs_occupied_housing_units)
  )

full_purpose_targeted_units <- safe_sum(
  parcels_full_purpose$units_calibrated_targeted
)
status_summary <- hex_audit %>%
  st_drop_geometry() %>%
  count(agreement_status, name = "hexes")
review_summary <- hex_audit %>%
  st_drop_geometry() %>%
  count(review_class, name = "hexes")

audit_summary <- bind_rows(
  tibble(
    metric_group = "configuration",
    metric = c("unit_threshold", "acs_year"),
    value = c(threshold, EWS_CONFIG$acs_current_year),
    note = c(
      "Threshold used only to classify source agreement.",
      "ACS five-year vintage used for small-area comparison."
    )
  ),
  tibble(
    metric_group = "h3_grid_totals",
    metric = c(
      "parcel_raw_units",
      "parcel_primary_units",
      "parcel_conservative_units",
      "parcel_targeted_units",
      "acs_total_housing_units",
      "acs_total_housing_units_moe_90",
      "acs_occupied_housing_units",
      "targeted_minus_acs_total_housing_units",
      "targeted_to_acs_total_housing_ratio"
    ),
    value = c(
      grid_totals$parcel_raw_units,
      grid_totals$parcel_primary_units,
      grid_totals$parcel_conservative_units,
      grid_totals$parcel_targeted_units,
      grid_totals$acs_total_housing_units,
      acs_project_total$moe_90,
      grid_totals$acs_occupied_housing_units,
      grid_totals$parcel_targeted_units -
        grid_totals$acs_total_housing_units,
      grid_totals$parcel_targeted_units /
        grid_totals$acs_total_housing_units
    ),
    note = NA_character_
  ),
  tibble(
    metric_group = "citywide_benchmark",
    metric = c(
      "parcel_targeted_units_full_purpose",
      "acs_2024_1yr_city_total",
      "parcel_to_acs_1yr_city_ratio"
    ),
    value = c(
      full_purpose_targeted_units,
      city_acs_benchmark,
      full_purpose_targeted_units / city_acs_benchmark
    ),
    note = c(
      "Parcel points inside the Austin full-purpose boundary.",
      "External benchmark retained by the unit-calibration audit.",
      "Not directly comparable to the H3 five-year allocation."
    )
  ),
  status_summary %>%
    transmute(
      metric_group = "threshold_agreement",
      metric = agreement_status,
      value = as.numeric(hexes),
      note = paste0("Hex count at the ", threshold, "-unit threshold.")
    ),
  review_summary %>%
    transmute(
      metric_group = "discordance_review_class",
      metric = review_class,
      value = as.numeric(hexes),
      note = "Robust classes remain discordant across parcel variants and the ACS 90% interval."
    ),
  block_group_method_summary %>%
    select(variant, aggregate_ratio, within_moe_share) %>%
    pivot_longer(
      cols = c(aggregate_ratio, within_moe_share),
      names_to = "measure",
      values_to = "value"
    ) %>%
    transmute(
      metric_group = "fully_contained_block_groups",
      metric = paste(variant, measure, sep = "_"),
      value,
      note = paste0(
        nrow(fully_contained_bg),
        " block groups with at least 95% of area inside Austin."
      )
    )
)

################################################################################
# Step 6: Save tables and figures
################################################################################

print_progress("Saving parcel/ACS audit outputs...")
hex_audit %>%
  st_drop_geometry() %>%
  write_csv(file.path(OUTPUT_DIR, "parcel_acs_hex_unit_audit.csv"))
write_csv(
  discordant_hexes,
  file.path(OUTPUT_DIR, "parcel_acs_discordant_hex_review.csv")
)
write_csv(
  discordant_method_audit,
  file.path(OUTPUT_DIR, "parcel_acs_discordant_method_audit.csv")
)
write_csv(
  county_unit_summary,
  file.path(OUTPUT_DIR, "parcel_acs_county_unit_summary.csv")
)
write_csv(
  block_group_audit,
  file.path(OUTPUT_DIR, "parcel_acs_block_group_unit_audit.csv")
)
write_csv(
  audit_summary,
  file.path(OUTPUT_DIR, "parcel_acs_unit_audit_summary.csv")
)

status_labels <- c(
  both_at_or_above_threshold = "Both >= 20",
  acs_only_at_or_above_threshold = "ACS only >= 20",
  parcel_only_at_or_above_threshold = "Parcel only >= 20",
  both_below_threshold = "Both < 20"
)
status_colors <- c(
  `Both >= 20` = "#009E73",
  `ACS only >= 20` = "#D55E00",
  `Parcel only >= 20` = "#7A5195",
  `Both < 20` = "#D9D9D9"
)

plot_data <- hex_audit %>%
  mutate(
    agreement_label = factor(
      status_labels[agreement_status],
      levels = unname(status_labels)
    )
  )

p_map <- ggplot(plot_data) +
  geom_sf(aes(fill = agreement_label), color = NA) +
  scale_fill_manual(values = status_colors, drop = FALSE) +
  coord_sf(datum = NA) +
  labs(
    title = "Parcel and ACS Housing-Unit Threshold Agreement",
    subtitle = paste0(
      "Targeted parcel units versus ACS total housing units; threshold = ",
      threshold
    ),
    fill = "Source agreement"
  ) +
  theme_void(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(
  file.path(FIGURES_DIR, "02o_parcel_acs_threshold_disagreement.png"),
  p_map,
  width = 9,
  height = 9,
  dpi = 300,
  bg = "white"
)

scatter_data <- plot_data %>%
  st_drop_geometry() %>%
  filter(parcel_targeted_units > 0 | acs_total_housing_units > 0)

p_scatter <- ggplot(
  scatter_data,
  aes(
    x = acs_total_housing_units,
    y = parcel_targeted_units,
    color = agreement_label
  )
) +
  geom_abline(slope = 1, intercept = 0, color = "#333333", linewidth = 0.5) +
  geom_point(alpha = 0.5, size = 1.2) +
  scale_x_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 1, 5, 20, 100, 500, 2000)
  ) +
  scale_y_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 1, 5, 20, 100, 500, 2000)
  ) +
  scale_color_manual(values = status_colors, drop = FALSE) +
  labs(
    title = "Parcel and ACS Housing Units by Hex",
    subtitle = "The diagonal indicates equal unit estimates",
    x = "ACS total housing units",
    y = "Targeted parcel housing units",
    color = "Source agreement"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(
  file.path(FIGURES_DIR, "02o_parcel_acs_unit_scatter.png"),
  p_scatter,
  width = 9,
  height = 7,
  dpi = 300,
  bg = "white"
)

print_header("02o AUDIT RESULTS")
cat(
  "H3 targeted parcel units:",
  scales::comma(round(grid_totals$parcel_targeted_units)),
  "\n"
)
cat(
  "H3 ACS total housing units:",
  scales::comma(round(grid_totals$acs_total_housing_units)),
  "+/-",
  scales::comma(round(acs_project_total$moe_90)),
  "\n"
)
print(status_summary)
print(review_summary)
cat("\nDiagnostic only: no unit estimates or clustering inputs were changed.\n")
