################################################################################
# Audit Populated Zero-Unit Hexes
################################################################################
#
# Classifies populated hexes that retain zero parcel-derived residential units
# in the promoted canonical surface. The audit distinguishes pipeline
# omissions, cross-county allocation choices, unit-parcel source gaps, broader
# appraisal-parcel support, explicit exclusions, and ACS block-point fallback.
# The audit is diagnostic and does not modify parcel, feature, or cluster
# outputs.
#
# Outputs:
#   output/populated_zero_unit_hex_audit.rds/.csv
#   output/populated_zero_unit_category_summary.csv
#   output/populated_zero_unit_jurisdiction_summary.csv
#   output/populated_zero_unit_parcel_review.csv
#   output/populated_zero_unit_direct_project_repairs.csv
#   output/populated_zero_unit_full_parcel_support.csv
#   output/populated_zero_unit_full_residential_proxy_review.csv
#   output/populated_zero_unit_audit_summary.csv
#   figures/populated_zero_unit_hex_audit.png
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(sf)
  library(stringr)
  library(tibble)
  library(tidyr)
})

source(here::here("R", "utils.R"))

print_header("AUDIT POPULATED ZERO-UNIT HEXES")

OUTPUT_DIR <- here::here("output")
FIGURES_DIR <- here::here("figures")
DATA_DIR <- here::here("data")
ANALYSIS_CRS <- 3857
MINIMUM_MEANINGFUL_ACS_HOUSING <- 5

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

input_files <- c(
  hex_grid = file.path(OUTPUT_DIR, "hex_grid.rds"),
  current_features = file.path(OUTPUT_DIR, "hex_features.rds"),
  current_parcels = file.path(
    OUTPUT_DIR,
    "residential_parcels_unit_promoted.rds"
  ),
  shadow_parcels = file.path(
    OUTPUT_DIR,
    "residential_parcels_unit_shadow_integrated.rds"
  ),
  projects = file.path(OUTPUT_DIR, "residential_unit_projects.rds"),
  block_hex = file.path(
    OUTPUT_DIR,
    "acs_dasymetric_block_hex_allocation.rds"
  ),
  hex_bg = file.path(OUTPUT_DIR, "acs_dasymetric_hex_bg_crosswalk.rds"),
  exclusions = file.path(
    OUTPUT_DIR,
    "residential_unit_eligibility_exclusions.rds"
  ),
  reviews = file.path(
    OUTPUT_DIR,
    "residential_unit_eligibility_reviews.rds"
  ),
  residual_reviews = here::here(
    "config",
    "residual_unit_parcel_reviews.csv"
  ),
  jurisdictions = file.path(
    DATA_DIR,
    "BOUNDARIES_jurisdictions_20260429.geojson"
  ),
  travis_parcels = file.path(
    DATA_DIR,
    "raw_parcels",
    "travis",
    "Parcel_poly.zip"
  ),
  williamson_parcels = file.path(
    DATA_DIR,
    "raw_parcels",
    "williamson",
    "wcad_parcels.rds"
  ),
  hays_parcels = file.path(
    DATA_DIR,
    "raw_parcels",
    "hays",
    "hays_parcels.rds"
  )
)

missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Populated zero-unit audit is missing:\n- ",
    paste(missing_files, collapse = "\n- "),
    call. = FALSE
  )
}
if (!requireNamespace("terra", quietly = TRUE)) {
  stop("The populated zero-unit audit requires the terra package.", call. = FALSE)
}

safe_sum <- function(x) sum(as.numeric(x), na.rm = TRUE)

collapse_values <- function(x) {
  observed <- sort(unique(as.character(x[!is.na(x) & nzchar(x)])))
  if (length(observed) == 0L) {
    return(NA_character_)
  }
  paste(observed, collapse = " | ")
}

county_from_geoid <- function(x) {
  recode(
    substr(as.character(x), 1L, 5L),
    `48209` = "Hays",
    `48453` = "Travis",
    `48491` = "Williamson",
    .default = "Other"
  )
}

load_full_parcel_points <- function(target_crs) {
  print_progress("Loading compact points from the three full parcel maps...")

  travis_extract_dir <- tempfile("zero_unit_travis_parcels_")
  dir.create(travis_extract_dir)
  utils::unzip(
    input_files[["travis_parcels"]],
    exdir = travis_extract_dir
  )
  travis_shapefile <- file.path(travis_extract_dir, "Parcel_poly.shp")
  if (!file.exists(travis_shapefile)) {
    stop("Travis parcel archive does not contain Parcel_poly.shp.", call. = FALSE)
  }

  travis_vect <- terra::vect(travis_shapefile)
  travis_coordinates <- as_tibble(terra::geom(travis_vect)) %>%
    filter(!is.na(x), !is.na(y)) %>%
    group_by(geom) %>%
    summarise(
      x = (min(x) + max(x)) / 2,
      y = (min(y) + max(y)) / 2,
      .groups = "drop"
    )
  travis_attributes <- as_tibble(as.data.frame(travis_vect)) %>%
    transmute(
      geom = row_number(),
      full_parcel_id = as.character(PROP_ID)
    )
  travis_points <- travis_attributes %>%
    left_join(travis_coordinates, by = "geom") %>%
    filter(is.finite(x), is.finite(y)) %>%
    st_as_sf(
      coords = c("x", "y"),
      crs = terra::crs(travis_vect)
    ) %>%
    st_transform(target_crs) %>%
    transmute(
      source_county = "Travis",
      full_parcel_id,
      full_residential_proxy = NA,
      full_use_description = NA_character_,
      full_residential_floor_area = NA_real_,
      full_building_area = NA_real_,
      geometry
    )

  williamson_raw <- readRDS(input_files[["williamson_parcels"]])
  williamson_points <- suppressWarnings(
    st_point_on_surface(williamson_raw)
  ) %>%
    st_transform(target_crs) %>%
    mutate(
      residential_floor_area = parse_number(as.character(resflrarea)),
      building_area = parse_number(as.character(bldgarea)),
      full_residential_proxy =
        usedscrp == "Residential" |
          coalesce(residential_floor_area, 0) > 0
    ) %>%
    transmute(
      source_county = "Williamson",
      full_parcel_id = as.character(parcelid),
      full_residential_proxy,
      full_use_description = as.character(usedscrp),
      full_residential_floor_area = residential_floor_area,
      full_building_area = building_area,
      geometry
    )

  hays_raw <- readRDS(input_files[["hays_parcels"]])
  hays_points <- suppressWarnings(
    st_point_on_surface(hays_raw)
  ) %>%
    st_transform(target_crs) %>%
    transmute(
      source_county = "Hays",
      full_parcel_id = coalesce(as.character(TEXT), as.character(REFNAME)),
      full_residential_proxy = NA,
      full_use_description = NA_character_,
      full_residential_floor_area = NA_real_,
      full_building_area = NA_real_,
      geometry
    )

  bind_rows(travis_points, williamson_points, hays_points)
}

################################################################################
# Step 1: Define the residual population-weighted audit universe
################################################################################

print_progress("Defining current zero-unit hexes...")
hex_grid <- readRDS(input_files[["hex_grid"]]) %>%
  mutate(hex_id = as.character(hex_id)) %>%
  select(hex_id, geometry) %>%
  st_transform(ANALYSIS_CRS)
current_features <- readRDS(input_files[["current_features"]]) %>%
  st_drop_geometry() %>%
  transmute(
    hex_id = as.character(hex_id),
    current_residential_units = as.numeric(residential_units),
    total_population = as.numeric(total_pop),
    household_population = as.numeric(population_in_occupied_housing),
    acs_total_housing_units = as.numeric(total_housing_units),
    acs_total_housing_units_moe = as.numeric(total_housing_units_moe)
  )
hex_comparison <- hex_grid %>%
  left_join(current_features, by = "hex_id", relationship = "one-to-one") %>%
  mutate(
    current_zero_units = coalesce(current_residential_units, 0) == 0
  )

zero_hexes <- hex_comparison %>%
  filter(
    current_zero_units,
    coalesce(total_population, 0) > 0
  )
if (nrow(zero_hexes) == 0L) {
  stop("No populated current zero-unit hexes were found.", call. = FALSE)
}
if (anyDuplicated(zero_hexes$hex_id)) {
  stop("The zero-unit audit universe contains duplicate hex IDs.", call. = FALSE)
}

################################################################################
# Step 2: Trace unit-parcel, project, and exclusion evidence
################################################################################

print_progress("Tracing zero-unit hexes to parcel and project evidence...")
current_values <- readRDS(input_files[["current_parcels"]]) %>%
  transmute(
    source_county,
    parcel_id,
    current_units = as.numeric(promoted_units),
    unit_land_use_validation_excluded = coalesce(
      unit_land_use_validation_excluded,
      FALSE
    )
  )
current_parcels <- readRDS(input_files[["shadow_parcels"]]) %>%
  mutate(
    project_id = as.character(project_id)
  ) %>%
  left_join(
    current_values,
    by = c("source_county", "parcel_id"),
    relationship = "one-to-one"
  ) %>%
  mutate(
    current_units = coalesce(current_units, 0),
    current_units_changed =
      coalesce(shadow_units_changed, FALSE) |
        coalesce(unit_land_use_validation_excluded, FALSE)
  )
residual_reviews <- read_csv(
  input_files[["residual_reviews"]],
  col_types = cols(.default = col_character()),
  show_col_types = FALSE
)
unit_residual_reviews <- residual_reviews %>%
  filter(review_layer == "unit_parcel") %>%
  transmute(
    source_county,
    parcel_id = as.character(parcel_id),
    residual_review_outcome = review_outcome,
    residual_review_basis = review_basis
  )
if (
  anyDuplicated(
    paste(
      unit_residual_reviews$source_county,
      unit_residual_reviews$parcel_id
    )
  )
) {
  stop("Residual unit-parcel reviews are not record-unique.", call. = FALSE)
}
project_current_totals <- current_parcels %>%
  group_by(project_id) %>%
  summarise(
    project_current_units = safe_sum(current_units),
    .groups = "drop"
  )
project_evidence <- readRDS(input_files[["projects"]]) %>%
  mutate(project_id = as.character(project_id)) %>%
  select(
    project_id,
    selected_observed_units,
    selected_observed_tier,
    selected_observed_source,
    training_label_eligible,
    model_candidate,
    source_conflict_requires_review,
    project_model_floor_area,
    project_is_multifamily_like
  ) %>%
  left_join(
    project_current_totals,
    by = "project_id",
    relationship = "one-to-one"
  )

unit_parcels_in_zero_hexes <- current_parcels %>%
  st_as_sf(
    coords = c("lon", "lat"),
    crs = 4326,
    remove = FALSE
  ) %>%
  st_transform(ANALYSIS_CRS) %>%
  st_join(
    zero_hexes %>% select(hex_id),
    join = st_within,
    left = FALSE
  ) %>%
  st_drop_geometry() %>%
  mutate(hex_id = as.character(hex_id)) %>%
  left_join(
    project_evidence,
    by = "project_id",
    relationship = "many-to-one"
  ) %>%
  left_join(
    unit_residual_reviews,
    by = c("source_county", "parcel_id"),
    relationship = "many-to-one"
  )

if (anyDuplicated(unit_parcels_in_zero_hexes$parcel_id)) {
  stop("A unit parcel joined to multiple zero-unit hexes.", call. = FALSE)
}

direct_project_repairs <- unit_parcels_in_zero_hexes %>%
  filter(
    selected_observed_tier == "strict_direct_project_total",
    coalesce(selected_observed_units, 0) > 0,
    abs(
      coalesce(project_current_units, 0) -
        selected_observed_units
    ) > 1e-6
  ) %>%
  group_by(hex_id, project_id) %>%
  summarise(
    source_county = collapse_values(source_county),
    parcel_ids = collapse_values(parcel_id),
    situs_addresses = collapse_values(situs_address),
    project_parcels_in_hex = n_distinct(parcel_id),
    selected_observed_units = first(selected_observed_units),
    selected_observed_tier = first(selected_observed_tier),
    selected_observed_source = first(selected_observed_source),
    training_label_eligible = first(training_label_eligible),
    current_targeted_units = safe_sum(current_targeted_units),
    current_units = safe_sum(current_units),
    project_current_units = first(project_current_units),
    .groups = "drop"
  )

direct_project_hex_counts <- direct_project_repairs %>%
  count(project_id, name = "zero_hex_count")
if (any(direct_project_hex_counts$zero_hex_count > 1L)) {
  stop(
    "A direct project repair spans multiple zero-unit hexes; ",
    "project allocation must be reviewed before summarising units.",
    call. = FALSE
  )
}

direct_repairs_by_hex <- direct_project_repairs %>%
  group_by(hex_id) %>%
  summarise(
    direct_project_repairs = n_distinct(project_id),
    direct_project_units_not_integrated = safe_sum(
      selected_observed_units
    ),
    direct_project_sources = collapse_values(selected_observed_source),
    .groups = "drop"
  )

unit_parcel_summary <- unit_parcels_in_zero_hexes %>%
  group_by(hex_id) %>%
  summarise(
    unit_parcel_records = n(),
    unit_projects = n_distinct(project_id),
    zero_unit_parcels = sum(coalesce(current_units, 0) == 0),
    current_positive_parcels = sum(coalesce(current_targeted_units, 0) > 0),
    changed_current_parcels = sum(coalesce(current_units_changed, FALSE)),
    positive_improvement_parcels = sum(
      coalesce(improvement_sqft, 0) > 0
    ),
    improvement_sqft = safe_sum(improvement_sqft),
    multifamily_signal_parcels = sum(
      coalesce(is_multifamily_like, FALSE) |
        coalesce(county_model_candidate_signal, FALSE)
    ),
    multifamily_model_area_parcels = sum(
      (
        coalesce(is_multifamily_like, FALSE) |
          coalesce(county_model_candidate_signal, FALSE)
      ) &
        coalesce(model_main_area, model_improvement_sqft, 0) > 0
    ),
    reviewed_land_only_parcels = sum(
      residual_review_outcome == "retain_zero_tcad_land_only",
      na.rm = TRUE
    ),
    residual_review_outcomes = collapse_values(
      residual_review_outcome
    ),
    cross_county_relocated_parcels = sum(
      allocation_method ==
        "cross_county_overlap_use_travis_representation",
      na.rm = TRUE
    ),
    parcel_source_counties = collapse_values(source_county),
    parcel_coordinate_sources = collapse_values(coord_source),
    .groups = "drop"
  )

exclusion_records <- bind_rows(
  readRDS(input_files[["exclusions"]]) %>%
    mutate(eligibility_record_type = "excluded"),
  readRDS(input_files[["reviews"]]) %>%
    mutate(eligibility_record_type = "review")
) %>%
  filter(is.finite(lat), is.finite(lon)) %>%
  st_as_sf(
    coords = c("lon", "lat"),
    crs = 4326,
    remove = FALSE
  ) %>%
  st_transform(ANALYSIS_CRS) %>%
  st_join(
    zero_hexes %>% select(hex_id),
    join = st_within,
    left = FALSE
  ) %>%
  st_drop_geometry() %>%
  mutate(hex_id = as.character(hex_id))

exclusions_by_hex <- exclusion_records %>%
  group_by(hex_id) %>%
  summarise(
    excluded_or_review_records = n(),
    excluded_records = sum(eligibility_record_type == "excluded"),
    manual_review_records = sum(eligibility_record_type == "review"),
    excluded_raw_units = safe_sum(units_raw),
    exclusion_reasons = collapse_values(wcad_unit_exclusion_reason),
    review_reasons = collapse_values(wcad_unit_review_reason),
    .groups = "drop"
  )

################################################################################
# Step 3: Audit broader parcel-map and ACS allocation support
################################################################################

full_parcel_points <- load_full_parcel_points(ANALYSIS_CRS)
full_parcels_in_zero_hexes <- suppressWarnings(
  st_join(
    full_parcel_points,
    zero_hexes %>% select(hex_id),
    join = st_within,
    left = FALSE
  )
) %>%
  st_drop_geometry() %>%
  mutate(hex_id = as.character(hex_id))

full_parcel_support <- full_parcels_in_zero_hexes %>%
  group_by(hex_id) %>%
  summarise(
    full_parcel_centroids = n(),
    full_residential_proxy_parcels = sum(
      coalesce(full_residential_proxy, FALSE)
    ),
    full_parcel_counties = collapse_values(source_county),
    .groups = "drop"
  )

full_residential_proxy_review <- full_parcels_in_zero_hexes %>%
  filter(coalesce(full_residential_proxy, FALSE)) %>%
  select(
    hex_id,
    source_county,
    full_parcel_id,
    full_residential_proxy,
    full_use_description,
    full_residential_floor_area,
    full_building_area
  )

block_allocation <- readRDS(input_files[["block_hex"]]) %>%
  mutate(hex_id = as.character(hex_id)) %>%
  filter(hex_id %in% zero_hexes$hex_id) %>%
  group_by(hex_id) %>%
  summarise(
    contributing_census_blocks = n_distinct(block_geoid),
    census_block_population_control = safe_sum(
      block_population_contribution
    ),
    census_block_housing_control = safe_sum(
      block_housing_units_contribution
    ),
    acs_parcel_support_points = safe_sum(
      project_parcel_support_points
    ),
    parcel_supported_block_population = safe_sum(
      if_else(
        block_hex_allocation_method ==
          "residential_parcel_floor_area_proxy",
        block_population_contribution,
        0
      )
    ),
    parcel_supported_block_housing = safe_sum(
      if_else(
        block_hex_allocation_method ==
          "residential_parcel_floor_area_proxy",
        block_housing_units_contribution,
        0
      )
    ),
    point_fallback_block_population = safe_sum(
      if_else(
        block_hex_allocation_method ==
          "block_point_no_residential_parcel_support",
        block_population_contribution,
        0
      )
    ),
    point_fallback_block_housing = safe_sum(
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
    point_fallback_population_share = if_else(
      census_block_population_control > 0,
      point_fallback_block_population /
        census_block_population_control,
      0
    ),
    point_fallback_housing_share = if_else(
      census_block_housing_control > 0,
      point_fallback_block_housing /
        census_block_housing_control,
      0
    )
  )

dominant_acs_source <- readRDS(input_files[["hex_bg"]]) %>%
  mutate(
    hex_id = as.character(hex_id),
    acs_source_county = county_from_geoid(source_geoid)
  ) %>%
  filter(hex_id %in% zero_hexes$hex_id) %>%
  arrange(
    hex_id,
    desc(project_block_housing_units),
    desc(project_block_population),
    source_geoid
  ) %>%
  group_by(hex_id) %>%
  slice_head(n = 1L) %>%
  ungroup() %>%
  transmute(
    hex_id,
    dominant_acs_source_geoid = source_geoid,
    acs_source_county,
    acs_population_allocation_basis = population_allocation_basis,
    acs_housing_allocation_basis = housing_allocation_basis,
    dominant_acs_ancillary = dominant_ancillary,
    dominant_acs_ancillary_basis = dominant_ancillary_basis
  )

################################################################################
# Step 4: Attach jurisdiction context and classify residual causes
################################################################################

print_progress("Assigning jurisdiction context and residual categories...")
jurisdictions <- st_read(
  input_files[["jurisdictions"]],
  quiet = TRUE
) %>%
  st_make_valid() %>%
  st_transform(ANALYSIS_CRS) %>%
  select(
    jurisdiction_label,
    jurisdiction_type,
    jurisdiction_type_specifics
  )

zero_hex_points <- suppressWarnings(
  st_point_on_surface(zero_hexes)
) %>%
  select(hex_id, geometry)
jurisdiction_lookup <- suppressWarnings(
  st_join(
    zero_hex_points,
    jurisdictions,
    join = st_within,
    left = TRUE
  )
) %>%
  mutate(
    jurisdiction_priority = case_when(
      jurisdiction_type == "FULL" ~ 1L,
      jurisdiction_type == "LTD" ~ 2L,
      jurisdiction_type == "2MILE" ~ 3L,
      jurisdiction_type == "5MILE" ~ 4L,
      TRUE ~ 9L
    )
  ) %>%
  arrange(hex_id, jurisdiction_priority, jurisdiction_label) %>%
  group_by(hex_id) %>%
  slice_head(n = 1L) %>%
  ungroup() %>%
  st_drop_geometry() %>%
  transmute(
    hex_id,
    jurisdiction_label = coalesce(
      jurisdiction_label,
      "Outside mapped Austin jurisdiction"
    ),
    jurisdiction_type = coalesce(jurisdiction_type, "OUTSIDE"),
    jurisdiction_type_specifics
  )

zero_audit <- zero_hexes %>%
  left_join(unit_parcel_summary, by = "hex_id", relationship = "one-to-one") %>%
  left_join(direct_repairs_by_hex, by = "hex_id", relationship = "one-to-one") %>%
  left_join(exclusions_by_hex, by = "hex_id", relationship = "one-to-one") %>%
  left_join(full_parcel_support, by = "hex_id", relationship = "one-to-one") %>%
  left_join(block_allocation, by = "hex_id", relationship = "one-to-one") %>%
  left_join(dominant_acs_source, by = "hex_id", relationship = "one-to-one") %>%
  left_join(jurisdiction_lookup, by = "hex_id", relationship = "one-to-one") %>%
  mutate(
    across(
      c(
        unit_parcel_records,
        unit_projects,
        zero_unit_parcels,
        current_positive_parcels,
        changed_current_parcels,
        positive_improvement_parcels,
        improvement_sqft,
        multifamily_signal_parcels,
        multifamily_model_area_parcels,
        reviewed_land_only_parcels,
        cross_county_relocated_parcels,
        direct_project_repairs,
        direct_project_units_not_integrated,
        excluded_or_review_records,
        excluded_records,
        manual_review_records,
        excluded_raw_units,
        full_parcel_centroids,
        full_residential_proxy_parcels,
        contributing_census_blocks,
        census_block_population_control,
        census_block_housing_control,
        acs_parcel_support_points,
        parcel_supported_block_population,
        parcel_supported_block_housing,
        point_fallback_block_population,
        point_fallback_block_housing,
        point_fallback_population_share,
        point_fallback_housing_share
      ),
      ~replace_na(as.numeric(.x), 0)
    ),
    acs_housing_relative_moe = if_else(
      acs_total_housing_units > 0,
      acs_total_housing_units_moe / acs_total_housing_units,
      NA_real_
    ),
    acs_allocation_support = case_when(
      point_fallback_population_share >= 0.999 &
        point_fallback_housing_share >= 0.999 ~ "all_point_fallback",
      pmax(
        point_fallback_population_share,
        point_fallback_housing_share
      ) >= 0.5 ~ "majority_point_fallback",
      pmax(
        point_fallback_population_share,
        point_fallback_housing_share
      ) > 0 ~ "minority_point_fallback",
      TRUE ~ "parcel_supported_or_no_block_housing"
    ),
    zero_unit_category = case_when(
      direct_project_units_not_integrated > 0 ~
        "direct_project_count_not_integrated",
      cross_county_relocated_parcels > 0 ~
        "cross_county_project_relocated",
      unit_parcel_records == 0 &
        full_residential_proxy_parcels > 0 ~
        "residential_proxy_missing_from_unit_universe",
      reviewed_land_only_parcels > 0 &
        reviewed_land_only_parcels == unit_parcel_records ~
        "reviewed_land_only_multifamily_zoning",
      multifamily_signal_parcels > 0 &
        multifamily_model_area_parcels == 0 ~
        "multifamily_signal_without_model_area",
      multifamily_signal_parcels > 0 ~
        "multifamily_signal_not_selected_for_model",
      unit_parcel_records == 0 &
        excluded_records > 0 ~
        "excluded_nonresidential_accounts_only",
      unit_parcel_records == 0 &
        acs_total_housing_units < MINIMUM_MEANINGFUL_ACS_HOUSING ~
        "population_without_meaningful_housing_control",
      unit_parcel_records == 0 &
        full_parcel_centroids == 0 ~
        "no_full_parcel_centroid_support",
      unit_parcel_records == 0 &
        pmax(
          point_fallback_population_share,
          point_fallback_housing_share
        ) >= 0.5 ~
        "acs_point_fallback_without_unit_parcel",
      unit_parcel_records == 0 ~ "no_unit_parcel_other",
      positive_improvement_parcels > 0 ~
        "other_improved_zero_unit_parcel",
      TRUE ~ "other_zero_unit_parcel"
    ),
    category_label = recode(
      zero_unit_category,
      direct_project_count_not_integrated =
        "Direct count not integrated",
      cross_county_project_relocated =
        "Cross-county count relocated",
      residential_proxy_missing_from_unit_universe =
        "Residential proxy missing",
      reviewed_land_only_multifamily_zoning =
        "Reviewed land-only, MF zoning",
      multifamily_signal_without_model_area =
        "MF signal, no model area",
      multifamily_signal_not_selected_for_model =
        "MF signal not modeled",
      excluded_nonresidential_accounts_only =
        "Reviewed nonresidential exclusions",
      population_without_meaningful_housing_control =
        "Population with <5 ACS units",
      no_full_parcel_centroid_support =
        "No full-parcel centroid",
      acs_point_fallback_without_unit_parcel =
        "ACS point fallback, no unit parcel",
      no_unit_parcel_other =
        "No unit parcel, other",
      other_improved_zero_unit_parcel =
        "Other improved zero-unit parcel",
      other_zero_unit_parcel =
        "Other zero-unit parcel"
    ),
    repair_priority = case_when(
      zero_unit_category == "direct_project_count_not_integrated" ~ 1L,
      zero_unit_category ==
        "residential_proxy_missing_from_unit_universe" ~ 2L,
      zero_unit_category ==
        "reviewed_land_only_multifamily_zoning" ~ 3L,
      zero_unit_category %in% c(
        "multifamily_signal_without_model_area",
        "multifamily_signal_not_selected_for_model"
      ) ~ 4L,
      zero_unit_category == "cross_county_project_relocated" ~ 4L,
      zero_unit_category == "other_improved_zero_unit_parcel" ~ 5L,
      zero_unit_category == "excluded_nonresidential_accounts_only" ~ 6L,
      zero_unit_category == "acs_point_fallback_without_unit_parcel" ~ 7L,
      zero_unit_category == "no_full_parcel_centroid_support" ~ 8L,
      zero_unit_category ==
        "population_without_meaningful_housing_control" ~ 9L,
      TRUE ~ 10L
    ),
    recommended_action = case_when(
      zero_unit_category == "direct_project_count_not_integrated" ~
        "Apply selected direct project totals before unresolved-candidate modeling.",
      zero_unit_category == "cross_county_project_relocated" ~
        "Review whether one-parcel cross-county allocation distorts hex support.",
      zero_unit_category ==
        "residential_proxy_missing_from_unit_universe" ~
        "Trace full WCAD residential proxies into the unit-bearing source extract.",
      zero_unit_category ==
        "reviewed_land_only_multifamily_zoning" ~
        "Retain zero units; TCAD confirms land-only status and no improvement area.",
      zero_unit_category == "multifamily_signal_without_model_area" ~
        "Review parcel identity and floor-area evidence; do not impute automatically.",
      zero_unit_category == "multifamily_signal_not_selected_for_model" ~
        "Review model-candidate eligibility and source-conflict flags.",
      zero_unit_category == "excluded_nonresidential_accounts_only" ~
        "Retain exclusion unless independent residential evidence is found.",
      zero_unit_category ==
        "population_without_meaningful_housing_control" ~
        "Treat as likely group-quarters or ACS population-allocation case; no unit backfill.",
      zero_unit_category == "no_full_parcel_centroid_support" ~
        "Inspect block/hex geometry and large-parcel centroid placement.",
      zero_unit_category == "acs_point_fallback_without_unit_parcel" ~
        "Audit parcel-universe coverage; do not use ACS as an automatic unit backfill.",
      zero_unit_category == "other_improved_zero_unit_parcel" ~
        "Review land use and current unit-estimation method.",
      TRUE ~ "Retain for lower-priority manual review."
    )
  ) %>%
  arrange(repair_priority, desc(total_population), hex_id)

if (
  nrow(zero_audit) != nrow(zero_hexes) ||
    anyDuplicated(zero_audit$hex_id) ||
    any(is.na(zero_audit$zero_unit_category))
) {
  stop("The zero-unit classification failed row/category QA.", call. = FALSE)
}

################################################################################
# Step 5: Summarise findings and save outputs
################################################################################

category_summary <- zero_audit %>%
  st_drop_geometry() %>%
  group_by(
    repair_priority,
    zero_unit_category,
    category_label,
    recommended_action
  ) %>%
  summarise(
    hexes = n(),
    population = safe_sum(total_population),
    household_population = safe_sum(household_population),
    acs_housing_units = safe_sum(acs_total_housing_units),
    unit_parcel_records = safe_sum(unit_parcel_records),
    reviewed_land_only_parcels = safe_sum(
      reviewed_land_only_parcels
    ),
    full_parcel_centroids = safe_sum(full_parcel_centroids),
    full_residential_proxy_parcels = safe_sum(
      full_residential_proxy_parcels
    ),
    direct_project_repairs = safe_sum(direct_project_repairs),
    direct_project_units_not_integrated = safe_sum(
      direct_project_units_not_integrated
    ),
    .groups = "drop"
  ) %>%
  mutate(
    population_share_of_zero_unit_audit =
      population / sum(population),
    acs_housing_share_of_zero_unit_audit =
      acs_housing_units / sum(acs_housing_units)
  ) %>%
  arrange(repair_priority, desc(population))

jurisdiction_summary <- zero_audit %>%
  st_drop_geometry() %>%
  group_by(
    jurisdiction_type,
    jurisdiction_label,
    acs_source_county,
    zero_unit_category,
    category_label
  ) %>%
  summarise(
    hexes = n(),
    population = safe_sum(total_population),
    acs_housing_units = safe_sum(acs_total_housing_units),
    .groups = "drop"
  ) %>%
  arrange(
    jurisdiction_type,
    jurisdiction_label,
    acs_source_county,
    desc(population)
  )

grid_totals <- hex_comparison %>%
  st_drop_geometry() %>%
  summarise(
    hexes = n(),
    study_total_population = safe_sum(total_population),
    acs_housing_units = safe_sum(acs_total_housing_units),
    current_units = safe_sum(current_residential_units),
    current_zero_hexes = sum(current_zero_units),
    current_populated_zero_hexes = sum(
      current_zero_units & total_population > 0,
      na.rm = TRUE
    ),
    current_zero_population = safe_sum(
      total_population[current_zero_units]
    )
  )

direct_project_total <- direct_project_repairs %>%
  distinct(project_id, .keep_all = TRUE) %>%
  summarise(
    projects = n(),
    units = safe_sum(selected_observed_units)
  )

audit_summary <- tribble(
  ~metric, ~value, ~note,
  "study_hexes",
  grid_totals$hexes,
  "All H3 cells in the analysis grid.",
  "current_zero_unit_hexes",
  grid_totals$current_zero_hexes,
  "All current-feature hexes with zero parcel units.",
  "current_populated_zero_unit_hexes",
  grid_totals$current_populated_zero_hexes,
  "Current zero-unit hexes with positive allocated population.",
  "current_zero_unit_population",
  grid_totals$current_zero_population,
  "Allocated population in the current audit universe.",
  "current_zero_unit_population_share",
  grid_totals$current_zero_population / grid_totals$study_total_population,
  "Share of all allocated population in current zero-unit hexes.",
  "current_zero_unit_acs_housing_units",
  safe_sum(zero_audit$acs_total_housing_units),
  "ACS housing units allocated to the current audit universe.",
  "current_mapped_parcel_units",
  grid_totals$current_units,
  "Current promoted units mapped to the H3 grid.",
  "mapped_acs_housing_units",
  grid_totals$acs_housing_units,
  "Allocated ACS housing units on the same grid.",
  "current_minus_acs_units",
  grid_totals$current_units - grid_totals$acs_housing_units,
  "Aggregate comparison only; ACS is not a calibration target.",
  "direct_projects_not_integrated",
  direct_project_total$projects,
  paste(
    "Strict selected project counts whose integrated project-level shadow",
    "total does not match the selected total."
  ),
  "direct_project_units_not_integrated",
  direct_project_total$units,
  "Selected totals for project-level integration mismatches.",
  "all_point_fallback_hexes",
  sum(zero_audit$acs_allocation_support == "all_point_fallback"),
  "Audit hexes whose block population and housing controls use point fallback.",
  "full_residential_proxy_gap_hexes",
  sum(
    zero_audit$zero_unit_category ==
      "residential_proxy_missing_from_unit_universe"
  ),
  "Hexes with full-parcel residential proxies but no unit-universe parcel."
)

saveRDS(
  zero_audit,
  file.path(OUTPUT_DIR, "populated_zero_unit_hex_audit.rds")
)
zero_audit %>%
  st_drop_geometry() %>%
  write_csv(file.path(OUTPUT_DIR, "populated_zero_unit_hex_audit.csv"))
write_csv(
  category_summary,
  file.path(OUTPUT_DIR, "populated_zero_unit_category_summary.csv")
)
write_csv(
  jurisdiction_summary,
  file.path(OUTPUT_DIR, "populated_zero_unit_jurisdiction_summary.csv")
)
write_csv(
  unit_parcels_in_zero_hexes,
  file.path(OUTPUT_DIR, "populated_zero_unit_parcel_review.csv")
)
write_csv(
  direct_project_repairs,
  file.path(OUTPUT_DIR, "populated_zero_unit_direct_project_repairs.csv")
)
write_csv(
  full_parcel_support,
  file.path(OUTPUT_DIR, "populated_zero_unit_full_parcel_support.csv")
)
write_csv(
  full_residential_proxy_review,
  file.path(
    OUTPUT_DIR,
    "populated_zero_unit_full_residential_proxy_review.csv"
  )
)
write_csv(
  exclusion_records,
  file.path(OUTPUT_DIR, "populated_zero_unit_exclusion_review.csv")
)
write_csv(
  audit_summary,
  file.path(OUTPUT_DIR, "populated_zero_unit_audit_summary.csv")
)

category_colors <- c(
  "Direct count not integrated" = "#B2182B",
  "Cross-county count relocated" = "#7B3294",
  "Residential proxy missing" = "#E08214",
  "Reviewed land-only, MF zoning" = "#4D4D4D",
  "MF signal, no model area" = "#D6604D",
  "MF signal not modeled" = "#F4A582",
  "Reviewed nonresidential exclusions" = "#878787",
  "Population with <5 ACS units" = "#4393C3",
  "No full-parcel centroid" = "#2166AC",
  "ACS point fallback, no unit parcel" = "#67A9CF",
  "No unit parcel, other" = "#92C5DE",
  "Other improved zero-unit parcel" = "#FDB863",
  "Other zero-unit parcel" = "#BDBDBD"
)

map_boundaries <- jurisdictions %>%
  filter(jurisdiction_type %in% c("FULL", "LTD", "2MILE")) %>%
  summarise()
audit_map <- ggplot() +
  geom_sf(
    data = hex_grid,
    fill = "#F3F3F3",
    color = NA
  ) +
  geom_sf(
    data = map_boundaries,
    fill = NA,
    color = "#666666",
    linewidth = 0.25
  ) +
  geom_sf(
    data = zero_audit,
    aes(fill = category_label),
    color = "#2B2B2B",
    linewidth = 0.12
  ) +
  scale_fill_manual(
    values = category_colors,
    drop = FALSE,
    name = "Audit category"
  ) +
  coord_sf(datum = NA) +
  labs(
    title = "Populated Hexes with Zero Promoted Residential Units",
    subtitle = paste0(
      scales::comma(nrow(zero_audit)),
      " hexes; categories distinguish unit-pipeline gaps from ACS support artifacts"
    ),
    caption = paste0(
      "Population and ACS housing controls are allocated estimates. ",
      "No ACS count is used as an automatic parcel-unit backfill."
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.key.height = grid::unit(0.45, "cm"),
    plot.title = element_text(face = "bold"),
    plot.caption = element_text(color = "#555555")
  )

ggsave(
  file.path(FIGURES_DIR, "populated_zero_unit_hex_audit.png"),
  audit_map,
  width = 12,
  height = 8.5,
  dpi = 300,
  bg = "white"
)

print_progress("Populated zero-unit categories:")
print(
  category_summary %>%
    select(
      category_label,
      hexes,
      population,
      acs_housing_units,
      direct_project_units_not_integrated
    )
)
print_progress("Audit metrics:")
print(audit_summary)
print_progress(
  paste0(
    "Saved current populated zero-unit audit for ",
    scales::comma(nrow(zero_audit)),
    " hexes. No parcel, feature, or cluster estimate was modified."
  )
)
