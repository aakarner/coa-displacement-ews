################################################################################
# Validate Williamson Residential Unit Counts
################################################################################
#
# Audits the unresolved projects that touch Williamson County. The stage:
#   * distinguishes appraisal accounts from physical developments;
#   * checks official source coverage without treating noncoverage as zero;
#   * adds documented project counts for selected validation gaps/anomalies; and
#   * tests whether a comparable main/living-area measure transfers better than
#     the total-improvement-area predictor used by the shadow 02r model.
#
# This is a shadow validation stage. It does not replace production parcel
# counts or clustering inputs.
#
# Optional environment variable:
#   REFRESH_UNIT_SOURCES  Set to "true" to refresh official source caches
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
source(here::here("R", "unit_count_helpers.R"))
source(here::here("R", "unit_count_modeling.R"))

print_header("02s - VALIDATE WILLIAMSON RESIDENTIAL UNIT COUNTS")

OUTPUT_DIR <- here::here("output")
UNIT_SOURCE_DIR <- here::here("data", "raw_parcels", "unit_sources")
GROUP_FILE <- here::here("config", "williamson_project_groups.csv")
MANUAL_SOURCE_FILE <- here::here(
  "config",
  "williamson_unit_validation_sources.csv"
)
TRAINING_FILE <- file.path(OUTPUT_DIR, "residential_unit_training_table.rds")
CANDIDATE_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_model_candidates.rds"
)
PREDICTION_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_model_predictions.rds"
)
CV_PREDICTION_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_model_cv_predictions.rds"
)
FOLD_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_model_folds.csv"
)
MEMBERSHIP_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_project_membership.rds"
)
PARCEL_FILE <- file.path(
  OUTPUT_DIR,
  "residential_parcels_unit_source_attributes.rds"
)
REFRESH <- str_to_lower(Sys.getenv("REFRESH_UNIT_SOURCES", unset = "false")) %in%
  c("true", "t", "1", "yes", "y")

AEGB_CACHE <- file.path(
  UNIT_SOURCE_DIR,
  "aegb_multifamily_projects.csv"
)
TDHCA_CACHE <- file.path(
  UNIT_SOURCE_DIR,
  "tdhca_multifamily_property_inventory.geojson"
)
AEGB_URL <- paste0(
  "https://data.austintexas.gov/resource/p6d8-mube.csv",
  "?$limit=50000"
)
TDHCA_URL <- paste0(
  "https://services2.arcgis.com/lVTEPvSytcCDW86m/arcgis/rest/services/",
  "June_2026_Property_Inventory/FeatureServer/0/query",
  "?where=1%3D1",
  "&geometry=-97.95%2C30.30%2C-97.60%2C30.60",
  "&geometryType=esriGeometryEnvelope",
  "&inSR=4326",
  "&spatialRel=esriSpatialRelIntersects",
  "&outFields=*",
  "&returnGeometry=true",
  "&f=geojson"
)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(UNIT_SOURCE_DIR, recursive = TRUE, showWarnings = FALSE)

required_files <- c(
  GROUP_FILE,
  MANUAL_SOURCE_FILE,
  TRAINING_FILE,
  CANDIDATE_FILE,
  PREDICTION_FILE,
  CV_PREDICTION_FILE,
  FOLD_FILE,
  MEMBERSHIP_FILE,
  PARCEL_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Run 02p, 02q, and 02r before 02s. Missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

cache_optional_public_source <- function(cache_file, temporary_file, url, label) {
  if (file.exists(cache_file) && !REFRESH) {
    return("cached")
  }
  if (file.exists(temporary_file) && !REFRESH) {
    copied <- file.copy(temporary_file, cache_file, overwrite = TRUE)
    if (copied) {
      return("cached_from_temporary_file")
    }
  }

  print_progress(paste0("Downloading optional ", label, "..."))
  download_error <- tryCatch(
    {
      utils::download.file(
        url,
        cache_file,
        mode = "wb",
        quiet = TRUE,
        method = "libcurl"
      )
      NULL
    },
    error = function(e) e
  )
  if (!is.null(download_error) || !file.exists(cache_file)) {
    warning(
      "Could not refresh optional ",
      label,
      "; source coverage will be reported as unavailable.",
      call. = FALSE
    )
    return("unavailable")
  }
  "downloaded"
}

safe_first_character <- function(x) {
  observed <- unique(x[!is.na(x) & nzchar(x)])
  if (length(observed) == 0L) {
    return(NA_character_)
  }
  observed[[1]]
}

safe_first_number <- function(x) {
  observed <- unique(x[is.finite(x)])
  if (length(observed) == 0L) {
    return(NA_real_)
  }
  observed[[1]]
}

collapse_values <- function(x) {
  observed <- sort(unique(as.character(x[!is.na(x) & nzchar(x)])))
  if (length(observed) == 0L) {
    return(NA_character_)
  }
  paste(observed, collapse = " | ")
}

relative_difference <- function(left, right) {
  ifelse(
    is.finite(left) & is.finite(right) & pmax(left, right) > 0,
    abs(left - right) / pmax(left, right),
    NA_real_
  )
}

source_class_rank <- function(x) {
  case_when(
    x == "public_agency_project_document" ~ 1L,
    x == "public_agency_filing" ~ 2L,
    x == "owner_transaction_announcement" ~ 3L,
    x == "nonprofit_housing_inventory" ~ 4L,
    x == "industry_finance_report" ~ 5L,
    x == "industry_transaction_report" ~ 6L,
    x == "industry_property_inventory" ~ 7L,
    x == "verified_property_listing" ~ 8L,
    x == "property_management_listing" ~ 9L,
    TRUE ~ 99L
  )
}

training <- readRDS(TRAINING_FILE) %>%
  arrange(project_id)
candidates <- readRDS(CANDIDATE_FILE) %>%
  arrange(project_id)
predictions <- readRDS(PREDICTION_FILE) %>%
  arrange(project_id)
existing_cv_predictions <- readRDS(CV_PREDICTION_FILE)
model_folds <- read_csv(
  FOLD_FILE,
  col_types = cols(
    .default = col_character(),
    .row_id = col_integer()
  ),
  show_col_types = FALSE
)
membership <- readRDS(MEMBERSHIP_FILE) %>%
  mutate(
    project_id = as.character(project_id),
    parcel_id = as.character(parcel_id)
  )
parcels <- readRDS(PARCEL_FILE) %>%
  mutate(
    parcel_id = as.character(parcel_id),
    situs_address = as.character(situs_address),
    situs_city = as.character(situs_city),
    situs_zip = as.character(situs_zip)
  )
group_overrides <- read_csv(
  GROUP_FILE,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE
)
manual_sources <- read_csv(
  MANUAL_SOURCE_FILE,
  col_types = cols(
    .default = col_character(),
    reported_units = col_double()
  ),
  show_col_types = FALSE
) %>%
  mutate(source_rank = source_class_rank(source_class))

williamson_candidates <- candidates %>%
  filter(str_detect(project_counties, fixed("Williamson"))) %>%
  left_join(
    predictions %>%
      select(
        project_id,
        prediction_fixed_ratio,
        prediction_stratified_ratio,
        prediction_negative_binomial_gam,
        prediction_monotonic_xgboost,
        prediction_lower_80,
        prediction_upper_80
      ),
    by = "project_id",
    relationship = "one-to-one"
  ) %>%
  mutate(
    validation_development_id = project_id
  )

if (nrow(williamson_candidates) == 0L) {
  stop("No Williamson-touching model candidates were found.", call. = FALSE)
}
parcel_detail <- membership %>%
  filter(project_id %in% williamson_candidates$project_id) %>%
  left_join(
    parcels %>%
      select(
        parcel_id,
        source_county,
        situs_address,
        situs_city,
        situs_zip,
        lat,
        lon,
        model_improvement_sqft,
        model_main_area,
        wcad_property_id,
        wcad_property_type,
        wcad_living_area,
        wcad_dba,
        wcad_legal_description,
        wcad_property_comment,
        wcad_use_description
      ),
    by = "parcel_id",
    relationship = "many-to-one"
  ) %>%
  left_join(
    group_overrides,
    by = "parcel_id",
    relationship = "many-to-one"
  ) %>%
  mutate(
    address_key = normalize_unit_address(
      strip_unit_address_locality(situs_address, situs_city)
    ),
    zip5 = unit_zip5(situs_zip),
    wcad_reference_only_text = source_county == "Williamson" &
      str_detect(
        str_to_upper(coalesce(wcad_legal_description, "")),
        fixed("REFERENCE ONLY")
      )
  )

project_detail <- parcel_detail %>%
  group_by(project_id) %>%
  summarise(
    parcel_ids = collapse_values(parcel_id),
    parcel_addresses = collapse_values(situs_address),
    address_keys = collapse_values(address_key),
    parcel_zip5 = collapse_values(zip5),
    wcad_property_ids = collapse_values(wcad_property_id),
    wcad_property_types = collapse_values(wcad_property_type),
    wcad_dbas = collapse_values(wcad_dba),
    reviewed_group_ids = collapse_values(reviewed_group_id),
    reviewed_group_member_count = sum(!is.na(reviewed_group_id)),
    grouping_reason = collapse_values(grouping_reason),
    reviewed_allocation_parcel_id = safe_first_character(
      allocation_parcel_id
    ),
    wcad_reference_only_account = any(wcad_reference_only_text, na.rm = TRUE),
    wcad_parcel_count = sum(source_county == "Williamson", na.rm = TRUE),
    travis_parcel_count = sum(source_county == "Travis", na.rm = TRUE),
    .groups = "drop"
  )

williamson_candidates <- williamson_candidates %>%
  left_join(
    project_detail,
    by = "project_id",
    relationship = "one-to-one"
  ) %>%
  mutate(
    grouping_reason = coalesce(
      grouping_reason,
      "One candidate project treated as one validation development."
    )
  )

reviewed_group_resolution <- parcel_detail %>%
  filter(!is.na(reviewed_group_id)) %>%
  group_by(reviewed_group_id) %>%
  summarise(
    member_parcels = n_distinct(parcel_id),
    resulting_projects = n_distinct(project_id),
    .groups = "drop"
  )
if (
  nrow(reviewed_group_resolution) !=
    n_distinct(group_overrides$reviewed_group_id) ||
    any(reviewed_group_resolution$resulting_projects != 1L)
) {
  stop(
    "Reviewed Williamson companion accounts were not grouped upstream.",
    call. = FALSE
  )
}

# Refit the two transparent ratio methods on the canonical main/living-area
# field. Recreate the former total-improvement-area specification separately so
# the measurement change remains directly testable.
main_area_training <- training %>%
  mutate(project_model_floor_area = project_main_area)
main_area_candidates <- candidates %>%
  filter(project_id %in% williamson_candidates$project_id) %>%
  mutate(project_model_floor_area = project_main_area)
main_training_features <- prepare_unit_model_features(main_area_training)
main_candidate_features <- prepare_unit_model_features(
  main_area_candidates,
  medians = main_training_features$medians
)
main_fixed_model <- fit_unit_count_model(
  "fixed_ratio",
  main_training_features$data,
  seed = 42L
)
main_stratified_model <- fit_unit_count_model(
  "stratified_ratio",
  main_training_features$data,
  seed = 42L
)
main_area_predictions <- main_area_candidates %>%
  transmute(
    project_id,
    prediction_main_area_fixed = predict_unit_count_model(
      "fixed_ratio",
      main_fixed_model,
      main_candidate_features$data
    ),
    prediction_main_area_stratified = predict_unit_count_model(
      "stratified_ratio",
      main_stratified_model,
      main_candidate_features$data
    )
  )

total_improvement_training <- training %>%
  mutate(project_model_floor_area = project_improvement_sqft)
total_improvement_candidates <- candidates %>%
  filter(project_id %in% williamson_candidates$project_id) %>%
  mutate(project_model_floor_area = project_improvement_sqft)
total_training_features <- prepare_unit_model_features(
  total_improvement_training
)
total_candidate_features <- prepare_unit_model_features(
  total_improvement_candidates,
  medians = total_training_features$medians
)
total_stratified_model <- fit_unit_count_model(
  "stratified_ratio",
  total_training_features$data,
  seed = 42L
)
total_area_predictions <- total_improvement_candidates %>%
  transmute(
    project_id,
    prediction_total_improvement_area_stratified =
      predict_unit_count_model(
        "stratified_ratio",
        total_stratified_model,
        total_candidate_features$data
      )
  )

print_progress(
  "Validating the main-area ratio specifications on the existing folds..."
)
total_improvement_cv_predictions <- run_unit_count_cross_validation(
  total_improvement_training,
  model_folds,
  methods = c("fixed_ratio", "stratified_ratio"),
  seed = 42L
)
floor_area_cv_comparison <- bind_rows(
  summarise_unit_count_metrics(
    existing_cv_predictions %>%
      filter(model %in% c("fixed_ratio", "stratified_ratio")),
    c("validation_scheme", "model", "model_display")
  ) %>%
    mutate(floor_area_definition = "main_or_living_area"),
  summarise_unit_count_metrics(
    total_improvement_cv_predictions,
    c("validation_scheme", "model", "model_display")
  ) %>%
    mutate(floor_area_definition = "total_improvement_area")
) %>%
  select(
    floor_area_definition,
    validation_scheme,
    model,
    model_display,
    everything()
  )

williamson_candidates <- williamson_candidates %>%
  left_join(
    main_area_predictions,
    by = "project_id",
    relationship = "one-to-one"
  ) %>%
  left_join(
    total_area_predictions,
    by = "project_id",
    relationship = "one-to-one"
  )

development_summary <- williamson_candidates %>%
  group_by(validation_development_id) %>%
  summarise(
    candidate_project_count = n(),
    candidate_project_ids = collapse_values(project_id),
    project_counties = collapse_values(project_counties),
    cross_county_project = any(project_cross_county),
    cross_county_address_overlap = any(
      project_cross_county_address_overlap
    ),
    project_area_aggregation_method = safe_first_character(
      project_area_aggregation_method
    ),
    grouping_reason = safe_first_character(grouping_reason),
    parcel_count = sum(project_parcel_count),
    parcel_ids = collapse_values(parcel_ids),
    parcel_addresses = collapse_values(parcel_addresses),
    address_keys = collapse_values(address_keys),
    parcel_zip5 = collapse_values(parcel_zip5),
    wcad_property_ids = collapse_values(wcad_property_ids),
    wcad_property_types = collapse_values(wcad_property_types),
    wcad_dbas = collapse_values(wcad_dbas),
    reviewed_group_ids = collapse_values(reviewed_group_ids),
    reviewed_group_member_count = sum(reviewed_group_member_count),
    reviewed_allocation_parcel_id = safe_first_character(
      reviewed_allocation_parcel_id
    ),
    wcad_reference_only_account = any(wcad_reference_only_account),
    project_model_floor_area = sum(
      project_model_floor_area,
      na.rm = TRUE
    ),
    project_main_area = sum(project_main_area, na.rm = TRUE),
    project_main_area_raw_sum = sum(
      project_main_area_raw_sum,
      na.rm = TRUE
    ),
    project_improvement_sqft = sum(project_improvement_sqft, na.rm = TRUE),
    project_improvement_sqft_raw_sum = sum(
      project_improvement_sqft_raw_sum,
      na.rm = TRUE
    ),
    project_land_sqft = sum(project_land_sqft, na.rm = TRUE),
    current_primary_units = sum(current_primary_units),
    current_conservative_units = sum(current_conservative_units),
    prediction_total_improvement_area_stratified = sum(
      prediction_total_improvement_area_stratified
    ),
    prediction_main_area_fixed = sum(prediction_main_area_fixed),
    prediction_main_area_stratified = sum(
      prediction_main_area_stratified
    ),
    uro_estimate_count = n_distinct(
      uro_sensitivity_units[is.finite(uro_sensitivity_units)]
    ),
    uro_estimated_units = safe_first_number(uro_sensitivity_units),
    .groups = "drop"
  ) %>%
  mutate(
    companion_account_group = !is.na(reviewed_group_ids),
    uro_conflict = uro_estimate_count > 1L
  )

if (any(development_summary$uro_conflict)) {
  stop(
    "A validation development has conflicting URO estimates.",
    call. = FALSE
  )
}
unknown_manual_developments <- setdiff(
  manual_sources$validation_development_id,
  development_summary$validation_development_id
)
if (length(unknown_manual_developments) > 0L) {
  stop(
    "Manual validation file contains unknown developments: ",
    paste(unknown_manual_developments, collapse = ", "),
    call. = FALSE
  )
}

manual_summary <- manual_sources %>%
  arrange(validation_development_id, source_rank, source_name) %>%
  group_by(validation_development_id) %>%
  summarise(
    manual_source_count = n_distinct(validation_record_id),
    manual_source_names = collapse_values(source_name),
    manual_source_classes = collapse_values(source_class),
    strongest_manual_source_class = source_class[[which.min(source_rank)]],
    strongest_manual_source_name = source_name[[which.min(source_rank)]],
    strongest_manual_source_url = source_url[[which.min(source_rank)]],
    manual_min_units = min(reported_units),
    manual_max_units = max(reported_units),
    manual_reported_units = median(reported_units),
    manual_relative_spread = unit_relative_spread(reported_units),
    .groups = "drop"
  )

development_validation <- development_summary %>%
  left_join(
    manual_summary,
    by = "validation_development_id",
    relationship = "one-to-one"
  ) %>%
  mutate(
    manual_uro_relative_difference = relative_difference(
      manual_reported_units,
      uro_estimated_units
    ),
    manual_uro_agree = is.finite(manual_uro_relative_difference) &
      manual_uro_relative_difference <= 0.10,
    validation_reference_units = coalesce(
      manual_reported_units,
      uro_estimated_units
    ),
    validation_reference_source = case_when(
      is.finite(manual_reported_units) & manual_uro_agree ~
        "documented_source_and_uro_agree",
      is.finite(manual_reported_units) ~ strongest_manual_source_class,
      is.finite(uro_estimated_units) ~ "austin_uro_estimate_only",
      TRUE ~ "no_external_unit_reference"
    ),
    validation_confidence = case_when(
      is.finite(manual_reported_units) & manual_uro_agree ~
        "corroborated",
      strongest_manual_source_class ==
        "public_agency_project_document" ~ "high",
      is.finite(manual_reported_units) ~ "moderate",
      is.finite(uro_estimated_units) ~ "provisional",
      TRUE ~ "unvalidated"
    ),
    main_area_sqft_per_reference_unit = if_else(
      is.finite(validation_reference_units) &
        validation_reference_units > 0,
      project_main_area / validation_reference_units,
      NA_real_
    ),
    improvement_sqft_per_reference_unit = if_else(
      is.finite(validation_reference_units) &
        validation_reference_units > 0,
      project_improvement_sqft / validation_reference_units,
      NA_real_
    ),
    main_area_measure_anomaly = is.finite(
      main_area_sqft_per_reference_unit
    ) & (
      main_area_sqft_per_reference_unit < 400 |
        main_area_sqft_per_reference_unit > 1600
    ),
    appraisal_structure_flag = case_when(
      cross_county_project ~ "cross_county_overlapping_appraisal_records",
      companion_account_group ~ "companion_wcad_accounts",
      wcad_reference_only_account ~ "wcad_reference_only_text",
      TRUE ~ "one_appraisal_project"
    ),
    validation_status = case_when(
      validation_confidence == "unvalidated" ~
        "manual_unit_research_needed",
      main_area_measure_anomaly ~
        "unit_count_supported_but_area_not_model_ready",
      cross_county_project ~
        "unit_count_supported_cross_county_area_review",
      TRUE ~ "usable_for_williamson_sensitivity_validation"
    )
  )

metric_values <- function(observed, predicted) {
  keep <- is.finite(observed) & observed > 0 &
    is.finite(predicted) & predicted > 0
  if (!any(keep)) {
    return(
      tibble(
        developments = 0L,
        observed_units = 0,
        predicted_units = 0,
        wape = NA_real_,
        bias = NA_real_,
        median_ape = NA_real_
      )
    )
  }
  observed <- observed[keep]
  predicted <- predicted[keep]
  tibble(
    developments = length(observed),
    observed_units = sum(observed),
    predicted_units = sum(predicted),
    wape = sum(abs(predicted - observed)) / sum(observed),
    bias = sum(predicted - observed) / sum(observed),
    median_ape = median(abs(predicted - observed) / observed)
  )
}

strategy_long <- development_validation %>%
  select(
    validation_development_id,
    project_counties,
    cross_county_project,
    companion_account_group,
    main_area_measure_anomaly,
    validation_confidence,
    validation_reference_units,
    manual_reported_units,
    uro_estimated_units,
    current_primary_units,
    current_conservative_units,
    prediction_total_improvement_area_stratified,
    prediction_main_area_fixed,
    prediction_main_area_stratified
  ) %>%
  pivot_longer(
    cols = c(
      current_primary_units,
      current_conservative_units,
      prediction_total_improvement_area_stratified,
      prediction_main_area_fixed,
      prediction_main_area_stratified
    ),
    names_to = "strategy",
    values_to = "predicted_units"
  ) %>%
  mutate(
    evaluation_group = case_when(
      cross_county_project ~ "cross_county",
      main_area_measure_anomaly ~ "pure_williamson_area_anomaly",
      TRUE ~ "pure_williamson_comparable_area"
    )
  )

strategy_comparison <- bind_rows(
  strategy_long %>%
    filter(is.finite(validation_reference_units)) %>%
    group_by(strategy) %>%
    group_modify(
      ~metric_values(
        .x$validation_reference_units,
        .x$predicted_units
      )
    ) %>%
    ungroup() %>%
    mutate(evaluation_group = "all_reference_developments"),
  strategy_long %>%
    filter(is.finite(manual_reported_units)) %>%
    group_by(strategy) %>%
    group_modify(
      ~metric_values(
        .x$manual_reported_units,
        .x$predicted_units
      )
    ) %>%
    ungroup() %>%
    mutate(evaluation_group = "documented_reference_developments"),
  strategy_long %>%
    filter(
      is.finite(manual_reported_units),
      !main_area_measure_anomaly,
      !cross_county_project
    ) %>%
    group_by(strategy) %>%
    group_modify(
      ~metric_values(
        .x$manual_reported_units,
        .x$predicted_units
      )
    ) %>%
    ungroup() %>%
    mutate(
      evaluation_group = "documented_comparable_area_developments"
    ),
  strategy_long %>%
    filter(
      is.finite(uro_estimated_units),
      !is.finite(manual_reported_units)
    ) %>%
    group_by(strategy) %>%
    group_modify(
      ~metric_values(
        .x$uro_estimated_units,
        .x$predicted_units
      )
    ) %>%
    ungroup() %>%
    mutate(evaluation_group = "uro_only_reference_developments"),
  strategy_long %>%
    filter(is.finite(validation_reference_units)) %>%
    group_by(evaluation_group, strategy) %>%
    group_modify(
      ~metric_values(
        .x$validation_reference_units,
        .x$predicted_units
      )
    ) %>%
    ungroup()
) %>%
  select(
    evaluation_group,
    strategy,
    developments,
    observed_units,
    predicted_units,
    wape,
    bias,
    median_ape
  )

official_source_status <- tibble(
  source_name = c(
    "Austin Energy Green Building multifamily projects",
    "TDHCA multifamily property inventory",
    "HUD multifamily properties"
  ),
  source_url = c(
    "https://data.austintexas.gov/d/p6d8-mube",
    paste0(
      "https://services2.arcgis.com/lVTEPvSytcCDW86m/arcgis/rest/",
      "services/June_2026_Property_Inventory/FeatureServer/0"
    ),
    paste0(
      "https://egis.hud.gov/arcgis/rest/services/cpdmaps/",
      "HudMfProps/MapServer/1"
    )
  ),
  source_scope = c(
    "Austin Energy rated or participating multifamily projects",
    "TDHCA-financed multifamily properties",
    "HUD insured or assisted multifamily properties"
  ),
  source_status = c(
    cache_optional_public_source(
      AEGB_CACHE,
      "/tmp/aegb_multifamily.csv",
      AEGB_URL,
      "AEGB multifamily inventory"
    ),
    cache_optional_public_source(
      TDHCA_CACHE,
      "/tmp/tdhca_inventory_local.geojson",
      TDHCA_URL,
      "TDHCA multifamily inventory"
    ),
    "unavailable_after_api_timeout"
  )
)

development_addresses <- development_validation %>%
  select(validation_development_id) %>%
  left_join(
    williamson_candidates %>%
      select(project_id, validation_development_id) %>%
      left_join(
        parcel_detail %>%
          select(project_id, address_key, zip5),
        by = "project_id",
        relationship = "one-to-many"
      ),
    by = "validation_development_id",
    relationship = "one-to-many"
  ) %>%
  filter(!is.na(address_key)) %>%
  distinct(validation_development_id, address_key, zip5)

official_matches <- tibble(
  validation_development_id = character(),
  official_source = character(),
  official_source_record_id = character(),
  official_reported_units = double(),
  match_method = character()
)

if (file.exists(AEGB_CACHE)) {
  aegb <- read_csv(
    AEGB_CACHE,
    col_types = cols(.default = col_character()),
    show_col_types = FALSE
  ) %>%
    transmute(
      official_source_record_id = paste0("aegb:", aegb_id),
      address_key = normalize_unit_address(project_address),
      zip5 = unit_zip5(zip_code),
      official_reported_units = unit_numeric(of_residential_units)
    ) %>%
    filter(
      !is.na(address_key),
      is.finite(official_reported_units),
      official_reported_units > 0
    )
  official_matches <- bind_rows(
    official_matches,
    development_addresses %>%
      inner_join(
        aegb,
        by = c("address_key", "zip5"),
        relationship = "many-to-many"
      ) %>%
      transmute(
        validation_development_id,
        official_source = "AEGB",
        official_source_record_id,
        official_reported_units,
        match_method = "exact_normalized_address_zip"
      )
  )
}

if (file.exists(TDHCA_CACHE)) {
  tdhca <- st_read(TDHCA_CACHE, quiet = TRUE) %>%
    st_drop_geometry() %>%
    transmute(
      official_source_record_id = paste0(
        "tdhca:",
        coalesce(as.character(TDHCA_), as.character(FID))
      ),
      address_key = normalize_unit_address(Project_Ad),
      zip5 = unit_zip5(Zip_Code),
      official_reported_units = unit_numeric(Total_Unit)
    ) %>%
    filter(
      !is.na(address_key),
      is.finite(official_reported_units),
      official_reported_units > 0
    )
  official_matches <- bind_rows(
    official_matches,
    development_addresses %>%
      inner_join(
        tdhca,
        by = c("address_key", "zip5"),
        relationship = "many-to-many"
      ) %>%
      transmute(
        validation_development_id,
        official_source = "TDHCA",
        official_source_record_id,
        official_reported_units,
        match_method = "exact_normalized_address_zip"
      )
  )
}

official_coverage <- official_source_status %>%
  mutate(
    matched_developments = case_when(
      str_detect(source_name, fixed("Austin Energy")) ~
        n_distinct(
          official_matches$validation_development_id[
            official_matches$official_source == "AEGB"
          ]
        ),
      str_detect(source_name, fixed("TDHCA")) ~
        n_distinct(
          official_matches$validation_development_id[
            official_matches$official_source == "TDHCA"
          ]
        ),
      TRUE ~ NA_integer_
    ),
    candidate_developments = nrow(development_validation),
    nonmatch_interpretation = case_when(
      str_detect(source_name, fixed("Austin Energy")) ~
        "Nonmatch means no rated/participating-project record, not zero units.",
      str_detect(source_name, fixed("TDHCA")) ~
        "Nonmatch means no TDHCA-financed record, not zero units.",
      TRUE ~
        "Service was unavailable; no negative inference is permitted."
    )
  )

measurement_qa <- bind_rows(
  training %>%
    transmute(
      comparison_group = "Travis direct-label training",
      main_area_sqft_per_unit = project_main_area / unit_count,
      improvement_sqft_per_unit = project_improvement_sqft / unit_count
    ),
  development_validation %>%
    filter(is.finite(validation_reference_units)) %>%
    transmute(
      comparison_group = case_when(
        cross_county_project ~ "Williamson-touching cross-county",
        companion_account_group ~ "Williamson companion-account group",
        TRUE ~ "Williamson single-project"
      ),
      main_area_sqft_per_unit = main_area_sqft_per_reference_unit,
      improvement_sqft_per_unit = improvement_sqft_per_reference_unit
    )
) %>%
  group_by(comparison_group) %>%
  summarise(
    projects = n(),
    median_main_area_sqft_per_unit = median(
      main_area_sqft_per_unit,
      na.rm = TRUE
    ),
    main_area_q25 = quantile(
      main_area_sqft_per_unit,
      0.25,
      na.rm = TRUE
    ),
    main_area_q75 = quantile(
      main_area_sqft_per_unit,
      0.75,
      na.rm = TRUE
    ),
    median_improvement_sqft_per_unit = median(
      improvement_sqft_per_unit,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

manual_review <- development_validation %>%
  filter(
    validation_confidence %in% c("provisional", "unvalidated") |
      main_area_measure_anomaly |
      cross_county_project |
      companion_account_group
  ) %>%
  mutate(
    review_priority = case_when(
      validation_confidence == "unvalidated" ~ "high",
      cross_county_project |
        companion_account_group |
        main_area_measure_anomaly ~ "high",
      TRUE ~ "medium"
    ),
    review_reason = case_when(
      validation_confidence == "unvalidated" ~
        "no_external_unit_reference",
      cross_county_project ~
        "cross_county_area_may_duplicate_or_mix_appraisal_measurements",
      companion_account_group ~
        "multiple_wcad_accounts_represent_one_development",
      main_area_measure_anomaly ~
        "appraisal_main_area_implausible_for_supported_unit_count",
      TRUE ~ "uro_estimate_not_independently_corroborated"
    )
  )

candidate_audit <- williamson_candidates %>%
  select(
    project_id,
    validation_development_id,
    project_counties,
    project_cross_county,
    project_cross_county_address_overlap,
    project_area_aggregation_method,
    project_parcel_count,
    parcel_ids,
    parcel_addresses,
    wcad_property_ids,
    wcad_property_types,
    wcad_dbas,
    reviewed_group_ids,
    reviewed_allocation_parcel_id,
    wcad_reference_only_account,
    project_model_floor_area,
    project_main_area,
    project_main_area_raw_sum,
    project_improvement_sqft,
    project_improvement_sqft_raw_sum,
    current_primary_units,
    current_conservative_units,
    uro_sensitivity_units,
    prediction_stratified_ratio,
    prediction_total_improvement_area_stratified,
    prediction_main_area_fixed,
    prediction_main_area_stratified,
    grouping_reason
  )

save_output(
  development_validation,
  file.path(OUTPUT_DIR, "residential_unit_williamson_validation.rds"),
  "Williamson development unit validation"
)
write_csv(
  development_validation,
  file.path(OUTPUT_DIR, "residential_unit_williamson_validation.csv")
)
write_csv(
  candidate_audit,
  file.path(OUTPUT_DIR, "residential_unit_williamson_candidate_audit.csv")
)
write_csv(
  manual_sources %>% select(-source_rank),
  file.path(
    OUTPUT_DIR,
    "residential_unit_williamson_manual_sources.csv"
  )
)
write_csv(
  official_matches,
  file.path(
    OUTPUT_DIR,
    "residential_unit_williamson_official_matches.csv"
  )
)
write_csv(
  official_coverage,
  file.path(
    OUTPUT_DIR,
    "residential_unit_williamson_source_coverage.csv"
  )
)
write_csv(
  measurement_qa,
  file.path(
    OUTPUT_DIR,
    "residential_unit_williamson_measurement_qa.csv"
  )
)
write_csv(
  strategy_comparison,
  file.path(
    OUTPUT_DIR,
    "residential_unit_williamson_strategy_comparison.csv"
  )
)
write_csv(
  total_improvement_cv_predictions,
  file.path(
    OUTPUT_DIR,
    "residential_unit_total_improvement_model_cv_predictions.csv"
  )
)
write_csv(
  floor_area_cv_comparison,
  file.path(
    OUTPUT_DIR,
    "residential_unit_floor_area_cv_comparison.csv"
  )
)
write_csv(
  manual_review,
  file.path(
    OUTPUT_DIR,
    "residential_unit_williamson_manual_review.csv"
  )
)

print_progress(
  paste0(
    "Audited ",
    nrow(williamson_candidates),
    " candidate projects as ",
    nrow(development_validation),
    " physical developments."
  )
)
print_progress(
  paste0(
    "External reference coverage: ",
    sum(is.finite(development_validation$validation_reference_units)),
    " developments; corroborated/high confidence: ",
    sum(
      development_validation$validation_confidence %in%
        c("corroborated", "high")
    ),
    "."
  )
)
print_progress(
  paste0(
    "Main-area measurement anomalies: ",
    sum(development_validation$main_area_measure_anomaly, na.rm = TRUE),
    "; high-priority structural reviews: ",
    sum(manual_review$review_priority == "high"),
    "."
  )
)

best_comparable_strategy <- strategy_comparison %>%
  filter(
    evaluation_group == "pure_williamson_comparable_area",
    is.finite(wape)
  ) %>%
  arrange(wape, abs(bias)) %>%
  slice_head(n = 1)
if (nrow(best_comparable_strategy) == 1L) {
  print_progress(
    paste0(
      "Lowest WAPE for comparable-area Williamson developments: ",
      best_comparable_strategy$strategy,
      " (",
      scales::percent(best_comparable_strategy$wape, accuracy = 0.1),
      ")."
    )
  )
}

print_progress(
  paste0(
    "Williamson validation complete. Production counts remain unchanged. ",
    "Review output/residential_unit_williamson_manual_review.csv before ",
    "integrating any model."
  )
)
