################################################################################
# Fit and Compare Residential Unit Count Models
################################################################################
#
# Compares physical-property models for unresolved multifamily projects. This
# is a shadow stage: predictions are not inserted into production parcel counts
# or clustering inputs. Direct, deterministic, and rule-based counts remain
# untouched.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(mgcv)
  library(readr)
  library(scales)
  library(tibble)
  library(tidyr)
  library(xgboost)
})

source(here::here("R", "utils.R"))
source(here::here("R", "unit_count_modeling.R"))

print_header("02r - FIT AND COMPARE RESIDENTIAL UNIT COUNT MODELS")

OUTPUT_DIR <- here::here("output")
FIGURE_DIR <- here::here("figures")
TRAINING_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_training_table.rds"
)
CANDIDATE_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_model_candidates.rds"
)
RANDOM_SEED <- 42L
PROJECT_FOLDS <- 5L

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

missing_inputs <- c(TRAINING_FILE, CANDIDATE_FILE)[
  !file.exists(c(TRAINING_FILE, CANDIDATE_FILE))
]
if (length(missing_inputs) > 0L) {
  stop(
    "Run the unit source and project targets before model fitting. Missing: ",
    paste(missing_inputs, collapse = ", "),
    call. = FALSE
  )
}

training <- readRDS(TRAINING_FILE) %>%
  arrange(project_id)
candidates <- readRDS(CANDIDATE_FILE) %>%
  arrange(project_id)

if (anyDuplicated(training$project_id) || anyDuplicated(candidates$project_id)) {
  stop("Training and candidate project IDs must be unique.", call. = FALSE)
}
if (any(training$unit_count < 5 | !is.finite(training$unit_count))) {
  stop("Training labels must be finite project counts of at least five.", call. = FALSE)
}
if (length(intersect(training$project_id, candidates$project_id)) > 0L) {
  stop("Training and unresolved candidate projects must not overlap.", call. = FALSE)
}

print_progress(
  paste0(
    "Training projects: ",
    comma(nrow(training)),
    "; unresolved candidates: ",
    comma(nrow(candidates))
  )
)

folds <- make_unit_model_folds(
  training,
  k = PROJECT_FOLDS,
  seed = RANDOM_SEED
)
fold_qa <- folds %>%
  left_join(
    training %>%
      transmute(
        project_id,
        label_source,
        observed_size_band = unit_model_size_band(unit_count),
        unit_count
      ),
    by = "project_id",
    relationship = "many-to-one"
  ) %>%
  group_by(validation_scheme, fold_id) %>%
  summarise(
    projects = n(),
    observed_units = sum(unit_count),
    minimum_units = min(unit_count),
    median_units = median(unit_count),
    maximum_units = max(unit_count),
    label_sources = n_distinct(label_source),
    size_bands = n_distinct(observed_size_band),
    .groups = "drop"
  )

print_progress("Running project, spatial, and source-holdout validation...")
cv_predictions <- run_unit_count_cross_validation(
  training,
  folds,
  methods = unit_model_methods(),
  seed = RANDOM_SEED
)

expected_cv_rows <- nrow(training) *
  length(unit_model_methods()) *
  3L
if (nrow(cv_predictions) != expected_cv_rows) {
  stop(
    "Cross-validation output has ",
    nrow(cv_predictions),
    " rows; expected ",
    expected_cv_rows,
    ".",
    call. = FALSE
  )
}
if (
  anyDuplicated(
    cv_predictions[
      c("validation_scheme", "project_id", "model")
    ]
  )
) {
  stop("Cross-validation predictions are not project-unique.", call. = FALSE)
}

fold_metrics <- summarise_unit_count_metrics(
  cv_predictions,
  c("validation_scheme", "fold_id", "model", "model_display")
)
pooled_metrics <- summarise_unit_count_metrics(
  cv_predictions,
  c("validation_scheme", "model", "model_display")
)
size_metrics <- summarise_unit_count_metrics(
  cv_predictions,
  c(
    "validation_scheme",
    "model",
    "model_display",
    "observed_size_band"
  )
)
source_metrics <- summarise_unit_count_metrics(
  cv_predictions,
  c("validation_scheme", "model", "model_display", "label_source")
)

model_recommendation <- select_unit_count_model(
  fold_metrics,
  size_metrics
)
recommended_model <- model_recommendation$recommended_model[[1]]
if (!recommended_model %in% unit_model_methods()) {
  stop("Model recommendation did not return a known method.", call. = FALSE)
}

print_progress(
  paste0(
    "Validation recommendation: ",
    unit_model_display_name(recommended_model)
  )
)

training_features <- prepare_unit_model_features(training)
candidate_features <- prepare_unit_model_features(
  candidates,
  medians = training_features$medians
)

print_progress("Fitting all candidate methods on the complete training set...")
fitted_models <- list()
candidate_prediction_columns <- list()
for (method_index in seq_along(unit_model_methods())) {
  method <- unit_model_methods()[[method_index]]
  fitted_models[[method]] <- fit_unit_count_model(
    method,
    training_features$data,
    seed = RANDOM_SEED + method_index
  )
  candidate_prediction_columns[[method]] <- predict_unit_count_model(
    method,
    fitted_models[[method]],
    candidate_features$data
  )
}

candidate_predictions <- candidates %>%
  transmute(
    project_id,
    source_county,
    project_counties,
    project_cross_county,
    project_cross_county_address_overlap,
    project_area_aggregation_method,
    project_model_floor_area,
    project_model_floor_area_source,
    project_improvement_sqft,
    project_improvement_sqft_raw_sum,
    project_main_area,
    project_main_area_raw_sum,
    project_land_sqft,
    project_year_built,
    project_max_stories,
    project_parcel_count,
    project_address_count,
    project_condo_account_count,
    project_b1_parcel_count,
    project_has_mf_zoning,
    project_has_commercial_mixed_zoning,
    current_primary_units,
    current_conservative_units,
    current_floor_area_estimate,
    current_needs_multifamily_estimate,
    uro_sensitivity_units
  )

for (method in unit_model_methods()) {
  candidate_predictions[[paste0("prediction_", method)]] <-
    candidate_prediction_columns[[method]]
}

recommended_prediction_column <- paste0(
  "prediction_",
  recommended_model
)
candidate_predictions <- candidate_predictions %>%
  mutate(
    recommended_model = recommended_model,
    recommended_prediction = .data[[recommended_prediction_column]],
    recommended_prediction_integer = pmax(
      round(recommended_prediction),
      2
    ),
    prediction_size_band = unit_model_size_band(recommended_prediction)
  )

interval_calibration <- unit_prediction_interval_calibration(
  cv_predictions,
  recommended_model
)
interval_validation <- validate_unit_prediction_intervals(
  cv_predictions,
  recommended_model
)
candidate_predictions <- candidate_predictions %>%
  left_join(
    interval_calibration %>%
      select(
        prediction_size_band,
        interval_calibration_n = n,
        lower_multiplier,
        upper_multiplier,
        interval_used_global_calibration = used_global_calibration
      ),
    by = "prediction_size_band",
    relationship = "many-to-one"
  ) %>%
  mutate(
    prediction_lower_80 = pmax(
      recommended_prediction * lower_multiplier,
      1
    ),
    prediction_upper_80 = pmax(
      recommended_prediction * upper_multiplier,
      prediction_lower_80
    ),
    prediction_lower_80_integer = pmax(round(prediction_lower_80), 1),
    prediction_upper_80_integer = pmax(
      round(prediction_upper_80),
      recommended_prediction_integer
    )
  )

training_ranges <- unit_model_training_ranges(training)
range_value <- function(predictor, bound) {
  training_ranges[[bound]][training_ranges$predictor == predictor]
}

candidate_predictions <- candidate_predictions %>%
  mutate(
    county_transfer_flag = project_counties != "Travis",
    missing_core_predictor = is.na(project_land_sqft) |
      is.na(project_year_built) |
      is.na(project_max_stories),
    floor_area_outside_training_range =
      project_model_floor_area <
        range_value("project_model_floor_area", "minimum") |
        project_model_floor_area >
          range_value("project_model_floor_area", "maximum"),
    land_area_outside_training_range = !is.na(project_land_sqft) &
      (
        project_land_sqft <
          range_value("project_land_sqft", "minimum") |
          project_land_sqft >
            range_value("project_land_sqft", "maximum")
      ),
    year_outside_training_range = !is.na(project_year_built) &
      (
        project_year_built <
          range_value("project_year_built", "minimum") |
          project_year_built >
            range_value("project_year_built", "maximum")
      ),
    stories_outside_training_range = !is.na(project_max_stories) &
      (
        project_max_stories <
          range_value("project_max_stories", "minimum") |
          project_max_stories >
            range_value("project_max_stories", "maximum")
      ),
    outside_training_range = floor_area_outside_training_range |
      land_area_outside_training_range |
      year_outside_training_range |
      stories_outside_training_range,
    floor_area_outside_central_range =
      project_model_floor_area <
        range_value("project_model_floor_area", "central_minimum") |
        project_model_floor_area >
          range_value("project_model_floor_area", "central_maximum"),
    below_training_unit_scope = recommended_prediction < 5,
    model_out_of_domain = county_transfer_flag |
      missing_core_predictor |
      outside_training_range,
    production_prediction_eligible = !model_out_of_domain &
      !below_training_unit_scope,
    prediction_caution = case_when(
      county_transfer_flag ~ "county_transfer",
      missing_core_predictor ~ "missing_predictor_imputed",
      outside_training_range ~ "outside_training_range",
      below_training_unit_scope ~ "below_training_unit_scope",
      floor_area_outside_central_range ~ "tail_of_training_distribution",
      TRUE ~ "within_training_domain"
    )
  )

if (
  any(!is.finite(candidate_predictions$recommended_prediction)) ||
    any(candidate_predictions$recommended_prediction <= 0) ||
    any(
      candidate_predictions$prediction_lower_80 >
        candidate_predictions$prediction_upper_80
    )
) {
  stop("Candidate predictions or intervals failed validation.", call. = FALSE)
}

model_candidate_long <- candidate_predictions %>%
  select(
    project_id,
    project_counties,
    uro_sensitivity_units,
    all_of(paste0("prediction_", unit_model_methods()))
  ) %>%
  pivot_longer(
    starts_with("prediction_"),
    names_prefix = "prediction_",
    names_to = "model",
    values_to = "predicted_units"
  )

current_candidate_long <- bind_rows(
  candidate_predictions %>%
    transmute(
      project_id,
      project_counties,
      uro_sensitivity_units,
      model = "current_primary",
      predicted_units = current_primary_units
    ),
  candidate_predictions %>%
    transmute(
      project_id,
      project_counties,
      uro_sensitivity_units,
      model = "current_conservative",
      predicted_units = current_conservative_units
    )
)
candidate_comparison_long <- bind_rows(
  current_candidate_long,
  model_candidate_long
)

uro_sensitivity_rows <- candidate_comparison_long %>%
  filter(
    is.finite(uro_sensitivity_units),
    uro_sensitivity_units > 0
  ) %>%
  mutate(
    model_display = unit_model_display_name(model),
    county_validation_group = if_else(
      project_counties == "Travis",
      "Travis",
      "Williamson transfer"
    )
  )
uro_sensitivity_metrics <- uro_sensitivity_rows %>%
  group_by(model, model_display) %>%
  group_modify(
    ~unit_count_metric_values(
      .x$uro_sensitivity_units,
      .x$predicted_units
    )
  ) %>%
  ungroup()

uro_sensitivity_by_county <- uro_sensitivity_rows %>%
  group_by(model, model_display, county_validation_group) %>%
  group_modify(
    ~unit_count_metric_values(
      .x$uro_sensitivity_units,
      .x$predicted_units
    )
  ) %>%
  ungroup()

candidate_summary <- candidate_comparison_long %>%
  group_by(model) %>%
  summarise(
    projects = n(),
    total_predicted_units = sum(predicted_units),
    median_prediction = median(predicted_units),
    maximum_prediction = max(predicted_units),
    .groups = "drop"
  ) %>%
  mutate(model_display = unit_model_display_name(model))

integration_scenarios <- tibble(
  scenario = c(
    "current_primary_all_candidates",
    "current_conservative_all_candidates",
    "stratified_all_candidates_for_comparison",
    "eligible_stratified_primary_review_fallback",
    "eligible_stratified_conservative_review_fallback"
  ),
  model_projects = c(
    0L,
    0L,
    nrow(candidate_predictions),
    sum(candidate_predictions$production_prediction_eligible),
    sum(candidate_predictions$production_prediction_eligible)
  ),
  fallback_projects = c(
    nrow(candidate_predictions),
    nrow(candidate_predictions),
    0L,
    sum(!candidate_predictions$production_prediction_eligible),
    sum(!candidate_predictions$production_prediction_eligible)
  ),
  candidate_units = c(
    sum(candidate_predictions$current_primary_units),
    sum(candidate_predictions$current_conservative_units),
    sum(candidate_predictions$recommended_prediction),
    sum(
      candidate_predictions$recommended_prediction[
        candidate_predictions$production_prediction_eligible
      ]
    ) +
      sum(
        candidate_predictions$current_primary_units[
          !candidate_predictions$production_prediction_eligible
        ]
      ),
    sum(
      candidate_predictions$recommended_prediction[
        candidate_predictions$production_prediction_eligible
      ]
    ) +
      sum(
        candidate_predictions$current_conservative_units[
          !candidate_predictions$production_prediction_eligible
        ]
      )
  ),
  production_status = c(
    "current",
    "current_sensitivity",
    "shadow_comparison_only",
    "shadow_scenario_only",
    "shadow_scenario_only"
  )
)

out_of_domain_qa <- candidate_predictions %>%
  summarise(
    candidates = n(),
    county_transfer = sum(county_transfer_flag),
    missing_core_predictor = sum(missing_core_predictor),
    outside_training_range = sum(outside_training_range),
    below_training_unit_scope = sum(below_training_unit_scope),
    floor_area_outside_central_range =
      sum(floor_area_outside_central_range),
    model_out_of_domain = sum(model_out_of_domain),
    production_prediction_eligible_units = sum(
      if_else(
        production_prediction_eligible,
        recommended_prediction,
        0
      )
    ),
    production_prediction_eligible_projects =
      sum(production_prediction_eligible)
  ) %>%
  pivot_longer(
    everything(),
    names_to = "metric",
    values_to = "value"
  )

xgb_importance <- xgboost::xgb.importance(
  feature_names = unit_xgboost_predictors(),
  model = fitted_models$monotonic_xgboost
)

xgb_monotonicity_features <- candidate_features$data
xgb_floor_area_increase <- xgb_monotonicity_features
xgb_floor_area_increase$.model_floor_area <-
  xgb_floor_area_increase$.model_floor_area * 1.10
xgb_floor_area_increase$.model_log_floor_area <-
  log(xgb_floor_area_increase$.model_floor_area)
xgb_floor_area_increase$.model_log_far <- log(
  pmax(
    xgb_floor_area_increase$.model_floor_area /
      pmax(xgb_floor_area_increase$.model_land_area, 1),
    0.001
  )
)
xgb_base_prediction <- predict_monotonic_unit_xgboost(
  fitted_models$monotonic_xgboost,
  xgb_monotonicity_features
)
xgb_increased_prediction <- predict_monotonic_unit_xgboost(
  fitted_models$monotonic_xgboost,
  xgb_floor_area_increase
)
xgb_monotonicity_qa <- tibble(
  metric = c(
    "candidate_rows_tested",
    "floor_area_increase_percent",
    "monotonicity_violations",
    "minimum_prediction_change",
    "median_prediction_change"
  ),
  value = c(
    nrow(xgb_monotonicity_features),
    10,
    sum(xgb_increased_prediction + 1e-8 < xgb_base_prediction),
    min(xgb_increased_prediction - xgb_base_prediction),
    median(xgb_increased_prediction - xgb_base_prediction)
  )
)

model_bundle <- list(
  models = fitted_models,
  recommended_model = recommended_model,
  feature_medians = training_features$medians,
  training_ranges = training_ranges,
  interval_calibration = interval_calibration,
  interval_validation = interval_validation,
  model_recommendation = model_recommendation,
  random_seed = RANDOM_SEED,
  training_project_ids = training$project_id
)

save_output(
  model_bundle,
  file.path(OUTPUT_DIR, "residential_unit_count_models.rds"),
  "residential unit-count model bundle"
)
save_output(
  cv_predictions,
  file.path(OUTPUT_DIR, "residential_unit_model_cv_predictions.rds"),
  "residential unit-count cross-validation predictions"
)
save_output(
  candidate_predictions,
  file.path(OUTPUT_DIR, "residential_unit_model_predictions.rds"),
  "unresolved-project unit-count predictions"
)

write_csv(
  folds,
  file.path(OUTPUT_DIR, "residential_unit_model_folds.csv")
)
write_csv(
  fold_qa,
  file.path(OUTPUT_DIR, "residential_unit_model_fold_qa.csv")
)
write_csv(
  cv_predictions,
  file.path(OUTPUT_DIR, "residential_unit_model_cv_predictions.csv")
)
write_csv(
  fold_metrics,
  file.path(OUTPUT_DIR, "residential_unit_model_fold_metrics.csv")
)
write_csv(
  pooled_metrics,
  file.path(OUTPUT_DIR, "residential_unit_model_metrics.csv")
)
write_csv(
  size_metrics,
  file.path(OUTPUT_DIR, "residential_unit_model_size_metrics.csv")
)
write_csv(
  source_metrics,
  file.path(OUTPUT_DIR, "residential_unit_model_source_metrics.csv")
)
write_csv(
  model_recommendation,
  file.path(OUTPUT_DIR, "residential_unit_model_recommendation.csv")
)
write_csv(
  interval_calibration,
  file.path(OUTPUT_DIR, "residential_unit_model_interval_calibration.csv")
)
write_csv(
  interval_validation,
  file.path(OUTPUT_DIR, "residential_unit_model_interval_validation.csv")
)
write_csv(
  training_ranges,
  file.path(OUTPUT_DIR, "residential_unit_model_training_ranges.csv")
)
write_csv(
  candidate_predictions,
  file.path(OUTPUT_DIR, "residential_unit_model_predictions.csv")
)
write_csv(
  candidate_summary,
  file.path(OUTPUT_DIR, "residential_unit_model_prediction_summary.csv")
)
write_csv(
  integration_scenarios,
  file.path(OUTPUT_DIR, "residential_unit_model_integration_scenarios.csv")
)
write_csv(
  uro_sensitivity_metrics,
  file.path(OUTPUT_DIR, "residential_unit_model_uro_sensitivity.csv")
)
write_csv(
  uro_sensitivity_by_county,
  file.path(
    OUTPUT_DIR,
    "residential_unit_model_uro_sensitivity_by_county.csv"
  )
)
write_csv(
  out_of_domain_qa,
  file.path(OUTPUT_DIR, "residential_unit_model_out_of_domain_qa.csv")
)
write_csv(
  as_tibble(xgb_importance),
  file.path(OUTPUT_DIR, "residential_unit_model_xgboost_importance.csv")
)
write_csv(
  xgb_monotonicity_qa,
  file.path(OUTPUT_DIR, "residential_unit_model_xgboost_monotonicity_qa.csv")
)
write_csv(
  candidate_predictions %>%
    filter(
      model_out_of_domain |
        below_training_unit_scope
    ),
  file.path(OUTPUT_DIR, "residential_unit_model_prediction_review.csv")
)

project_cv_plot <- cv_predictions %>%
  filter(validation_scheme == "project_grouped") %>%
  ggplot(aes(x = observed_units, y = predicted_units)) +
  geom_abline(slope = 1, intercept = 0, color = "grey45", linewidth = 0.5) +
  geom_point(alpha = 0.28, size = 0.9, color = "#147D92") +
  scale_x_log10(labels = label_number()) +
  scale_y_log10(labels = label_number()) +
  facet_wrap(vars(model_display), ncol = 2) +
  labs(
    title = "Held-out residential unit predictions",
    subtitle = "Five project-stratified folds; axes use logarithmic scales",
    x = "Reported project units",
    y = "Predicted project units"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold")
  )
ggsave(
  file.path(FIGURE_DIR, "02r_unit_model_cv_observed_predicted.png"),
  project_cv_plot,
  width = 10,
  height = 8,
  dpi = 180
)

performance_plot <- pooled_metrics %>%
  ggplot(
    aes(
      x = reorder(model_display, wape),
      y = wape,
      fill = validation_scheme
    )
  ) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  coord_flip() +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_fill_manual(
    values = c(
      project_grouped = "#147D92",
      source_holdout = "#C44E52",
      spatial_cluster = "#D49A2A"
    ),
    labels = c(
      project_grouped = "Project folds",
      source_holdout = "Source holdout",
      spatial_cluster = "Spatial holdout"
    )
  ) +
  labs(
    title = "Residential unit model validation error",
    x = NULL,
    y = "Weighted absolute percentage error",
    fill = "Validation"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.y = element_blank())
ggsave(
  file.path(FIGURE_DIR, "02r_unit_model_validation_wape.png"),
  performance_plot,
  width = 10,
  height = 6,
  dpi = 180
)

print_progress(
  paste0(
    "Recommended method predicts ",
    comma(round(sum(candidate_predictions$recommended_prediction))),
    " units across ",
    comma(nrow(candidate_predictions)),
    " unresolved projects."
  )
)
print_progress(
  paste0(
    "Out-of-domain candidates: ",
    comma(sum(candidate_predictions$model_out_of_domain)),
    "; county-transfer candidates: ",
    comma(sum(candidate_predictions$county_transfer_flag))
  )
)

print_header("02r COMPLETE")
