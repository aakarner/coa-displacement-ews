################################################################################
# Residential Unit Count Modeling Helpers
################################################################################

unit_model_methods <- function() {
  c(
    "fixed_ratio",
    "stratified_ratio",
    "negative_binomial_gam",
    "monotonic_xgboost"
  )
}

unit_model_display_name <- function(method) {
  labels <- c(
    current_primary = "Current primary estimate",
    current_conservative = "Current conservative estimate",
    fixed_ratio = "Fixed sqft/unit ratio",
    stratified_ratio = "Stratified sqft/unit ratio",
    negative_binomial_gam = "Negative-binomial GAM",
    monotonic_xgboost = "Monotonic gradient boosting"
  )
  unname(labels[method])
}

unit_model_size_band <- function(units) {
  cut(
    units,
    breaks = c(-Inf, 19, 49, 99, 249, Inf),
    labels = c("5-19", "20-49", "50-99", "100-249", "250+"),
    ordered_result = TRUE
  )
}

unit_model_area_band <- function(floor_area) {
  cut(
    floor_area,
    breaks = c(-Inf, 25000, 100000, 300000, Inf),
    labels = c("<25k", "25k-100k", "100k-300k", "300k+"),
    ordered_result = TRUE
  )
}

unit_model_story_band <- function(stories) {
  dplyr::case_when(
    stories <= 1 ~ "1",
    stories <= 2 ~ "2",
    TRUE ~ "3+"
  )
}

unit_model_required_predictors <- function() {
  c(
    "project_model_floor_area",
    "project_land_sqft",
    "project_year_built",
    "project_max_stories",
    "project_parcel_count",
    "project_address_count",
    "project_condo_account_count",
    "project_b1_parcel_count",
    "project_has_mf_zoning",
    "project_has_commercial_mixed_zoning"
  )
}

unit_model_median <- function(x, fallback = 0) {
  observed <- x[is.finite(x)]
  if (length(observed) == 0L) {
    return(fallback)
  }
  stats::median(observed)
}

prepare_unit_model_features <- function(data, medians = NULL) {
  missing_columns <- setdiff(unit_model_required_predictors(), names(data))
  if (length(missing_columns) > 0L) {
    stop(
      "Unit model input is missing predictors: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  if (is.null(medians)) {
    medians <- c(
      land_area = unit_model_median(data$project_land_sqft, 1),
      year_built = unit_model_median(data$project_year_built, 2000),
      max_stories = unit_model_median(data$project_max_stories, 1),
      parcel_count = unit_model_median(data$project_parcel_count, 1),
      address_count = unit_model_median(data$project_address_count, 1),
      condo_count = unit_model_median(data$project_condo_account_count, 0),
      b1_count = unit_model_median(data$project_b1_parcel_count, 0)
    )
  }

  prepared <- data %>%
    dplyr::mutate(
      .model_floor_area = pmax(as.numeric(project_model_floor_area), 1),
      .model_land_area = dplyr::coalesce(
        as.numeric(project_land_sqft),
        medians[["land_area"]]
      ),
      .model_year_built = dplyr::coalesce(
        as.numeric(project_year_built),
        medians[["year_built"]]
      ),
      .model_max_stories = pmax(
        dplyr::coalesce(
          as.numeric(project_max_stories),
          medians[["max_stories"]]
        ),
        1
      ),
      .model_parcel_count = pmax(
        dplyr::coalesce(
          as.numeric(project_parcel_count),
          medians[["parcel_count"]]
        ),
        1
      ),
      .model_address_count = pmax(
        dplyr::coalesce(
          as.numeric(project_address_count),
          medians[["address_count"]]
        ),
        1
      ),
      .model_condo_count = pmax(
        dplyr::coalesce(
          as.numeric(project_condo_account_count),
          medians[["condo_count"]]
        ),
        0
      ),
      .model_b1_count = pmax(
        dplyr::coalesce(
          as.numeric(project_b1_parcel_count),
          medians[["b1_count"]]
        ),
        0
      ),
      .model_has_mf_zoning = as.numeric(
        dplyr::coalesce(project_has_mf_zoning, FALSE)
      ),
      .model_has_commercial_mixed_zoning = as.numeric(
        dplyr::coalesce(project_has_commercial_mixed_zoning, FALSE)
      ),
      .model_log_floor_area = log(.model_floor_area),
      .model_log_land_area = log(pmax(.model_land_area, 1)),
      .model_log_far = log(
        pmax(.model_floor_area / pmax(.model_land_area, 1), 0.001)
      ),
      .model_log_stories = log(.model_max_stories),
      .model_log_parcel_count = log1p(.model_parcel_count),
      .model_log_address_count = log1p(.model_address_count),
      .model_log_condo_count = log1p(.model_condo_count),
      .model_log_b1_count = log1p(.model_b1_count),
      .model_area_band = unit_model_area_band(.model_floor_area),
      .model_story_band = unit_model_story_band(.model_max_stories)
    )

  list(data = prepared, medians = medians)
}

make_unit_model_folds <- function(training_data, k = 5L, seed = 42L) {
  if (!all(c("project_id", "unit_count", "label_source") %in% names(training_data))) {
    stop("Training data lacks fold-assignment fields.", call. = FALSE)
  }
  if (nrow(training_data) < k * 10L) {
    stop("Training data is too small for the requested folds.", call. = FALSE)
  }

  set.seed(seed)
  project_folds <- training_data %>%
    dplyr::mutate(
      .row_id = dplyr::row_number(),
      .size_band = unit_model_size_band(unit_count)
    ) %>%
    dplyr::group_by(label_source, .size_band) %>%
    dplyr::mutate(
      .random_order = sample.int(dplyr::n()),
      fold_id = as.character((.random_order - 1L) %% k + 1L)
    ) %>%
    dplyr::ungroup() %>%
    dplyr::transmute(
      validation_scheme = "project_grouped",
      fold_id,
      .row_id,
      project_id
    )

  mean_latitude <- mean(training_data$project_lat, na.rm = TRUE)
  spatial_coordinates <- cbind(
    x = training_data$project_lon * cos(mean_latitude * pi / 180),
    y = training_data$project_lat
  )
  spatial_coordinates <- scale(spatial_coordinates)
  if (any(!is.finite(spatial_coordinates))) {
    stop("Spatial fold coordinates contain non-finite values.", call. = FALSE)
  }

  set.seed(seed + 1L)
  spatial_clusters <- stats::kmeans(
    spatial_coordinates,
    centers = k,
    nstart = 100,
    iter.max = 100
  )$cluster
  spatial_folds <- tibble::tibble(
    validation_scheme = "spatial_cluster",
    fold_id = as.character(spatial_clusters),
    .row_id = seq_len(nrow(training_data)),
    project_id = training_data$project_id
  )

  source_folds <- training_data %>%
    dplyr::mutate(.row_id = dplyr::row_number()) %>%
    dplyr::transmute(
      validation_scheme = "source_holdout",
      fold_id = as.character(label_source),
      .row_id,
      project_id
    )

  folds <- dplyr::bind_rows(project_folds, spatial_folds, source_folds)
  fold_counts <- folds %>%
    dplyr::count(validation_scheme, .row_id)
  if (
    nrow(fold_counts) != nrow(training_data) * 3L ||
      any(fold_counts$n != 1L)
  ) {
    stop("Every project must occur once in each validation scheme.", call. = FALSE)
  }
  folds
}

fit_fixed_unit_ratio <- function(training_data) {
  ratios <- training_data$.model_floor_area / training_data$unit_count
  ratios <- ratios[is.finite(ratios) & ratios > 0]
  if (length(ratios) == 0L) {
    stop("Fixed-ratio model has no valid training ratios.", call. = FALSE)
  }
  list(global_ratio = stats::median(ratios))
}

predict_fixed_unit_ratio <- function(model, new_data) {
  pmax(new_data$.model_floor_area / model$global_ratio, 1)
}

fit_stratified_unit_ratio <- function(
    training_data,
    joint_minimum = 15L,
    fallback_minimum = 20L) {
  ratio_data <- training_data %>%
    dplyr::mutate(
      .ratio = .model_floor_area / unit_count,
      .joint_key = paste(.model_area_band, .model_story_band, sep = "|")
    ) %>%
    dplyr::filter(is.finite(.ratio), .ratio > 0)

  global_ratio <- stats::median(ratio_data$.ratio)
  joint <- ratio_data %>%
    dplyr::group_by(.joint_key) %>%
    dplyr::summarise(
      n = dplyr::n(),
      ratio = stats::median(.ratio),
      .groups = "drop"
    ) %>%
    dplyr::filter(n >= joint_minimum)
  area <- ratio_data %>%
    dplyr::group_by(.model_area_band) %>%
    dplyr::summarise(
      n = dplyr::n(),
      ratio = stats::median(.ratio),
      .groups = "drop"
    ) %>%
    dplyr::filter(n >= fallback_minimum)
  story <- ratio_data %>%
    dplyr::group_by(.model_story_band) %>%
    dplyr::summarise(
      n = dplyr::n(),
      ratio = stats::median(.ratio),
      .groups = "drop"
    ) %>%
    dplyr::filter(n >= fallback_minimum)

  list(
    global_ratio = global_ratio,
    joint = stats::setNames(joint$ratio, joint$.joint_key),
    area = stats::setNames(area$ratio, as.character(area$.model_area_band)),
    story = stats::setNames(story$ratio, story$.model_story_band)
  )
}

predict_stratified_unit_ratio <- function(model, new_data) {
  joint_key <- paste(
    new_data$.model_area_band,
    new_data$.model_story_band,
    sep = "|"
  )
  ratio <- unname(model$joint[joint_key])
  area_ratio <- unname(model$area[as.character(new_data$.model_area_band)])
  story_ratio <- unname(model$story[new_data$.model_story_band])
  ratio <- dplyr::coalesce(
    as.numeric(ratio),
    as.numeric(area_ratio),
    as.numeric(story_ratio),
    model$global_ratio
  )
  pmax(new_data$.model_floor_area / ratio, 1)
}

fit_negative_binomial_unit_gam <- function(training_data) {
  mgcv::gam(
    unit_count ~
      offset(.model_log_floor_area) +
      s(.model_year_built, k = 6, bs = "cr") +
      s(.model_log_far, k = 6, bs = "cr") +
      s(.model_log_stories, k = 4, bs = "cr") +
      .model_log_parcel_count +
      .model_has_mf_zoning +
      .model_has_commercial_mixed_zoning,
    data = training_data,
    family = mgcv::nb(link = "log"),
    method = "REML",
    select = TRUE,
    gamma = 1.2
  )
}

predict_negative_binomial_unit_gam <- function(model, new_data) {
  prediction <- stats::predict(model, newdata = new_data, type = "response")
  pmax(as.numeric(prediction), 1)
}

unit_xgboost_predictors <- function() {
  c(
    ".model_log_floor_area",
    ".model_log_land_area",
    ".model_log_far",
    ".model_year_built",
    ".model_log_stories",
    ".model_log_parcel_count",
    ".model_log_address_count",
    ".model_log_condo_count",
    ".model_log_b1_count",
    ".model_has_mf_zoning",
    ".model_has_commercial_mixed_zoning"
  )
}

unit_xgboost_matrix <- function(data) {
  matrix_data <- as.matrix(data[, unit_xgboost_predictors(), drop = FALSE])
  storage.mode(matrix_data) <- "double"
  matrix_data
}

fit_monotonic_unit_xgboost <- function(training_data, seed = 42L) {
  predictors <- unit_xgboost_predictors()
  positive_predictors <- c(
    ".model_log_floor_area",
    ".model_log_far",
    ".model_log_condo_count",
    ".model_log_b1_count"
  )
  monotonicity <- ifelse(predictors %in% positive_predictors, 1L, 0L)
  constraint <- paste0("(", paste(monotonicity, collapse = ","), ")")
  matrix_data <- unit_xgboost_matrix(training_data)
  dtrain <- xgboost::xgb.DMatrix(
    data = matrix_data,
    label = training_data$unit_count
  )

  model <- xgboost::xgb.train(
    params = list(
      objective = "count:poisson",
      eval_metric = "poisson-nloglik",
      eta = 0.03,
      max_depth = 3L,
      min_child_weight = 10,
      subsample = 0.85,
      colsample_bytree = 0.85,
      lambda = 2,
      alpha = 0.1,
      monotone_constraints = constraint,
      seed = seed,
      nthread = 1L
    ),
    data = dtrain,
    nrounds = 400L,
    verbose = 0
  )
  attr(model, "unit_predictors") <- predictors
  model
}

predict_monotonic_unit_xgboost <- function(model, new_data) {
  prediction <- stats::predict(
    model,
    xgboost::xgb.DMatrix(unit_xgboost_matrix(new_data))
  )
  pmax(as.numeric(prediction), 1)
}

fit_unit_count_model <- function(method, training_data, seed = 42L) {
  switch(
    method,
    fixed_ratio = fit_fixed_unit_ratio(training_data),
    stratified_ratio = fit_stratified_unit_ratio(training_data),
    negative_binomial_gam = fit_negative_binomial_unit_gam(training_data),
    monotonic_xgboost = fit_monotonic_unit_xgboost(training_data, seed),
    stop("Unknown unit-count model: ", method, call. = FALSE)
  )
}

predict_unit_count_model <- function(method, model, new_data) {
  switch(
    method,
    fixed_ratio = predict_fixed_unit_ratio(model, new_data),
    stratified_ratio = predict_stratified_unit_ratio(model, new_data),
    negative_binomial_gam =
      predict_negative_binomial_unit_gam(model, new_data),
    monotonic_xgboost = predict_monotonic_unit_xgboost(model, new_data),
    stop("Unknown unit-count model: ", method, call. = FALSE)
  )
}

unit_count_metric_values <- function(observed, predicted) {
  keep <- is.finite(observed) & observed > 0 &
    is.finite(predicted) & predicted >= 0
  observed <- observed[keep]
  predicted <- predicted[keep]
  error <- predicted - observed

  tibble::tibble(
    n = length(observed),
    observed_units = sum(observed),
    predicted_units = sum(predicted),
    mae = mean(abs(error)),
    rmse = sqrt(mean(error^2)),
    wape = sum(abs(error)) / sum(observed),
    bias = sum(error) / sum(observed),
    median_ape = stats::median(abs(error) / observed),
    r_squared = if (length(observed) > 1L) {
      stats::cor(observed, predicted)^2
    } else {
      NA_real_
    }
  )
}

summarise_unit_count_metrics <- function(predictions, group_columns) {
  predictions %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_columns))) %>%
    dplyr::group_modify(
      ~unit_count_metric_values(.x$observed_units, .x$predicted_units)
    ) %>%
    dplyr::ungroup()
}

run_unit_count_cross_validation <- function(
    training_data,
    folds,
    methods = unit_model_methods(),
    seed = 42L) {
  fold_keys <- folds %>%
    dplyr::distinct(validation_scheme, fold_id) %>%
    dplyr::arrange(validation_scheme, fold_id)
  predictions <- vector("list", nrow(fold_keys) * length(methods))
  result_index <- 0L

  for (fold_index in seq_len(nrow(fold_keys))) {
    scheme <- fold_keys$validation_scheme[[fold_index]]
    fold_id <- fold_keys$fold_id[[fold_index]]
    test_rows <- folds %>%
      dplyr::filter(
        validation_scheme == scheme,
        .data$fold_id == !!fold_id
      ) %>%
      dplyr::pull(.row_id)

    train_raw <- training_data[-test_rows, , drop = FALSE]
    test_raw <- training_data[test_rows, , drop = FALSE]
    train_features <- prepare_unit_model_features(train_raw)
    test_features <- prepare_unit_model_features(
      test_raw,
      medians = train_features$medians
    )

    for (method_index in seq_along(methods)) {
      method <- methods[[method_index]]
      model_seed <- seed + fold_index * 100L + method_index
      fitted_model <- fit_unit_count_model(
        method,
        train_features$data,
        seed = model_seed
      )
      predicted <- predict_unit_count_model(
        method,
        fitted_model,
        test_features$data
      )
      if (any(!is.finite(predicted)) || any(predicted <= 0)) {
        stop(
          "Non-finite prediction from ",
          method,
          " in ",
          scheme,
          " fold ",
          fold_id,
          ".",
          call. = FALSE
        )
      }

      result_index <- result_index + 1L
      predictions[[result_index]] <- tibble::tibble(
        validation_scheme = scheme,
        fold_id = fold_id,
        project_id = test_raw$project_id,
        label_source = test_raw$label_source,
        observed_units = test_raw$unit_count,
        observed_size_band = unit_model_size_band(test_raw$unit_count),
        model = method,
        model_display = unit_model_display_name(method),
        predicted_units = predicted
      )
    }
  }

  dplyr::bind_rows(predictions)
}

select_unit_count_model <- function(
    fold_metrics,
    pooled_size_metrics,
    minimum_improvement = 0.10,
    maximum_absolute_bias = 0.10,
    maximum_absolute_size_band_bias = 0.20,
    maximum_large_project_penalty = 0.05,
    maximum_gam_benchmark_penalty = 0.10) {
  grouped <- fold_metrics %>%
    dplyr::filter(validation_scheme == "project_grouped") %>%
    dplyr::group_by(model, model_display) %>%
    dplyr::summarise(
      median_fold_wape = stats::median(wape),
      mean_fold_wape = mean(wape),
      median_fold_bias = stats::median(bias),
      maximum_absolute_fold_bias = max(abs(bias)),
      .groups = "drop"
    )
  large <- pooled_size_metrics %>%
    dplyr::filter(
      validation_scheme == "project_grouped",
      as.character(observed_size_band) == "250+"
    ) %>%
    dplyr::select(model, large_project_wape = wape)
  size_bias <- pooled_size_metrics %>%
    dplyr::filter(validation_scheme == "project_grouped") %>%
    dplyr::group_by(model) %>%
    dplyr::summarise(
      maximum_absolute_size_band_bias = max(abs(bias)),
      .groups = "drop"
    )
  comparison <- grouped %>%
    dplyr::left_join(large, by = "model") %>%
    dplyr::left_join(size_bias, by = "model")

  baseline_wape <- comparison$median_fold_wape[
    comparison$model == "fixed_ratio"
  ]
  baseline_large_wape <- comparison$large_project_wape[
    comparison$model == "fixed_ratio"
  ]
  benchmark_large_wape <- comparison$large_project_wape[
    comparison$model == "monotonic_xgboost"
  ]

  comparison <- comparison %>%
    dplyr::mutate(
      improvement_over_fixed = 1 - median_fold_wape / baseline_wape,
      large_project_penalty_vs_fixed =
        large_project_wape / baseline_large_wape - 1,
      large_project_penalty_vs_benchmark =
        large_project_wape / benchmark_large_wape - 1,
      interpretable_candidate = model %in% c(
        "fixed_ratio",
        "stratified_ratio",
        "negative_binomial_gam"
      ),
      passes_improvement = model == "fixed_ratio" |
        improvement_over_fixed >= minimum_improvement,
      passes_bias = maximum_absolute_fold_bias <= maximum_absolute_bias,
      passes_size_band_bias =
        maximum_absolute_size_band_bias <=
          .env$maximum_absolute_size_band_bias,
      passes_large_project_check = model == "fixed_ratio" |
        large_project_penalty_vs_fixed <= maximum_large_project_penalty,
      passes_benchmark_check = model != "negative_binomial_gam" |
        large_project_penalty_vs_benchmark <=
          maximum_gam_benchmark_penalty,
      eligible_for_production = model == "fixed_ratio" |
        (
          interpretable_candidate &
            passes_improvement &
            passes_bias &
            passes_size_band_bias &
            passes_large_project_check &
            passes_benchmark_check
        )
    )

  eligible <- comparison %>%
    dplyr::filter(eligible_for_production) %>%
    dplyr::arrange(median_fold_wape)
  if (nrow(eligible) == 0L) {
    recommended_model <- "fixed_ratio"
  } else {
    recommended_model <- eligible$model[[1]]
  }

  comparison %>%
    dplyr::mutate(
      recommended_model = recommended_model,
      recommended = model == recommended_model,
      decision = dplyr::case_when(
        recommended ~ "recommended",
        model == "fixed_ratio" ~ "baseline_not_selected",
        !interpretable_candidate & !passes_size_band_bias ~
          "benchmark_size_band_bias",
        !interpretable_candidate ~ "benchmark_only",
        !passes_improvement ~ "insufficient_wape_improvement",
        !passes_bias ~ "material_fold_bias",
        !passes_size_band_bias ~ "material_size_band_bias",
        !passes_large_project_check ~ "large_project_penalty",
        !passes_benchmark_check ~ "worse_than_large_project_benchmark",
        TRUE ~ "not_selected"
      )
    )
}

unit_prediction_interval_calibration <- function(
    project_predictions,
    model,
    minimum_band_n = 50L) {
  residuals <- project_predictions %>%
    dplyr::filter(
      validation_scheme == "project_grouped",
      .data$model == !!model
    ) %>%
    dplyr::mutate(
      prediction_size_band = unit_model_size_band(predicted_units),
      observed_to_predicted_ratio = observed_units / predicted_units
    )
  global <- stats::quantile(
    residuals$observed_to_predicted_ratio,
    probs = c(0.10, 0.90),
    na.rm = TRUE,
    names = FALSE
  )
  residuals %>%
    dplyr::group_by(prediction_size_band) %>%
    dplyr::summarise(
      n = dplyr::n(),
      lower_multiplier = if (dplyr::n() >= minimum_band_n) {
        stats::quantile(
          observed_to_predicted_ratio,
          0.10,
          na.rm = TRUE,
          names = FALSE
        )
      } else {
        global[[1]]
      },
      upper_multiplier = if (dplyr::n() >= minimum_band_n) {
        stats::quantile(
          observed_to_predicted_ratio,
          0.90,
          na.rm = TRUE,
          names = FALSE
        )
      } else {
        global[[2]]
      },
      used_global_calibration = dplyr::n() < minimum_band_n,
      .groups = "drop"
    )
}

validate_unit_prediction_intervals <- function(
    project_predictions,
    model,
    minimum_band_n = 50L) {
  residuals <- project_predictions %>%
    dplyr::filter(
      validation_scheme == "project_grouped",
      .data$model == !!model
    ) %>%
    dplyr::mutate(
      prediction_size_band = unit_model_size_band(predicted_units),
      observed_to_predicted_ratio = observed_units / predicted_units
    )
  fold_ids <- sort(unique(residuals$fold_id))
  validated <- vector("list", length(fold_ids))

  for (fold_index in seq_along(fold_ids)) {
    held_out_fold <- fold_ids[[fold_index]]
    calibration_rows <- residuals %>%
      dplyr::filter(fold_id != held_out_fold)
    validation_rows <- residuals %>%
      dplyr::filter(fold_id == held_out_fold)
    global <- stats::quantile(
      calibration_rows$observed_to_predicted_ratio,
      probs = c(0.10, 0.90),
      na.rm = TRUE,
      names = FALSE
    )
    fold_calibration <- calibration_rows %>%
      dplyr::group_by(prediction_size_band) %>%
      dplyr::summarise(
        calibration_n = dplyr::n(),
        lower_multiplier = if (dplyr::n() >= minimum_band_n) {
          stats::quantile(
            observed_to_predicted_ratio,
            0.10,
            na.rm = TRUE,
            names = FALSE
          )
        } else {
          global[[1]]
        },
        upper_multiplier = if (dplyr::n() >= minimum_band_n) {
          stats::quantile(
            observed_to_predicted_ratio,
            0.90,
            na.rm = TRUE,
            names = FALSE
          )
        } else {
          global[[2]]
        },
        used_global_calibration = dplyr::n() < minimum_band_n,
        .groups = "drop"
      )

    validated[[fold_index]] <- validation_rows %>%
      dplyr::left_join(
        fold_calibration,
        by = "prediction_size_band",
        relationship = "many-to-one"
      ) %>%
      dplyr::mutate(
        interval_lower = predicted_units * lower_multiplier,
        interval_upper = predicted_units * upper_multiplier,
        interval_covered = observed_units >= interval_lower &
          observed_units <= interval_upper,
        relative_interval_width =
          (interval_upper - interval_lower) / predicted_units
      )
  }

  validated <- dplyr::bind_rows(validated)
  dplyr::bind_rows(
    validated %>%
      dplyr::summarise(
        validation_group = "overall",
        projects = dplyr::n(),
        empirical_coverage = mean(interval_covered),
        median_relative_interval_width = stats::median(
          relative_interval_width
        )
      ),
    validated %>%
      dplyr::group_by(prediction_size_band) %>%
      dplyr::summarise(
        validation_group = paste0(
          "predicted_",
          as.character(dplyr::first(prediction_size_band))
        ),
        projects = dplyr::n(),
        empirical_coverage = mean(interval_covered),
        median_relative_interval_width = stats::median(
          relative_interval_width
        ),
        .groups = "drop"
      ) %>%
      dplyr::select(-prediction_size_band)
  )
}

unit_model_training_ranges <- function(training_data) {
  tibble::tibble(
    predictor = c(
      "project_model_floor_area",
      "project_land_sqft",
      "project_year_built",
      "project_max_stories"
    ),
    minimum = c(
      min(training_data$project_model_floor_area, na.rm = TRUE),
      min(training_data$project_land_sqft, na.rm = TRUE),
      min(training_data$project_year_built, na.rm = TRUE),
      min(training_data$project_max_stories, na.rm = TRUE)
    ),
    central_minimum = c(
      stats::quantile(
        training_data$project_model_floor_area,
        0.01,
        na.rm = TRUE
      ),
      stats::quantile(training_data$project_land_sqft, 0.01, na.rm = TRUE),
      stats::quantile(training_data$project_year_built, 0.01, na.rm = TRUE),
      stats::quantile(training_data$project_max_stories, 0.01, na.rm = TRUE)
    ),
    central_maximum = c(
      stats::quantile(
        training_data$project_model_floor_area,
        0.99,
        na.rm = TRUE
      ),
      stats::quantile(training_data$project_land_sqft, 0.99, na.rm = TRUE),
      stats::quantile(training_data$project_year_built, 0.99, na.rm = TRUE),
      stats::quantile(training_data$project_max_stories, 0.99, na.rm = TRUE)
    ),
    maximum = c(
      max(training_data$project_model_floor_area, na.rm = TRUE),
      max(training_data$project_land_sqft, na.rm = TRUE),
      max(training_data$project_year_built, na.rm = TRUE),
      max(training_data$project_max_stories, na.rm = TRUE)
    )
  )
}
