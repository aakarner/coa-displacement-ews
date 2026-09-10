# Fixed-component Part 2 scoring; source counts, coverage and denominators stay
# unchanged. Reviewed bounds are retained for unchanged measurement components.
part2_index_components <- function(index_name) {
  specification <- list(
    ownership_pressure_index = c("pct_corporate_units", "corporate_owned_units_per_km2", "pct_financialized_owner_parcels"),
    eviction_pressure_index = c("eviction_latest_12mo_per_100_units", "eviction_latest_12mo_rate_change_per_100_units"))
  if (length(index_name) != 1L || is.na(index_name) || !index_name %in% names(specification)) {
    stop("Unknown Part 2 event index.", call. = FALSE)
  }
  specification[[index_name]]
}

part2_index_component_inputs <- function(features, components, index_name) {
  if (!identical(components, part2_index_components(index_name))) {
    stop("Event component definitions/order do not match the selected index.", call. = FALSE)
  }
  if (!is.data.frame(features) || !nrow(features) || !all(components %in% names(features))) {
    stop("Missing event components or empty feature data.", call. = FALSE)
  }
  inputs <- as.data.frame(features)[components]
  if (any(vapply(inputs, function(x) !is.numeric(x) || any(!is.finite(x) & !is.na(x)), logical(1)))) {
    stop("Event component inputs must be numeric, finite or missing.", call. = FALSE)
  }
  inputs
}

part2_fit_index_scaling <- function(features, components, index_name, reference_date, preserved_scaling = NULL) {
  reference_date <- as.Date(reference_date)
  if (length(reference_date) != 1L || is.na(reference_date)) stop("Invalid event reference date.", call. = FALSE)
  inputs <- part2_index_component_inputs(features, components, index_name)
  if ("analysis_as_of_date" %in% names(features) &&
      (anyNA(features$analysis_as_of_date) || any(as.Date(features$analysis_as_of_date) != reference_date))) {
    stop("Fit event bounds only on the earlier reference date.", call. = FALSE)
  }
  signed <- if (identical(index_name, "eviction_pressure_index")) "eviction_latest_12mo_rate_change_per_100_units" else character()
  unchanged <- setdiff(components, signed)
  if (!is.null(preserved_scaling)) {
    p <- preserved_scaling
    b <- p$bounds
    if (!is.list(p) || !p$schema_version %in% c("part2-index-scaling-v1", "part2-index-scaling-v2") ||
        !identical(p$index_name, index_name) || !identical(as.Date(p$reference_date), reference_date) ||
        !identical(p$probabilities, c(.01, .99)) || !identical(p$quantile_type, 7L) ||
        !is.data.frame(b) || !all(c("component", "lower_bound", "upper_bound",
          "reference_available_hexes", "degenerate_range") %in% names(b)) ||
        anyDuplicated(b$component) || !all(unchanged %in% b$component) ||
        any(!is.finite(b$lower_bound)) || any(!is.finite(b$upper_bound)) ||
        any(b$lower_bound > b$upper_bound) || !is.logical(b$degenerate_range) ||
        anyNA(b$degenerate_range) || any(b$degenerate_range != (b$lower_bound == b$upper_bound)) ||
        any(!is.finite(b$reference_available_hexes) | b$reference_available_hexes <= 0)) {
      stop("Invalid preserved earlier component bounds.", call. = FALSE)
    }
  }
  bounds <- do.call(rbind, lapply(components, function(component) {
    x <- inputs[[component]]
    if (all(is.na(x))) stop("No earlier observed values for ", component, call. = FALSE)
    is_signed <- component %in% signed
    preserved <- !is_signed && !is.null(preserved_scaling)
    available <- sum(!is.na(x))
    if (preserved) {
      b <- preserved_scaling$bounds[match(component, preserved_scaling$bounds$component), ]
      q <- c(b$lower_bound, b$upper_bound)
      available <- b$reference_available_hexes
    } else if (is_signed) {
      bound <- as.numeric(stats::quantile(abs(x), .99, na.rm = TRUE, type = 7))
      q <- c(-bound, bound)
    } else {
      q <- as.numeric(stats::quantile(x, c(.01, .99), na.rm = TRUE, type = 7))
    }
    data.frame(component = component, lower_bound = q[1], upper_bound = q[2],
      reference_available_hexes = available, degenerate_range = q[1] == q[2],
      scaling_method = if (is_signed) "symmetric_abs_change_q99" else "baseline_percentile_01_99",
      bounds_preserved = preserved, neutral_score = if (is_signed) 50 else NA_real_,
      equal_weight = 1 / length(components))
  }))
  list(schema_version = "part2-index-scaling-v2", index_name = index_name,
    components = components, reference_date = reference_date, probabilities = c(.01, .99),
    change_abs_quantile = .99, quantile_type = 7L, bounds = bounds,
    missing_policy = "all_components_required_fixed_equal_weights",
    degenerate_range_policy = "unchanged_zero_signed_change_neutral50",
    measurement_version = "fixed_components_v2")
}

part2_apply_index_scaling <- function(features, scaling) {
  if (!is.list(scaling) || !identical(scaling$schema_version, "part2-index-scaling-v2") ||
      !identical(scaling$change_abs_quantile, .99) ||
      !identical(scaling$measurement_version, "fixed_components_v2") ||
      !identical(scaling$probabilities, c(.01, .99)) || !identical(scaling$quantile_type, 7L) ||
      !identical(scaling$missing_policy, "all_components_required_fixed_equal_weights") ||
      !identical(scaling$degenerate_range_policy, "unchanged_zero_signed_change_neutral50") ||
      length(scaling$reference_date) != 1L || is.na(scaling$reference_date)) {
    stop("Invalid event scaling contract.", call. = FALSE)
  }
  inputs <- part2_index_component_inputs(features, scaling$components, scaling$index_name)
  if ("analysis_as_of_date" %in% names(features) &&
      (anyNA(features$analysis_as_of_date) || any(as.Date(features$analysis_as_of_date) < scaling$reference_date))) {
    stop("Event scaling cannot use a future reference date.", call. = FALSE)
  }
  bounds <- scaling$bounds
  signed <- if (identical(scaling$index_name, "eviction_pressure_index")) "eviction_latest_12mo_rate_change_per_100_units" else character()
  if (!is.data.frame(bounds) || !all(c("component", "lower_bound", "upper_bound", "degenerate_range",
      "scaling_method", "neutral_score", "equal_weight", "bounds_preserved") %in% names(bounds)) ||
      anyDuplicated(bounds$component) || !setequal(bounds$component, scaling$components) ||
      any(!is.finite(bounds$lower_bound)) || any(!is.finite(bounds$upper_bound)) ||
      any(bounds$upper_bound < bounds$lower_bound) || !is.logical(bounds$degenerate_range) ||
      anyNA(bounds$degenerate_range) ||
      any(bounds$degenerate_range != (bounds$lower_bound == bounds$upper_bound)) ||
      any(!is.finite(bounds$equal_weight)) || any(bounds$equal_weight != 1 / length(scaling$components)) ||
      anyNA(bounds$scaling_method) || any(bounds$scaling_method != ifelse(bounds$component %in% signed,
        "symmetric_abs_change_q99", "baseline_percentile_01_99")) ||
      any(bounds$lower_bound[bounds$component %in% signed] != -bounds$upper_bound[bounds$component %in% signed]) ||
      any(!bounds$neutral_score[bounds$component %in% signed] %in% 50)) {
    stop("Invalid event component bounds.", call. = FALSE)
  }
  result <- as.data.frame(features)
  scores <- inputs
  qa <- lapply(scaling$components, function(component) {
    b <- bounds[match(component, bounds$component), ]
    x <- inputs[[component]]
    score <- if (b$degenerate_range) ifelse(is.na(x), NA_real_, if (component %in% signed) 50 else 0) else
      100 * (pmin(pmax(x, b$lower_bound), b$upper_bound) - b$lower_bound) /
      (b$upper_bound - b$lower_bound)
    # Avoid tiny excursions beyond endpoints from floating-point arithmetic.
    score <- pmin(pmax(score, 0), 100)
    scores[[component]] <<- score
    data.frame(component = component, available_hexes = sum(!is.na(x)), missing_hexes = sum(is.na(x)),
      lower_bound = b$lower_bound, upper_bound = b$upper_bound, degenerate_range = b$degenerate_range,
      scaling_method = b$scaling_method, bounds_preserved = b$bounds_preserved,
      neutral_score = b$neutral_score, equal_weight = b$equal_weight,
      below_lower_bound = sum(x < b$lower_bound, na.rm = TRUE),
      above_upper_bound = sum(x > b$upper_bound, na.rm = TRUE))
  })
  for (component in scaling$components) result[[paste0(component, "_score")]] <- scores[[component]]
  n_available <- rowSums(!is.na(scores))
  complete <- n_available == length(scaling$components)
  index <- rowMeans(scores, na.rm = FALSE)
  index[!complete] <- NA_real_
  result[[scaling$index_name]] <- index
  result[[paste0(scaling$index_name, "_components_available")]] <- n_available
  result[[paste0(scaling$index_name, "_components_required")]] <- length(scaling$components)
  result[[paste0(scaling$index_name, "_components_complete")]] <- complete
  result[[paste0(scaling$index_name, "_measurement_version")]] <- "fixed_components_v2"
  result[[paste0(scaling$index_name, "_scaling_reference_as_of_date")]] <- scaling$reference_date
  list(features = result, qa = do.call(rbind, qa))
}
