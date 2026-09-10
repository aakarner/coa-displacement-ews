# Part 2 ACS composites: same selected Part 1 components, with common-dollar
# income and a single earlier-vintage normalization reference. No clustering.
acs_component_spec <- function() {
  data.frame(
    component = c("rent_level", "rent_growth", "rent_acceleration", "low_income",
                  "renters", "poverty", "rent_burden", "low_college"),
    input = c("acs_rent_current_real", "acs_rent_growth_recent_annualized_pct",
              "acs_rent_acceleration_pp", "median_income_real", "pct_renter",
              "poverty_rate", "pct_rent_burden_30plus", "pct_college"),
    direction = c(1, 1, 1, -1, 1, 1, 1, -1),
    group = c(rep("rent", 3), rep("vulnerability", 5)),
    stringsAsFactors = FALSE)
}

acs_component_inputs <- function(features) {
  spec <- acs_component_spec()
  if (!all(c(spec$input, "acs_rent_trend_reliable") %in% names(features)) ||
      !nrow(features)) stop("Missing ACS component inputs or empty data.", call. = FALSE)
  if (!is.logical(features$acs_rent_trend_reliable)) {
    stop("ACS trend reliability must be logical.", call. = FALSE)
  }
  inputs <- setNames(lapply(seq_len(nrow(spec)), function(i) {
    x <- features[[spec$input[i]]]
    if (!is.numeric(x) || any(!is.finite(x) & !is.na(x))) {
      stop("ACS inputs must be numeric, finite or missing: ", spec$input[i], call. = FALSE)
    }
    x <- x * spec$direction[i]
    if (spec$group[i] == "rent") {
      x[!features$acs_rent_trend_reliable %in% TRUE] <- NA_real_
    }
    x
  }), spec$component)
  as.data.frame(inputs)
}

acs_fit_scaling <- function(features, reference_date = as.Date("2025-04-01"), dollar_base_year = 2024L) {
  inputs <- acs_component_inputs(features)
  bounds <- do.call(rbind, lapply(names(inputs), function(component) {
    x <- inputs[[component]]
    if (all(is.na(x))) stop("No earlier observations to scale: ", component, call. = FALSE)
    q <- as.numeric(stats::quantile(x, c(.01, .99), na.rm = TRUE, type = 7))
    data.frame(component = component, lower_bound = q[1], upper_bound = q[2],
      reference_available_hexes = sum(!is.na(x)), degenerate_range = q[1] == q[2])
  }))
  list(schema_version = "acs-scaling-v2", reference_date = as.Date(reference_date),
       dollar_base_year = as.integer(dollar_base_year), probabilities = c(.01, .99),
       quantile_type = 7L, component_spec = acs_component_spec(), bounds = bounds,
       missing_policy = "all_required_components_fixed_equal_weights",
       degenerate_range_policy = "zero_score_for_observed_inputs")
}

acs_preserve_scaling_reference <- function(scaling) {
  if (!is.list(scaling) || !scaling$schema_version %in% c("acs-scaling-v1", "acs-scaling-v2") ||
      !identical(scaling$component_spec, acs_component_spec()) ||
      !identical(scaling$probabilities, c(.01, .99)) || !identical(scaling$quantile_type, 7L) ||
      !scaling$missing_policy %in% c("mean_available_components_all_missing_is_NA", "all_required_components_fixed_equal_weights") ||
      !identical(scaling$degenerate_range_policy, "zero_score_for_observed_inputs")) {
    stop("Invalid existing ACS normalization reference.", call. = FALSE)
  }
  # Only the composite recipe changes. Never refit the bounds on the newly
  # complete cohort, and never mutate/reorder the original eight bound rows.
  scaling$schema_version <- "acs-scaling-v2"
  scaling$missing_policy <- "all_required_components_fixed_equal_weights"
  scaling$normalization_origin <- "preserved_initial_2025_full_grid_component_bounds"
  scaling
}

acs_apply_scaling <- function(features, scaling) {
  spec <- acs_component_spec()
  if (!identical(scaling$schema_version, "acs-scaling-v2") ||
      !identical(scaling$component_spec, spec) ||
      !identical(scaling$probabilities, c(.01, .99)) ||
      !identical(scaling$missing_policy, "all_required_components_fixed_equal_weights") ||
      !identical(scaling$degenerate_range_policy, "zero_score_for_observed_inputs")) {
    stop("Invalid ACS normalization contract.", call. = FALSE)
  }
  b <- scaling$bounds
  if (!all(c("component", "lower_bound", "upper_bound", "degenerate_range") %in% names(b)) ||
      anyDuplicated(b$component) || !setequal(b$component, spec$component) ||
      any(!is.finite(b$lower_bound)) || any(!is.finite(b$upper_bound)) ||
      any(b$upper_bound < b$lower_bound) || anyNA(b$degenerate_range) ||
      any(b$degenerate_range != (b$upper_bound == b$lower_bound))) {
    stop("Invalid ACS component bounds.", call. = FALSE)
  }
  if (!"acs_dollar_base_year" %in% names(features) ||
      anyNA(features$acs_dollar_base_year) ||
      any(features$acs_dollar_base_year != scaling$dollar_base_year)) {
    stop("ACS scoring requires the reference dollar base.", call. = FALSE)
  }
  inputs <- acs_component_inputs(features)
  scores <- inputs
  qa <- lapply(names(inputs), function(component) {
    bound <- b[match(component, b$component), ]
    x <- inputs[[component]]
    scores[[component]] <<- if (bound$degenerate_range) ifelse(is.na(x), NA_real_, 0) else
      100 * (pmin(pmax(x, bound$lower_bound), bound$upper_bound) - bound$lower_bound) /
      (bound$upper_bound - bound$lower_bound)
    data.frame(component = component, available_hexes = sum(!is.na(x)),
      missing_hexes = sum(is.na(x)), below_lower_bound = sum(x < bound$lower_bound, na.rm = TRUE),
      above_upper_bound = sum(x > bound$upper_bound, na.rm = TRUE),
      lower_bound = bound$lower_bound, upper_bound = bound$upper_bound)
  })
  result <- as.data.frame(features)
  for (component in names(scores)) result[[paste0("acs_score_", component)]] <- scores[[component]]
  for (group in unique(spec$group)) {
    group_scores <- scores[spec$component[spec$group == group]]
    n <- rowSums(!is.na(group_scores))
    index <- rowMeans(group_scores, na.rm = FALSE)
    index_name <- if (group == "rent") "rent_pressure_citywide_index" else "demographic_vulnerability_index"
    result[[index_name]] <- index
    result[[paste0("acs_", group, "_components_available")]] <- n
    result[[paste0("acs_", group, "_components_required")]] <- ncol(group_scores)
    result[[paste0("acs_", group, "_complete")]] <- n == ncol(group_scores)
  }
  result$acs_composite_missing_policy <- "all_required_components_fixed_equal_weights"
  result$acs_scaling_reference_as_of_date <- scaling$reference_date
  list(features = result, qa = do.call(rbind, qa))
}
