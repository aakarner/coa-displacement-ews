# Pure scoring helpers for a frozen, auditable amenity normalization reference.
# Exposure construction and geocoding remain in scripts/data/amenities.R.

amenity_count_confirmed <- function(flags, event_ids = seq_along(flags)) {
  if (length(flags) != length(event_ids) || !is.logical(flags)) {
    stop("Amenity confirmation flags must be logical and match the event IDs.", call. = FALSE)
  }
  if (all(is.na(flags))) return(NA_integer_)
  length(unique(event_ids[which(flags %in% TRUE)]))
}

amenity_component_inputs <- function(
  features,
  categories = c("cafe", "full_service_restaurant", "drinking_place")
) {
  required <- as.vector(outer(categories, c("recent", "previous"), paste, sep = "_"))
  if (!all(required %in% names(features))) {
    stop("Amenity exposure columns are missing: ",
         paste(setdiff(required, names(features)), collapse = ", "), call. = FALSE)
  }
  if (!nrow(features) || any(vapply(features[required], function(x) {
    !is.numeric(x) || any(!is.finite(x)) || any(x < 0)
  }, logical(1)))) {
    stop("Amenity exposures must be finite, nonnegative numeric values on a nonempty grid.",
         call. = FALSE)
  }
  components <- list()
  for (category in categories) {
    recent <- as.numeric(features[[paste0(category, "_recent")]])
    previous <- as.numeric(features[[paste0(category, "_previous")]])
    components[[paste0(category, "_recent")]] <- recent
    components[[paste0(category, "_positive_change")]] <- pmax(recent - previous, 0)
  }
  components
}

amenity_fit_scaling <- function(
  features,
  analysis_as_of_date = as.Date(NA),
  categories = c("cafe", "full_service_restaurant", "drinking_place"),
  probabilities = c(0.01, 0.99)
) {
  if (length(probabilities) != 2L || any(!is.finite(probabilities)) ||
      probabilities[1] < 0 || probabilities[2] > 1 ||
      probabilities[1] >= probabilities[2]) {
    stop("Amenity scaling requires two increasing probabilities between zero and one.",
         call. = FALSE)
  }
  inputs <- amenity_component_inputs(as.data.frame(features), categories)
  components <- do.call(rbind, lapply(names(inputs), function(component) {
    x <- inputs[[component]]
    bounds <- as.numeric(stats::quantile(x, probabilities, names = FALSE, type = 7))
    data.frame(
      component = component,
      lower_probability = probabilities[1],
      upper_probability = probabilities[2],
      lower_bound = bounds[1],
      upper_bound = bounds[2],
      degenerate_baseline_range = bounds[1] == bounds[2],
      baseline_hexes = length(x),
      baseline_raw_min = min(x),
      baseline_raw_max = max(x),
      stringsAsFactors = FALSE
    )
  }))
  list(
    schema_version = "amenity-scaling-v1",
    analysis_as_of_date = as.Date(analysis_as_of_date),
    categories = categories,
    quantile_type = 7L,
    degenerate_range_policy = "zero_score_with_qa_flag",
    components = components
  )
}

amenity_apply_scaling <- function(features, scaling) {
  if (!is.list(scaling) || !identical(scaling$schema_version, "amenity-scaling-v1") ||
      !identical(scaling$degenerate_range_policy, "zero_score_with_qa_flag") ||
      !identical(scaling$categories,
                 c("cafe", "full_service_restaurant", "drinking_place"))) {
    stop("Unrecognized amenity scaling reference or category contract.", call. = FALSE)
  }
  result <- as.data.frame(features)
  inputs <- amenity_component_inputs(result, scaling$categories)
  bounds <- scaling$components
  required <- c("component", "lower_bound", "upper_bound", "degenerate_baseline_range")
  if (!is.data.frame(bounds) || !all(required %in% names(bounds)) ||
      anyDuplicated(bounds$component) ||
      !setequal(bounds$component, names(inputs)) ||
      any(!is.finite(bounds$lower_bound)) || any(!is.finite(bounds$upper_bound)) ||
      any(bounds$upper_bound < bounds$lower_bound) ||
      !is.logical(bounds$degenerate_baseline_range) ||
      anyNA(bounds$degenerate_baseline_range) ||
      any(bounds$degenerate_baseline_range != (bounds$lower_bound == bounds$upper_bound))) {
    stop("Amenity scaling reference has invalid or incomplete component bounds.", call. = FALSE)
  }
  normalized <- list()
  qa <- lapply(names(inputs), function(component) {
    x <- inputs[[component]]
    bound <- bounds[match(component, bounds$component), ]
    # Match normalize_robust_to_100() exactly, including zero-range behavior.
    score <- if (bound$degenerate_baseline_range) {
      rep(0, length(x))
    } else {
      clipped <- pmin(pmax(x, bound$lower_bound), bound$upper_bound)
      (clipped - bound$lower_bound) / (bound$upper_bound - bound$lower_bound) * 100
    }
    normalized[[component]] <<- score
    data.frame(
      component = component,
      lower_bound = bound$lower_bound,
      upper_bound = bound$upper_bound,
      degenerate_baseline_range = bound$degenerate_baseline_range,
      hexes = length(x),
      raw_min = min(x),
      raw_max = max(x),
      below_lower_bound_hexes = sum(x < bound$lower_bound),
      above_upper_bound_hexes = sum(x > bound$upper_bound),
      below_lower_bound_pct = 100 * mean(x < bound$lower_bound),
      above_upper_bound_pct = 100 * mean(x > bound$upper_bound),
      positive_input_zero_score_hexes = sum(x > 0 & score == 0),
      score_min = min(score),
      score_max = max(score),
      stringsAsFactors = FALSE
    )
  })
  for (category in scaling$categories) {
    result[[paste0("amenity_", category, "_weighted_change")]] <-
      result[[paste0(category, "_recent")]] - result[[paste0(category, "_previous")]]
    result[[paste0("amenity_", category, "_positive_weighted_change")]] <-
      inputs[[paste0(category, "_positive_change")]]
    result[[paste0("amenity_", category, "_score")]] <- rowMeans(cbind(
      normalized[[paste0(category, "_recent")]],
      normalized[[paste0(category, "_positive_change")]]
    ))
  }
  result$amenity_change_index <- rowMeans(
    result[paste0("amenity_", scaling$categories, "_score")]
  )
  result$amenity_scaling_degenerate_components <- sum(bounds$degenerate_baseline_range)
  list(features = result, qa = do.call(rbind, qa))
}

amenity_coverage_contract <- function(contract = NULL, legacy_mode = FALSE) {
  if (is.null(contract)) {
    return(list(
      window_complete = if (legacy_mode) TRUE else NA,
      status = if (legacy_mode) "legacy_source_audit_assumption" else "unverified",
      retrospective_usable = if (legacy_mode) TRUE else NA
    ))
  }
  required <- c("window_complete", "status", "retrospective_usable")
  if (!is.list(contract) || !all(required %in% names(contract)) ||
      !is.logical(contract$window_complete) || length(contract$window_complete) != 1L ||
      !is.logical(contract$retrospective_usable) || length(contract$retrospective_usable) != 1L ||
      !is.character(contract$status) || length(contract$status) != 1L ||
      is.na(contract$status) || !nzchar(contract$status)) {
    stop("Amenity coverage_contract must provide scalar logical window_complete and ",
         "retrospective_usable plus a nonempty status string.", call. = FALSE)
  }
  contract
}
