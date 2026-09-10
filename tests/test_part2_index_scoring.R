# Pure regression tests for corrected ownership/eviction scoring; no output writes.
source("R/part2_index_scoring.R")
source("R/utils.R")
expect_error <- function(expr) stopifnot(inherits(tryCatch({force(expr); NULL}, error = identity), "error"))
eq <- function(x, y) stopifnot(isTRUE(all.equal(x, y, tolerance = 1e-12)))
specification <- list(
  ownership_pressure_index = c("pct_corporate_units", "corporate_owned_units_per_km2",
    "pct_financialized_owner_parcels"),
  eviction_pressure_index = c("eviction_latest_12mo_per_100_units",
    "eviction_latest_12mo_rate_change_per_100_units"))
earlier_date <- as.Date("2025-04-01")
later_date <- as.Date("2026-04-01")

for (index in names(specification)) {
  components <- specification[[index]]; n <- length(components)
  stopifnot(identical(part2_index_components(index), components))
  signed <- components[grepl("rate_change_per_100_units$", components)]
  unchanged <- setdiff(components, signed)
  x <- as.data.frame(setNames(rep(list(as.numeric(1:100)), n), components))
  if (length(signed)) x[[signed]] <- as.numeric(-50:49)
  x$analysis_as_of_date <- earlier_date
  ref <- part2_fit_index_scaling(x, components, index, earlier_date)
  scored <- part2_apply_index_scaling(x, ref)
  result <- scored$features
  score_names <- paste0(components, "_score")
  count_name <- paste0(index, "_components_available")
  stopifnot(ref$schema_version == "part2-index-scaling-v2",
    ref$measurement_version == "fixed_components_v2",
    ref$missing_policy == "all_components_required_fixed_equal_weights",
    all(ref$bounds$equal_weight == 1/n), all(scored$qa$equal_weight == 1/n),
    all(result[[count_name]] == n),
    all(result[[paste0(index, "_components_required")]] == n),
    all(result[[paste0(index, "_components_complete")]]),
    all(result[[paste0(index, "_measurement_version")]] == "fixed_components_v2"))

  # Unchanged terms retain the original p1/p99 recipe; signed change is different.
  for (component in unchanged) {
    eq(result[[paste0(component, "_score")]], normalize_robust_to_100(x[[component]]))
    b <- ref$bounds[ref$bounds$component == component, ]
    eq(c(b$lower_bound, b$upper_bound), as.numeric(quantile(x[[component]], c(.01,.99), type=7)))
    stopifnot(b$scaling_method == "baseline_percentile_01_99", is.na(b$neutral_score))
  }
  if (length(signed)) {
    bound <- as.numeric(quantile(abs(x[[signed]]), .99, type=7))
    b <- ref$bounds[ref$bounds$component == signed, ]
    eq(c(b$lower_bound, b$upper_bound), c(-bound, bound))
    stopifnot(b$scaling_method == "symmetric_abs_change_q99", b$neutral_score == 50)
    eq(result[[paste0(signed, "_score")]], 50 + 50*pmax(pmin(x[[signed]]/bound, 1), -1))
    edges <- x[1:6, ]; edges[[signed]] <- c(-2*bound, -bound, 0, bound, 2*bound, NA_real_)
    eq(part2_apply_index_scaling(edges, ref)$features[[paste0(signed, "_score")]], c(0,0,50,100,100,NA))
    bad <- ref; bad$bounds$lower_bound[bad$bounds$component == signed] <- -bound + 1
    expect_error(part2_apply_index_scaling(x, bad))
    bad <- ref; bad$bounds$neutral_score[bad$bounds$component == signed] <- 0
    expect_error(part2_apply_index_scaling(x, bad))
  }
  eq(result[[index]], rowMeans(result[score_names], na.rm=FALSE))

  # Missing any individual required term invalidates the index, not its peers.
  for (component in components) {
    missing <- x; missing[1, components] <- NA_real_; missing[2, component] <- NA_real_
    r <- part2_apply_index_scaling(missing, ref)$features
    stopifnot(all(is.na(r[[index]][1:2])), r[[count_name]][1] == 0L,
      r[[count_name]][2] == n-1L, all(!r[[paste0(index, "_components_complete")]][1:2]))
    eq(r[3:nrow(r), score_names], result[3:nrow(result), score_names])
  }

  # All observed zeros are valid. Signed B=0 is neutral50, while missing stays NA.
  zero <- x; zero[components] <- 0; zero[1, components] <- NA_real_
  zero_ref <- part2_fit_index_scaling(zero, components, index, earlier_date)
  zr <- part2_apply_index_scaling(zero, zero_ref)$features
  stopifnot(all(zero_ref$bounds$degenerate_range), is.na(zr[[index]][1]),
    all(zr[[index]][-1] == if (length(signed)) 25 else 0))
  for (component in unchanged) stopifnot(all(zr[[paste0(component, "_score")]][-1] == 0))
  if (length(signed)) {
    zero[[signed]][2:4] <- c(-10,0,10); zero$analysis_as_of_date <- later_date
    eq(part2_apply_index_scaling(zero, zero_ref)$features[[paste0(signed, "_score")]][1:4], c(NA,50,50,50))
  }

  # Later values use earlier bounds; neither an implicit refit nor a future ref is allowed.
  later <- x; later$analysis_as_of_date <- later_date; later[[components[1]]] <- later[[components[1]]] + 100
  frozen <- part2_apply_index_scaling(later, ref)$features
  stopifnot(all(frozen[[score_names[1]]] == 100),
    all(frozen[[paste0(index, "_scaling_reference_as_of_date")]] == earlier_date))
  refitted <- part2_fit_index_scaling(later, components, index, later_date)
  stopifnot(!isTRUE(all.equal(frozen[[index]], part2_apply_index_scaling(later, refitted)$features[[index]])))
  expect_error(part2_apply_index_scaling(x, refitted))
  expect_error(part2_fit_index_scaling(later, components, index, earlier_date))

  # A realistic legacy eviction contract still has its two discarded diagnostics.
  legacy_components <- if (length(signed)) c(components[1],
    "eviction_cases_latest_12mo_change_pct", "eviction_recent_share") else components
  legacy <- list(schema_version="part2-index-scaling-v1", index_name=index,
    components=legacy_components, reference_date=earlier_date, probabilities=c(.01,.99),
    quantile_type=7L, bounds=data.frame(component=legacy_components, lower_bound=10,
      upper_bound=90, reference_available_hexes=42L, degenerate_range=FALSE))
  if (length(signed)) {
    legacy$bounds$lower_bound[2:3] <- c(-100,0)
    legacy$bounds$upper_bound[2:3] <- c(100,1)
  }
  altered <- x; altered[unchanged] <- lapply(altered[unchanged], function(z) z+1000)
  retained <- part2_fit_index_scaling(altered, components, index, earlier_date, preserved_scaling=legacy)
  keep <- retained$bounds$component %in% unchanged
  stopifnot(all(retained$bounds$lower_bound[keep] == 10), all(retained$bounds$upper_bound[keep] == 90),
    all(retained$bounds$reference_available_hexes[keep] == 42L),
    all(retained$bounds$bounds_preserved[keep]),
    all(!retained$bounds$bounds_preserved[!keep]),
    identical(retained, part2_fit_index_scaling(altered, components, index, earlier_date, preserved_scaling=retained)))
  if (length(signed)) eq(retained$bounds$upper_bound[!keep], bound)
  expect_error(part2_apply_index_scaling(x, legacy))
  bad <- legacy; bad$reference_date <- later_date
  expect_error(part2_fit_index_scaling(x, components, index, earlier_date, preserved_scaling=bad))
  bad <- legacy; bad$bounds$upper_bound[1] <- -1
  expect_error(part2_fit_index_scaling(x, components, index, earlier_date, preserved_scaling=bad))
  bad <- legacy; bad$bounds <- bad$bounds[-1, ]
  expect_error(part2_fit_index_scaling(x, components, index, earlier_date, preserved_scaling=bad))

  # Invalid data, component definitions, or declared scoring policies fail closed.
  bad <- x; bad[[components[1]]] <- NA_real_
  expect_error(part2_fit_index_scaling(bad, components, index, earlier_date))
  bad <- x; bad[[components[1]]][1] <- Inf
  expect_error(part2_apply_index_scaling(bad, ref))
  bad <- x; bad[[components[1]]] <- as.character(bad[[components[1]]])
  expect_error(part2_apply_index_scaling(bad, ref))
  expect_error(part2_fit_index_scaling(x, rev(components), index, earlier_date))
  for (field in c("schema_version", "missing_policy", "degenerate_range_policy", "measurement_version")) {
    bad <- ref; bad[[field]] <- "invalid_contract"
    expect_error(part2_apply_index_scaling(x, bad))
  }
  bad <- ref; bad$bounds$equal_weight[1] <- 1
  expect_error(part2_apply_index_scaling(x, bad))
  bad <- ref; bad$bounds$component[2] <- bad$bounds$component[1]
  expect_error(part2_apply_index_scaling(x, bad))
  bad <- ref; bad$bounds$upper_bound[1] <- -Inf
  expect_error(part2_apply_index_scaling(x, bad))
  bad <- ref; bad$bounds$scaling_method[1] <- "incorrect_scaling"
  expect_error(part2_apply_index_scaling(x, bad))
  bad <- ref; bad$bounds$degenerate_range[1] <- TRUE
  expect_error(part2_apply_index_scaling(x, bad))
  bad <- ref; bad$probabilities <- c(.05,.95)
  expect_error(part2_apply_index_scaling(x, bad))
}
cat("Part 2 ownership/eviction v2 fixed recipes, signed bounds, strict missingness and preservation tests passed.\n")
