# Pure tests for every corrected Part 2 event/ownership recipe; no output writes.
source("R/part2_event_scoring.R")
source("R/part2_index_scoring.R")
source("R/utils.R")
expect_error <- function(expr) stopifnot(inherits(tryCatch({force(expr); NULL}, error = identity), "error"))
for (family in c("event", "index")) {
  get_components <- get(paste0("part2_", family, "_components"))
  fit <- get(paste0("part2_fit_", family, "_scaling"))
  apply <- get(paste0("part2_apply_", family, "_scaling"))
  indices <- if (family == "event") c("sr_311_pressure_index", "demolition_pressure_index") else
    c("eviction_pressure_index", "ownership_pressure_index")
  for (index in indices) {
    components <- get_components(index); n <- length(components)
    signed <- components[grepl("rate_change_per_100_units$", components)]
    x <- as.data.frame(setNames(rep(list(as.numeric(1:100)), n), components))
    if (length(signed)) x[[signed]] <- seq(-50, 49)
    x$analysis_as_of_date <- as.Date("2025-04-01")
    ref <- fit(x, components, index, as.Date("2025-04-01"))
    result <- apply(x, ref)$features
    score_names <- paste0(components, "_score")
    count_name <- paste0(index, "_components_available")
    stopifnot(all(result[[count_name]] == n), all(result[[paste0(index, "_components_required")]] == n),
      all(result[[paste0(index, "_components_complete")]]),
      all(result[[paste0(index, "_measurement_version")]] == "fixed_components_v2"))
    for (component in components) {
      expected <- if (!component %in% signed) normalize_robust_to_100(x[[component]]) else {
        bound <- as.numeric(quantile(abs(x[[component]]), .99, type = 7))
        50 + 50 * pmax(pmin(x[[component]] / bound, 1), -1)
      }
      stopifnot(isTRUE(all.equal(result[[paste0(component, "_score")]], expected)))
    }
    stopifnot(isTRUE(all.equal(result[[index]], rowMeans(result[score_names]))))
    later <- x; later$analysis_as_of_date <- as.Date("2026-04-01")
    later[[components[1]]] <- later[[components[1]]] + 100
    frozen <- apply(later, ref)$features
    stopifnot(all(frozen[[score_names[1]]] == 100))
    refitted <- fit(later, components, index, as.Date("2026-04-01"))
    stopifnot(!isTRUE(all.equal(frozen[[index]], apply(later, refitted)$features[[index]])))
    expect_error(apply(x, refitted))
    expect_error(fit(later, components, index, as.Date("2025-04-01")))
    missing <- x; missing[1, components] <- NA_real_; missing[2, components[1]] <- NA_real_
    m <- apply(missing, ref)$features
    stopifnot(all(is.na(m[[index]][1:2])), m[[count_name]][1] == 0L, m[[count_name]][2] == n-1L,
      all(!m[[paste0(index, "_components_complete")]][1:2]))
    constant <- x; constant[components] <- 0; constant[1, components] <- NA_real_
    zero_ref <- fit(constant, components, index, as.Date("2025-04-01"))
    z <- apply(constant, zero_ref)$features
    stopifnot(is.na(z[[index]][1]), all(z[[index]][-1] == if (length(signed)) 50/n else 0),
      all(zero_ref$bounds$degenerate_range))
    if (length(signed)) {
      stopifnot(all(z[[paste0(signed, "_score")]][-1] == 50))
      constant[[signed]][2:4] <- c(-10, 0, 10)
      neutral <- apply(constant, zero_ref)$features
      stopifnot(all(neutral[[paste0(signed, "_score")]][2:4] == 50))
      edge <- x[1:4, ]; edge[[signed]] <- c(-ref$bounds$upper_bound[ref$bounds$component == signed], 0,
        ref$bounds$upper_bound[ref$bounds$component == signed], NA_real_)
      stopifnot(isTRUE(all.equal(apply(edge, ref)$features[[paste0(signed, "_score")]], c(0,50,100,NA))))
    }
    # Existing unchanged bounds survive even if input distributions change.
    altered <- x; altered[[components[1]]] <- altered[[components[1]]] + 1000
    legacy <- ref; legacy$schema_version <- paste0("part2-", family, "-scaling-v1")
    legacy$bounds <- legacy$bounds[c("component", "lower_bound", "upper_bound", "reference_available_hexes", "degenerate_range")]
    retained <- fit(altered, components, index, as.Date("2025-04-01"), preserved_scaling = legacy)
    unchanged <- !retained$bounds$component %in% signed
    stopifnot(identical(retained$bounds$lower_bound[unchanged], ref$bounds$lower_bound[unchanged]),
      identical(retained$bounds$upper_bound[unchanged], ref$bounds$upper_bound[unchanged]),
      all(retained$bounds$bounds_preserved[unchanged]),
      identical(retained, fit(altered, components, index, as.Date("2025-04-01"), preserved_scaling = retained)))
    bad <- x; bad[[components[1]]] <- NA_real_
    expect_error(fit(bad, components, index, as.Date("2025-04-01")))
    bad <- x; bad[[components[1]]][1] <- Inf
    expect_error(apply(bad, ref))
    bad_ref <- ref; bad_ref$bounds$component[2] <- bad_ref$bounds$component[1]
    expect_error(apply(x, bad_ref))
    expect_error(fit(x, components, index, as.Date("2025-04-01"), preserved_scaling = bad_ref))
    bad_ref <- ref; bad_ref$bounds$upper_bound[1] <- -Inf
    expect_error(apply(x, bad_ref))
    bad_ref <- ref; bad_ref$missing_policy <- "available_mean"
    expect_error(apply(x, bad_ref))
    expect_error(fit(x, rev(components), index, as.Date("2025-04-01")))
  }
}
cat("Part 2 fixed recipes, preserved/frozen scoring, signed neutral-zero and missingness tests passed.\n")
