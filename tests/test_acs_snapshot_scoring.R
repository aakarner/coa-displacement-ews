# Synthetic tests: no data downloads, no artifact writes.
source("R/acs_snapshot_scoring.R")
source("R/utils.R")
expect_error <- function(expr) stopifnot(inherits(tryCatch({force(expr); NULL}, error = identity), "error"))
spec <- acs_component_spec()
x <- as.data.frame(setNames(rep(list(as.numeric(1:100)), nrow(spec)), spec$input))
x$acs_dollar_base_year <- 2024L
x$acs_rent_trend_reliable <- TRUE
ref <- acs_fit_scaling(x)
scored <- acs_apply_scaling(x, ref)$features
stopifnot(all(c("acs_rent_components_available", "acs_vulnerability_components_available",
  "rent_pressure_citywide_index", "demographic_vulnerability_index") %in% names(scored)))
inputs <- acs_component_inputs(x)
for (component in names(inputs)) stopifnot(isTRUE(all.equal(
  scored[[paste0("acs_score_", component)]], normalize_robust_to_100(inputs[[component]]))))
stopifnot(abs(scored$acs_score_low_income[1] - 100) < 1e-10, scored$acs_score_low_income[100] == 0)
later <- x
later$acs_rent_current_real <- later$acs_rent_current_real + 100
frozen <- acs_apply_scaling(later, ref)$features
stopifnot(all(abs(frozen$acs_score_rent_level - 100) < 1e-10))
stopifnot(!isTRUE(all.equal(frozen$acs_score_rent_level,
  acs_apply_scaling(later, acs_fit_scaling(later))$features$acs_score_rent_level)))
missing <- x
missing[1, spec$input] <- NA_real_
missing$acs_rent_trend_reliable[2:3] <- c(FALSE, NA)
missing$median_income_real[4] <- NA_real_
out <- acs_apply_scaling(missing, ref)$features
stopifnot(is.na(out$rent_pressure_citywide_index[1]), is.na(out$demographic_vulnerability_index[1]),
  out$acs_rent_components_available[1] == 0L,
  all(out$acs_rent_components_available[2:3] == 0L),
  out$acs_vulnerability_components_available[4] == 4L,
  is.na(out$demographic_vulnerability_index[4]),
  all(is.na(out$acs_score_rent_level[2:3])),
  all(is.na(out$rent_pressure_citywide_index[2:3])),
  all(out$acs_rent_components_required == 3L), all(out$acs_vulnerability_components_required == 5L))
# One missing term may never promote a variable-weight proxy for the full index.
for (input in spec$input) {
  missing_one <- x; missing_one[[input]][1] <- NA_real_
  result <- acs_apply_scaling(missing_one, ref)$features
  group <- spec$group[spec$input == input]
  index <- if (group == "rent") "rent_pressure_citywide_index" else "demographic_vulnerability_index"
  stopifnot(is.na(result[[index]][1]))
}
old_reference <- ref
old_reference$schema_version <- "acs-scaling-v1"
old_reference$missing_policy <- "mean_available_components_all_missing_is_NA"
upgraded <- acs_preserve_scaling_reference(old_reference)
stopifnot(identical(upgraded$bounds, old_reference$bounds),
  identical(upgraded$missing_policy, "all_required_components_fixed_equal_weights"),
  identical(acs_preserve_scaling_reference(upgraded), upgraded))
expect_error(acs_apply_scaling(x, old_reference))
constant <- x
constant$poverty_rate <- 5
constant$poverty_rate[1] <- NA_real_
out <- acs_apply_scaling(constant, acs_fit_scaling(constant))$features
stopifnot(is.na(out$acs_score_poverty[1]), all(out$acs_score_poverty[-1] == 0))
bad <- x; bad$acs_dollar_base_year <- 2023L
expect_error(acs_apply_scaling(bad, ref))
bad <- x; bad$pct_renter[1] <- Inf
expect_error(acs_fit_scaling(bad))
bad <- x; bad$pct_renter <- NA_real_
expect_error(acs_fit_scaling(bad))
bad_ref <- ref; bad_ref$bounds$upper_bound[1] <- -Inf
expect_error(acs_apply_scaling(x, bad_ref))
bad_ref <- ref; bad_ref$bounds$component[1] <- bad_ref$bounds$component[2]
expect_error(acs_apply_scaling(x, bad_ref))
cat("ACS snapshot scoring tests passed.\n")
