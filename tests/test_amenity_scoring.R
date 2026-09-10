# Run from the repository root: Rscript tests/test_amenity_scoring.R
# Pure synthetic tests: no geocoding, downloads, or project data required.
source("R/utils.R")
source("R/amenity_scoring.R")

expect_equal <- function(actual, expected) {
  stopifnot(isTRUE(all.equal(actual, expected, check.attributes = FALSE)))
}
expect_error <- function(expr, pattern) {
  result <- tryCatch(force(expr), error = identity)
  stopifnot(inherits(result, "error"), grepl(pattern, conditionMessage(result)))
}

stopifnot(is.na(amenity_count_confirmed(c(NA, NA))),
          amenity_count_confirmed(c(TRUE, NA, FALSE)) == 1L,
          amenity_count_confirmed(c(FALSE, FALSE)) == 0L,
          amenity_count_confirmed(c(TRUE, TRUE, NA), c("a", "a", "b")) == 1L)
expect_error(amenity_count_confirmed(c("TRUE", "FALSE")), "must be logical")

categories <- c("cafe", "full_service_restaurant", "drinking_place")
features <- data.frame(hex_id = paste0("h", 1:101))
for (i in seq_along(categories)) {
  features[[paste0(categories[i], "_recent")]] <- (0:100)^i / 100
  features[[paste0(categories[i], "_previous")]] <- (100:0)^i / 120
}
scaling <- amenity_fit_scaling(features, as.Date("2025-04-01"))
scored <- amenity_apply_scaling(features, scaling)
stopifnot(nrow(scaling$components) == 6L, nrow(scored$qa) == 6L)
stopifnot(identical(scaling$analysis_as_of_date, as.Date("2025-04-01")))
for (category in categories) {
  recent <- features[[paste0(category, "_recent")]]
  previous <- features[[paste0(category, "_previous")]]
  expected <- rowMeans(cbind(
    normalize_robust_to_100(recent),
    normalize_robust_to_100(pmax(recent - previous, 0))
  ))
  expect_equal(scored$features[[paste0("amenity_", category, "_score")]], expected)
  expect_equal(scored$features[[paste0("amenity_", category, "_weighted_change")]],
               recent - previous)
}
expect_equal(scored$features$amenity_change_index,
             rowMeans(scored$features[paste0("amenity_", categories, "_score")]))
expect_equal(scored$features[names(features)], features)

# Later values must use the baseline bounds, not the later distribution.
later <- features
later$cafe_recent <- later$cafe_recent * 2
frozen <- amenity_apply_scaling(later, scaling)
refitted <- amenity_apply_scaling(later, amenity_fit_scaling(later))
stopifnot(!isTRUE(all.equal(frozen$features$amenity_cafe_score,
                          refitted$features$amenity_cafe_score)))
expect_equal(frozen$qa$upper_bound, scaling$components$upper_bound)
stopifnot(frozen$qa$above_upper_bound_hexes[1] > scored$qa$above_upper_bound_hexes[1])
stopifnot(all(frozen$features$amenity_change_index >= 0),
          all(frozen$features$amenity_change_index <= 100))

# Reordering a reference or a grid cannot change its scoring semantics.
reordered <- scaling
reordered$components <- scaling$components[6:1, ]
expect_equal(amenity_apply_scaling(features, reordered)$features, scored$features)
reverse_rows <- nrow(features):1
expect_equal(amenity_apply_scaling(features[reverse_rows, ], scaling)$features,
             scored$features[reverse_rows, ])

# Entirely absent / degenerate baseline categories retain legacy zero scores,
# even if new events appear later. Flag the otherwise invisible later change.
zeros <- features
for (column in setdiff(names(zeros), "hex_id")) zeros[[column]] <- 0
zero_reference <- amenity_fit_scaling(zeros)
zero_scores <- amenity_apply_scaling(zeros, zero_reference)
stopifnot(all(zero_reference$components$degenerate_baseline_range),
          all(zero_scores$features$amenity_change_index == 0),
          all(zero_scores$features$amenity_scaling_degenerate_components == 6))
new_events <- zeros
new_events$cafe_recent[1] <- 5
new_scores <- amenity_apply_scaling(new_events, zero_reference)
stopifnot(all(new_scores$features$amenity_change_index == 0),
          all(new_scores$qa$positive_input_zero_score_hexes[1:2] == 1),
          all(new_scores$qa$above_upper_bound_hexes[1:2] == 1))

constant <- zeros
constant$cafe_recent <- 3
constant$cafe_previous <- 1
constant_scores <- amenity_apply_scaling(constant, amenity_fit_scaling(constant))
expect_equal(constant_scores$features$amenity_cafe_score,
             normalize_robust_to_100(rep(3, nrow(constant))))

bad <- features; bad$cafe_recent[1] <- NA_real_
expect_error(amenity_fit_scaling(bad), "finite, nonnegative")
bad <- features; bad$cafe_previous[1] <- -1
expect_error(amenity_fit_scaling(bad), "finite, nonnegative")
expect_error(amenity_fit_scaling(features[FALSE, ]), "nonempty grid")
expect_error(amenity_fit_scaling(features[, -2]), "columns are missing")
expect_error(amenity_fit_scaling(features, probabilities = c(.99, .01)), "increasing")
bad_reference <- scaling; bad_reference$components <- bad_reference$components[-1, ]
expect_error(amenity_apply_scaling(features, bad_reference), "invalid or incomplete")
bad_reference <- scaling; bad_reference$components$upper_bound[1] <- -1
expect_error(amenity_apply_scaling(features, bad_reference), "invalid or incomplete")
bad_reference <- scaling; bad_reference$components$degenerate_baseline_range[1] <- TRUE
expect_error(amenity_apply_scaling(features, bad_reference), "invalid or incomplete")
bad_reference <- scaling; bad_reference$schema_version <- "unknown"
expect_error(amenity_apply_scaling(features, bad_reference), "Unrecognized")

# Custom historical inputs cannot acquire a fabricated completeness flag.
unknown <- amenity_coverage_contract()
stopifnot(is.na(unknown$window_complete), is.na(unknown$retrospective_usable),
          unknown$status == "unverified")
legacy <- amenity_coverage_contract(legacy_mode = TRUE)
stopifnot(legacy$window_complete, legacy$retrospective_usable,
          legacy$status == "legacy_source_audit_assumption")
contract <- list(window_complete = NA, status = "retrospective_reconstructed",
                 retrospective_usable = TRUE, reason = "Fixture source history")
stopifnot(identical(amenity_coverage_contract(contract), contract))
bad_contract <- contract; bad_contract$window_complete <- "TRUE"
expect_error(amenity_coverage_contract(bad_contract), "scalar logical")
bad_contract <- contract; bad_contract$status <- ""
expect_error(amenity_coverage_contract(bad_contract), "nonempty status")
bad_contract <- contract; bad_contract$retrospective_usable <- NULL
expect_error(amenity_coverage_contract(bad_contract), "scalar logical")

cat("Amenity scoring, frozen-reference, and coverage-contract tests passed.\n")
