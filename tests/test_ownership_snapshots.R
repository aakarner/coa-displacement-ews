# Run from the repository root: Rscript tests/test_ownership_snapshots.R
# Synthetic regressions only; no private inputs or downloaded data are needed.
source("R/ownership_snapshots.R")

expect_error <- function(expr, pattern = NULL) {
  result <- tryCatch(force(expr), error = identity)
  stopifnot(inherits(result, "error"))
  if (!is.null(pattern)) stopifnot(grepl(pattern, conditionMessage(result)))
  invisible(result)
}

expect_equal <- function(actual, expected) {
  stopifnot(isTRUE(all.equal(actual, expected, check.attributes = FALSE)))
}

stopifnot(identical(
  ownership_bool(c("TRUE", "false", " True ", "NA", "", NA_character_)),
  c(TRUE, FALSE, TRUE, NA, NA, NA)
))
stopifnot(identical(ownership_bool(c(TRUE, FALSE, NA)), c(TRUE, FALSE, NA)))
for (invalid in c("1", "0", "yes", "no", "unknown")) {
  expect_error(ownership_bool(invalid), "Invalid ownership boolean")
}
expect_equal(ownership_safe_share(c(1, 0, 2), c(2, 0, 0)), c(.5, NA, NA))

targets <- data.frame(
  parcel_id = c("A", "B", "C", "D", "E"),
  source_county = c("Travis", "Travis", "Travis", "Williamson", "Hays")
)
rows <- data.frame(
  parcel_id = rep(targets$parcel_id, 2),
  source_county = rep(targets$source_county, 2),
  tax_year = rep(c(2024L, 2025L), each = 5),
  owner_ids = "owner-id",
  owner_names = "Synthetic owner",
  is_owner_occupied = c("FALSE", "NA", "NA", "TRUE", "FALSE",
                        "TRUE", "NA", "FALSE", "TRUE", "FALSE"),
  has_financialized_owner = c("TRUE", "FALSE", "NA", "FALSE", "TRUE",
                             "FALSE", "FALSE", "TRUE", "FALSE", "TRUE"),
  is_corporate_owned = c("TRUE", "FALSE", "NA", "FALSE", "TRUE",
                        "FALSE", "FALSE", "TRUE", "FALSE", "TRUE"),
  classification_status = c("matched_classified", "matched_evidence_insufficient",
                            "source_parcel_not_found", "matched_classified", "matched_classified",
                            "matched_classified", "matched_evidence_insufficient",
                            "matched_classified", "matched_classified", "matched_classified"),
  classification_rule_version = "test-owner-rule-v1",
  source_snapshot_id = rep(c("2024-fixture", "2025-fixture"), each = 5)
)
validate <- function(x, target = targets) {
  ownership_validate_rows(x, target, c(2024L, 2025L), "test-owner-rule-v1")
}
validated <- validate(rows)
stopifnot(is.logical(validated$is_corporate_owned), is.na(validated$is_corporate_owned[3]))
# Input order may differ from the target order.
stopifnot(nrow(validate(rows[nrow(rows):1, ])) == nrow(rows))

expect_error(validate(rows[-1, ]), "enumerate every target parcel")
expect_error(validate(rbind(rows, rows[1, ])), "duplicate parcel-year")
bad <- rows; bad$parcel_id[1] <- NA_character_
expect_error(validate(bad), "Missing or duplicate")
bad <- rows; bad$parcel_id[1] <- "UNEXPECTED"
expect_error(validate(bad), "enumerate every target parcel")
bad <- rows; bad$tax_year[1] <- 2023L
expect_error(validate(bad), "year or classifier")
bad <- rows; bad$tax_year[1] <- NA_integer_
expect_error(validate(bad))
bad <- rows; bad$source_county[1] <- "Hays"
expect_error(validate(bad), "source county")
bad <- rows; bad$classification_rule_version[1] <- "other-rule"
expect_error(validate(bad), "year or classifier")
bad <- rows; bad$source_snapshot_id <- NULL
expect_error(validate(bad), "schema is incomplete")
bad <- rows; bad$is_corporate_owned[3] <- "FALSE"
expect_error(validate(bad), "Missing source parcel")
bad <- rows; bad$has_financialized_owner[3] <- "FALSE"
expect_error(validate(bad), "Missing source parcel")
bad <- rows; bad$classification_status[3] <- "source_snapshot_unavailable"
stopifnot(is.na(validate(bad)$is_corporate_owned[3]))
bad$is_corporate_owned[3] <- "FALSE"
expect_error(validate(bad), "Missing source parcel")
bad <- rows; bad$is_owner_occupied[1] <- "NA"
expect_error(validate(bad), "Complete classification")
bad <- rows; bad$is_owner_occupied[1] <- "TRUE"
expect_error(validate(bad), "Corporate flag contradicts")
bad <- rows; bad$has_financialized_owner[1] <- "FALSE"
expect_error(validate(bad), "Corporate flag contradicts")
bad <- rows; bad$is_corporate_owned[1] <- "yes"
expect_error(validate(bad), "Invalid ownership boolean")

panel <- validated
panel$residential_units <- rep(c(10, 30, 60, 10, 1000), 2)
panel$hex_id <- rep(c("h1", "h1", "h1", "h2", NA_character_), 2)
grid <- data.frame(hex_id = c("h1", "h2", "empty"), area_km2 = c(2, 1, 1))

summarized <- ownership_summarise(panel[panel$hex_id %in% "h1", ], "tax_year")
expect_equal(summarized$residential_units, c(100, 100))
expect_equal(summarized$matched_units, c(40, 100))
expect_equal(summarized$complete_units, c(10, 70))
expect_equal(summarized$corporate_unknown_units, c(60, 0))
expect_equal(summarized$corporate_owned_units_observed, c(10, 60))
expect_equal(summarized$pct_corporate_units_lower, c(10, 60))
expect_equal(summarized$pct_corporate_units_upper, c(70, 60))
expect_equal(summarized$pct_corporate_units, c(NA, 60))
expect_equal(summarized$pct_financialized_owner_parcels, c(NA, 100 / 3))
expect_equal(summarized$corporate_owned_parcels, c(NA, 1))
# Masked unit sums must use the original parcel weights, independently of order.
expect_equal(summarized, ownership_summarise(panel[rev(which(panel$hex_id %in% "h1")), ], "tax_year"))

output <- ownership_hex_outputs(panel, grid)
stopifnot(nrow(output$full) == 6L, nrow(output$common) == 6L, nrow(output$change) == 3L)
stopifnot(output$pair_support$common_owner_support[output$pair_support$parcel_id == "B"])
stopifnot(!output$pair_support$common_owner_support[output$pair_support$parcel_id == "C"])
stopifnot(!"E" %in% output$pair_support$parcel_id)
common <- output$common[output$common$hex_id == "h1", ]
expect_equal(common$common_parcels, c(2, 2))
expect_equal(common$common_units, c(40, 40))
expect_equal(common$common_parcel_coverage, rep(2 / 3, 2))
expect_equal(common$common_unit_coverage, c(.4, .4))
expect_equal(common$pct_corporate_units, c(25, 0))
expect_equal(common$corporate_owned_units_per_km2, c(5, 0))
expect_equal(common$pct_financialized_owner_parcels, c(50, 0))
stopifnot(!any(common$comparison_ready))
change <- output$change[output$change$hex_id == "h1", ]
expect_equal(change$delta_pct_corporate_units, -25)
expect_equal(change$delta_corporate_owned_units_per_km2, -5)
expect_equal(change$delta_pct_financialized_owner_parcels, -50)

permissive <- ownership_hex_outputs(panel, grid, minimum_coverage = .4, minimum_units = 20)
stopifnot(permissive$change$comparison_ready[permissive$change$hex_id == "h1"])
# Complete ownership alone does not pass the minimum-units gate.
stopifnot(!permissive$change$comparison_ready[permissive$change$hex_id == "h2"])

empty <- output$full[output$full$hex_id == "empty", ]
stopifnot(!any(empty$has_residential_support), !any(empty$full_ownership_complete))
expect_equal(empty$residential_parcels, c(0, 0))
expect_equal(empty$residential_units, c(0, 0))
stopifnot(all(is.na(empty$pct_corporate_units)), all(is.na(empty$corporate_owned_units_per_km2)),
          all(is.na(empty$pct_financialized_owner_parcels)))
empty_common <- output$common[output$common$hex_id == "empty", ]
stopifnot(all(is.na(empty_common$common_unit_coverage)), !any(empty_common$comparison_ready))
stopifnot(is.na(output$change$delta_pct_corporate_units[output$change$hex_id == "empty"]))

# No jointly known parcel is valid input: preserve full bounds and an empty cohort.
unknown <- panel
unknown[c("is_owner_occupied", "has_financialized_owner", "is_corporate_owned")] <- NA
unknown$classification_status <- "source_parcel_not_found"
unknown_output <- ownership_hex_outputs(unknown, grid)
stopifnot(!any(unknown_output$pair_support$common_owner_support),
          all(unknown_output$common$common_parcels == 0),
          all(is.na(unknown_output$common$pct_corporate_units)))
known_support <- unknown_output$full$has_residential_support
expect_equal(unknown_output$full$pct_corporate_units_lower[known_support], rep(0, 4))
expect_equal(unknown_output$full$pct_corporate_units_upper[known_support], rep(100, 4))
stopifnot(all(is.na(unknown_output$full$pct_corporate_units)))

# Unmapped parcels must never enter a hex denominator, even if all are unmapped.
unmapped <- panel
unmapped$hex_id <- NA_character_
unmapped_output <- ownership_hex_outputs(unmapped, grid)
stopifnot(all(unmapped_output$full$residential_units == 0),
          all(is.na(unmapped_output$full$pct_corporate_units)),
          !any(unmapped_output$change$comparison_ready))

message("Ownership snapshot synthetic tests passed.")
