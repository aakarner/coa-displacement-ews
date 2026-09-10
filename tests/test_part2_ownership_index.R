# Synthetic regressions: no private source data, downloads, or output writes.
source("R/ownership_snapshots.R")
source("R/part2_ownership_index.R")
source("R/part2_index_scoring.R")
expect_error <- function(expr, pattern = NULL) {
  result <- tryCatch(force(expr), error = identity)
  stopifnot(inherits(result, "error"))
  if (!is.null(pattern)) stopifnot(grepl(pattern, conditionMessage(result)))
}
expect_equal <- function(actual, expected) stopifnot(isTRUE(all.equal(actual, expected, check.attributes = FALSE)))
grid <- data.frame(hex_id = seq_len(6), area_km2 = c(2, .5, 1, 1, 1, 1))
panel <- data.frame(parcel_id = rep(LETTERS[1:7], 2), tax_year = rep(c(2024L, 2025L), each = 7),
  hex_id = rep(c(1L, 1L, 2L, 3L, 4L, 4L, 5L), 2), residential_units = rep(c(50, 50, 20, 19, 20, 1, 20), 2),
  is_corporate_owned = c(FALSE, FALSE, TRUE, FALSE, FALSE, NA, NA, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),
  has_financialized_owner = c(FALSE, FALSE, TRUE, FALSE, FALSE, NA, NA, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE))
panel$classification_status <- ifelse(is.na(panel$is_corporate_owned), "source_parcel_not_found", "matched_classified")
out <- ownership_hex_outputs(panel, grid)
variants <- setNames(rep(list(out$change), 3), c("certified_only", "source_agreement", "prior_2024_certified_only"))
prepare <- function(common = out$common, full = out$full, reference = grid, sensitivities = variants) {
  part2_prepare_ownership_index(common, full, reference, sensitivities)
}
x <- prepare()
stopifnot(nrow(x) == 12L, identical(x$hex_id, rep(grid$hex_id, 2)),
  identical(x$analysis_as_of_date, rep(as.Date(c("2025-04-01", "2026-04-01")), each = 6)),
  identical(x$ownership_comparison_ready, rep(c(TRUE, TRUE, FALSE, FALSE, FALSE, FALSE), 2)),
  !any(x$ownership_certified_only_components_differ_from_main),
  !any(x$ownership_source_agreement_readiness_differs_from_main))
expect_equal(x$pct_corporate_units, c(0, 100, NA, NA, NA, NA, 50, 0, NA, NA, NA, NA))
expect_equal(x$corporate_owned_units_per_km2, c(0, 40, NA, NA, NA, NA, 25, 0, NA, NA, NA, NA))
expect_equal(x$pct_financialized_owner_parcels, c(0, 100, NA, NA, NA, NA, 50, 0, NA, NA, NA, NA))
stopifnot(x$ownership_coverage_status[3] == "common_units_below_20",
  x$ownership_coverage_status[4] == "common_coverage_below_95pct",
  x$ownership_coverage_status[5] == "no_jointly_known_parcels",
  x$ownership_coverage_status[6] == "no_residential_support",
  is.na(x$full_pct_corporate_units[4]), x$full_corporate_unknown_units[4] == 1,
  is.na(x$full_corporate_unknown_units[6]),
  x$pct_corporate_units_common_support_unmasked[3] == 0,
  is.na(x$pct_corporate_units[3]))
# Source order and incoming numeric key representation never alter canonical IDs.
shuffled <- out$common[rev(seq_len(nrow(out$common))), ]
shuffled$hex_id <- as.character(shuffled$hex_id)
expect_equal(prepare(shuffled), x)
stopifnot(is.integer(prepare(shuffled)$hex_id))

# Corporate share is unit-weighted, whereas the financialized share is parcel-
# weighted; they must not accidentally become the same denominator.
weighted_panel <- panel
weighted_panel$residential_units[weighted_panel$parcel_id == "A"] <- 90
weighted_panel$residential_units[weighted_panel$parcel_id == "B"] <- 10
weighted_output <- ownership_hex_outputs(weighted_panel, grid)
weighted_variants <- setNames(rep(list(weighted_output$change), 3), names(variants))
weighted <- part2_prepare_ownership_index(weighted_output$common, weighted_output$full, grid, weighted_variants)
expect_equal(weighted$pct_corporate_units[7], 90)
expect_equal(weighted$pct_financialized_owner_parcels[7], 50)
expect_equal(weighted$corporate_owned_units_per_km2[7], 45)

# Exactly 95% jointly observed coverage is eligible even with some full-support
# unknowns; dropping below 95% masks the entire index, not just the unknown units.
boundary_panel <- data.frame(parcel_id = rep(paste0("p", 1:20), 2),
  tax_year = rep(c(2024L, 2025L), each = 20), hex_id = 1L, residential_units = 2,
  is_corporate_owned = FALSE, has_financialized_owner = FALSE,
  classification_status = "matched_classified")
boundary_panel$is_corporate_owned[1] <- boundary_panel$has_financialized_owner[1] <- NA
boundary_panel$classification_status[1] <- "source_parcel_not_found"
boundary_grid <- grid[1, ]
boundary_output <- ownership_hex_outputs(boundary_panel, boundary_grid)
boundary_variants <- setNames(rep(list(boundary_output$change), 3), names(variants))
boundary <- part2_prepare_ownership_index(boundary_output$common, boundary_output$full,
  boundary_grid, boundary_variants)
stopifnot(all(boundary$common_parcel_coverage == .95), all(boundary$common_unit_coverage == .95),
  all(boundary$ownership_comparison_ready), boundary$full_corporate_unknown_units[1] == 2,
  boundary$pct_corporate_units[1] == 0, is.na(boundary$full_pct_corporate_units[1]))
boundary_panel$is_corporate_owned[2] <- boundary_panel$has_financialized_owner[2] <- NA
boundary_panel$classification_status[2] <- "source_parcel_not_found"
boundary_output <- ownership_hex_outputs(boundary_panel, boundary_grid)
boundary_variants <- setNames(rep(list(boundary_output$change), 3), names(variants))
boundary <- part2_prepare_ownership_index(boundary_output$common, boundary_output$full,
  boundary_grid, boundary_variants)
stopifnot(all(boundary$common_parcel_coverage == .9), !any(boundary$ownership_comparison_ready),
  all(is.na(boundary$pct_corporate_units)))

bad <- out$common; bad$pct_corporate_units <- bad$pct_corporate_units / 100
expect_error(prepare(bad), "component/denominator")
bad <- out$common; bad$corporate_owned_units_per_km2 <- bad$corporate_owned_units_per_km2 * 2
expect_error(prepare(bad), "component/denominator")
bad <- out$common; bad$pct_financialized_owner_parcels <- bad$pct_financialized_owner_parcels / 100
expect_error(prepare(bad), "component/denominator")
bad <- out$common; bad$common_unit_coverage <- bad$common_unit_coverage * 100
expect_error(prepare(bad), "coverage proportion")
bad <- out$common; bad$common_units[1] <- bad$common_units[1] + 1
expect_error(prepare(bad), "fixed common_units")
bad <- out$common; bad$comparison_ready[1] <- FALSE
expect_error(prepare(bad), "screen differs")
bad <- grid; bad$area_km2[1] <- 1
expect_error(prepare(reference = bad), "canonical common area")
expect_error(prepare(common = out$common[-1, ]), "full canonical grid")
expect_error(prepare(common = rbind(out$common, out$common[1, ])), "duplicate")
expect_error(prepare(sensitivities = variants[-1]), "sensitivity variants")
bad <- variants; bad$source_agreement$comparison_ready_2024[1] <- NA
expect_error(prepare(sensitivities = bad), "Unknown sensitivity readiness")
expect_error(part2_prepare_ownership_index(out$common, out$full, grid, variants,
  cutoffs = as.Date(c("2024-04-01", "2025-04-01"))), "tax-year/cutoff")

# Unknown required evidence masks the WHOLE index, never a two-component mean.
bad <- out$common
unknown_row <- which(bad$hex_id == 1 & bad$tax_year == 2024)
bad$corporate_unknown_units[unknown_row] <- 1
bad$corporate_unknown_parcels[unknown_row] <- 1L
bad$corporate_owned_units[unknown_row] <- NA_real_
bad$pct_corporate_units[unknown_row] <- NA_real_
bad$corporate_owned_units_per_km2[unknown_row] <- NA_real_
unknown <- prepare(bad)
stopifnot(unknown$comparison_ready[1], !unknown$ownership_comparison_ready[1],
  unknown$ownership_coverage_status[1] == "required_ownership_evidence_unknown",
  all(is.na(unknown[1, part2_ownership_components()])),
  unknown$pct_financialized_owner_parcels_common_support_unmasked[1] == 0)

bad <- variants
bad$certified_only$comparison_ready_2024[1] <- FALSE
bad$certified_only$pct_corporate_units_2025[1] <- 25
sensitive <- prepare(sensitivities = bad)
stopifnot(sensitive$ownership_certified_only_readiness_differs_from_main[1],
  sensitive$ownership_certified_only_components_differ_from_main[7],
  sensitive$pct_corporate_units[7] == 50)

earlier <- x[1:6, ]; later <- x[7:12, ]
scaling <- part2_fit_index_scaling(earlier, part2_ownership_components(), "ownership_pressure_index", as.Date("2025-04-01"))
a <- part2_apply_index_scaling(earlier, scaling)$features
b <- part2_apply_index_scaling(later, scaling)$features
stopifnot(all(scaling$bounds$reference_available_hexes == 2L), a$ownership_pressure_index[1] == 0,
  a$ownership_pressure_index[2] == 100, all(is.na(a$ownership_pressure_index[3:6])),
  identical(a$ownership_pressure_index_components_available, c(3, 3, 0, 0, 0, 0)))
change <- part2_ownership_changes(a, b)
stopifnot(identical(change$hex_id, grid$hex_id), change$ownership_change_direction[1] == "increase",
  change$ownership_change_direction[2] == "decrease", all(change$ownership_change_direction[3:6] == "unavailable"))
expect_equal(change$delta_pct_corporate_units, c(50, -100, NA, NA, NA, NA))
expect_error(part2_ownership_changes(a, b[6:1, ]), "ordered canonical grid")

p <- "R/part2_ownership_index.R"
sha <- digest::digest(file = p, algo = "sha256")
entries <- part2_ownership_manifest_entries(list(inputs = list(list(path = p, sha256 = sha)),
  nested = list(list(path = normalizePath(p), sha256 = sha))))
stopifnot(nrow(entries) == 1L, all(part2_ownership_verify_hashes(entries)$verified))
bad <- entries; bad$sha256 <- paste(rep("0", 64), collapse = "")
expect_error(part2_ownership_verify_hashes(bad), "checksum mismatch")
expect_error(part2_ownership_manifest_entries(list(list(path = p, sha256 = sha),
  list(path = normalizePath(p), sha256 = bad$sha256))), "conflicting checksums")
cat("Paired ownership index synthetic tests passed.\n")
