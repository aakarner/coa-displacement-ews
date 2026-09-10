# Read-only integration audit of the completed paired ownership index artifacts.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
source("R/part2_ownership_index.R")
source("R/part2_index_scoring.R")
root <- "output/part2/ownership_index"
manifest <- jsonlite::read_json(file.path(root, "ownership_index_run_manifest.json"))
stopifnot(identical(manifest$status, "paired_ownership_index_complete_v2"), isTRUE(manifest$retrospective),
  identical(manifest$source_variant, "main"), isTRUE(manifest$no_cluster_or_ml_changes))
entries <- part2_ownership_manifest_entries(manifest)
stopifnot(all(part2_ownership_verify_hashes(entries)$verified))
source_preservation <- read_csv(file.path(root, "ownership_source_directory_preservation.csv"), show_col_types = FALSE)
stopifnot(all(part2_ownership_verify_hashes(source_preservation)$verified))
grid <- readRDS("output/hex_grid.rds")
common <- readRDS("output/part2/ownership/ownership_common_support_by_hex_year.rds")
full <- readRDS("output/part2/ownership/ownership_features_by_hex_year.rds")
cutoffs <- as.Date(c("2025-04-01", "2026-04-01")); years <- c(2024L, 2025L)
components <- part2_ownership_components(); features <- list(); bounds <- list()
for (i in seq_along(cutoffs)) {
  x <- readRDS(file.path(root, as.character(cutoffs[[i]]), "ownership_features_by_hex.rds"))
  scaling <- readRDS(file.path(root, as.character(cutoffs[[i]]), "ownership_scaling.rds"))
  c <- common[common$tax_year == years[[i]], ]; c <- c[match(grid$hex_id, c$hex_id), ]
  f <- full[full$tax_year == years[[i]], ]; f <- f[match(grid$hex_id, f$hex_id), ]
  ready <- x$ownership_comparison_ready
  stopifnot(nrow(x) == 7027L, identical(x$hex_id, grid$hex_id), identical(x$area_km2, as.numeric(grid$area_km2)),
    all(x$analysis_as_of_date == cutoffs[[i]]), all(x$tax_year == years[[i]]), sum(ready) == 3227L,
    identical(ready, c$comparison_ready), all(is.finite(x$ownership_pressure_index[ready])),
    all(is.na(x$ownership_pressure_index[!ready])),
    all(x$ownership_pressure_index_components_available == ifelse(ready, 3L, 0L)),
    all(x$ownership_pressure_index >= 0 & x$ownership_pressure_index <= 100, na.rm = TRUE),
    all(x$ownership_common_support_provisional),
    all(x$ownership_scaling_reference_as_of_date == cutoffs[[1]]),
    all(x$ownership_source_variant == "main"),
    all(x$ownership_area_basis == "canonical_full_hex_area_not_common_parcel_area"))
  for (component in components) {
    part2_ownership_assert_numeric(x[[paste0(component, "_common_support_unmasked")]], c[[component]], component)
    expected <- c[[component]]; expected[!ready] <- NA_real_
    part2_ownership_assert_numeric(x[[component]], expected, component)
  }
  for (field in c("common_parcels", "common_units", "common_parcel_coverage", "common_unit_coverage")) {
    part2_ownership_assert_numeric(x[[field]], c[[field]], field)
  }
  for (field in setdiff(names(f), c("hex_id", "tax_year", "area_km2"))) {
    stopifnot(isTRUE(all.equal(x[[paste0("full_", field)]], f[[field]])))
  }
  recomputed <- part2_apply_index_scaling(x, scaling)$features
  stopifnot(identical(recomputed$ownership_pressure_index, x$ownership_pressure_index))
  features[[i]] <- x; bounds[[i]] <- scaling
}
stopifnot(identical(bounds[[1]], bounds[[2]]),
  identical(bounds[[1]], part2_fit_index_scaling(features[[1]], components,
    "ownership_pressure_index", cutoffs[[1]], preserved_scaling = bounds[[1]])), all(bounds[[1]]$bounds$reference_available_hexes == 3227L),
  identical(features[[1]]$ownership_comparison_ready, features[[2]]$ownership_comparison_ready),
  identical(features[[1]]$common_units, features[[2]]$common_units),
  identical(features[[1]]$common_parcels, features[[2]]$common_parcels))
paired <- readRDS(file.path(root, "ownership_features_paired.rds"))
changes <- readRDS(file.path(root, "ownership_feature_changes_by_hex.rds"))
stopifnot(identical(paired, bind_rows(features)), nrow(paired) == 14054L,
  !anyDuplicated(paired[c("hex_id", "analysis_as_of_date")]),
  identical(changes, part2_ownership_changes(features[[1]], features[[2]])),
  sum(changes$ownership_change_available) == 3227L)
for (variant in c("certified_only", "source_agreement", "prior_2024_certified_only")) {
  filename <- switch(variant, certified_only = "ownership_certified_only_hex_change.csv",
    source_agreement = "ownership_source_agreement_hex_change.csv",
    prior_2024_certified_only = "ownership_2024_certified_only_hex_change.csv")
  source <- read_csv(file.path("output/part2/ownership", filename), show_col_types = FALSE)
  source <- source[match(grid$hex_id, source$hex_id), ]
  for (i in seq_along(years)) stopifnot(identical(
    features[[i]][[paste0("ownership_", variant, "_comparison_ready")]], source[[paste0("comparison_ready_", years[[i]])]]))
}
cat("Paired ownership index output audit passed: 7,027 hexes/date; 3,227 score-ready;",
  nrow(entries), "pinned source/code/output checks and", nrow(source_preservation), "preserved source-directory files.\n")
