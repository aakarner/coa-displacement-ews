# Read-only integration audit. Requires build_acs_snapshots.R outputs.
suppressPackageStartupMessages({library(dplyr); library(sf)})
source("R/acs_snapshot_scoring.R")
source("R/analysis_config.R")
root <- "output/part2/acs"
eq <- function(x, y) stopifnot(isTRUE(all.equal(x, y, check.attributes = FALSE, tolerance = 1e-9)))
manifest <- jsonlite::read_json(file.path(root, "acs_run_manifest.json"), simplifyVector = FALSE)
stopifnot(identical(manifest$status, "paired_acs_features_complete_v2"), manifest$part1_outputs_unchanged,
  manifest$raw_caches_and_crosswalks_unchanged)
hash_checks <- 0L
for (entry in c(manifest$inputs, manifest$outputs)) {
  stopifnot(file.exists(entry$path), identical(digest::digest(file = entry$path, algo = "sha256"), entry$sha256))
  hash_checks <- hash_checks + 1L
}
before <- read.csv(file.path(root, "part1_preservation_before.csv"))
after <- read.csv(file.path(root, "part1_preservation_after.csv"))
eq(before[c("path", "sha256")], after[c("path", "sha256")])
if (file.exists(file.path(root, "part1_inputs_before.rds"))) {
  first <- readRDS(file.path(root, "part1_inputs_before.rds"))
  for (i in seq_len(nrow(first))) stopifnot(identical(
    digest::digest(file = first$path[i], algo = "sha256"), first$sha256[i]))
}
grid_ids <- as.character(readRDS("output/hex_grid.rds")$hex_id)
reference <- readRDS(file.path(root, "acs_scaling.rds"))
stopifnot(identical(reference$schema_version, "acs-scaling-v2"),
  identical(reference$missing_policy, "all_required_components_fixed_equal_weights"),
  identical(digest::digest(reference$bounds, algo = "sha256"),
    "045e6d84ddc67880dee5fe7528c2279cdbde35adfc1f33d1866c539c7c96df94"),
  identical(reference$preserved_bounds_sha256, manifest$preserved_normalization_bounds_sha256))
sets <- list()
for (i in 1:2) {
  cutoff <- as.Date(c("2025-04-01", "2026-04-01")[i])
  current <- c(2023L, 2024L)[i]
  years <- current - c(10L, 5L, 0L)
  directory <- file.path(root, as.character(cutoff))
  f <- readRDS(file.path(directory, "acs_features_by_hex.rds"))
  demo <- st_drop_geometry(readRDS(file.path(directory, "acs_demographics_by_hex.rds")))
  vintage <- st_drop_geometry(readRDS(file.path(directory, "acs_rent_by_hex_vintage.rds")))
  stopifnot(all(c("hex_id", "analysis_as_of_date", "acs_dollar_base_year",
    "acs_current_year", "acs_release_date", "acs_period_start", "acs_period_end",
    "acs_rent_components_available", "acs_vulnerability_components_available",
    "acs_rent_recent_interval_years", "acs_rent_prior_interval_years",
    "acs_scaling_reference_as_of_date", "rent_pressure_citywide_index",
    "demographic_vulnerability_index") %in% names(f)),
    all(c("acs_preserve_missing", "median_income_real", "median_income_moe_real") %in% names(demo)))
  stopifnot(identical(as.character(f$hex_id), grid_ids), !anyDuplicated(f$hex_id),
    all(f$analysis_as_of_date == cutoff), all(f$acs_dollar_base_year == 2024L),
    all(f$acs_current_year == current), all(f$acs_release_date < cutoff),
    all(f$acs_period_start == current - 4L), all(f$acs_period_end == current),
    all(demo$acs_preserve_missing), all(f$acs_rent_recent_interval_years == 5L),
    all(f$acs_rent_prior_interval_years == 5L), setequal(unique(vintage$acs_year), years),
    all(f$acs_scaling_reference_as_of_date == as.Date("2025-04-01")))
  cpi <- EWS_CONFIG$acs_cpi_u
  eq(demo$median_income_real, demo$median_income * cpi[["2024"]] / cpi[[as.character(current)]])
  eq(demo$median_income_moe_real, demo$median_income_moe * cpi[["2024"]] / cpi[[as.character(current)]])
  eq(vintage$median_rent_real, vintage$median_rent * cpi[["2024"]] / cpi[as.character(vintage$acs_year)])
  by_year <- lapply(years, function(year) {
    x <- vintage[vintage$acs_year == year, ]
    stopifnot(nrow(x) == length(grid_ids), !anyDuplicated(x$hex_id))
    x[match(grid_ids, x$hex_id), ]
  })
  growth <- function(new, old) {
    ok <- is.finite(new) & new > 0 & is.finite(old) & old > 0
    result <- rep(NA_real_, length(new))
    result[ok] <- 100 * (log(new[ok]) - log(old[ok])) / 5
    result
  }
  recent <- growth(by_year[[3]]$median_rent_real, by_year[[2]]$median_rent_real)
  previous <- growth(by_year[[2]]$median_rent_real, by_year[[1]]$median_rent_real)
  eq(f$acs_rent_growth_recent_annualized_pct, recent)
  eq(f$acs_rent_acceleration_pp, recent - previous)
  expected_reliable <- Reduce(`&`, lapply(by_year, function(x)
    is.finite(x$median_rent) & x$median_rent > 0 &
    is.finite(x$median_rent_relative_moe) & x$median_rent_relative_moe >= 0 &
    x$median_rent_relative_moe <= .30))
  eq(f$acs_rent_trend_reliable, expected_reliable)
  rescored <- acs_apply_scaling(f, reference)$features
  scores <- c(paste0("acs_score_", acs_component_spec()$component),
              "rent_pressure_citywide_index", "demographic_vulnerability_index")
  eq(f[scores], rescored[scores])
  for (column in scores) stopifnot(all(is.na(f[[column]]) |
    (is.finite(f[[column]]) & f[[column]] >= -1e-10 & f[[column]] <= 100 + 1e-10)))
  stopifnot(all(is.na(f$acs_score_rent_growth[!expected_reliable])),
    all(is.na(f$acs_score_rent_acceleration[!expected_reliable])),
    all(is.na(f$acs_score_rent_level[!expected_reliable])),
    all(is.na(f$demographic_vulnerability_index[f$acs_vulnerability_components_available < 5L])),
    all(is.na(f$rent_pressure_citywide_index[f$acs_rent_components_available < 3L])),
    all(f$acs_rent_components_available %in% c(0L, 3L)),
    all(f$acs_rent_components_required == 3L), all(f$acs_vulnerability_components_required == 5L),
    identical(f$acs_rent_complete, f$acs_rent_components_available == 3L),
    identical(f$acs_vulnerability_complete, f$acs_vulnerability_components_available == 5L),
    all(is.finite(f$rent_pressure_citywide_index) == f$acs_rent_complete),
    all(is.finite(f$demographic_vulnerability_index) == f$acs_vulnerability_complete))
  eq(f$rent_pressure_citywide_index, rowMeans(as.matrix(f[paste0("acs_score_", c("rent_level", "rent_growth", "rent_acceleration"))])))
  eq(f$demographic_vulnerability_index, rowMeans(as.matrix(f[paste0("acs_score_", c("low_income", "renters", "poverty", "rent_burden", "low_college"))])))
  for (column in c("pct_renter", "pct_college", "poverty_rate", "pct_rent_burden_30plus")) {
    stopifnot(all(is.na(demo[[column]]) | (demo[[column]] >= 0 & demo[[column]] <= 100 + 1e-8)))
  }
  qa <- read.csv(file.path(directory, "acs_dasymetric_allocation_qa.csv"))
  stopifnot(sum(qa$qa_group == "count_conservation") == 24L,
            all(abs(qa$value[qa$qa_group == "count_conservation"]) < 1e-6))
  sets[[i]] <- f
}
eq(sets[[1]]$acs_rent_source_geography, sets[[2]]$acs_rent_source_geography)
eq(sets[[1]]$acs_rent_series_supported, sets[[2]]$acs_rent_series_supported)
eq(readRDS(file.path(root, "2025-04-01", "acs_dasymetric_block_hex_allocation.rds")),
   readRDS(file.path(root, "2026-04-01", "acs_dasymetric_block_hex_allocation.rds")))
eq(readRDS(file.path(root, "acs_features_paired.rds")), bind_rows(sets))
changes <- read.csv(file.path(root, "acs_feature_changes_by_hex.csv"))
eq(as.character(changes$hex_id), grid_ids)
for (index in c("rent_pressure_citywide_index", "demographic_vulnerability_index")) {
  eq(changes[[paste0("delta_", index)]], sets[[2]][[index]] - sets[[1]][[index]])
}
a <- lapply(sets, function(x) !is.na(acs_component_inputs(x)))
eq(changes$acs_same_component_availability, rowSums(a[[1]] != a[[2]]) == 0L)
eq(changes$acs_all_eight_components_both, rowSums(a[[1]] & a[[2]]) == 8L)
cat("ACS snapshot output audit passed:", hash_checks, "SHA-256 checks;", length(grid_ids), "hexes per date.\n")
