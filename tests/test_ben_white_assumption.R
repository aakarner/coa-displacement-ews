# These incremental assertions describe the completed first-batch/Ben White
# transition. Replay its saved endpoint once batch 2 supersedes production;
# test_residential_property_batch2.R verifies the current production transition.
historical_endpoint <- "output/residential_property_batch2/before"
historical_output <- function(path) {
  archived <- file.path(historical_endpoint, path)
  if (!file.exists(archived)) stop("Missing preserved historical endpoint: ", path)
  archived
}
# Focused check of the incremental user-approved 170-unit assumption.
suppressPackageStartupMessages({library(dplyr); library(sf)})
source("R/reviewed_unit_properties.R")
before <- "output/ben_white_170_20261007/before"
p <- readRDS(historical_output("output/residential_parcels_unit_promoted.rds"))
old <- readRDS(file.path(before, "output/residential_parcels_unit_promoted.rds"))
stopifnot(nrow(p) == nrow(old) + 1L, !anyDuplicated(p$parcel_id),
  setequal(setdiff(p$parcel_id, old$parcel_id), "291453"))
j <- match(old$parcel_id, p$parcel_id)
stopifnot(identical(old$units_calibrated_targeted, p$units_calibrated_targeted[j]),
  identical(old$lon, p$lon[j]), identical(old$lat, p$lat[j]))
b <- p[p$parcel_id == "291453", ]
stopifnot(b$units_raw == 0, b$propertyProf_imprvStateCd == "F1",
  b$units_calibrated_targeted == 170, b$unit_review_count_status == "provisional_assumption",
  b$unit_model_selection_method == "reviewed_assumed_project_total",
  b$unit_estimation_confidence_targeted == "low", !b$unit_model_used)
validate_reviewed_unit_surface(p, read_reviewed_unit_properties(historical_output("config/residential_unit_property_reviews.json")))
bad <- p; bad$unit_estimation_confidence_targeted[bad$parcel_id == "291453"] <- "high"
err <- tryCatch({validate_reviewed_unit_surface(bad, read_reviewed_unit_properties(historical_output("config/residential_unit_property_reviews.json"))); NULL}, error = identity)
stopifnot(inherits(err, "error"))
record <- jsonlite::read_json("config/ben_white_170_assumption.json")
stopifnot(record$adopted_units == 170L, record$count_status == "provisional_assumption",
  record$sources[[1]]$reported_quantity == 178L,
  record$sources[[2]]$reported_quantity == 100L,
  all(vapply(record$sources, function(s) grepl("^https://", s$url), logical(1))))
u <- st_drop_geometry(readRDS(historical_output("output/corporate_ownership_by_hex.rds")))
v <- st_drop_geometry(readRDS(file.path(before, "output/corporate_ownership_by_hex.rds")))
delta <- u$residential_units - v$residential_units[match(u$hex_id, v$hex_id)]
stopifnot(identical(u$hex_id[abs(delta) > 1e-7], 6260L), abs(sum(delta) - 170) < 1e-7)
a <- readRDS(historical_output("output/part2/evictions/eviction_case_ledger.rds"))
old_a <- readRDS(file.path(before, "output/part2/evictions/eviction_case_ledger.rds"))
old_a <- old_a[match(a$case_number, old_a$case_number), ]
stopifnot(identical(a$case_number, old_a$case_number),
  identical(a$assignment_status, old_a$assignment_status),
  identical(a$assigned_hex_key, old_a$assigned_hex_key))
f <- a %>% filter(property_denominator_status == "provisional_assumed_170_units")
stopifnot(nrow(f) == 249L, all(f$assigned_hex_key == "6260"), all(f$property_id == "parcel:291453"))
x <- readRDS(historical_output("output/part2/evictions/eviction_features_paired.rds")) %>%
  filter(hex_id == 6260, analysis_as_of_date == as.Date("2026-04-01"))
stopifnot(nrow(x) == 1L, x$residential_units == 170, x$eviction_recent_observed_cases == 60L,
  abs(x$eviction_latest_12mo_per_100_units - 60/170*100) < 1e-9)
ownership <- readRDS(historical_output("output/part2/ownership/parcel_ownership_snapshots.rds")) %>%
  filter(parcel_id == "291453")
stopifnot(nrow(ownership) == 2L, setequal(ownership$tax_year, c(2024L, 2025L)),
  all(is.na(ownership$owner_names)), all(is.na(ownership$is_corporate_owned)),
  all(ownership$classification_status == "source_snapshot_unavailable"))
cat("Ben White adds exactly 170 provisional units in one cell, preserves every case assignment, retains source discrepancies and produces the expected filing rate.\n")
