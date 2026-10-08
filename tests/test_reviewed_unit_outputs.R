# These incremental assertions describe the completed first-batch/Ben White
# transition. Replay its saved endpoint once batch 2 supersedes production;
# test_residential_property_batch2.R verifies the current production transition.
historical_endpoint <- "output/residential_property_batch2/before"
historical_output <- function(path) {
  archived <- file.path(historical_endpoint, path)
  if (!file.exists(archived)) stop("Missing preserved historical endpoint: ", path)
  archived
}
# End-to-end conservation and independently specified October 7 review outcomes.
suppressPackageStartupMessages({library(dplyr); library(sf); library(readr)})
source("R/reviewed_unit_properties.R")
root <- "output/reviewed_units_20261007"
before <- file.path(root, "before")
reviews <- read_reviewed_unit_properties(historical_output("config/residential_unit_property_reviews.json"))
p <- readRDS(historical_output("output/residential_parcels_unit_promoted.rds"))
old <- readRDS(file.path(before, "output/residential_parcels_unit_promoted.rds"))
validate_reviewed_unit_surface(p, reviews)
totals <- p %>% group_by(unit_model_project_id) %>% summarise(units = sum(units_calibrated_targeted), .groups = "drop")
expected <- c(`878332` = 400, `533185` = 330, `513751` = 452, `774341` = 412, `774412` = 26, `291453` = 170)
names(expected) <- paste0("project:", names(expected))
stopifnot(all(abs(totals$units[match(names(expected), totals$unit_model_project_id)] - expected) < 1e-7),
  nrow(p) == nrow(old) + 2L, p$units_calibrated_targeted[p$parcel_id == "533185"] == 0,
  p$units_calibrated_targeted[p$parcel_id == "737158"] == 390)
review_ids <- unlist(lapply(reviews$projects, `[[`, "parcel_ids"))
i <- !old$parcel_id %in% review_ids; j <- match(old$parcel_id[i], p$parcel_id)
stopifnot(identical(old$units_calibrated_targeted[i], p$units_calibrated_targeted[j]),
  identical(old$lon[i], p$lon[j]), identical(old$lat[i], p$lat[j]))
protected <- jsonlite::read_json(file.path(before, "protected_inputs.json"), simplifyVector = TRUE)
for (i in seq_len(nrow(protected))) stopifnot(identical(protected$sha256[i],
  digest::digest(file = protected$path[i], algo = "sha256")))

old_hex <- st_drop_geometry(readRDS(file.path(before, "output/corporate_ownership_by_hex.rds")))
new_hex <- st_drop_geometry(readRDS(historical_output("output/corporate_ownership_by_hex.rds")))
new_hex <- new_hex[match(old_hex$hex_id, new_hex$hex_id), ]
changed <- abs(new_hex$residential_units - old_hex$residential_units) > 1e-7
stopifnot(setequal(new_hex$hex_id[changed], c(3459L, 3485L, 3486L, 6260L, 6670L, 6964L, 6973L)),
  all(new_hex$residential_units[match(c(3459, 3485, 3486, 6670, 6964, 6973), new_hex$hex_id)] == c(26, 412, 0, 400, 330, 464)),
  new_hex$residential_units[new_hex$hex_id == 6260] == 170,
  abs(sum(new_hex$residential_units) - sum(old_hex$residential_units) - 243.52097864614245) < 1e-6)

a <- readRDS(historical_output("output/part2/evictions/eviction_case_ledger.rds"))
b <- readRDS(file.path(before, "output/part2/evictions/eviction_case_ledger.rds"))
b <- b[match(a$case_number, b$case_number), ]
stopifnot(identical(a$case_number, b$case_number), identical(a$assignment_status, b$assignment_status),
  identical(a$original_assigned_hex_key, b$original_assigned_hex_key),
  identical(is.na(a$assigned_hex_key), is.na(b$assigned_hex_key)))
changed <- which(!is.na(a$assigned_hex_key) & a$assigned_hex_key != b$assigned_hex_key)
stopifnot(length(changed) == 36L, all(b$property_id[changed] == "parcel:737158"),
  all(b$assigned_hex_key[changed] == "3454"),
  identical(a$assigned_hex_key[changed], a$original_assigned_hex_key[changed]),
  all(a$property_assignment_status[changed] == "unverified_property_link_original_hex_retained"),
  all(table(a$assigned_hex_key[changed])[c("3455", "3459")] == c(8L, 28L)))
# The separate reviewed-case test verifies the 71 prior manual links and annual
# parity; none is among these withdrawn automatic polygon matches.
stopifnot(all(is.na(a$property_review_id[changed]) | a$property_review_id[changed] == ""))
facility <- a %>% filter(!is.na(property_facility_type), nzchar(property_facility_type))
stopifnot(nrow(facility) == 249L,
  all(facility$property_facility_type == "probable_sro_rooming_house"),
  all(facility$property_denominator_status == "provisional_assumed_170_units"),
  sum(facility$file_date >= as.Date("2025-04-02") & facility$file_date <= as.Date("2026-04-01")) == 60L,
  identical(facility$assigned_hex_key, b$assigned_hex_key[match(facility$case_number, b$case_number)]))
ownership <- readRDS(historical_output("output/part2/ownership/parcel_ownership_snapshots.rds")) %>% filter(parcel_id == "774412")
stopifnot(nrow(ownership) == 2L, setequal(ownership$tax_year, c(2024L, 2025L)),
  all(is.na(ownership$owner_names)), all(is.na(ownership$is_corporate_owned)),
  all(ownership$classification_status == "source_snapshot_unavailable"))
m <- st_drop_geometry(readRDS(historical_output("output/part1/measurement/current_measurement.rds")))
stopifnot(sum(m$eviction_recent_observed_cases, na.rm = TRUE) == 11784L,
  sum(m$primary_cluster_eligible) == 2676L,
  m$primary_cluster_eligible[m$hex_id == 3485], !m$primary_cluster_eligible[m$hex_id == 3459],
  m$eviction_recent_observed_cases[m$hex_id == 3485] == 31L,
  sum(m$eviction_recent_observed_cases[m$residential_units < 20], na.rm = TRUE) == 246L)
cat("Reviewed housing totals, seven-cell geography changes, unchanged raw sources/exclusions, 36 conservative fallbacks, and provisional facility/unknown ownership provenance pass.\n")
