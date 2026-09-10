# Independent read-only audit of joined values and the common eligibility mask.
suppressPackageStartupMessages(library(sf))
source("R/part2_feature_matrix.R")
root <- "output/part2/matrix"
grid <- readRDS("output/hex_grid.rds")
ids <- grid$hex_id
dates <- as.Date(c("2025-04-01", "2026-04-01"))
indices <- part2_matrix_indices()
paired <- readRDS(file.path(root, "part2_features_paired.rds"))
eligibility <- readRDS(file.path(root, "part2_eligibility_by_hex.rds"))
registry <- read.csv(file.path(root, "part2_source_registry.csv"))
domains <- setNames(lapply(registry$feature_path, readRDS), registry$domain)
eq <- function(a, b) stopifnot(isTRUE(all.equal(a, b, check.attributes = FALSE, tolerance = 1e-9)))
stopifnot(nrow(paired) == 2L * length(ids), identical(paired$hex_id, rep(ids, 2L)),
  identical(paired$analysis_as_of_date, rep(dates, each = length(ids))),
  identical(eligibility$hex_id, ids), !anyDuplicated(paired[c("hex_id", "analysis_as_of_date")]),
  all(paired$retrospective_reconstruction), !any(paired$cluster_standardization_fitted))
keys <- paste(paired$hex_id, paired$analysis_as_of_date)
for (domain in names(domains)) {
  specification <- part2_matrix_contract()[[domain]]
  x <- domains[[domain]]
  x <- x[match(keys, paste(x$hex_id, x[[specification$date]])), ]
  stopifnot(!anyNA(x$hex_id))
  for (index in names(specification$scores)) {
    eq(paired[[index]], x[[index]])
    eq(paired[[paste0(index, "_terms_available")]], rowSums(!is.na(x[specification$scores[[index]]])))
  }
  domains[[domain]] <- x
}
units <- st_drop_geometry(readRDS("output/corporate_ownership_by_hex.rds"))
fixed_units <- units$residential_units[match(ids, units$hex_id)]
expected <- rep(is.finite(fixed_units) & fixed_units >= 20, 2L) &
  domains$sr311$sr_311_in_current_city_scope & domains$sr311$sr_311_poc_coverage_usable &
  domains$demolitions$demolition_comparison_ready & domains$evictions$eviction_count_observed &
  domains$ownership$ownership_comparison_ready & domains$amenities$amenity_retrospective_usable &
  rowSums(is.finite(as.matrix(paired[indices]))) == 7L
stopifnot(!anyNA(expected))
common <- expected[seq_along(ids)] & expected[length(ids) + seq_along(ids)]
eq(paired$eligible_this_snapshot, expected)
eq(eligibility$common_comparison_ready, common)
eq(paired$common_comparison_ready, rep(common, 2L))
eq(paired$residential_units, rep(fixed_units, 2L))
eq(paired$area_km2, rep(grid$area_km2, 2L))
# The overlapping 12-month interval must reconcile on the SAME cells despite
# the differing per-date ambiguity masks outside this common sample.
eviction_earlier <- domains$evictions[seq_along(ids), ]
eviction_later <- domains$evictions[length(ids) + seq_along(ids), ]
eq(eviction_earlier$eviction_cases_latest_12mo[common], eviction_later$eviction_cases_previous_12mo[common])
for (date in as.character(dates)) {
  x <- readRDS(file.path(root, paste0("part2_analysis_matrix_", date, ".rds")))
  stopifnot(identical(x$hex_id, ids[common]), all(is.finite(as.matrix(x[indices]))),
    all(as.matrix(x[indices]) >= -1e-8), all(as.matrix(x[indices]) <= 100 + 1e-8))
  eq(x, paired[paired$analysis_as_of_date == as.Date(date) & paired$common_comparison_ready,
    c("hex_id", "analysis_as_of_date", indices)])
}
summary <- read.csv(file.path(root, "part2_exclusion_summary.csv"))
stopifnot(sum(summary$primary_hexes) == length(ids),
  summary$primary_hexes[summary$reason == "included"] == sum(common),
  all(eligibility$primary_exclusion[common] == "included"))
long <- read.csv(file.path(root, "part2_exclusions_long.csv"))
for (reason in summary$reason[summary$reason != "included"]) {
  stopifnot(setequal(long$hex_id[long$reason == reason], ids[eligibility[[paste0("exclude_", reason)]]]))
}
manifest <- jsonlite::read_json(file.path(root, "part2_matrix_run_manifest.json"), simplifyVector = FALSE)
stopifnot(identical(manifest$status, "paired_seven_feature_matrix_complete_v2"),
  identical(manifest$measurement_version, "part2-fixed-components-v2"),
  identical(manifest$clustering_fitted, FALSE), identical(manifest$cluster_standardization_fitted, FALSE),
  identical(manifest$ml_work_paused, TRUE), manifest$common_eligible_hexes == sum(common))
stopifnot(all(paired$all_required_components_available[paired$common_comparison_ready]),
  all(paired$measurement_version == "part2-fixed-components-v2"))
for (index in indices) stopifnot(all(paired[[paste0(index, "_terms_available")]][paired$common_comparison_ready] ==
  paired[[paste0(index, "_terms_total")]][paired$common_comparison_ready]))
checks <- 0L
for (entry in c(manifest$inputs, manifest$outputs)) {
  stopifnot(file.exists(entry$path), identical(digest::digest(file = entry$path, algo = "sha256"), entry$sha256))
  checks <- checks + 1L
}
before <- readRDS(file.path(root, "existing_outputs_before.rds"))
source("tests/current_preservation_policy.R")
before <- current_preservation_entries(before)
for (i in seq_len(nrow(before))) stopifnot(identical(
  digest::digest(file = before$path[i], algo = "sha256"), before$sha256[i]))
cat("Paired seven-feature integration audit passed:", sum(common), "common cells;", checks,
  "matrix manifest checks;", nrow(before), "previous outputs unchanged.\n")
