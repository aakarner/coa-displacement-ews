# Current 2026 measurement, rebuilt from reviewed raw-domain reconstructions.
# No downloads, new geocoding, prior-year eligibility, or model fitting.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
source("R/pipeline.R")
source("R/analysis_config.R")
source("R/acs_snapshot_scoring.R")
source("R/ownership_snapshots.R")
source("R/part2_index_scoring.R")
source("R/part2_event_scoring.R")
source("R/amenity_scoring.R")
source("R/part2_feature_matrix.R")
source("R/current_measurement.R")
cutoff <- EWS_CONFIG$analysis_as_of_date
stopifnot(identical(cutoff, as.Date("2026-04-01")))
root <- "output/part1/measurement"
paths <- c(grid = "output/hex_grid.rds", units = "output/corporate_ownership_by_hex.rds",
  counties = "config/hex_county_assignment_2024.csv",
  rent = "output/part2/acs/acs_rent_source_candidates.rds",
  demographics = "output/part2/acs/2026-04-01/acs_demographics_by_hex.rds",
  ownership = "output/part2/ownership/parcel_ownership_snapshots.rds",
  sr311 = "output/part2/311/311_features_paired.rds",
  demolitions = "output/part2/demolitions/demolition_features_paired.rds",
  evictions = "output/part2/evictions/eviction_features_paired.rds",
  amenities = "output/part2/amenities/amenity_features_paired.rds")
registry <- data.frame(domain = c("acs", "amenities", "311", "demolitions", "evictions", "ownership_index"),
  stem = c("acs", "amenity", "311", "demolition", "eviction", "ownership"),
  status = c("paired_acs_features_complete_v2", "paired_features_complete", "paired_311_features_complete_v2",
    "paired_demolition_features_complete_v2", "paired_eviction_features_complete_v2", "paired_ownership_index_complete_v2"))
pins <- bind_rows(lapply(seq_len(nrow(registry)), function(i) {
  d <- registry$domain[i]; stem <- registry$stem[i]
  part2_matrix_verify_source(file.path("output/part2", d, paste0(ifelse(d == "ownership_index", d, stem), "_run_manifest.json")),
    file.path("output/part2", d, paste0(stem, "_features_paired.rds")), registry$status[i])
}))
before <- build_file_manifest(unique(normalizePath(c(paths, pins$path), mustWork = TRUE)), require_all = TRUE, hash_files = TRUE)
grid <- readRDS(paths[["grid"]]); ids <- grid$hex_id
units <- st_drop_geometry(readRDS(paths[["units"]]))
support <- st_drop_geometry(grid)[c("hex_id", "area_km2")]
support$residential_units <- units$residential_units[match(ids, units$hex_id)]
order_rows <- function(x) {
  x <- st_drop_geometry(x)
  stopifnot(!anyDuplicated(x$hex_id), setequal(x$hex_id, ids))
  x <- as.data.frame(x[match(ids, x$hex_id), ])
  x$hex_id <- ids
  x
}
rent <- current_rent_series(readRDS(paths[["rent"]]), ids, c(2014L, 2019L, 2024L), EWS_CONFIG$acs_cpi_u)
acs <- order_rows(readRDS(paths[["demographics"]]))
for (name in setdiff(names(rent), "hex_id")) acs[[name]] <- rent[[name]]
acs$analysis_as_of_date <- cutoff
scaling <- list(acs = acs_fit_scaling(acs, cutoff, 2024L))
domains <- list(acs = acs_apply_scaling(acs, scaling$acs)$features)
own <- current_ownership_features(readRDS(paths[["ownership"]]), support)
own$analysis_as_of_date <- cutoff
scaling$ownership <- part2_fit_index_scaling(own, part2_index_components("ownership_pressure_index"), "ownership_pressure_index", cutoff)
domains$ownership <- part2_apply_index_scaling(own, scaling$ownership)$features
for (domain in c("sr311", "demolitions", "evictions")) {
  x <- readRDS(paths[[domain]]); x <- order_rows(x[x$analysis_as_of_date == cutoff, ])
  for (field in intersect(c("residential_units", "area_km2"), names(x)))
    stopifnot(isTRUE(all.equal(x[[field]], support[[field]], check.attributes = FALSE)))
  index <- c(sr311 = "sr_311_pressure_index", demolitions = "demolition_pressure_index", evictions = "eviction_pressure_index")[[domain]]
  if (domain == "evictions") {
    stopifnot("eviction_eligibility_rule" %in% names(x),
      all(x$eviction_eligibility_rule == "rolling_scored_24_months_v1"),
      all(x$eviction_eligibility_window_start == as.Date("2024-04-02")))
    scaling[[domain]] <- part2_fit_index_scaling(x, part2_index_components(index), index, cutoff)
    domains[[domain]] <- part2_apply_index_scaling(x, scaling[[domain]])$features
  } else {
    scaling[[domain]] <- part2_fit_event_scaling(x, part2_event_components(index), index, cutoff)
    domains[[domain]] <- part2_apply_event_scaling(x, scaling[[domain]])$features
  }
}
a <- readRDS(paths[["amenities"]]); a <- order_rows(a[a$amenity_analysis_as_of_date == cutoff, ])
scaling$amenities <- amenity_fit_scaling(a, cutoff)
domains$amenities <- amenity_apply_scaling(a, scaling$amenities)$features
domains$amenities$amenity_scaling_reference_as_of_date <- cutoff
domains$amenities$amenity_scaling_mode <- "current_snapshot_reference"
for (domain in names(domains)) {
  for (field in grep("scaling_reference_as_of_date$", names(domains[[domain]]), value = TRUE))
    domains[[domain]][[field]] <- cutoff
  for (field in grep("scaling_mode$", names(domains[[domain]]), value = TRUE))
    domains[[domain]][[field]] <- "fit_current_snapshot"
}
x <- support
# Promote current raw fields as well as scores: map tooltips and profiles must
# not combine corrected cluster scores with the superseded raw-event measures.
for (domain in names(domains)) {
  d <- domains[[domain]]
  stopifnot(identical(d$hex_id, ids))
  fields <- switch(domain,
    acs = names(d), ownership = names(d),
    sr311 = grep("^sr_311", names(d), value = TRUE),
    demolitions = grep("^(demo_|demolition_|hex_center_)", names(d), value = TRUE),
    evictions = grep("^eviction", names(d), value = TRUE),
    amenities = names(d))
  for (name in setdiff(fields, c("hex_id", "area_km2", "residential_units", "analysis_as_of_date"))) x[[name]] <- d[[name]]
}
for (domain in c("sr311", "demolitions", "evictions")) {
  x[[paste0(domain, "_scoring_reference")]] <- cutoff
}
indices <- part2_matrix_indices(); contract <- part2_matrix_contract()
for (domain in names(contract)) for (index in names(contract[[domain]]$scores)) {
  terms <- contract[[domain]]$scores[[index]]
  expected <- rowMeans(as.matrix(x[terms]))
  stopifnot(isTRUE(all.equal(expected, x[[index]], check.attributes = FALSE)))
  x[[paste0(index, "_terms_complete")]] <- rowSums(is.finite(as.matrix(x[terms]))) == length(terms)
  x[[paste0(index, "_terms_available")]] <- rowSums(is.finite(as.matrix(x[terms])))
  x[[paste0(index, "_terms_total")]] <- length(terms)
}
x$in_current_city_scope <- domains$sr311$sr_311_in_current_city_scope
stopifnot(identical(x$in_current_city_scope, domains$evictions$eviction_inside_current_city),
  identical(x$in_current_city_scope, domains$demolitions$hex_center_inside_current_austin_full))
x$minimum_unit_support <- is.finite(x$residential_units) & x$residential_units >= 20
x$all_required_components_available <- rowSums(is.finite(as.matrix(x[indices]))) == length(indices)
x$boundary_straddling_hex <- domains$sr311$sr_311_boundary_straddling_hex
x$source_county <- read_csv(paths[["counties"]], show_col_types = FALSE)$source_county[match(ids, read_csv(paths[["counties"]], show_col_types = FALSE)$hex_id)]
x$analysis_as_of_date <- cutoff
x$measurement_version <- "complete-components-v2"
x$measurement_scope <- "single_cutoff_current_sample"
x$component_scaling_reference_as_of_date <- cutoff
x$retrospective_reconstruction <- TRUE
x <- current_measurement_eligibility(x)
dir.create(root, recursive = TRUE, showWarnings = FALSE)
saveRDS(x, file.path(root, "current_measurement.rds"))
saveRDS(scaling, file.path(root, "current_component_scaling.rds"))
write_csv(rent, file.path(root, "rent_source_selection.csv"))
write_csv(x[c("hex_id", "source_county", "in_current_city_scope", "primary_cluster_eligible", "primary_exclusion",
  "acs_rent_source_geography", "ownership_current_usable", "ownership_unit_coverage", "ownership_parcel_coverage")],
  file.path(root, "current_eligibility.csv"))
write_csv(count(x, source_county, primary_exclusion, name = "hexes"), file.path(root, "current_exclusions.csv"))
write_csv(data.frame(index = indices, available = vapply(indices, function(i) sum(is.finite(x[[i]])), integer(1))),
  file.path(root, "current_index_readiness.csv"))
after <- build_file_manifest(before$path, require_all = TRUE, hash_files = TRUE)
stopifnot(identical(before$sha256, after$sha256))
code <- c("scripts/features/build_current_measurement.R", "R/current_measurement.R", "R/analysis_config.R",
  "R/acs_snapshot_scoring.R", "R/part2_index_scoring.R", "R/part2_event_scoring.R", "R/amenity_scoring.R",
  "R/ownership_snapshots.R", "R/part2_feature_matrix.R", "R/pipeline.R")
manifest <- list(schema_version = 1L, status = "current_measurement_complete_v2", cutoff = as.character(cutoff),
  measurement_version = "complete-components-v2", eligible_cells = sum(x$primary_cluster_eligible),
  rent = "BG reliable across current triplet, else tract across current triplet; no prior-snapshot gate",
  ownership = "Current-year jointly known parcels, units>=20 and >=95% parcel/unit coverage; no prior-year gate",
  scoring = "Same complete equal-weight recipes as Part2; component bounds estimated on current 2026 domain support; signed zero change=50",
  comparison_caution = "Part2 uses paired source support and frozen 2025 component/cluster scales; equal formula does not mean identical scores",
  source_products_unchanged = TRUE, inputs = build_file_manifest(unique(normalizePath(c(before$path, code), mustWork = TRUE)), require_all = TRUE, hash_files = TRUE),
  outputs = build_file_manifest(list.files(root, full.names = TRUE, pattern = "[.](csv|rds)$"), require_all = TRUE, hash_files = TRUE))
jsonlite::write_json(manifest, file.path(root, "current_measurement_manifest.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
print(count(x, primary_exclusion, name = "hexes"))
cat("Corrected current measurement:", sum(x$primary_cluster_eligible), "eligible cells.\n")
