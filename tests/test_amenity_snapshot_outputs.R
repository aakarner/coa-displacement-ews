# Read-only integration audit of completed amenity snapshots.
# Run: Rscript tests/test_amenity_snapshot_outputs.R
# Requires the local pinned sources and outputs from build_amenity_snapshots.R.
suppressPackageStartupMessages(library(dplyr))
source("R/amenity_scoring.R")
source("R/amenity_classification.R")

expect_equal <- function(actual, expected) {
  stopifnot(isTRUE(all.equal(actual, expected, check.attributes = FALSE)))
}
root <- "output/part2/amenities"
manifest_path <- file.path(root, "amenity_run_manifest.json")
stopifnot(file.exists(manifest_path))
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
stopifnot(identical(manifest$status, "paired_features_complete"), isTRUE(manifest$retrospective))
sha_checks <- 0L
verify_hash <- function(path, expected) {
  if (!file.exists(path)) stop("Missing manifest file: ", path)
  actual <- digest::digest(file = path, algo = "sha256")
  if (!identical(actual, expected)) stop("Manifest SHA-256 mismatch: ", path)
  sha_checks <<- sha_checks + 1L
}
verify_hash(manifest$source_manifest, manifest$source_manifest_sha256)
source_manifest <- jsonlite::read_json(manifest$source_manifest, simplifyVector = FALSE)
stopifnot(length(source_manifest$sources) == 2L)
for (source in source_manifest$sources) {
  verify_hash(source$normalized_path, source$normalized_sha256)
  verify_hash(source$raw_path, source$raw_sha256)
}
for (entry in c(manifest$inputs, manifest$outputs)) verify_hash(entry$path, entry$sha256)

grid <- readRDS("output/hex_grid.rds")
grid_ids <- as.character(grid$hex_id)
stopifnot(!anyDuplicated(grid_ids), !anyNA(grid_ids))
cutoffs <- as.Date(unlist(manifest$cutoffs))
stopifnot(identical(cutoffs, as.Date(c("2025-04-01", "2026-04-01"))))
feature_sets <- list()
event_sets <- list()
references <- list()
scores <- c(paste0("amenity_", c("cafe", "full_service_restaurant", "drinking_place"), "_score"),
            "amenity_change_index")
spatial_flags <- c("within_study_buffer", "within_hex_grid",
                   "within_access_radius_of_hex_centroid", "contributes_positive_exposure")
for (i in seq_along(cutoffs)) {
  cutoff <- cutoffs[[i]]
  directory <- file.path(root, as.character(cutoff))
  candidate <- readRDS(file.path(directory, "amenity_source_candidates.rds"))
  features <- readRDS(file.path(directory, "amenity_change_features_by_hex.rds"))
  events <- readRDS(file.path(directory, "amenity_events_all_audit.rds"))
  spatial <- readRDS(file.path(directory, "amenity_events_geocoded.rds"))
  reference <- readRDS(file.path(directory, "amenity_scaling.rds"))
  stopifnot(identical(as.character(features$hex_id), grid_ids),
            nrow(features) == nrow(grid), !anyDuplicated(features$hex_id),
            all(features$amenity_analysis_as_of_date == cutoff))
  for (column in scores) {
    stopifnot(all(is.finite(features[[column]])),
              all(features[[column]] >= 0), all(features[[column]] <= 100))
  }
  stopifnot(all(is.na(features$amenity_window_complete)),
            all(features$amenity_retrospective_usable),
            all(features$amenity_coverage_status == "retrospective_reconstructed_completeness_unverified"),
            all(features$amenity_scaling_reference_as_of_date == cutoffs[[1]]))
  expected_mode <- if (i == 1L) "fit_this_vintage" else "frozen_reference"
  stopifnot(all(features$amenity_scaling_mode == expected_mode))
  rescored <- amenity_apply_scaling(features, reference)$features
  expect_equal(as.data.frame(features)[scores], rescored[scores])

  # Candidate and final event inventories agree; retained openings must obey
  # both rolling-window and permit-date rules. Closed openings remain eligible.
  eligible <- candidate$sales_tax_locations %>% filter(core_index_eligible)
  windows <- amenity_windows(cutoff)
  stopifnot(identical(candidate$analysis_as_of_date, cutoff),
            identical(candidate$previous_window_start, windows$previous_window_start),
            identical(candidate$recent_window_start, windows$recent_window_start),
            is.na(candidate$coverage_contract$window_complete),
            isTRUE(candidate$coverage_contract$retrospective_usable),
            !anyDuplicated(events$event_id), !anyNA(events$event_id),
            setequal(eligible$event_id, events$event_id), nrow(eligible) == nrow(events),
            all(events$opening_date >= windows$previous_window_start),
            all(events$opening_date <= cutoff),
            all(is.na(events$permit_date) | events$permit_date <= cutoff),
            all(events$core_index_eligible), !any(events$duplicate_opening_alias),
            !anyDuplicated(events$opening_id),
            all(events$opening_id == events$event_id),
            all(events$event_window %in% c("previous", "recent")),
            all(is.na(events$mixed_beverage_address_match)),
            all(is.na(events$austin_food_address_match)))
  stopifnot(all(events$opening_date[events$event_window == "previous"] < windows$recent_window_start),
            all(events$opening_date[events$event_window == "recent"] >= windows$recent_window_start))
  closed_ids <- eligible$event_id[!eligible$active_as_of]
  stopifnot(all(closed_ids %in% events$event_id))

  # Unknown locations are not counted as outside the study area. Matched events
  # receive explicit spatial scope/influence flags, including matched outsiders.
  stopifnot(!anyNA(events$geocode_matched),
            identical(events$geometry_available, events$geocode_matched),
            sum(events$geocode_matched) == nrow(spatial),
            setequal(events$event_id[events$geocode_matched], spatial$event_id))
  for (column in spatial_flags) {
    stopifnot(all(is.na(events[[column]][!events$geocode_matched])),
              !anyNA(events[[column]][events$geocode_matched]))
  }
  stopifnot(all(is.na(events$direct_hex_id[!events$geocode_matched])),
            all(!events$contributes_positive_exposure[events$geocode_matched] |
                  events$within_access_radius_of_hex_centroid[events$geocode_matched]),
            all(!events$contributes_positive_exposure[events$geocode_matched] |
                  events$within_study_buffer[events$geocode_matched]))
  qa <- readr::read_csv(file.path(directory, "amenity_geocoding_qa.csv"), show_col_types = FALSE)
  stopifnot(sum(qa$opening_events) == nrow(events),
            sum(qa$geocoded_events) == sum(events$geocode_matched),
            sum(qa$unmatched_events_scope_unknown) == sum(!events$geocode_matched),
            all(is.na(qa$mixed_beverage_address_matches)),
            all(is.na(qa$austin_food_address_matches)),
            all(qa$corroboration_status == "not_rebuilt_not_used_in_index"))
  verify_hash(candidate$source_manifest_path, candidate$source_manifest_sha256)
  feature_sets[[i]] <- features
  event_sets[[i]] <- events
  references[[i]] <- reference
}

# Later scaling is the exact baseline object, not separately refitted bounds.
stopifnot(identical(references[[1]], references[[2]]))
expect_equal(references[[1]], amenity_fit_scaling(feature_sets[[1]], cutoffs[[1]]))
combined <- readRDS(file.path(root, "amenity_features_paired.rds"))
expect_equal(combined, bind_rows(feature_sets))
stopifnot(nrow(combined) == 2L * nrow(grid),
          !anyDuplicated(paste(combined$hex_id, combined$amenity_analysis_as_of_date)))

# The same event ledger supplies both dates: a continuing ID cannot change its
# event date, category, or address just because the comparison cutoff advances.
shared <- inner_join(event_sets[[1]], event_sets[[2]], by = "event_id", suffix = c("_old", "_new"))
for (column in c("opening_date", "address_key", "category_classified", "selected_source_id")) {
  expect_equal(shared[[paste0(column, "_old")]], shared[[paste0(column, "_new")]])
}
changes <- readr::read_csv(file.path(root, "amenity_feature_changes_by_hex.csv"),
  col_types = readr::cols(hex_id = readr::col_character()), show_col_types = FALSE)
expect_equal(changes$hex_id, as.character(feature_sets[[1]]$hex_id))
for (delta in names(changes)[startsWith(names(changes), "delta_")]) {
  measure <- sub("^delta_", "", delta)
  expect_equal(changes[[delta]], feature_sets[[2]][[measure]] - feature_sets[[1]][[measure]])
}
cat("Amenity snapshot integration audit passed:", nrow(grid), "hexes per cutoff;",
    paste(vapply(event_sets, nrow, integer(1)), collapse = "/"), "events;",
    sha_checks, "SHA-256 checks.\n")
