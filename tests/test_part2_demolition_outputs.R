# Read-only integration audit of the completed, local paired demolition run.
suppressPackageStartupMessages({library(dplyr); library(sf); library(readr)})
source("R/part2_event_scoring.R")
root <- "output/part2/demolitions"
manifest <- jsonlite::read_json(file.path(root, "demolition_run_manifest.json"), simplifyVector = FALSE)
stopifnot(identical(manifest$status, "paired_demolition_features_complete_v2"), isTRUE(manifest$retrospective))
checks <- 0L
for (entry in c(manifest$inputs, manifest$outputs)) {
  stopifnot(file.exists(entry$path), identical(digest::digest(file = entry$path, algo = "sha256"), entry$sha256))
  checks <- checks + 1L
}
grid <- readRDS("output/hex_grid.rds")
cutoffs <- as.Date(c("2025-04-01", "2026-04-01"))
features <- list(); scaling <- list()
components <- part2_event_components("demolition_pressure_index")
for (i in seq_along(cutoffs)) {
  cutoff <- cutoffs[[i]]
  directory <- file.path(root, as.character(cutoff))
  x <- readRDS(file.path(directory, "demolition_features_by_hex.rds"))
  c <- readRDS(file.path(directory, "demolition_window_coverage.rds"))
  e <- readRDS(file.path(directory, "demolition_events_audit.rds"))
  s <- readRDS(file.path(directory, "demolition_scaling.rds"))
  stopifnot(identical(x$hex_id, grid$hex_id), identical(x$area_km2, as.numeric(grid$area_km2)),
            nrow(x) == 7027L, nrow(c) == 2L * nrow(x), !anyDuplicated(x$hex_id),
            all(x$analysis_as_of_date == cutoff), sum(x$hex_center_inside_current_austin_full) == 6060L,
            all(x$demolition_area_basis == "canonical_full_hex_area_not_city_clipped"),
            all(x$demolition_scaling_reference_as_of_date == cutoffs[[1]]),
            !anyDuplicated(e$permit_id), all(e$is_residential), all(e$is_demolition))
  stopifnot(all(is.finite(x$demolition_pressure_index[x$demolition_comparison_ready])),
            all(is.na(x$demolition_pressure_index[!x$demolition_comparison_ready])),
            all(x$demolition_pressure_index >= 0 & x$demolition_pressure_index <= 100, na.rm = TRUE),
            all(x$demolition_pressure_index_components_available[x$demolition_comparison_ready] == 3L),
            all(x$demolition_pressure_index_components_available[!x$demolition_comparison_ready] == 0L))
  for (component in components) stopifnot(all(is.na(x[[component]][!x$demolition_comparison_ready])))
  rescored <- part2_apply_event_scaling(x, s)$features
  stopifnot(isTRUE(all.equal(rescored$demolition_pressure_index, x$demolition_pressure_index)))
  for (window in c("previous", "recent")) {
    coverage <- c[c$event_window == window, ]
    stopifnot(!anyDuplicated(coverage$hex_id), setequal(coverage$hex_id, as.character(grid$hex_id)),
              all(coverage$source_period_complete))
    events <- e[e$event_window == window, ]
    stopifnot(all(events$issue_date >= coverage$window_start[1]),
              all(events$issue_date <= coverage$window_end[1]), all(events$issue_date <= cutoff))
    accepted <- events %>% filter(spatial_status == "mapped_current_city_study_hex")
    stopifnot(all(accepted$point_inside_current_austin_full), all(accepted$inside_city_study_hex))
    counts <- accepted %>% count(hex_id, name = "observed")
    expected <- counts$observed[match(as.character(x$hex_id), counts$hex_id)]
    expected[is.na(expected)] <- 0L
    stopifnot(identical(x[[paste0("demo_", window, "_observed_permits")]], expected))
    availability <- x[[paste0("demo_", window, "_count_available")]]
    count_values <- x[[paste0("demo_", window, "_permits")]]
    stopifnot(all(is.na(count_values[!availability])), all(!is.na(count_values[availability])),
              all(count_values[availability] == expected[availability]))
  }
  invalid <- !e$valid_coordinates
  stopifnot(all(is.na(e$point_inside_current_austin_full[invalid])), all(is.na(e$hex_id[invalid])))
  features[[i]] <- x; scaling[[i]] <- s
}
stopifnot(identical(scaling[[1]], scaling[[2]]))
expected_scaling <- part2_fit_event_scaling(features[[1]], components, "demolition_pressure_index", cutoffs[[1]], preserved_scaling = scaling[[1]])
stopifnot(identical(scaling[[1]], expected_scaling))
paired <- readRDS(file.path(root, "demolition_features_paired.rds"))
stopifnot(identical(paired, bind_rows(features)),
          !anyDuplicated(paste(paired$hex_id, paired$analysis_as_of_date)))
changes <- read_csv(file.path(root, "demolition_feature_changes_by_hex.csv"), show_col_types = FALSE)
stopifnot(all(changes$hex_id == grid$hex_id))
for (column in names(changes)[startsWith(names(changes), "delta_")]) {
  measure <- sub("^delta_", "", column)
  expected <- ifelse(changes$demolition_change_available, features[[2]][[measure]] - features[[1]][[measure]], NA_real_)
  stopifnot(isTRUE(all.equal(changes[[column]], expected)))
}
qa <- read_csv(file.path(root, "demolition_window_event_qa.csv"), show_col_types = FALSE)
stopifnot(all(qa$source_unique_permits == qa$invalid_coordinate_permits +
  qa$excluded_outside_exact_current_city + qa$excluded_inside_city_outside_center_selected_hex +
  qa$outside_grid_permits + qa$ambiguous_permits + qa$mapped_current_city_study_permits))
stopifnot(all(qa$single_hex_any_city_status_permits == qa$excluded_grid_points_outside_exact_current_city +
  qa$excluded_inside_city_outside_center_selected_hex + qa$mapped_current_city_study_permits))
cat("Paired demolition integration audit passed:", nrow(grid), "hexes/date;",
    paste(vapply(features, function(x) sum(x$demolition_comparison_ready), integer(1)), collapse = "/"),
    "coverage-ready;", checks, "SHA-256 checks.\n")
