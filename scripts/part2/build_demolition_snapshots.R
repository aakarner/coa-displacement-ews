# Build only paired retrospective demolition features from already-local data.
# No network calls and no writes outside output/part2/demolitions/.
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(sf)
  library(readr)
})
source("R/pipeline.R")
source("R/demolition_panel.R")
source("R/demolition_coverage_history.R")
source("R/part2_demolitions.R")
source("R/part2_event_scoring.R")

output_root <- "output/part2/demolitions"
# Load before any overwrite; v1 and v2 both preserve unchanged baseline bounds.
preserved_scaling_path <- file.path(output_root, "2025-04-01", "demolition_scaling.rds")
preserved_scaling <- if (file.exists(preserved_scaling_path)) readRDS(preserved_scaling_path) else NULL
raw_path <- "data/Issued_Construction_Permits_20260401.csv"
grid_path <- "output/hex_grid.rds"
boundary_path <- "data/BOUNDARIES_jurisdictions_20260429.geojson"
coverage_config_path <- "config/demolition_jurisdiction_sources.csv"
source_start <- as.Date("2009-10-02")
source_end <- as.Date("2026-04-01")
raw_expected_sha256 <- "841940499a2866279d5dbf092f97914255e15240f68b76cec297ea72436afd70"
cutoffs <- as.Date(c("2025-04-01", "2026-04-01"))
components <- c("demo_recent_density", "demo_trend_positive", "demo_total_recent_density")
index_name <- "demolition_pressure_index"
required <- c(raw_path, grid_path, boundary_path, coverage_config_path)
if (!all(file.exists(required))) stop("Missing local demolition input: ", paste(required[!file.exists(required)], collapse = ", "))
if (!identical(digest::digest(file = raw_path, algo = "sha256"), raw_expected_sha256)) {
  stop("Demolition source checksum differs from the reviewed April 2026 extract.", call. = FALSE)
}
coverage_config <- read_csv(coverage_config_path, show_col_types = FALSE) %>% filter(status == "active")
if (nrow(coverage_config) != 2L || !setequal(coverage_config$coverage_role, c("historical_baseline", "dated_actions"))) {
  stop("Expected exactly one active historical baseline and one dated-action source.")
}
for (row in seq_len(nrow(coverage_config))) {
  if (!file.exists(coverage_config$local_path[row]) || !identical(
    digest::digest(file = coverage_config$local_path[row], algo = "sha256"), coverage_config$sha256[row])) {
    stop("Historical jurisdiction source checksum mismatch.")
  }
}
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
raw <- read_csv(raw_path, col_types = cols(.default = col_character()), show_col_types = FALSE)
hex_grid <- readRDS(grid_path)
if (!inherits(hex_grid, "sf") || nrow(hex_grid) != 7027L || anyNA(hex_grid$hex_id) || anyDuplicated(hex_grid$hex_id)) {
  stop("The canonical 7,027-cell demolition grid is invalid.")
}
city <- part2_demolition_city_reference(hex_grid, st_read(boundary_path, quiet = TRUE))
if (sum(city$reference$hex_center_inside_current_austin_full) != 6060L) {
  stop("The fixed current-FULL study geography no longer contains 6,060 cells.")
}
cat("Preparing unique issued residential demolition permits...\n")
prepared <- part2_prepare_demolitions(raw, source_end)
prepared$source_qa <- bind_rows(prepared$source_qa, tibble::tibble(
  metric = "eligible_permits_missing_description",
  value = sum(is.na(prepared$events$Description) | !nzchar(trimws(prepared$events$Description)))))
write_csv(prepared$source_audit, file.path(output_root, "demolition_source_row_audit.csv"))
write_csv(prepared$source_qa, file.path(output_root, "demolition_source_qa.csv"))
write_csv(prepared$duplicate_groups, file.path(output_root, "demolition_duplicate_permit_qa.csv"))
write_csv(prepared$classification_disagreements, file.path(output_root, "demolition_legacy_classification_differences.csv"))
write_csv(city$reference, file.path(output_root, "demolition_fixed_city_hex_reference.csv"))
assigned <- part2_assign_demolitions(prepared$events, hex_grid, city$reference, city$boundary)
saveRDS(assigned$events, file.path(output_root, "demolition_event_locations.rds"))
write_csv(assigned$events, file.path(output_root, "demolition_event_locations.csv"))
write_csv(assigned$ambiguous_hex_links, file.path(output_root, "demolition_ambiguous_hex_links.csv"))

cat("Preparing effective-dated historical jurisdiction evidence...\n")
context <- part2_demolition_coverage_context(hex_grid,
  st_read(coverage_config$local_path[coverage_config$coverage_role == "historical_baseline"], quiet = TRUE),
  st_read(coverage_config$local_path[coverage_config$coverage_role == "dated_actions"], quiet = TRUE))
features_by_date <- list()
coverage_by_date <- list()
summary_by_date <- list()
window_qa <- list()
scaling <- NULL
for (i in seq_along(cutoffs)) {
  cutoff <- cutoffs[[i]]
  date_dir <- file.path(output_root, as.character(cutoff))
  dir.create(date_dir, recursive = TRUE, showWarnings = FALSE)
  cat("Replaying exact 24-month windows for ", as.character(cutoff), "...\n", sep = "")
  windows <- part2_demolition_windows(cutoff)
  coverage <- part2_demolition_window_coverage(hex_grid, context, windows, city$reference, source_start, source_end)
  features <- part2_demolition_features(hex_grid, assigned, coverage, windows)
  features$hex_id <- hex_grid$hex_id[match(features$hex_id, as.character(hex_grid$hex_id))]
  if (is.null(scaling)) scaling <- part2_fit_event_scaling(features, components, index_name, cutoffs[[1]], preserved_scaling = preserved_scaling)
  scored <- part2_apply_event_scaling(features, scaling)
  features <- scored$features
  features$demolition_scaling_reference_as_of_date <- cutoffs[[1]]
  features$demolition_scaling_mode <- if (i == 1L && !is.null(preserved_scaling)) "preserved_earlier_bounds" else if (i == 1L) "fit_this_vintage" else "frozen_reference"
  if (any(!is.na(features[[index_name]][!features$demolition_comparison_ready])) ||
      any(!is.finite(features[[index_name]][features$demolition_comparison_ready]))) {
    stop("Demolition score does not respect the full-window coverage mask.")
  }
  events <- assigned$events %>% mutate(analysis_as_of_date = cutoff,
    event_window = case_when(issue_date >= windows$window_start[2] & issue_date <= cutoff ~ "recent",
      issue_date >= windows$window_start[1] & issue_date <= windows$window_end[1] ~ "previous",
      TRUE ~ "outside"))
  saveRDS(events, file.path(date_dir, "demolition_events_audit.rds"))
  write_csv(events, file.path(date_dir, "demolition_events_audit.csv"))
  saveRDS(coverage, file.path(date_dir, "demolition_window_coverage.rds"))
  write_csv(coverage, file.path(date_dir, "demolition_window_coverage.csv"))
  saveRDS(features, file.path(date_dir, "demolition_features_by_hex.rds"))
  write_csv(features, file.path(date_dir, "demolition_features_by_hex.csv"))
  saveRDS(scaling, file.path(date_dir, "demolition_scaling.rds"))
  write_csv(scaling$bounds, file.path(date_dir, "demolition_scaling_bounds.csv"))
  write_csv(scored$qa, file.path(date_dir, "demolition_scaling_qa.csv"))
  period_qa <- events %>% filter(event_window != "outside") %>%
    group_by(analysis_as_of_date, event_window) %>% summarise(
      source_unique_permits = n(),
      source_total_demolition_permits = sum(is_total_demolition),
      source_permits_missing_description = sum(is.na(Description) | !nzchar(trimws(Description))),
      invalid_coordinate_permits = sum(!valid_coordinates),
      single_hex_any_city_status_permits = sum(!is.na(hex_id)),
      excluded_outside_exact_current_city = sum(spatial_status == "outside_current_austin_full_purpose"),
      excluded_grid_points_outside_exact_current_city = sum(!is.na(hex_id) & spatial_status == "outside_current_austin_full_purpose"),
      excluded_inside_city_outside_center_selected_hex = sum(spatial_status == "inside_city_point_outside_center_selected_hex"),
      outside_grid_permits = sum(spatial_status == "outside_grid"),
      ambiguous_permits = sum(spatial_status == "multiple_hex_matches"),
      mapped_current_city_study_permits = sum(spatial_status == "mapped_current_city_study_hex"),
      mapped_unsupported_window_permits = sum(spatial_status == "mapped_current_city_study_hex" &
        !hex_id %in% coverage$hex_id[coverage$event_window == first(event_window) & coverage$window_usable]),
      .groups = "drop")
  write_csv(period_qa, file.path(date_dir, "demolition_window_event_qa.csv"))
  write_csv(coverage %>% count(event_window, coverage_reason, historical_source_covered, window_usable, name = "hexes"),
    file.path(date_dir, "demolition_window_coverage_qa.csv"))
  summary_by_date[[i]] <- tibble::tibble(analysis_as_of_date = cutoff, hexes = nrow(features),
    fixed_city_hexes = sum(features$hex_center_inside_current_austin_full),
    straddling_city_study_hexes = sum(features$hex_center_inside_current_austin_full & features$hex_straddles_current_austin_full),
    comparison_ready_hexes = sum(features$demolition_comparison_ready),
    covered_zero_recent_hexes = sum(features$demo_latest_24mo == 0, na.rm = TRUE),
    recent_available_permits = sum(features$demo_latest_24mo, na.rm = TRUE),
    previous_available_permits = sum(features$demo_previous_24mo, na.rm = TRUE),
    score_min = min(features[[index_name]], na.rm = TRUE), score_max = max(features[[index_name]], na.rm = TRUE))
  features_by_date[[i]] <- features
  coverage_by_date[[i]] <- coverage
  window_qa[[i]] <- period_qa
}
paired <- bind_rows(features_by_date)
saveRDS(paired, file.path(output_root, "demolition_features_paired.rds"))
write_csv(paired, file.path(output_root, "demolition_features_paired.csv"))
write_csv(bind_rows(coverage_by_date), file.path(output_root, "demolition_window_coverage_paired.csv"))
write_csv(bind_rows(summary_by_date), file.path(output_root, "demolition_snapshot_summary.csv"))
write_csv(bind_rows(window_qa), file.path(output_root, "demolition_window_event_qa.csv"))
measures <- c(components, index_name, "demo_latest_24mo", "demo_previous_24mo", "demo_total_latest_24mo", "demo_total_previous_24mo")
changes <- inner_join(select(features_by_date[[1]], hex_id, demolition_comparison_ready, all_of(measures)),
  select(features_by_date[[2]], hex_id, demolition_comparison_ready, all_of(measures)),
  by = "hex_id", suffix = c("_2025", "_2026"))
changes$demolition_change_available <- changes$demolition_comparison_ready_2025 & changes$demolition_comparison_ready_2026
for (measure in measures) changes[[paste0("delta_", measure)]] <- ifelse(changes$demolition_change_available,
  changes[[paste0(measure, "_2026")]] - changes[[paste0(measure, "_2025")]], NA_real_)
changes$demolition_change_direction <- case_when(!changes$demolition_change_available ~ "unavailable",
  changes$delta_demolition_pressure_index > 1e-9 ~ "increase", changes$delta_demolition_pressure_index < -1e-9 ~ "decrease",
  TRUE ~ "unchanged")
write_csv(changes, file.path(output_root, "demolition_feature_changes_by_hex.csv"))
code_paths <- c("R/demolition_panel.R", "R/demolition_coverage_history.R", "R/part2_demolitions.R",
  "R/part2_event_scoring.R", "scripts/part2/build_demolition_snapshots.R")
manifest <- list(schema_version = 2L, status = "paired_demolition_features_complete_v2",
  generated_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  cutoffs = as.character(cutoffs), source_observed_start = as.character(source_start), source_observed_end = as.character(source_end),
  retrospective = TRUE, permit_definition = "unique issued residential demolition permit; not proof of completed demolition",
  total_demolition_definition = "Description matches total whitespace demo; missing description is unclassified, not verified non-total demolition",
  coverage = "Current FULL point+center study mask, historical FULL/LTD/2MILE continuous rolling-interval source coverage",
  denominator = "Canonical full-hex square kilometers; boundary-straddling densities are not city-clipped densities",
  scoring = "Fixed equal thirds; reviewed earlier p1/p99 component bounds retained and frozen across dates; all three components required, unknown is never zero.",
  measurement_version = "fixed_components_v2",
  component_missing_policy = "all_components_required_fixed_equal_weights",
  inputs = build_file_manifest(c(required, coverage_config$local_path, code_paths), require_all = TRUE, hash_files = TRUE),
  outputs = build_file_manifest(output_root, recursive = TRUE, hash_files = TRUE))
# Exclude any previous manifest to avoid self-referential hashes on reruns.
manifest$outputs <- manifest$outputs[basename(manifest$outputs$path) != "demolition_run_manifest.json", ]
jsonlite::write_json(manifest, file.path(output_root, "demolition_run_manifest.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
print(bind_rows(summary_by_date))
cat("Paired demolition processing complete; no Part 1/Part 3 outputs modified.\n")
