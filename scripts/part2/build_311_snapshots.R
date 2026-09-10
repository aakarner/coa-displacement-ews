# Local-only selected 311 reconstruction. Never refreshes or overwrites Part 1.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
source("R/pipeline.R")
source("R/demolition_coverage_history.R")
source("R/part2_311.R")
source("R/part2_event_scoring.R")
root <- "output/part2/311"
# Load before any overwrite; v1 and v2 both preserve unchanged baseline bounds.
preserved_scaling_path <- file.path(root, "311_scaling.rds")
preserved_scaling <- if (file.exists(preserved_scaling_path)) readRDS(preserved_scaling_path) else NULL
dir.create(root, recursive = TRUE, showWarnings = FALSE)
raw_path <- "data/raw_311/austin_311_selected_20200101_20260401.rds"
raw_sha <- "6c508ded811d58e222cb54f4f52f518efbc2ec3838d1ca0646c2cbf482ee34d6"
stopifnot(identical(digest::digest(file = raw_path, algo = "sha256"), raw_sha))
type_path <- "config/311_smoke_signal_types.csv"
grid_path <- "output/hex_grid.rds"
units_path <- "output/corporate_ownership_by_hex.rds"
city_path <- "data/BOUNDARIES_jurisdictions_20260429.geojson"
jurisdiction_config_path <- "config/demolition_jurisdiction_sources.csv"
jurisdiction_config <- read_csv(jurisdiction_config_path, show_col_types = FALSE) %>% filter(status == "active")
stopifnot(identical(sort(jurisdiction_config$coverage_role), sort(c("historical_baseline", "dated_actions"))))
for (i in seq_len(nrow(jurisdiction_config))) stopifnot(identical(
  digest::digest(file = jurisdiction_config$local_path[i], algo = "sha256"), jurisdiction_config$sha256[i]))
protected <- c(grid_path, units_path, list.files("output", pattern = "^311_.*[.](rds|csv)$", full.names = TRUE))
before <- build_file_manifest(protected, require_all = TRUE, hash_files = TRUE)
write_csv(before, file.path(root, "part1_preservation_before.csv"))
cache <- readRDS(raw_path)
types <- read_csv(type_path, show_col_types = FALSE)
grid <- readRDS(grid_path)
stopifnot(nrow(grid) == 7027L)
units <- st_drop_geometry(readRDS(units_path))
stopifnot(!anyDuplicated(units$hex_id), setequal(grid$hex_id, units$hex_id))
support <- st_drop_geometry(grid) %>% select(hex_id, area_km2) %>%
  left_join(select(units, hex_id, residential_units), by = "hex_id")
city_rows <- st_read(city_path, quiet = TRUE) %>%
  filter(toupper(trimws(city_name)) == "CITY OF AUSTIN", toupper(trimws(jurisdiction_type)) == "FULL")
stopifnot(nrow(city_rows) > 0L)
city <- st_sf(geometry = st_union(st_transform(st_make_valid(city_rows), 3083)))
baselines <- st_read(jurisdiction_config$local_path[jurisdiction_config$coverage_role == "historical_baseline"], quiet = TRUE)
actions <- st_read(jurisdiction_config$local_path[jurisdiction_config$coverage_role == "dated_actions"], quiet = TRUE)
cutoffs <- as.Date(c("2025-04-01", "2026-04-01"))
cat("Preparing pinned 311 event ledger and spatial QA...\n")
prepared <- part2_311_prepare_events(cache, types, grid, city)
write_csv(prepared$events, file.path(root, "311_event_ledger.csv"))
saveRDS(prepared$events, file.path(root, "311_event_ledger.rds"))
write_csv(prepared$duplicates, file.path(root, "311_duplicate_id_audit.csv"))
write_csv(prepared$qa, file.path(root, "311_source_qa.csv"))
write_csv(prepared$events %>% filter(!event_date_valid | !coordinate_valid |
  spatial_assignment != "within_unique" | !event_inside_current_city %in% TRUE),
  file.path(root, "311_spatial_exception_audit.csv"))
months <- seq(as.Date(cache$start_date), as.Date(format(cache$analysis_as_of_date, "%Y-%m-01")), by = "month")
monthly <- prepared$events %>% filter(event_date_valid) %>%
  mutate(month = as.Date(format(sr_created_date, "%Y-%m-01"))) %>%
  count(month, sr_type_desc, name = "requests")
monthly <- tidyr::expand_grid(month = months, sr_type_desc = types$sr_type_desc) %>%
  left_join(monthly, by = c("month", "sr_type_desc")) %>% mutate(requests = coalesce(requests, 0L))
write_csv(monthly, file.path(root, "311_monthly_type_qa.csv"))
stopifnot(all((monthly %>% group_by(month) %>% summarise(n = sum(requests)))$n > 0L))
cat("Replaying FULL-only historical geography over each exact 24-month window...\n")
coverage <- part2_311_geography(grid, city, baselines, actions, cutoffs)
stopifnot(sum(coverage$sr_311_in_current_city_scope & coverage$analysis_as_of_date == cutoffs[1]) == 6060L)
write_csv(coverage, file.path(root, "311_historical_geography_by_hex_date.csv"))
sets <- list(); summaries <- list()
for (i in seq_along(cutoffs)) {
  cutoff <- cutoffs[i]
  directory <- file.path(root, as.character(cutoff)); dir.create(directory, showWarnings = FALSE)
  snapshot <- part2_311_snapshot(prepared$events, support, filter(coverage, analysis_as_of_date == cutoff),
    cutoff, cache$start_date, cache$analysis_as_of_date, cache$complete, 20)
  features <- snapshot$features
  if (i == 1L) {
    scaling <- part2_fit_event_scaling(features, part2_311_components(), "sr_311_pressure_index", cutoff, preserved_scaling = preserved_scaling)
    saveRDS(scaling, file.path(root, "311_scaling.rds"))
    write_csv(scaling$bounds, file.path(root, "311_scaling_bounds.csv"))
  }
  scored <- part2_apply_event_scaling(features, scaling)
  features <- scored$features; sets[[i]] <- features
  stopifnot(all(is.finite(features$sr_311_pressure_index) == features$sr_311_pressure_index_components_complete))
  saveRDS(features, file.path(directory, "311_features_by_hex.rds"))
  write_csv(features, file.path(directory, "311_features_by_hex.csv"))
  write_csv(snapshot$membership, file.path(directory, "311_event_window_membership.csv"))
  write_csv(scored$qa, file.path(directory, "311_scoring_qa.csv"))
  write_csv(count(features, sr_311_coverage_reason, sr_311_boundary_straddling_hex,
                  sr_311_pressure_index_components_available, name = "hexes"), file.path(directory, "311_coverage_qa.csv"))
  membership <- snapshot$membership
  summaries[[i]] <- data.frame(analysis_as_of_date = cutoff, hexes = nrow(features),
    current_city_hexes = sum(features$sr_311_in_current_city_scope),
    historical_FULL_screen_usable = sum(features$sr_311_poc_coverage_usable),
    index_available = sum(is.finite(features$sr_311_pressure_index)),
    recent_requests_before_city_filter = sum(membership$event_window == "recent" & !is.na(membership$hex_id)),
    recent_city_point_requests = sum(membership$event_window == "recent" & membership$mapped_city_event),
    recent_feature_requests = sum(features$sr_311_smoke_signal_latest_12mo, na.rm = TRUE),
    previous_feature_requests = sum(features$sr_311_smoke_signal_previous_12mo, na.rm = TRUE),
    usable_zero_recent_hexes = sum(features$sr_311_valid_zero_latest),
    usable_rate_hexes = sum(is.finite(features$sr_311_smoke_signal_latest_12mo_per_100_units)),
    usable_rate_change_hexes = sum(is.finite(features$sr_311_smoke_signal_latest_12mo_rate_change_per_100_units)),
    usable_legacy_percent_change_hexes = sum(is.finite(features$sr_311_smoke_signal_latest_12mo_change_pct)))
}
paired <- bind_rows(sets)
saveRDS(paired, file.path(root, "311_features_paired.rds")); write_csv(paired, file.path(root, "311_features_paired.csv"))
write_csv(bind_rows(summaries), file.path(root, "311_snapshot_summary.csv"))
measures <- c(part2_311_components(), "sr_311_pressure_index", "sr_311_smoke_signal_latest_12mo", "sr_311_smoke_signal_previous_12mo")
changes <- inner_join(select(sets[[1]], hex_id, all_of(measures)), select(sets[[2]], hex_id, all_of(measures)),
                      by = "hex_id", suffix = c("_2025", "_2026"))
for (measure in measures) changes[[paste0("delta_", measure)]] <- changes[[paste0(measure, "_2026")]] - changes[[paste0(measure, "_2025")]]
a <- lapply(sets, function(x) !is.na(x[part2_311_components()]))
changes$sr_311_same_component_availability <- rowSums(a[[1]] != a[[2]]) == 0L
changes$sr_311_all_three_components_both <- rowSums(a[[1]] & a[[2]]) == 3L
write_csv(changes, file.path(root, "311_feature_changes_by_hex.csv"))
after <- build_file_manifest(protected, require_all = TRUE, hash_files = TRUE)
stopifnot(identical(before$sha256, after$sha256)); write_csv(after, file.path(root, "part1_preservation_after.csv"))
inputs <- c(raw_path, type_path, grid_path, units_path, city_path, jurisdiction_config_path,
  jurisdiction_config$local_path, "R/part2_311.R", "R/part2_event_scoring.R", "R/demolition_coverage_history.R",
  "R/pipeline.R", "scripts/part2/build_311_snapshots.R")
manifest <- list(schema_version = 2L, status = "paired_311_features_complete_v2", analysis_cutoffs = as.character(cutoffs),
  processed_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  git_commit = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  source_url = "https://data.austintexas.gov/api/v3/views/xwdj-i9he/query.json",
  source_fetched_at_utc = format(cache$fetched_at, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  source_start = as.character(cache$start_date), source_end = as.character(cache$analysis_as_of_date),
  source_query_complete = cache$complete, selected_types = cache$selected_type_descriptions,
  query_contract = "Selected types; sr_created_date >=2020-01-01 and <2026-04-02; latitude/longitude not null; ordered date and SR number; complete pagination",
  event_time = "source floating timestamp calendar date; no timezone shift",
  all_requests_coverage_verified = FALSE,
  geography = "Fixed 2026-04-29 FULL city-center study mask, point-level current-city filter, continuous FULL jurisdiction across each24monthwindow as POCscreen only",
  zero_definition = "zero mapped selected requests in complete coordinate-required cache, conditional on geography screen; not zero complaints",
  missing_definition = "outside scope, failed/unresolved FULL replay or query period, or ambiguous point assignment; zero previous count makes percentage change undefined",
  denominator = "canonical promoted residential units, >=20 for rates; fixed full hex areas; boundary-cell rates approximate hex scales, not clipped City rates",
  source_limitations = c("Geocoding omissions and complaint/reporting biases are not measured", "FULL replay is not independently verified 311 service coverage",
    "Current city/parcel support is fixed retrospectively, not a historical boundary or housing reconstruction"),
  scoring = "Fixed equal thirds: current rate/density retain reviewed earlier p1/p99 bounds; signed rate change uses symmetric earlier q99 absolute-change bounds, zero50 (including degenerate B0); all components required. Two thirds of weight measures current activity.",
  measurement_version = "fixed_components_v2",
  component_missing_policy = "all_components_required_fixed_equal_weights",
  part1_outputs_unchanged = TRUE,
  inputs = build_file_manifest(inputs, require_all = TRUE, hash_files = TRUE),
  outputs = build_file_manifest(list.files(root, recursive = TRUE, full.names = TRUE, pattern = "[.](rds|csv)$"), require_all = TRUE, hash_files = TRUE))
jsonlite::write_json(manifest, file.path(root, "311_run_manifest.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
print(bind_rows(summaries)); cat("Paired 311 processing complete.\n")
