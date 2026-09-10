# Local-only paired mapped-filing proxies; never refreshes or overwrites Part 1/3.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
source("R/pipeline.R")
source("R/eviction_panel.R")
source("R/eviction_coverage.R")
source("R/part2_evictions.R")
source("R/part2_index_scoring.R")
root <- "output/part2/evictions"
# Load before any overwrite; v1 and v2 both preserve unchanged baseline bounds.
preserved_scaling_path <- file.path(root, "eviction_scaling.rds")
preserved_scaling <- if (file.exists(preserved_scaling_path)) readRDS(preserved_scaling_path) else NULL
dir.create(root, recursive = TRUE, showWarnings = FALSE)
cutoffs <- as.Date(c("2025-04-01", "2026-04-01")); history_start <- as.Date("2022-01-01")
grid_path <- "output/hex_grid.rds"; units_path <- "output/corporate_ownership_by_hex.rds"
city_path <- "data/BOUNDARIES_jurisdictions_20260429.geojson"
county_path <- "config/hex_county_assignment_2024.csv"
jp_path <- "config/williamson_jp_hex_assignment.csv"
jp_metadata_path <- "config/williamson_jp_hex_assignment_metadata.csv"
source_path <- "config/eviction_sources.csv"
filing_paths <- c(Travis = "output/eviction_filings_prepared_for_geocoding.csv",
  Williamson = "output/williamson_eviction_filings_prepared_for_geocoding.csv")
geocode_paths <- c(Travis = "output/eviction_addresses_geocoded.csv",
  Williamson = "output/williamson_eviction_addresses_geocoded_with_arcgis.csv")
registries <- c(Travis = "travis_reviewed", Williamson = "williamson_geocode_cascade")
protected <- unique(c(grid_path, units_path,
  list.files("output", pattern = "^(eviction|williamson_eviction|williamson_address_reference).*[.](rds|csv)$", full.names = TRUE),
  list.files("output/part3", pattern = "^eviction.*[.](rds|csv)$", full.names = TRUE)))
before <- build_file_manifest(protected, require_all = TRUE, hash_files = TRUE)
write_csv(before, file.path(root, "canonical_preservation_before.csv"))
grid <- readRDS(grid_path); units <- st_drop_geometry(readRDS(units_path))
counties <- read_csv(county_path, show_col_types = FALSE) %>% mutate(hex_id = as.integer(hex_id))
jps <- read_csv(jp_path, show_col_types = FALSE) %>% mutate(hex_id = as.integer(hex_id))
jp_metadata <- read_csv(jp_metadata_path, show_col_types = FALSE)
source_config <- read_csv(source_path, show_col_types = FALSE)
stopifnot(nrow(grid) == 7027L, is.integer(grid$hex_id), !anyDuplicated(units$hex_id),
  setequal(grid$hex_id, units$hex_id), !anyDuplicated(counties$hex_id), setequal(grid$hex_id, counties$hex_id),
  identical(as.character(grid$h3_index), as.character(counties$h3_index[match(grid$hex_id, counties$hex_id)])),
  all(jps$h3_index == as.character(grid$h3_index[match(jps$hex_id, grid$hex_id)])),
  all(jp_metadata$grid_sha256 == digest::digest(file = grid_path, algo = "sha256")),
  all(jp_metadata$county_reference_sha256 == digest::digest(file = county_path, algo = "sha256")))
city_rows <- st_read(city_path, quiet = TRUE) %>%
  filter(toupper(trimws(city_name)) == "CITY OF AUSTIN", toupper(trimws(jurisdiction_type)) == "FULL")
stopifnot(nrow(city_rows) > 0L)
city <- st_sf(geometry = st_union(st_transform(st_make_valid(city_rows), 3083)))
city_reference <- part2_eviction_city_reference(grid, city)
stopifnot(sum(city_reference$eviction_inside_current_city) == 6060L)
support <- st_drop_geometry(grid) %>% select(hex_id, area_km2) %>%
  left_join(select(units, hex_id, residential_units), by = "hex_id") %>%
  left_join(select(counties, hex_id, source_county), by = "hex_id") %>%
  left_join(city_reference, by = "hex_id")
filing_sets <- lapply(names(filing_paths), function(county) part2_eviction_read_filings(filing_paths[county], county, registries[county]))
filings <- bind_rows(filing_sets)
geocodes <- bind_rows(lapply(seq_along(filing_sets), function(i) part2_eviction_read_geocodes(
  geocode_paths[i], registries[i], filing_sets[[i]])))
coverage_sets <- lapply(seq_along(cutoffs), function(i) part2_eviction_source_coverage(
  select(counties, hex_id, source_county), source_config, jps, cutoffs[i], history_start))
# Full date/court evidence is retained for resolving conflicting case rows.
# This registry is not the cell eligibility test: each snapshot checks only
# its scored 24 months, including missing dates that could be in that window.
resolution_coverage <- build_eviction_hex_year_coverage(select(counties, hex_id, source_county),
  seq(as.integer(format(history_start, "%Y")), as.integer(format(max(cutoffs), "%Y"))),
  source_config, jps, max(cutoffs))
cat("Resolving namespaced eviction cases from prepared records and final geocode registries...\n")
resolved <- part2_eviction_resolve(filings, geocodes, grid, select(counties, hex_id, source_county), city,
  city_reference, resolution_coverage, history_start, max(cutoffs))
saveRDS(resolved$cases, file.path(root, "eviction_case_ledger.rds"))
write_csv(resolved$cases, file.path(root, "eviction_case_ledger.csv"))
write_csv(resolved$issues, file.path(root, "eviction_case_assignment_issues.csv"))
write_csv(resolved$row_qc, file.path(root, "eviction_case_row_qa.csv"))
# Keep source-row dates and all reliable candidate evidence locally for audits;
# no defendant names or raw source address columns are copied.
write_csv(select(resolved$source, -address_for_geocoding), file.path(root, "eviction_source_case_dates.csv"))
write_csv(select(resolved$evidence, -address_for_geocoding), file.path(root, "eviction_spatial_evidence.csv"))
write_csv(resolved$candidates, file.path(root, "eviction_candidate_hexes.csv"))
write_csv(count(resolved$cases, source_county, source_jp_district, assignment_status, name = "unique_cases_or_unidentified_rows"),
  file.path(root, "eviction_assignment_summary.csv"))
write_csv(resolved$evidence %>% distinct(row_id, source_county, jp_district, source_geography_status, point_hex_matches) %>%
  count(source_county, jp_district, source_geography_status, point_hex_matches, name = "reliable_rows"),
  file.path(root, "eviction_point_geography_qa.csv"))
source_qa <- bind_rows(lapply(seq_along(filing_sets), function(i) data.frame(source_county = names(filing_paths)[i],
  prepared_rows = nrow(filing_sets[[i]]), unique_identified_cases = n_distinct(filing_sets[[i]]$case_number[filing_sets[[i]]$case_identity_valid]),
  unidentified_source_rows = sum(!filing_sets[[i]]$case_identity_valid),
  registry_addresses = sum(geocodes$geocode_registry == registries[i]),
  filing_date_min = min(filing_sets[[i]]$file_date), filing_date_max = max(filing_sets[[i]]$file_date),
  geocode_registry = registries[i], all_filing_locations_complete = FALSE)))
write_csv(source_qa, file.path(root, "eviction_source_qa.csv"))
sets <- list(); summaries <- list()
for (i in seq_along(cutoffs)) {
  cutoff <- cutoffs[i]; directory <- file.path(root, as.character(cutoff)); dir.create(directory, showWarnings = FALSE)
  coverage <- coverage_sets[[i]]
  snapshot <- part2_eviction_snapshot(resolved, support, coverage$coverage, cutoff, history_start, 20)
  features <- snapshot$features
  if (i == 1L) {
    scaling <- part2_fit_index_scaling(features, part2_eviction_components(), "eviction_pressure_index", cutoff, preserved_scaling = preserved_scaling)
    saveRDS(scaling, file.path(root, "eviction_scaling.rds"))
    write_csv(scaling$bounds, file.path(root, "eviction_scaling_bounds.csv"))
  }
  scored <- part2_apply_index_scaling(features, scaling); features <- scored$features; sets[[i]] <- features
  stopifnot(is.integer(features$hex_id), !anyNA(features$eviction_count_observed),
    all(is.na(features$eviction_pressure_index[!features$eviction_count_observed])),
    all(is.finite(features$eviction_pressure_index) == features$eviction_pressure_index_components_complete))
  saveRDS(features, file.path(directory, "eviction_features_by_hex.rds"))
  write_csv(features, file.path(directory, "eviction_features_by_hex.csv"))
  write_csv(snapshot$membership, file.path(directory, "eviction_case_window_membership.csv"))
  write_csv(snapshot$uncertainty, file.path(directory, "eviction_localizable_uncertainty.csv"))
  write_csv(coverage$segments, file.path(directory, "eviction_source_coverage_segments.csv"))
  write_csv(scored$qa, file.path(directory, "eviction_scoring_qa.csv"))
  write_csv(part2_eviction_court_window_qa(resolved, cutoff, history_start), file.path(directory, "eviction_court_window_qa.csv"))
  write_csv(count(features, source_county, eviction_coverage_jps, eviction_coverage_reason,
    eviction_boundary_straddling_hex, eviction_pressure_index_components_available, name = "hexes"),
    file.path(directory, "eviction_coverage_qa.csv"))
  summaries[[i]] <- data.frame(analysis_as_of_date = cutoff, hexes = nrow(features),
    eligibility_window_start = unique(features$eviction_eligibility_window_start),
    eligibility_window_end = unique(features$eviction_eligibility_window_end),
    current_city_hexes = sum(features$eviction_inside_current_city),
    source_covered_hexes = sum(features$eviction_source_covered),
    localizable_ambiguity_hexes = sum(features$eviction_source_covered & features$eviction_unresolved_candidate_cases > 0),
    count_observed_hexes = sum(features$eviction_count_observed),
    index_available = sum(is.finite(features$eviction_pressure_index)),
    all_required_components = sum(features$eviction_pressure_index_components_complete),
    recent_feature_filings = sum(features$eviction_cases_latest_12mo, na.rm = TRUE),
    previous_feature_filings = sum(features$eviction_cases_previous_12mo, na.rm = TRUE),
    history_feature_filings = sum(features$eviction_cases_total, na.rm = TRUE),
    valid_zero_recent_hexes = sum(features$eviction_valid_zero_recent),
    usable_rate_hexes = sum(is.finite(features$eviction_latest_12mo_per_100_units)),
    usable_rate_change_hexes = sum(is.finite(features$eviction_latest_12mo_rate_change_per_100_units)),
    usable_legacy_percent_change_hexes = sum(is.finite(features$eviction_cases_latest_12mo_change_pct)),
    usable_recent_share_hexes = sum(is.finite(features$eviction_recent_share)),
    history_days = unique(features$eviction_history_days), recent_window_days = unique(features$eviction_recent_window_days))
}
paired <- bind_rows(sets)
saveRDS(paired, file.path(root, "eviction_features_paired.rds")); write_csv(paired, file.path(root, "eviction_features_paired.csv"))
write_csv(bind_rows(summaries), file.path(root, "eviction_snapshot_summary.csv"))
measures <- c(part2_eviction_components(), "eviction_pressure_index", "eviction_cases_latest_12mo", "eviction_cases_previous_12mo", "eviction_cases_total")
changes <- inner_join(select(sets[[1]], hex_id, all_of(measures)), select(sets[[2]], hex_id, all_of(measures)),
  by = "hex_id", suffix = c("_2025", "_2026"))
for (measure in measures) changes[[paste0("delta_", measure)]] <- changes[[paste0(measure, "_2026")]] - changes[[paste0(measure, "_2025")]]
available <- lapply(sets, function(x) !is.na(x[part2_eviction_components()]))
changes$eviction_same_component_availability <- rowSums(available[[1]] != available[[2]]) == 0L
changes$eviction_all_required_components_both <- rowSums(available[[1]] & available[[2]]) == length(part2_eviction_components())
write_csv(changes, file.path(root, "eviction_feature_changes_by_hex.csv"))
after <- build_file_manifest(protected, require_all = TRUE, hash_files = TRUE)
stopifnot(identical(before$sha256, after$sha256)); write_csv(after, file.path(root, "canonical_preservation_after.csv"))
qa_paths <- c("output/williamson_eviction_geocode_qa.csv", "output/williamson_eviction_geocode_arcgis_qa.csv",
  "output/williamson_address_reference_qa.csv", "output/williamson_eviction_geocode_local_qa.csv", "output/williamson_eviction_geocode_coa_qa.csv")
inputs <- unique(c(grid_path, units_path, city_path, county_path, jp_path, jp_metadata_path,
  "config/hex_county_assignment_2024_metadata.csv", source_path, source_config$path, filing_paths, geocode_paths,
  qa_paths, "R/pipeline.R", "R/eviction_panel.R", "R/eviction_coverage.R", "R/part2_evictions.R",
  "R/part2_index_scoring.R", "scripts/part2/build_eviction_snapshots.R"))
manifest <- list(schema_version = 2L, status = "paired_eviction_features_complete_v2",
  analysis_cutoffs = as.character(cutoffs), history_start = as.character(history_start),
  eligibility_rule = "rolling_scored_24_months_v1",
  eligibility_window_starts = c("2023-04-02", "2024-04-02"),
  processed_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  git_commit = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  event_definition = "unique county/court-namespaced recorded filings; source filing date, not judgment/displacement date",
  prepared_geocodes = "existing Travis reviewed registry and final Williamson local/City/Census/ArcGIS cascade; no new geocoding",
  reliable_location = "status M or T; score >=90; valid coordinates; unique consistent case hex; source county/effective court and fixed city point+center",
  temporal_reconstruction = "retrospective; current prepared records and geocodes, not an as-published historical extraction",
  windows = "Inclusive Apr2..Apr1 recent and preceding windows; coverage and ambiguity checks restricted to those scored 24 months. Since2022 counts are unscored diagnostics only",
  source_coverage = "Every calendar-year segment of the scored 24 months covers its actual requested dates; final year ends April1, not Dec31. No pre-window coverage requirement",
  geography = "fixed 2026-04-29 Austin FULL city-center mask and exact event-point footprint; Travis all 5 courts; Williamson effective JP1/2; Hays/JP3/unassigned missing",
  zero_definition = "Zero recorded reliably mapped filings among covered local sources and court/City geography, with no localizable unresolved case potentially in either scored annual window",
  missing_definition = "Unsupported county/court/scored source period, outside city, or any potentially in-window localizable case/date ambiguity; wholly unlocated cases do not suppress whole courts",
  all_filing_locations_complete = FALSE,
  ambiguity_rule = "Retain all original dates/locations for potentially relevant cases; any candidate date within the scored 24 months, or missing date, masks every physical candidate city hex. Older-only ambiguities do not exclude cells; invalid case IDs remain audit-only",
  denominator = "fixed promoted residential units >=20 for rates; fixed whole-hex areas/unit counts; boundary rates are hex-scale approximations not clipped City rates",
  recent_share_caveat = "Unscored diagnostic: recent filings divided by mapped historical filings since2022. Older source completeness is no longer required, so this is not a coverage-qualified longitudinal measure",
  scoring = "Fixed equal halves: current rate retains reviewed earlier p1/p99 bounds; signed rate change uses symmetric earlier q99 absolute-change bounds, zero50 (including degenerate B0); both components required. Percent change/recent share are diagnostics only.",
  measurement_version = "fixed_components_v2",
  component_missing_policy = "all_components_required_fixed_equal_weights",
  canonical_outputs_unchanged = TRUE,
  inputs = build_file_manifest(inputs, require_all = TRUE, hash_files = TRUE),
  outputs = build_file_manifest(list.files(root, recursive = TRUE, full.names = TRUE, pattern = "[.](rds|csv)$"), require_all = TRUE, hash_files = TRUE))
jsonlite::write_json(manifest, file.path(root, "eviction_run_manifest.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
print(bind_rows(summaries)); cat("Paired eviction processing complete.\n")
