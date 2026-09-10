# Corrected paired ACS inputs overwrite superseded Part 2 ACS products.
# --assemble-only keeps demographics but always rebuilds rent from twelve raw
# caches and dominant-geography crosswalks, never from old rent trend outputs.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
source("R/pipeline.R")
source("R/analysis_config.R")
source("R/acs_snapshot_scoring.R")
source("R/part2_rent_fallback.R")
source("R/part2_acs_rent_products.R")
args <- commandArgs(trailingOnly = TRUE)
if (length(setdiff(args, "--assemble-only"))) stop("Only --assemble-only is supported.")
root <- "output/part2/acs"
dir.create(root, recursive = TRUE, showWarnings = FALSE)
profiles <- data.frame(analysis_as_of_date = as.Date(c("2025-04-01", "2026-04-01")),
  current_year = c(2023L, 2024L), rent_years = c("2013,2018,2023", "2014,2019,2024"),
  acs_period_start = c(2019L, 2020L), acs_period_end = c(2023L, 2024L),
  release_date = as.Date(c("2024-12-12", "2026-01-29")), dollar_base_year = 2024L)
dates <- profiles$analysis_as_of_date; years <- part2_rent_required_years()
geographies <- c("block_group", "tract")
stopifnot(all(profiles$release_date < dates))
fixed_paths <- c("output/hex_grid.rds", "output/residential_parcels_for_hex_sf.rds")
crosswalk_paths <- file.path(root, as.character(dates), "acs_rent_dominant_sources_by_hex_vintage.csv")
rent_cache_paths <- unlist(lapply(years, function(year) file.path("data/raw_acs",
  paste0("acs_", year, "_acs5_", geographies, "_median_rent.rds"))))
cache_paths <- c(rent_cache_paths, unlist(lapply(c(2023L, 2024L), function(year)
  file.path("data/raw_acs", paste0("acs_", year, "_acs5_", c("block_group_demographics", "tract_medians"), ".rds")))),
  file.path("data/raw_acs", paste0("decennial_2020_blocks_", c("travis", "hays", "williamson"), ".rds")))
protected_paths <- c(fixed_paths, cache_paths, crosswalk_paths,
  list.files("output", pattern = "^acs_.*[.](rds|csv)$", full.names = TRUE))
protected_before <- build_file_manifest(protected_paths, require_all = TRUE, hash_files = TRUE)
write_csv(protected_before, file.path(root, "part1_preservation_before.csv"))
grid <- readRDS(fixed_paths[1]); grid_ids <- grid$hex_id
stopifnot(nrow(grid) == 7027L, is.integer(grid_ids), !anyNA(grid_ids), !anyDuplicated(grid_ids))
# Preserve the original normalization rows, including original availability
# counts; do not estimate fresh bounds on the corrected complete-case cohort.
expected_bounds_sha256 <- "045e6d84ddc67880dee5fe7528c2279cdbde35adfc1f33d1866c539c7c96df94"
reference <- readRDS(file.path(root, "acs_scaling.rds"))
stopifnot(reference$reference_date == dates[1], reference$dollar_base_year == 2024L,
  identical(digest::digest(reference$bounds, algo = "sha256"), expected_bounds_sha256))
reference <- acs_preserve_scaling_reference(reference)
reference$preserved_bounds_sha256 <- expected_bounds_sha256
cpi <- EWS_CONFIG$acs_cpi_u
stopifnot(all(as.character(years) %in% names(cpi)))
crosswalk <- bind_rows(lapply(crosswalk_paths, read_csv, show_col_types = FALSE,
  col_types = cols(acs_year = col_integer(), source_geography = col_character(), hex_id = col_integer(),
    dominant_source_geoid = col_character(), dominant_source_share = col_double(), dominant_source_method = col_character())))
stopifnot(nrow(crosswalk) == length(grid_ids) * 12L, !anyDuplicated(crosswalk[c("hex_id", "acs_year", "source_geography")]),
  setequal(crosswalk$hex_id, grid_ids), setequal(crosswalk$acs_year, years))
cat("Reconstructing one fixed six-vintage rent series from twelve local extracts...\n")
candidates <- bind_rows(lapply(years, function(year) bind_rows(lapply(geographies, function(geography) {
  raw <- st_drop_geometry(readRDS(file.path("data/raw_acs", paste0("acs_", year, "_acs5_", geography, "_median_rent.rds"))))
  stopifnot(!anyDuplicated(raw$GEOID), all(raw$variable == "median_rent"))
  assigned <- crosswalk %>% filter(acs_year == year, source_geography == geography) %>%
    rename(source_geoid = dominant_source_geoid, source_residential_share = dominant_source_share,
      source_assignment_method = dominant_source_method) %>%
    left_join(transmute(raw, source_geoid = GEOID, estimate, moe, raw_source_matched = TRUE), by = "source_geoid", relationship = "many-to-one")
  stopifnot(nrow(assigned) == length(grid_ids), all(assigned$raw_source_matched %in% TRUE))
  assigned
}))))
selected <- part2_select_rent_series(candidates, relative_moe_limit = .30)
series <- part2_rent_series_features(selected$candidates, selected$selection, cpi, base_year = 2024L)
rent_products <- part2_acs_rent_products(selected$candidates, selected$selection, series, grid, cpi)
for (name in c("candidates", "selection")) {
  saveRDS(selected[[name]], file.path(root, paste0("acs_rent_source_", name, ".rds")))
  write_csv(selected[[name]], file.path(root, paste0("acs_rent_source_", name, ".csv")))
}
saveRDS(series, file.path(root, "acs_rent_fixed_series_features.rds"))
write_csv(series, file.path(root, "acs_rent_fixed_series_features.csv"))
write_csv(count(selected$candidates, acs_year, source_geography, failure_reason, reliable, name = "hexes"), file.path(root, "acs_rent_candidate_reliability_qa.csv"))
write_csv(count(selected$selection, selected_geography, fallback_reason, name = "hexes"), file.path(root, "acs_rent_fixed_source_qa.csv"))
saveRDS(reference, file.path(root, "acs_scaling.rds")); write_csv(reference$bounds, file.path(root, "acs_scaling_bounds.csv"))
feature_sets <- list(); summaries <- list()
for (i in seq_len(nrow(profiles))) {
  profile <- profiles[i, ]; date_dir <- file.path(root, as.character(profile$analysis_as_of_date))
  dir.create(date_dir, recursive = TRUE, showWarnings = FALSE)
  if (!"--assemble-only" %in% args) {
    env <- c(paste0("EWS_ANALYSIS_AS_OF_DATE=", profile$analysis_as_of_date), paste0("EWS_ACS_YEARS=", profile$rent_years),
      paste0("EWS_ACS_CURRENT_YEAR=", profile$current_year), "EWS_ACS_DOLLAR_BASE_YEAR=2024", "EWS_ACS_PRESERVE_MISSING=true",
      paste0("EWS_ACS_OUTPUT_DIR=", date_dir))
    status <- system2(file.path(R.home("bin"), "Rscript"), args = "scripts/data/acs_demographics.R", env = env)
    if (status != 0L) stop("ACS demographic processing failed: ", profile$analysis_as_of_date)
  }
  demographics <- st_drop_geometry(readRDS(file.path(date_dir, "acs_demographics_by_hex.rds")))
  rent <- st_drop_geometry(rent_products[[i]]$trends); rent_vintages <- rent_products[[i]]$vintage
  for (x in list(demographics, rent)) stopifnot(!anyDuplicated(x$hex_id), setequal(x$hex_id, grid_ids),
    all(x$analysis_as_of_date == profile$analysis_as_of_date), all(x$acs_dollar_base_year == 2024L))
  stopifnot(all(demographics$acs_year == profile$current_year), all(rent$acs_rent_current_year == profile$current_year), all(demographics$acs_preserve_missing))
  saveRDS(rent_vintages, file.path(date_dir, "acs_rent_by_hex_vintage.rds"))
  write_csv(st_drop_geometry(rent_vintages), file.path(date_dir, "acs_rent_by_hex_vintage.csv"))
  saveRDS(rent_products[[i]]$trends, file.path(date_dir, "acs_rent_trends_by_hex.rds")); write_csv(rent, file.path(date_dir, "acs_rent_trends_by_hex.csv"))
  demo_columns <- c("hex_id", "median_income", "median_income_real", "median_income_moe", "median_income_moe_real",
    "median_income_relative_moe", "median_income_reliable", "pct_renter", "pct_college", "poverty_rate", "pct_rent_burden_30plus")
  features <- data.frame(hex_id = grid_ids) %>% left_join(rent, by = "hex_id") %>% left_join(select(demographics, all_of(demo_columns)), by = "hex_id") %>%
    mutate(acs_current_year = profile$current_year, acs_period_start = profile$acs_period_start, acs_period_end = profile$acs_period_end,
      acs_release_date = profile$release_date, acs_retrospective_reconstruction = TRUE, acs_fixed_block_year = 2020L,
      acs_scaling_mode = "preserved_initial_2025_component_bounds")
  scored <- acs_apply_scaling(features, reference); features <- scored$features
  stopifnot(all(is.finite(features$rent_pressure_citywide_index) == features$acs_rent_complete),
    all(is.finite(features$demographic_vulnerability_index) == features$acs_vulnerability_complete))
  feature_sets[[i]] <- features
  saveRDS(features, file.path(date_dir, "acs_features_by_hex.rds")); write_csv(features, file.path(date_dir, "acs_features_by_hex.csv"))
  write_csv(mutate(scored$qa, analysis_as_of_date = profile$analysis_as_of_date), file.path(date_dir, "acs_scoring_qa.csv"))
  write_csv(count(features, analysis_as_of_date, acs_rent_components_available, acs_vulnerability_components_available, name = "hexes"), file.path(date_dir, "acs_component_availability_qa.csv"))
  summaries[[i]] <- data.frame(profile, hexes = nrow(features), rent_level_available = sum(is.finite(features$acs_rent_current_real)),
    reliable_rent_trend = sum(features$acs_rent_trend_reliable), rent_index_available = sum(is.finite(features$rent_pressure_citywide_index)),
    vulnerability_index_available = sum(is.finite(features$demographic_vulnerability_index)),
    all_five_vulnerability_components = sum(features$acs_vulnerability_components_available == 5L),
    selected_block_group_series = sum(features$acs_rent_source_geography == "block_group", na.rm = TRUE),
    selected_tract_series = sum(features$acs_rent_source_geography == "tract", na.rm = TRUE), reliable_income = sum(features$median_income_reliable %in% TRUE))
}
paired <- bind_rows(feature_sets)
write_csv(paired, file.path(root, "acs_features_paired.csv")); saveRDS(paired, file.path(root, "acs_features_paired.rds"))
write_csv(bind_rows(summaries), file.path(root, "acs_snapshot_summary.csv"))
measures <- c("rent_pressure_citywide_index", "demographic_vulnerability_index", "acs_rent_current_real", "median_income_real",
  "acs_rent_components_available", "acs_vulnerability_components_available")
changes <- inner_join(select(feature_sets[[1]], hex_id, all_of(measures)), select(feature_sets[[2]], hex_id, all_of(measures)), by = "hex_id", suffix = c("_2025", "_2026"))
for (measure in measures) changes[[paste0("delta_", measure)]] <- changes[[paste0(measure, "_2026")]] - changes[[paste0(measure, "_2025")]]
availability <- lapply(feature_sets, function(x) !is.na(acs_component_inputs(x)))
changes$acs_same_component_availability <- rowSums(availability[[1]] != availability[[2]]) == 0L
changes$acs_all_eight_components_both <- rowSums(availability[[1]] & availability[[2]]) == 8L
changes$acs_both_indices_both_dates <- is.finite(changes$rent_pressure_citywide_index_2025) & is.finite(changes$rent_pressure_citywide_index_2026) &
  is.finite(changes$demographic_vulnerability_index_2025) & is.finite(changes$demographic_vulnerability_index_2026)
write_csv(changes, file.path(root, "acs_feature_changes_by_hex.csv"))
protected_after <- build_file_manifest(protected_paths, require_all = TRUE, hash_files = TRUE)
stopifnot(identical(protected_before$path, protected_after$path), identical(protected_before$sha256, protected_after$sha256),
  identical(digest::digest(reference$bounds, algo = "sha256"), expected_bounds_sha256))
write_csv(protected_after, file.path(root, "part1_preservation_after.csv"))
code_paths <- c("R/analysis_config.R", "R/acs_dasymetric.R", "R/acs_snapshot_scoring.R", "R/part2_rent_fallback.R",
  "R/part2_acs_rent_products.R", "R/utils.R", "scripts/data/acs_demographics.R", "scripts/part2/build_acs_snapshots.R")
outputs <- list.files(root, recursive = TRUE, full.names = TRUE, pattern = "[.](rds|csv)$")
manifest <- list(schema_version = 2L, status = "paired_acs_features_complete_v2",
  processed_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), git_commit = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  reconstruction = "retrospective_release_shift_fixed_support", profiles = profiles, dollar_base_year = 2024L, cpi_u = as.list(cpi[as.character(years)]),
  overlap = "Adjacent five-year ACS releases share four survey years; changes are not independent annual estimates",
  rent_source_rule = "Block group reliable at all six vintages, else tract reliable at all six, else all three components missing at both dates",
  relative_moe_limit = .30, preserved_normalization_bounds_sha256 = expected_bounds_sha256,
  scoring = "Original 2025 full-grid component bounds preserved; fixed equal weights; all three rent and all five vulnerability components required",
  required_components = list(rent = 3L, vulnerability = 5L),
  limitations = c("Fixed 2020 block ancillary weights and current parcels, not harmonized historical geography",
    "Income reliability reported but not an additional vulnerability scoring gate",
    "Rent source selection uses both dates retrospectively; not an operational past-information-only choice",
    "Historical rent fallback audit is a superseded-baseline sensitivity artifact, not a dependency or current validation",
    "ACS completeness is not final seven-feature cluster eligibility"),
  part1_outputs_unchanged = TRUE, raw_caches_and_crosswalks_unchanged = TRUE, prior_failed_run_overwritten_without_archive = TRUE,
  inputs = build_file_manifest(c(cache_paths, crosswalk_paths, fixed_paths, code_paths), require_all = TRUE, hash_files = TRUE),
  outputs = build_file_manifest(outputs, require_all = TRUE, hash_files = TRUE))
jsonlite::write_json(manifest, file.path(root, "acs_run_manifest.json"), auto_unbox = TRUE, pretty = TRUE, na = "null", digits = NA)
print(bind_rows(summaries)); cat("Corrected ACS pair complete:", root, "\n")
