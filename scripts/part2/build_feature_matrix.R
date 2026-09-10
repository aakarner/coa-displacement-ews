# Local-only assembly of already-validated/scored snapshots. No clustering or ML.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
source("R/pipeline.R")
source("R/part2_feature_matrix.R")
root <- "output/part2/matrix"
dir.create(root, recursive = TRUE, showWarnings = FALSE)
sources <- data.frame(
  domain = c("acs", "amenities", "sr311", "demolitions", "evictions", "ownership"),
  directory = c("acs", "amenities", "311", "demolitions", "evictions", "ownership_index"),
  feature = c("acs", "amenity", "311", "demolition", "eviction", "ownership"),
  manifest = c("acs", "amenity", "311", "demolition", "eviction", "ownership_index"),
  expected_status = c("paired_acs_features_complete_v2", "paired_features_complete", "paired_311_features_complete_v2",
    "paired_demolition_features_complete_v2", "paired_eviction_features_complete_v2", "paired_ownership_index_complete_v2"))
sources$feature_path <- file.path("output/part2", sources$directory, paste0(sources$feature, "_features_paired.rds"))
sources$manifest_path <- file.path("output/part2", sources$directory, paste0(sources$manifest, "_run_manifest.json"))
cat("Verifying all six source manifests and their pinned inputs/outputs...\n")
verification <- bind_rows(lapply(seq_len(nrow(sources)), function(i) part2_matrix_verify_source(
  sources$manifest_path[i], sources$feature_path[i], sources$expected_status[i])))
source_before <- build_file_manifest(unique(normalizePath(c(sources$manifest_path, verification$path), mustWork = TRUE)),
  require_all = TRUE, hash_files = TRUE)
# The user requested replacement, not an archive of the failed Part 2 run.
# Preserve only outputs outside Part 2, and overwrite this checksum inventory.
protected_paths <- c(list.files("output", recursive = TRUE, full.names = TRUE,
                                pattern = "[.](rds|csv|json)$"),
                     list.files("figures", recursive = TRUE, full.names = TRUE))
protected_paths <- protected_paths[!startsWith(protected_paths, "output/part2/") &
                                     !startsWith(protected_paths, "figures/part2/")]
protected_before <- build_file_manifest(protected_paths, require_all = TRUE, hash_files = TRUE)
saveRDS(protected_before, file.path(root, "existing_outputs_before.rds"))
domains <- setNames(lapply(sources$feature_path, readRDS), sources$domain)
stopifnot(all(domains$evictions$eviction_eligibility_rule == "rolling_scored_24_months_v1"),
  "eviction_eligibility_rule" %in% names(domains$evictions))
grid <- readRDS("output/hex_grid.rds")
units <- st_drop_geometry(readRDS("output/corporate_ownership_by_hex.rds"))
counties <- read_csv("config/hex_county_assignment_2024.csv", show_col_types = FALSE)
stopifnot(nrow(grid) == 7027L, !anyDuplicated(units$hex_id), !anyDuplicated(counties$hex_id),
  setequal(grid$hex_id, units$hex_id), setequal(grid$hex_id, counties$hex_id),
  identical(as.character(grid$h3_index), counties$h3_index[match(grid$hex_id, counties$hex_id)]))
support <- st_drop_geometry(grid)[c("hex_id", "area_km2")]
support$residential_units <- units$residential_units[match(support$hex_id, units$hex_id)]
support$source_county <- counties$source_county[match(support$hex_id, counties$hex_id)]
stopifnot(all(c("acs_dollar_base_year", "acs_retrospective_reconstruction", "acs_current_year", "analysis_as_of_date") %in% names(domains$acs)),
  all(domains$acs$acs_dollar_base_year == 2024), all(domains$acs$acs_retrospective_reconstruction),
  all(domains$acs$acs_current_year == ifelse(domains$acs$analysis_as_of_date == as.Date("2025-04-01"), 2023, 2024)))
result <- part2_assemble_feature_matrix(domains, support)
write_pair <- function(x, stem) {
  saveRDS(x, file.path(root, paste0(stem, ".rds")))
  write_csv(x, file.path(root, paste0(stem, ".csv")))
}
write_pair(result$paired, "part2_features_paired")
write_pair(result$eligibility, "part2_eligibility_by_hex")
write_csv(result$exclusions, file.path(root, "part2_exclusions_long.csv"))
write_csv(result$exclusion_summary, file.path(root, "part2_exclusion_summary.csv"))
for (date in names(result$matrices)) write_pair(result$matrices[[date]], paste0("part2_analysis_matrix_", date))
write_csv(sources, file.path(root, "part2_source_registry.csv"))
write_csv(verification, file.path(root, "part2_source_hash_verification.csv"))
county_qa <- result$eligibility %>% group_by(source_county) %>% summarise(audit_hexes = n(),
  current_city_hexes = sum(in_current_city_scope), common_eligible_hexes = sum(common_comparison_ready),
  eligible_boundary_straddling_hexes = sum(common_comparison_ready & boundary_straddling_hex), .groups = "drop")
write_csv(county_qa, file.path(root, "part2_county_eligibility_summary.csv"))
indices <- part2_matrix_indices()
index_qa <- bind_rows(lapply(indices, function(index) {
  a <- result$paired[result$paired$analysis_as_of_date == as.Date("2025-04-01"), ]
  b <- result$paired[result$paired$analysis_as_of_date == as.Date("2026-04-01"), ]
  eligible <- result$eligibility$common_comparison_ready
  signature <- paste0(index, "_availability_signature")
  data.frame(index = index, earlier_available = sum(is.finite(a[[index]])), later_available = sum(is.finite(b[[index]])),
    available_both = sum(is.finite(a[[index]]) & is.finite(b[[index]])), common_sample_hexes = sum(eligible),
    common_sample_changed_term_availability = sum(a[[signature]][eligible] != b[[signature]][eligible]),
    common_sample_mean_2025 = mean(a[[index]][eligible]), common_sample_mean_2026 = mean(b[[index]][eligible]))
}))
write_csv(index_qa, file.path(root, "part2_index_readiness_summary.csv"))
source_after <- build_file_manifest(source_before$path, require_all = TRUE, hash_files = TRUE)
stopifnot(!anyDuplicated(source_before$path), setequal(source_before$path, source_after$path),
  identical(unname(source_before$sha256), unname(source_after$sha256[match(source_before$path, source_after$path)])))
write_csv(source_after, file.path(root, "part2_source_preservation_after.csv"))
after <- build_file_manifest(protected_before$path, require_all = TRUE, hash_files = TRUE)
stopifnot(identical(protected_before$path, after$path), identical(protected_before$sha256, after$sha256))
write_csv(after, file.path(root, "existing_outputs_preservation_after.csv"))
preserved_count <- nrow(protected_before)
manifest <- list(schema_version = 2L, status = "paired_seven_feature_matrix_complete_v2",
  processed_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  analysis_cutoffs = c("2025-04-01", "2026-04-01"), indices = indices,
  audit_hexes = nrow(support), paired_audit_rows = nrow(result$paired),
  common_eligible_hexes = sum(result$eligibility$common_comparison_ready),
  geography = "Fixed canonical integer-ID grid; fixed 2026-04-29 Austin FULL city-center mask; event points also inside current city",
  eligibility = "Both dates: city-center; fixed promoted units>=20; ownership common units>=20 and >=95% unit/parcel coverage; usable 311, demolition and mapped-eviction source coverage; retrospective-usable amenities; all seven indices finite",
  measurement_version = "part2-fixed-components-v2",
  eviction_eligibility_rule = "rolling_scored_24_months_v1",
  partial_components = "All required terms at both dates; fixed equal weights, never average available terms. Amenity categories each require complete exposure inputs",
  scoring = "Consume earlier-frozen 0-100 component scores unchanged; no renormalization on the common sample; final cluster z-scores and centroids not fit",
  exclusion_counts = "Nonexclusive reasons can overlap; primary reasons use the ordered gates in R/part2_feature_matrix.R and sum to the audit grid",
  retrospective_reconstruction = TRUE, cluster_standardization_fitted = FALSE, clustering_fitted = FALSE, ml_work_paused = TRUE,
  limitations = c("Adjacent ACS five-year releases overlap; changes are not independent annual observations",
    "Current housing support and city footprint are fixed retrospectively; ownership common support uses both years",
    "Evictions represent reliably mapped filings; unlocated cases cannot establish full city match rates; absent courts remain unknown",
    "Rent uses the same reliability-qualified BG or tract level across six vintages; historical boundaries are not harmonized",
    "Signed event rate-change scores anchor no change at50; composites are relative measurements, not probabilities or literal no-risk-zero scores",
    "311 is a coordinate-required selected-request universe; historical FULL geography is a POC screen, not independently verified service coverage",
    "Amenity historical source completeness is not proven; retrospective usability is not exhaustive coverage",
    "Boundary-straddling cells retain full-hex areas and unit denominators; rates are not exact city-clipped rates",
    "Common sample is a covered subset, not a complete Austin census or leakage-safe predictive panel"),
  existing_outputs_preserved = preserved_count, source_files_unchanged = TRUE,
  preservation_scope = "Non-Part2 analytical outputs and figures; failed Part2 outputs replaced in place at user request",
  inputs = build_file_manifest(unique(c(sources$feature_path, sources$manifest_path, "output/hex_grid.rds",
    "output/corporate_ownership_by_hex.rds", "config/hex_county_assignment_2024.csv", "R/pipeline.R",
    "R/part2_feature_matrix.R", "scripts/part2/build_feature_matrix.R")), require_all = TRUE, hash_files = TRUE),
  outputs = build_file_manifest(list.files(root, recursive = TRUE, full.names = TRUE, pattern = "[.](rds|csv)$"),
    require_all = TRUE, hash_files = TRUE))
jsonlite::write_json(manifest, file.path(root, "part2_matrix_run_manifest.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
print(result$exclusion_summary)
print(county_qa)
cat("Paired seven-feature matrix complete:", manifest$common_eligible_hexes, "common cells; no clusters fit.\n")
