# Assemble paired ownership scores from existing, pinned ownership summaries.
# No network, classifier, parcel rebuilding, or writes to the source directory.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
source("R/pipeline.R")
source("R/part2_ownership_index.R")
source("R/part2_index_scoring.R")

source_root <- "output/part2/ownership"
output_root <- "output/part2/ownership_index"
# Load before any overwrite; v1 and v2 both preserve unchanged baseline bounds.
preserved_scaling_path <- file.path(output_root, "ownership_scaling.rds")
preserved_scaling <- if (file.exists(preserved_scaling_path)) readRDS(preserved_scaling_path) else NULL
manifest_path <- file.path(source_root, "ownership_snapshot_manifest.json")
expected_manifest_sha256 <- "15f4331043892297e2f68ce90404efb20b571c8ec1f679ad4ea1bff814391fb1"
if (!file.exists(manifest_path) || !identical(digest::digest(file = manifest_path, algo = "sha256"),
    expected_manifest_sha256)) stop("Reviewed ownership snapshot manifest changed; review before rebuilding the index.")
upstream <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
specification <- upstream$specification
tax_years <- c(2024L, 2025L)
cutoffs <- as.Date(c("2025-04-01", "2026-04-01"))
components <- part2_ownership_components()
index_name <- "ownership_pressure_index"
if (!identical(as.integer(unlist(specification$tax_years)), tax_years) ||
    !identical(as.Date(unlist(specification$analysis_cutoffs)), cutoffs) ||
    specification$minimum_hex_units != 20 || specification$minimum_common_coverage != .95 ||
    specification$unit_field != "units_calibrated_targeted" ||
    upstream$promoted_unit_version != "v2_2026-07-31_land_use_validated" ||
    upstream$checks$comparison_ready_hexes != 3227L) stop("Reviewed ownership support contract changed.")
cat("Verifying the pinned ownership source, code and output checksums...\n")
pin_entries <- part2_ownership_manifest_entries(upstream)
verification <- part2_ownership_verify_hashes(pin_entries)
source_preservation <- build_file_manifest(source_root, recursive = TRUE, require_all = TRUE, hash_files = TRUE)
grid <- readRDS("output/hex_grid.rds")
if (!inherits(grid, "sf") || nrow(grid) != 7027L || !is.integer(grid$hex_id)) {
  stop("Expected the canonical 7,027-cell grid with integer identifiers.")
}
common_path <- file.path(source_root, "ownership_common_support_by_hex_year.rds")
full_path <- file.path(source_root, "ownership_features_by_hex_year.rds")
variant_files <- c(certified_only = "ownership_certified_only_hex_change.csv",
  source_agreement = "ownership_source_agreement_hex_change.csv",
  prior_2024_certified_only = "ownership_2024_certified_only_hex_change.csv")
variants <- lapply(variant_files, function(path) read_csv(file.path(source_root, path), show_col_types = FALSE))
variant_summary <- read_csv(file.path(source_root, "ownership_source_variant_summary.csv"), show_col_types = FALSE)
if (!identical(variant_summary$comparison_ready_hexes[variant_summary$source_variant == "main"], 3227)) {
  stop("Primary ownership source variant summary changed.")
}
panel <- part2_prepare_ownership_index(readRDS(common_path), readRDS(full_path), grid, variants,
  tax_years, cutoffs, specification$minimum_hex_units, specification$minimum_common_coverage)
panel$ownership_unit_field <- specification$unit_field
panel$ownership_promoted_unit_version <- upstream$promoted_unit_version
panel$ownership_classifier_commit <- specification$upstream_commit
panel$ownership_classification_rule_version <- specification$classification_rule_version
panel$ownership_source_manifest_sha256 <- expected_manifest_sha256
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
features_by_date <- list(); summaries <- list(); scaling_qa <- list(); sensitivity_qa <- list()
scaling <- NULL
for (i in seq_along(cutoffs)) {
  cutoff <- cutoffs[[i]]
  date_root <- file.path(output_root, as.character(cutoff))
  dir.create(date_root, recursive = TRUE, showWarnings = FALSE)
  features <- panel[panel$analysis_as_of_date == cutoff, , drop = FALSE]
  rownames(features) <- NULL
  if (sum(features$ownership_comparison_ready) != 3227L ||
      !identical(features$hex_id, grid$hex_id)) stop("The provisional ownership screen or canonical row order changed.")
  if (is.null(scaling)) scaling <- part2_fit_index_scaling(features, components, index_name, cutoffs[[1]], preserved_scaling = preserved_scaling)
  scored <- part2_apply_index_scaling(features, scaling)
  features <- scored$features
  features$ownership_scaling_reference_as_of_date <- cutoffs[[1]]
  features$ownership_scaling_mode <- if (i == 1L && !is.null(preserved_scaling)) "preserved_earlier_bounds" else if (i == 1L) "fit_this_vintage" else "frozen_reference"
  ready <- features$ownership_comparison_ready
  if (any(!is.finite(features[[index_name]][ready])) || any(!is.na(features[[index_name]][!ready])) ||
      any(features$ownership_pressure_index_components_available != ifelse(ready, 3L, 0L))) {
    stop("Ownership scores do not respect the complete common-support mask.")
  }
  saveRDS(features, file.path(date_root, "ownership_features_by_hex.rds"))
  write_csv(features, file.path(date_root, "ownership_features_by_hex.csv"))
  saveRDS(scaling, file.path(date_root, "ownership_scaling.rds"))
  write_csv(scaling$bounds, file.path(date_root, "ownership_scaling_bounds.csv"))
  scored$qa$analysis_as_of_date <- cutoff
  write_csv(scored$qa, file.path(date_root, "ownership_scaling_qa.csv"))
  coverage_qa <- features %>% count(analysis_as_of_date, ownership_coverage_status, ownership_comparison_ready, name = "hexes")
  write_csv(coverage_qa, file.path(date_root, "ownership_coverage_qa.csv"))
  summaries[[i]] <- data.frame(analysis_as_of_date = cutoff, tax_year = tax_years[[i]],
    hexes = nrow(features), residential_hexes = sum(features$ownership_has_residential_support),
    hexes_with_common_support = sum(features$common_parcels > 0), comparison_ready_hexes = sum(ready),
    unavailable_hexes = sum(!ready), common_parcels = sum(features$common_parcels),
    common_units = sum(features$common_units), comparison_ready_common_parcels = sum(features$common_parcels[ready]),
    comparison_ready_common_units = sum(features$common_units[ready]),
    full_support_parcels = sum(features$residential_parcels), full_support_units = sum(features$residential_units),
    full_support_corporate_unknown_parcels = sum(features$full_corporate_unknown_parcels, na.rm = TRUE),
    full_support_corporate_unknown_units = sum(features$full_corporate_unknown_units, na.rm = TRUE),
    full_support_financialized_unknown_parcels = sum(features$full_financialized_unknown_parcels, na.rm = TRUE),
    ready_hexes_with_full_support_unknowns = sum(ready &
      (features$full_corporate_unknown_parcels > 0 | features$full_financialized_unknown_parcels > 0), na.rm = TRUE),
    observed_zero_index_hexes = sum(features[[index_name]] == 0, na.rm = TRUE),
    score_min = min(features[[index_name]], na.rm = TRUE), score_max = max(features[[index_name]], na.rm = TRUE))
  sensitivity_qa[[i]] <- bind_rows(lapply(names(variants), function(variant) {
    prefix <- paste0("ownership_", variant, "_")
    data.frame(analysis_as_of_date = cutoff, source_variant = variant,
      comparison_ready_hexes = sum(features[[paste0(prefix, "comparison_ready")]]),
      readiness_differs_from_main_hexes = sum(features[[paste0(prefix, "readiness_differs_from_main")]]),
      components_differ_from_main_hexes = sum(features[[paste0(prefix, "components_differ_from_main")]]),
      main_ready_components_differ_hexes = sum(ready & features[[paste0(prefix, "components_differ_from_main")]]))
  }))
  features_by_date[[i]] <- features
  scaling_qa[[i]] <- scored$qa
}
paired <- bind_rows(features_by_date)
changes <- part2_ownership_changes(features_by_date[[1]], features_by_date[[2]])
saveRDS(paired, file.path(output_root, "ownership_features_paired.rds"))
write_csv(paired, file.path(output_root, "ownership_features_paired.csv"))
saveRDS(changes, file.path(output_root, "ownership_feature_changes_by_hex.rds"))
write_csv(changes, file.path(output_root, "ownership_feature_changes_by_hex.csv"))
saveRDS(scaling, file.path(output_root, "ownership_scaling.rds"))
write_csv(scaling$bounds, file.path(output_root, "ownership_scaling_bounds.csv"))
write_csv(bind_rows(scaling_qa), file.path(output_root, "ownership_scaling_qa.csv"))
write_csv(bind_rows(summaries), file.path(output_root, "ownership_snapshot_summary.csv"))
write_csv(bind_rows(sensitivity_qa), file.path(output_root, "ownership_source_sensitivity_qa.csv"))
write_csv(variant_summary, file.path(output_root, "ownership_source_variant_summary.csv"))
write_csv(verification, file.path(output_root, "ownership_upstream_hash_verification.csv"))
write_csv(data.frame(component = components, units = c("percent_0_to_100", "units_per_square_kilometer", "percent_0_to_100"),
  numerator = c("known_corporate_units_in_common_cohort", "known_corporate_units_in_common_cohort", "known_financialized_parcels_in_common_cohort"),
  denominator = c("common_cohort_validated_units", "canonical_full_hex_area_km2", "common_cohort_parcels"),
  multiplier = c(100, 1, 100), equal_component_weight = rep(1 / 3, 3)), file.path(output_root, "ownership_component_dictionary.csv"))

# Recheck the entire old ownership directory, including its manifest, after all
# writes; the old sources and outputs remain a read-only dependency of this stage.
preserved <- part2_ownership_verify_hashes(source_preservation)
current_paths <- build_file_manifest(source_root, recursive = TRUE, require_all = TRUE)$path
if (!setequal(current_paths, source_preservation$path)) stop("The source ownership directory file set changed.")
write_csv(preserved, file.path(output_root, "ownership_source_directory_preservation.csv"))
code_paths <- c("R/pipeline.R", "R/part2_ownership_index.R", "R/part2_index_scoring.R", "scripts/part2/build_ownership_index.R")
manifest <- list(schema_version = 2L, status = "paired_ownership_index_complete_v2",
  generated_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  cutoffs = as.character(cutoffs), tax_years = tax_years, source_variant = "main", retrospective = TRUE,
  source_manifest_sha256 = expected_manifest_sha256, source_specification = specification,
  source_variant_semantics = upstream$semantics$source_variants,
  sensitivity_caution = "Source variants rebuild their own common cohorts; component differences are not classification-only effects on identical parcels",
  support = "Same jointly known parcels, fixed coordinates and validated units; provisional 20-unit/95-percent coverage screen",
  coverage_caution = "Common-cohort coverage is evidence completeness, not representativeness; full-support unknown ownership is retained",
  cutoff_semantics = specification$semantics,
  component_denominators = c("common cohort units", "full canonical hex area", "common cohort parcels"),
  scoring = "Fixed equal thirds; reviewed earlier p1/p99 component bounds retained and frozen across dates; all three components required, unknown is never zero.",
  measurement_version = "fixed_components_v2",
  component_missing_policy = "all_components_required_fixed_equal_weights",
  required_components = components, scaling_reference_as_of_date = as.character(cutoffs[[1]]),
  no_cluster_or_ml_changes = TRUE, verified_pinned_upstream_files = nrow(verification),
  preserved_source_directory_files = nrow(preserved),
  inputs = build_file_manifest(c(manifest_path, common_path, full_path, "output/hex_grid.rds",
    file.path(source_root, variant_files), file.path(source_root, "ownership_source_variant_summary.csv"), code_paths),
    require_all = TRUE, hash_files = TRUE),
  pinned_upstream_files = pin_entries,
  outputs = build_file_manifest(output_root, recursive = TRUE, require_all = TRUE, hash_files = TRUE))
manifest$outputs <- manifest$outputs[basename(manifest$outputs$path) != "ownership_index_run_manifest.json", ]
jsonlite::write_json(manifest, file.path(output_root, "ownership_index_run_manifest.json"),
  auto_unbox = TRUE, pretty = TRUE, na = "null", digits = NA)
print(bind_rows(summaries))
cat("Paired ownership index complete; original ownership outputs unchanged.\n")
