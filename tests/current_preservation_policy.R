# A dated run's before/after evidence remains valid; it does not permanently
# lock a different stage's outputs against an approved subsequent rebuild.
# Only generated current Part1 products are released by decision 0013.
current_preservation_entries <- function(entries) {
  model_path <- "output/part1/baseline_cluster_model.rds"
  if (!file.exists(model_path)) return(entries)
  model <- readRDS(model_path)
  if (!identical(model$measurement_version, "complete-components-v2")) return(entries)
  stopifnot(file.exists("docs/decisions/0013-harmonized-measurement.md"),
    identical(model$measurement_scope, "single_cutoff_current_sample"),
    identical(model$measurement_manifest_sha256,
      digest::digest(file="output/part1/measurement/current_measurement_manifest.json",algo="sha256")),
    identical(model$interpretation$centroids_sha256,digest::digest(model$centroids,algo="sha256")),
    identical(model$interpretation$labels_sha256,digest::digest(file="config/amenity_cluster_labels.csv",algo="sha256")))
  p <- sub(paste0(normalizePath("."),"/"),"",entries$path,fixed=TRUE)
  replaced <- startsWith(p,"output/part1/") |
    p %in% c("output/hex_features.rds","output/feature_list.csv","output/feature_coverage_audit.csv",
      "output/part2/baseline_fixed_cluster_assignments.csv","output/part2/baseline_fixed_cluster_assignment_summary.csv") |
    grepl("^output/amenity_cluster_",p) | grepl("^figures/03[deg]_",p)
  # The older event-stage inventory pins this annual panel. Verify the new
  # policy before releasing that one derived file; retain its original hash.
  if (file.exists("docs/decisions/0014-eviction-ambiguity-keeps-cells.md")) {
    annual <- read.csv("output/eviction_filings_complete_by_hex_year.csv")
    stopifnot(all(annual$eviction_ambiguity_rule == "flag_unassigned_cases_keep_cells_v1"),
      identical(annual$count_observed, annual$source_covered & annual$period_complete),
      identical(annual$has_unassigned_ambiguous_cases, annual$unresolved_candidate_cases > 0L),
      identical(is.na(annual$eviction_cases), !annual$count_observed))
    replaced <- replaced | p == "output/eviction_filings_complete_by_hex_year.csv"
  }
  # Decision 0020 explicitly supersedes these derived aggregations. Raw
  # source caches are deliberately absent from this allowlist. Current source
  # manifests and the independent grid-coverage test validate replacements.
  if (file.exists("docs/decisions/0020-full-purpose-h3-grid.md")) {
    source("R/grid_contract.R")
    stopifnot(grid_contract()$version == "austin_full_20260429_stable_h3_v1")
    grid_products <- c("output/311_requests_by_hex_summary.csv", "output/311_requests_by_hex_summary.rds",
      "output/311_requests_by_hex_year.csv", "output/311_service_request_counts.csv",
      "output/311_service_request_selection.csv", "output/acs_dasymetric_allocation_qa.csv",
      "output/acs_dasymetric_block_hex_allocation.csv", "output/acs_dasymetric_block_hex_allocation.rds",
      "output/acs_dasymetric_hex_bg_crosswalk.csv", "output/acs_dasymetric_hex_bg_crosswalk.rds",
      "output/acs_demographics_by_hex.csv", "output/acs_demographics_by_hex.rds",
      "output/acs_rent_by_hex_vintage.csv", "output/acs_rent_by_hex_vintage.rds",
      "output/acs_rent_dasymetric_crosswalk_qa.csv", "output/acs_rent_dominant_sources_by_hex_vintage.csv",
      "output/acs_rent_trends_by_hex.csv", "output/acs_rent_trends_by_hex.rds",
      "output/amenity_change_features_by_hex.csv", "output/amenity_change_features_by_hex.rds",
      "output/amenity_events_geocoded.rds", "output/amenity_geocoding_method_qa.csv",
      "output/amenity_geocoding_qa.csv", "output/amenity_hex_distribution_qa.csv",
      "output/demolition_permits_annual_qa.csv", "output/demolition_permits_by_hex_year.csv",
      "output/demolition_permits_source_qa.csv", "output/demolition_permits_unmatched_qa.csv",
      "output/hex_grid.rds", "output/part2/acs/acs_features_paired.rds",
      "output/part2/amenities/amenity_features_paired.rds", "output/part2/ownership/ownership_common_support_by_hex_year.rds",
      "output/part3/demolition_coverage_current_snapshot_qa.csv", "output/part3/demolition_coverage_current_snapshot_summary.csv",
      "output/part3/demolition_historical_coverage_by_hex_year.csv",
      "output/part3/demolition_panel_source_manifest.csv")
    # The separately regenerated H3 demonstration consumes that same grid
    # and adopted boundary; it does not alter fitted analysis artifacts.
    grid_products <- c(grid_products, "figures/10_hex_grid_city_boundary.png",
      "figures/10_hex_grid_city_boundary.pdf")
    replaced <- replaced | p %in% grid_products
  }
  # Other files remain protected by their dated inventories.
  entries[!replaced,,drop=FALSE]
}
