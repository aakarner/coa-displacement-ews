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
  # Raw inputs, canonical unit/geometry surfaces, historical Part2 products,
  # and Part3 products are never exempted by this rule.
  entries[!replaced,,drop=FALSE]
}
