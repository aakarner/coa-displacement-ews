# A new unit review must invalidate every selected measure that consumes units.
suppressPackageStartupMessages({library(targets); library(igraph)})
n <- tar_network()
g <- graph_from_data_frame(n$edges, directed = TRUE)
stopifnot(is_dag(g))
consumers <- c("promoted_unit_surface", "corporate_features", "eviction_property_geography",
  "paired_eviction_snapshots", "paired_311_snapshots", "paired_acs_snapshots",
  "part2_ownership_snapshots", "paired_ownership_index", "current_measurement", "current_features")
stopifnot(all(is.finite(distances(g, v = "reviewed_unit_property_inputs", to = consumers, mode = "out"))))
paired <- c("paired_eviction_snapshots", "paired_311_snapshots", "paired_acs_snapshots", "paired_ownership_index")
stopifnot(all(is.finite(distances(g, v = paired, to = "current_measurement", mode = "out"))))
cat("Reviewed-unit inputs invalidate the selected parcel-dependent measures and current features; target graph is acyclic.\n")
