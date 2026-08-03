library(targets)

# Run with:
# targets::tar_make(script = "_targets_review.R", store = "_targets_review")

source("R/analysis_config.R")
source("R/pipeline.R")

tar_option_set(
  error = "stop",
  memory = "transient",
  garbage_collection = TRUE
)

list(
  tar_target(review_analysis_config, EWS_CONFIG),
  tar_target(
    part1_cluster_selection_review_script,
    "scripts/reviews/part1_cluster_selection.R",
    format = "file"
  ),
  tar_target(
    part1_cluster_selection_review_inputs,
    c(
      "output/hex_features.rds",
      "output/amenity_cluster_sensitivity.rds",
      "output/amenity_cluster_assignments.csv",
      "output/amenity_cluster_metrics.csv",
      "output/amenity_cluster_gap_statistics.csv",
      "output/amenity_cluster_stability.csv"
    ),
    format = "file"
  ),
  tar_target(
    part1_cluster_selection_review,
    run_r_script_stage(
      part1_cluster_selection_review_script,
      c(
        "output/part1/cluster_selection_block_schemes.csv",
        "output/part1/cluster_selection_stability_replicates.csv",
        "output/part1/cluster_selection_stability_summary.csv",
        "output/part1/cluster_selection_stability_by_cluster.csv",
        "output/part1/cluster_selection_assignment_confidence.csv",
        "output/part1/cluster_selection_confidence_by_cluster.csv",
        "output/part1/cluster_selection_profiles.csv",
        "output/part1/cluster_selection_signal_prevalence.csv",
        "output/part1/cluster_selection_signal_separation.csv",
        "output/part1/cluster_selection_spatial_behavior.csv",
        "output/part1/cluster_selection_scorecard.csv",
        "output/part1/cluster_selection_k6_k7_crosswalk.csv",
        "output/part1/cluster_selection_audit.rds",
        "figures/03f_cluster_selection_stability.png",
        "figures/03f_cluster_selection_k6_k7_maps.png",
        "figures/03f_cluster_selection_k6_k7_confidence.png",
        "figures/03f_cluster_selection_k6_k7_profiles.png",
        "figures/03f_cluster_selection_k6_k7_crosswalk.png",
        "figures/03f_cluster_selection_signal_prevalence.png"
      ),
      dependencies = list(
        part1_cluster_selection_review_inputs,
        review_analysis_config
      )
    ),
    format = "file"
  )
)
