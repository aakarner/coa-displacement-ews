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
  ),
  tar_target(
    code_complaint_audit_script,
    "scripts/audits/311_code_complaints.R",
    format = "file"
  ),
  tar_target(
    code_complaint_audit_inputs,
    c(
      "config/311_code_complaint_categories.csv",
      "data/austin_land_use_inventory_202607.csv",
      "output/hex_grid.rds",
      "output/residential_parcels_unit_promoted.rds",
      "output/311_requests_by_hex_summary.rds",
      "output/part1/baseline_cluster_assignments.csv"
    ),
    format = "file"
  ),
  tar_target(
    code_complaint_audit,
    run_r_script_stage(
      code_complaint_audit_script,
      c(
        "output/311_code_complaint_source_audit.csv",
        "output/311_code_complaint_linkage_methods.csv",
        "output/311_code_complaint_year_qa.csv",
        "output/311_code_complaint_month_qa.csv",
        "output/311_code_complaint_overlap_period_qa.csv",
        "output/311_code_complaint_hex_sparsity_audit.csv",
        "output/311_code_complaint_unfiltered_311_types.csv",
        "output/311_code_complaint_cardinality_audit.csv",
        "output/311_code_complaint_case_audit.rds"
      ),
      dependencies = list(
        code_complaint_audit_inputs,
        review_analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    high_risk_island_review_script,
    "scripts/reviews/high_risk_islands.R",
    format = "file"
  ),
  tar_target(
    high_risk_island_review_inputs,
    c(
      "config/high_risk_island_parameters.csv",
      "data/Issued_Construction_Permits_20260401.csv",
      "data/austin_land_use_inventory_202607.csv",
      "data/neighborhood_reporting_areas.geojson",
      "data/BOUNDARIES_jurisdictions_20260429.geojson",
      "output/hex_features.rds",
      "output/part1/baseline_cluster_assignments.csv",
      "output/part1/baseline_cluster_model.rds",
      "output/residential_parcels_unit_promoted.rds",
      "output/residential_unit_project_membership.rds",
      "output/eviction_filings_full_geocoded_hex.rds"
    ),
    format = "file"
  ),
  tar_target(
    high_risk_island_review,
    run_r_script_stage(
      high_risk_island_review_script,
      c(
        "output/part1/high_risk_island_hex_summary.csv",
        "output/part1/high_risk_island_neighbor_context.csv",
        "output/part1/high_risk_island_property_drivers.csv",
        "output/part1/high_risk_island_top_properties.csv",
        "output/part1/high_risk_island_counterfactuals.csv",
        "output/part1/high_risk_island_unmatched_events.csv",
        "output/part1/high_risk_island_attribution_qa.csv",
        "output/part1/high_risk_island_attribution_coverage.csv",
        "output/part1/high_risk_island_review_metrics.csv",
        "output/part1/high_risk_island_review.rds",
        "figures/03h_high_risk_islands.png",
        "figures/03h_high_risk_islands_interactive.html"
      ),
      dependencies = list(
        high_risk_island_review_inputs,
        code_complaint_audit,
        review_analysis_config
      )
    ),
    format = "file"
  )
)
