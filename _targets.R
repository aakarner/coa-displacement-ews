library(targets)

source("R/analysis_config.R")
source("R/pipeline.R")
source("R/cluster_assignment.R")
source("R/forecast_spec.R")

tar_option_set(
  error = "stop",
  memory = "transient",
  garbage_collection = TRUE
)

list(
  tar_target(analysis_config, EWS_CONFIG),

  # Fast metadata manifests detect changed source files without hashing large
  # appraisal archives on every pipeline startup.
  tar_target(
    parcel_input_manifest,
    build_file_manifest(
      c(
        "data/residential_parcels_for_hex.csv",
        "data/hays_residential_parcels_for_hex.csv",
        "data/williamson_residential_parcels_for_hex.csv",
        "data/corporate_owned_parcels.csv",
        "data/CoStarHistoric-clean.csv",
        "data/geocoded_buildings.csv",
        "data/BOUNDARIES_jurisdictions_20260429.geojson",
        "data/raw_parcels/unit_sources",
        "data/raw_parcels/williamson",
        "config/residual_unit_parcel_reviews.csv",
        "config/williamson_project_groups.csv",
        "config/williamson_unit_validation_sources.csv"
      ),
      recursive = TRUE
    ),
    cue = tar_cue(mode = "always")
  ),
  tar_target(
    appraisal_input_manifest,
    build_file_manifest(
      c(
        "data/raw_parcels/appraisal_history",
        "config/appraisal_sources.csv"
      ),
      recursive = TRUE
    ),
    cue = tar_cue(mode = "always")
  ),
  tar_target(
    eviction_input_manifest,
    build_file_manifest(
      c(
        "data/Alex Karner Eviction Report 1-1-20 to 5-22-26 (1).xlsx",
        paste0(
          "data/Odyssey-JobOutput-May 20, 2026 ",
          "16-54-45-3728695-1 (1).xlsx"
        ),
        "output/eviction_addresses_geocoded.csv"
      )
    ),
    cue = tar_cue(mode = "always")
  ),
  tar_target(
    demolition_input_manifest,
    build_file_manifest("data/Issued_Construction_Permits_20260401.csv"),
    cue = tar_cue(mode = "always")
  ),
  tar_target(
    amenity_input_manifest,
    build_file_manifest(
      c(
        "data/raw_amenities",
        "config/amenity_categories.csv"
      ),
      recursive = TRUE
    ),
    cue = tar_cue(mode = "always")
  ),
  tar_target(
    acs_cache_manifest,
    build_file_manifest("data/raw_acs", recursive = TRUE),
    cue = tar_cue(mode = "always")
  ),
  tar_target(
    landlord_mapper_manifest,
    build_file_manifest(
      c(
        "../landlord-mapper/output/property_profile.csv",
        "../landlord-mapper/output/travis_deeds.csv",
        "../landlord-mapper/output/austin_parcel_year_land_transactions.csv"
      )
    ),
    cue = tar_cue(mode = "always")
  ),
  tar_target(
    method_config_manifest,
    build_file_manifest(
      c(
        "R/analysis_config.R",
        "config/feature_dictionary.csv",
        "config/amenity_cluster_labels_k6.csv",
        "config/forecast_outcomes.csv"
      )
    ),
    cue = tar_cue(mode = "always")
  ),

  # Base geography.
  tar_target(grid_script, "01_create_hex_grid.R", format = "file"),
  tar_target(
    hex_grid,
    run_r_script_stage(
      grid_script,
      c(
        "output/hex_grid.rds",
        "figures/01_hex_grid_static.png"
      ),
      dependencies = analysis_config
    ),
    format = "file"
  ),

  # Residential unit hierarchy. This is upstream of corporate ownership,
  # dasymetric ACS allocation, rates, and clustering eligibility.
  tar_target(
    unit_calibration_script,
    "scripts/data/parcel_units_calibrate.R",
    format = "file"
  ),
  tar_target(
    unit_calibration,
    run_r_script_stage(
      unit_calibration_script,
      "output/residential_parcels_unit_calibrated.rds",
      dependencies = list(
        hex_grid,
        parcel_input_manifest,
        method_config_manifest
      )
    ),
    format = "file"
  ),
  tar_target(
    unit_validation_script,
    "scripts/data/parcel_units_validate.R",
    format = "file"
  ),
  tar_target(
    unit_validation,
    run_r_script_stage(
      unit_validation_script,
      "output/residential_parcels_unit_targeted.rds",
      dependencies = list(
        unit_calibration,
        parcel_input_manifest,
        method_config_manifest
      )
    ),
    format = "file"
  ),
  tar_target(
    unit_sources_script,
    "scripts/data/unit_counts/prepare_sources.R",
    format = "file"
  ),
  tar_target(
    unit_sources,
    run_r_script_stage(
      unit_sources_script,
      c(
        "output/residential_parcels_unit_source_attributes.rds",
        "output/residential_unit_source_records.rds",
        "output/residential_unit_source_parcel_links.rds"
      ),
      dependencies = list(
        unit_calibration,
        parcel_input_manifest,
        landlord_mapper_manifest
      )
    ),
    format = "file"
  ),
  tar_target(
    unit_projects_script,
    "scripts/data/unit_counts/build_projects.R",
    format = "file"
  ),
  tar_target(
    unit_projects,
    run_r_script_stage(
      unit_projects_script,
      c(
        "output/residential_unit_project_membership.rds",
        "output/residential_unit_projects.rds",
        "output/residential_unit_training_table.rds"
      ),
      dependencies = list(unit_sources, method_config_manifest)
    ),
    format = "file"
  ),
  tar_target(
    unit_models_script,
    "scripts/data/unit_counts/fit_models.R",
    format = "file"
  ),
  tar_target(
    unit_models,
    run_r_script_stage(
      unit_models_script,
      c(
        "output/residential_unit_count_models.rds",
        "output/residential_unit_model_predictions.rds"
      ),
      dependencies = unit_projects
    ),
    format = "file"
  ),
  tar_target(
    williamson_validation_script,
    "scripts/data/unit_counts/validate_williamson.R",
    format = "file"
  ),
  tar_target(
    williamson_validation,
    run_r_script_stage(
      williamson_validation_script,
      "output/residential_unit_williamson_validation.rds",
      dependencies = list(
        unit_projects,
        unit_models,
        parcel_input_manifest,
        method_config_manifest
      )
    ),
    format = "file"
  ),
  tar_target(
    unit_integration_script,
    "scripts/data/unit_counts/build_integration.R",
    format = "file"
  ),
  tar_target(
    unit_integration,
    run_r_script_stage(
      unit_integration_script,
      c(
        "output/residential_parcels_unit_shadow_integrated.rds",
        "output/corporate_ownership_by_hex_unit_shadow.rds"
      ),
      dependencies = list(
        hex_grid,
        unit_validation,
        unit_projects,
        unit_models,
        williamson_validation
      )
    ),
    format = "file"
  ),
  tar_target(
    unit_promotion_script,
    "scripts/data/unit_counts/promote_integration.R",
    format = "file"
  ),
  tar_target(
    promoted_unit_surface,
    run_r_script_stage(
      unit_promotion_script,
      c(
        "output/residential_parcels_unit_promoted.rds",
        "output/residential_unit_promotion_manifest.csv"
      ),
      dependencies = list(unit_validation, unit_integration)
    ),
    format = "file"
  ),

  # Source-specific current and historical streams.
  tar_target(
    corporate_script,
    "scripts/data/corporate_ownership.R",
    format = "file"
  ),
  tar_target(
    corporate_features,
    run_r_script_stage(
      corporate_script,
      c(
        "output/corporate_ownership_by_hex.rds",
        "output/residential_parcels_for_hex_sf.rds"
      ),
      dependencies = list(
        hex_grid,
        promoted_unit_surface,
        parcel_input_manifest
      )
    ),
    format = "file"
  ),
  tar_target(
    acs_demographics_script,
    "scripts/data/acs_demographics.R",
    format = "file"
  ),
  tar_target(
    acs_demographics,
    run_r_script_stage(
      acs_demographics_script,
      "output/acs_demographics_by_hex.rds",
      dependencies = list(
        hex_grid,
        corporate_features,
        acs_cache_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    acs_rent_script,
    "scripts/data/acs_rent_history.R",
    format = "file"
  ),
  tar_target(
    acs_rent_history,
    run_r_script_stage(
      acs_rent_script,
      c(
        "output/acs_rent_by_hex_vintage.rds",
        "output/acs_rent_trends_by_hex.rds"
      ),
      dependencies = list(
        hex_grid,
        corporate_features,
        acs_cache_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    eviction_prepare_script,
    "scripts/data/evictions_prepare.R",
    format = "file"
  ),
  tar_target(
    prepared_evictions,
    run_r_script_stage(
      eviction_prepare_script,
      c(
        "output/eviction_filings_prepared_for_geocoding.csv",
        "output/eviction_unique_addresses_for_geocoding.csv"
      ),
      dependencies = eviction_input_manifest
    ),
    format = "file"
  ),
  tar_target(
    eviction_process_script,
    "scripts/data/evictions_process.R",
    format = "file"
  ),
  tar_target(
    eviction_features,
    run_r_script_stage(
      eviction_process_script,
      c(
        "output/eviction_filings_by_hex_summary.rds",
        "output/eviction_filings_by_hex_year.csv"
      ),
      dependencies = list(
        hex_grid,
        prepared_evictions,
        eviction_input_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    requests_311_script,
    "scripts/data/austin_311.R",
    format = "file"
  ),
  tar_target(
    requests_311,
    run_cached_api_stage(
      requests_311_script,
      c(
        "output/311_requests_by_hex_summary.rds",
        "output/311_requests_by_hex_year.csv"
      ),
      dependencies = list(hex_grid, analysis_config),
      required_environment = c(
        "AUSTIN_DATA_API_KEY",
        "AUSTIN_DATA_API_SECRET"
      ),
      expected_as_of = analysis_config$analysis_as_of_date,
      as_of_field = "sr_311_analysis_as_of"
    ),
    format = "file"
  ),
  tar_target(
    appraisal_history_script,
    "scripts/data/appraisal_history.R",
    format = "file"
  ),
  tar_target(
    appraisal_history,
    run_r_script_stage(
      appraisal_history_script,
      c(
        "output/appraisal_values_by_parcel_year.rds",
        "output/appraisal_values_by_hex_year.rds"
      ),
      dependencies = list(
        corporate_features,
        appraisal_input_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    appraisal_adjustment_script,
    "scripts/data/appraisal_adjusted_trends.R",
    format = "file"
  ),
  tar_target(
    appraisal_adjusted_features,
    run_r_script_stage(
      appraisal_adjustment_script,
      "output/appraisal_adjusted_trends_by_hex.rds",
      dependencies = list(appraisal_history, analysis_config)
    ),
    format = "file"
  ),
  tar_target(
    ownership_audit_script,
    "scripts/audits/ownership_transactions.R",
    format = "file"
  ),
  tar_target(
    ownership_source_audit,
    run_r_script_stage(
      ownership_audit_script,
      "output/ownership_transaction_source_audit.csv",
      dependencies = list(
        unit_validation,
        appraisal_input_manifest,
        landlord_mapper_manifest
      )
    ),
    format = "file"
  ),
  tar_target(
    ownership_process_script,
    "scripts/data/ownership_transactions.R",
    format = "file"
  ),
  tar_target(
    ownership_transaction_features,
    run_r_script_stage(
      ownership_process_script,
      "output/ownership_transaction_features_by_hex.rds",
      dependencies = list(
        corporate_features,
        ownership_source_audit,
        appraisal_input_manifest,
        landlord_mapper_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    amenity_audit_script,
    "scripts/audits/amenity_sources.R",
    format = "file"
  ),
  tar_target(
    amenity_source_audit,
    run_r_script_stage(
      amenity_audit_script,
      "output/amenity_source_candidates.rds",
      dependencies = list(
        amenity_input_manifest,
        method_config_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    amenity_process_script,
    "scripts/data/amenities.R",
    format = "file"
  ),
  tar_target(
    amenity_features,
    run_r_script_stage(
      amenity_process_script,
      "output/amenity_change_features_by_hex.rds",
      dependencies = list(
        hex_grid,
        amenity_source_audit,
        amenity_input_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),

  # Shared feature layer for the baseline vintage.
  tar_target(
    current_features_script,
    "scripts/features/build_current_features.R",
    format = "file"
  ),
  tar_target(
    current_features,
    run_r_script_stage(
      current_features_script,
      c(
        "output/hex_features.rds",
        "output/feature_list.csv"
      ),
      dependencies = list(
        hex_grid,
        corporate_features,
        acs_demographics,
        acs_rent_history,
        eviction_features,
        requests_311,
        appraisal_adjusted_features,
        ownership_transaction_features,
        amenity_features,
        demolition_input_manifest,
        method_config_manifest
      )
    ),
    format = "file"
  ),
  tar_target(
    feature_audit_script,
    "scripts/audits/features.R",
    format = "file"
  ),
  tar_target(
    feature_audit,
    run_r_script_stage(
      feature_audit_script,
      "output/feature_coverage_audit.csv",
      dependencies = list(current_features, method_config_manifest)
    ),
    format = "file"
  ),

  # Part 1: fit the baseline typology and freeze every transformation required
  # to assign future vintages without redefining the clusters.
  tar_target(
    part1_cluster_script,
    "scripts/part1/fit_baseline_clusters.R",
    format = "file"
  ),
  tar_target(
    part1_cluster_analysis,
    run_r_script_stage(
      part1_cluster_script,
      c(
        "output/amenity_cluster_sensitivity.rds",
        "output/amenity_cluster_assignments.csv",
        "output/amenity_cluster_recommendations.csv"
      ),
      dependencies = list(
        current_features,
        feature_audit,
        method_config_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    part1_baseline_model,
    {
      current_features
      part1_cluster_analysis
      method_config_manifest
      freeze_baseline_cluster_model(
        feature_file = "output/hex_features.rds",
        cluster_results_file = "output/amenity_cluster_sensitivity.rds",
        label_file = "config/amenity_cluster_labels_k6.csv",
        output_file = "output/part1/baseline_cluster_model.rds",
        config = analysis_config
      )
    },
    format = "file"
  ),
  tar_target(
    part1_visualization_script,
    "scripts/part1/visualize_baseline_clusters.R",
    format = "file"
  ),
  tar_target(
    part1_visualizations,
    run_r_script_stage(
      part1_visualization_script,
      c(
        "figures/03e_amenity_clusters_tentative.png",
        "figures/03e_amenity_clusters_interactive.html"
      ),
      dependencies = list(
        current_features,
        part1_cluster_analysis,
        method_config_manifest
      )
    ),
    format = "file"
  ),

  # Part 2: self-reassignment is the first validation of the frozen model.
  # Future-vintage feature targets will feed the same assignment function.
  tar_target(
    part2_baseline_assignment,
    {
      current_features
      part1_cluster_analysis
      write_baseline_assignment_audit(
        feature_file = "output/hex_features.rds",
        model_file = part1_baseline_model,
        cluster_results_file = "output/amenity_cluster_sensitivity.rds",
        assignment_file =
          "output/part2/baseline_fixed_cluster_assignments.csv",
        summary_file =
          "output/part2/baseline_fixed_cluster_assignment_summary.csv"
      )
    },
    format = "file"
  ),

  # Part 3: define the four future proxy outcomes and report what historical
  # hex-year artifacts remain to be built before supervised forecasting.
  tar_target(
    forecast_outcome_spec,
    "config/forecast_outcomes.csv",
    format = "file"
  ),
  tar_target(
    part3_forecast_readiness,
    {
      eviction_features
      acs_rent_history
      appraisal_history
      build_forecast_readiness(
        outcome_spec_file = forecast_outcome_spec,
        output_file = "output/part3/forecast_readiness.csv",
        config = analysis_config,
        source_files = c(
          eviction_filings =
            "output/eviction_filings_by_hex_year.csv",
          residential_demolitions =
            "output/demolition_permits_by_hex_year.csv",
          rent_growth =
            "output/acs_rent_by_hex_vintage.rds",
          land_value_growth =
            "output/appraisal_values_by_hex_year.rds"
        )
      )
    },
    format = "file"
  )
)
