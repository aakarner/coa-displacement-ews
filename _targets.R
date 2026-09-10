library(targets)

source("R/analysis_config.R")
source("R/pipeline.R")
source("R/cluster_assignment.R")
source("R/forecast_labels.R")
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
    eviction_source_config,
    "config/eviction_sources.csv",
    format = "file"
  ),
  tar_target(
    eviction_input_manifest,
    {
      source_rows <- utils::read.csv(
        eviction_source_config,
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      build_file_manifest(
        unique(source_rows$path),
        require_all = TRUE,
        hash_files = TRUE
      )
    },
    cue = tar_cue(mode = "always")
  ),
  tar_target(
    eviction_geocode_input,
    "output/eviction_addresses_geocoded.csv",
    format = "file"
  ),
  tar_target(
    current_austin_jurisdiction_boundary,
    "data/BOUNDARIES_jurisdictions_20260429.geojson",
    format = "file"
  ),
  tar_target(
    williamson_address_reference_download_script,
    "scripts/data/williamson_address_reference_download.R",
    format = "file"
  ),
  tar_target(
    williamson_address_reference_input,
    {
      williamson_address_reference_download_script
      reference_paths <- c(
        "output/williamson_address_reference.csv",
        "output/williamson_address_reference_qa.csv"
      )
      if (!all(file.exists(reference_paths))) {
        stop(
          "Williamson public address reference is missing. Run: ",
          "WILLIAMSON_ADDRESS_REFERENCE_NETWORK=true Rscript ",
          "scripts/data/williamson_address_reference_download.R",
          call. = FALSE
        )
      }
      reference_qa <- utils::read.csv(
        reference_paths[[2]],
        stringsAsFactors = FALSE
      )
      qa_values <- stats::setNames(
        as.character(reference_qa$value),
        reference_qa$metric
      )
      grid_path <- hex_grid[basename(hex_grid) == "hex_grid.rds"]
      county_path <- hex_county_assignment_reference[
        basename(hex_county_assignment_reference) ==
          "hex_county_assignment_2024.csv"
      ]
      if (
        length(grid_path) != 1L ||
          length(county_path) != 1L ||
          !identical(
            unname(qa_values[["grid_sha256"]]),
            digest::digest(
              grid_path,
              algo = "sha256",
              file = TRUE
            )
          ) ||
          !identical(
            unname(qa_values[["county_reference_sha256"]]),
            digest::digest(
              county_path,
              algo = "sha256",
              file = TRUE
            )
          )
      ) {
        stop(
          "Williamson public address reference does not match the current ",
          "grid/county reference. Refresh it with the documented downloader.",
          call. = FALSE
        )
      }
      reference_paths
    },
    format = "file"
  ),
  tar_target(
    demolition_input_manifest,
    build_file_manifest(
      c(
        "data/Issued_Construction_Permits_20260401.csv",
        "data/BOUNDARIES_jurisdictions_20260429.geojson",
        "data/Austin_Historical_Jurisdiction_Baselines_20260904.geojson",
        "data/Austin_Annexation_History_20260904.geojson",
        "config/demolition_jurisdiction_sources.csv"
      ),
      require_all = TRUE,
      hash_files = TRUE
    ),
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
    land_use_input_manifest,
    build_file_manifest(
      c(
        "data/austin_land_use_inventory_202607.csv",
        "data/austin_land_use_inventory_202607.geojson",
        "data/raw_acs/acsdt1y2024-b25024.dat",
        "data/BOUNDARIES_jurisdictions_20260429.geojson"
      )
    ),
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
    feature_dictionary,
    "config/feature_dictionary.csv",
    format = "file"
  ),
  tar_target(
    land_use_codes,
    "config/austin_land_use_codes.csv",
    format = "file"
  ),
  tar_target(
    cluster_labels,
    "config/amenity_cluster_labels.csv",
    format = "file"
  ),
  tar_target(cluster_interpretation, "config/part1_cluster_interpretation.json", format = "file"),
  tar_target(
    neighborhood_reference_files,
    c(
      "data/neighborhood_reporting_areas.geojson",
      "data/neighborhood_reporting_areas_metadata.json",
      "data/BOUNDARIES_jurisdictions_20260429.geojson"
    ),
    format = "file"
  ),
  tar_target(
    pipeline_code_files,
    sort(unique(c(
      "00_requirements.R",
      "01_create_hex_grid.R",
      "run_analysis.R",
      "_targets.R",
      list.files(
        "R",
        pattern = "[.]R$",
        recursive = TRUE,
        full.names = TRUE
      ),
      list.files(
        "scripts/data",
        pattern = "[.]R$",
        recursive = TRUE,
        full.names = TRUE
      ),
      list.files(
        "scripts/features",
        pattern = "[.]R$",
        recursive = TRUE,
        full.names = TRUE
      ),
      list.files(
        "scripts/part1",
        pattern = "[.]R$",
        recursive = TRUE,
        full.names = TRUE
      ),
      list.files(
        "scripts/audits",
        pattern = "[.]R$",
        recursive = TRUE,
        full.names = TRUE
      )
    ))),
    format = "file",
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
  tar_target(
    map_orientation_script,
    "scripts/data/map_orientation.R",
    format = "file"
  ),
  tar_target(
    map_orientation_reference,
    run_r_script_stage(
      map_orientation_script,
      "output/map_orientation_reference.rds",
      dependencies = hex_grid
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
        parcel_input_manifest
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
        parcel_input_manifest
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
        "output/residential_unit_training_table.rds",
        "output/residential_unit_project_grouping_source_breakdown.csv"
      ),
      dependencies = unit_sources
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
        parcel_input_manifest
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
        "output/corporate_ownership_by_hex_unit_shadow.rds",
        "output/residential_unit_shadow_project_selection.csv"
      ),
      dependencies = list(
        hex_grid,
        unit_validation,
        unit_projects,
        unit_models,
        williamson_validation,
        analysis_config
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
        "output/residential_unit_promotion_manifest.csv",
        "output/residential_unit_land_use_exclusions.csv"
      ),
      dependencies = list(
        unit_validation,
        unit_integration,
        land_use_input_manifest,
        land_use_codes
      )
    ),
    format = "file"
  ),
  tar_target(
    land_use_unit_audit_script,
    "scripts/audits/land_use_unit_classification.R",
    format = "file"
  ),
  tar_target(
    land_use_unit_classification_audit,
    run_r_script_stage(
      land_use_unit_audit_script,
      c(
        "output/land_use_unit_classification_summary.csv",
        "output/land_use_unit_classification_benchmark.csv",
        "output/land_use_unit_classification_comparison.csv",
        "output/land_use_unit_classification_disagreements.csv",
        "figures/land_use_multifamily_classification_audit.png"
      ),
      dependencies = list(
        promoted_unit_surface,
        unit_projects,
        unit_integration,
        land_use_input_manifest,
        land_use_codes
      )
    ),
    format = "file"
  ),
  tar_target(
    nonresidential_unit_reconciliation_script,
    "scripts/audits/reconcile_nonresidential_unit_projects.R",
    format = "file"
  ),
  tar_target(
    nonresidential_unit_reconciliation_audit,
    run_r_script_stage(
      nonresidential_unit_reconciliation_script,
      c(
        "output/residential_unit_nonresidential_reconciliation_projects.csv",
        "output/residential_unit_nonresidential_reconciliation_summary.csv",
        "output/residential_unit_candidate_evidence_scope.csv",
        "output/residential_unit_nonresidential_reconciliation_impact.csv",
        "figures/residential_unit_nonresidential_reconciliation.png"
      ),
      dependencies = list(
        land_use_unit_classification_audit,
        promoted_unit_surface,
        unit_projects,
        unit_integration
      )
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
    acs_allocation_helpers,
    "R/acs_dasymetric.R",
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
        acs_allocation_helpers,
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
        acs_allocation_helpers,
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
      dependencies = eviction_input_manifest,
      environment = "GEOCODE_EVICTION_ADDRESSES=false"
    ),
    format = "file"
  ),
  tar_target(
    williamson_eviction_ingest_helpers,
    "R/williamson_eviction_ingest.R",
    format = "file"
  ),
  tar_target(
    williamson_eviction_prepare_script,
    "scripts/data/williamson_evictions_prepare.R",
    format = "file"
  ),
  tar_target(
    prepared_williamson_evictions,
    run_r_script_stage(
      williamson_eviction_prepare_script,
      c(
        "output/williamson_eviction_filings_prepared_for_geocoding.csv",
        "output/williamson_eviction_unique_addresses_for_geocoding.csv",
        "output/williamson_eviction_address_qc_summary.csv",
        "output/williamson_eviction_address_qc_by_source.csv",
        "output/williamson_eviction_source_period_qa.csv",
        "output/williamson_eviction_ingest_issues.csv"
      ),
      dependencies = list(
        eviction_source_config,
        eviction_input_manifest,
        williamson_eviction_ingest_helpers,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    williamson_eviction_local_geocode_script,
    "scripts/data/williamson_evictions_geocode_local.R",
    format = "file"
  ),
  tar_target(
    williamson_eviction_local_geocodes,
    run_r_script_stage(
      williamson_eviction_local_geocode_script,
      c(
        "output/williamson_eviction_addresses_geocoded_local.csv",
        "output/williamson_eviction_geocode_local_qa.csv"
      ),
      dependencies = list(
        prepared_williamson_evictions,
        williamson_address_reference_input,
        eviction_geocode_input,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    williamson_eviction_coa_geocode_script,
    "scripts/data/williamson_evictions_geocode_coa.R",
    format = "file"
  ),
  tar_target(
    williamson_eviction_coa_geocodes,
    run_r_script_stage(
      williamson_eviction_coa_geocode_script,
      c(
        "output/williamson_eviction_addresses_geocoded_coa.csv",
        "output/williamson_eviction_geocode_coa_qa.csv",
        "output/williamson_eviction_geocode_coa_review.csv"
      ),
      dependencies = list(
        prepared_williamson_evictions,
        williamson_eviction_local_geocodes,
        hex_grid
      ),
      environment = "WILLIAMSON_EVICTION_COA_NETWORK=false"
    ),
    format = "file"
  ),
  tar_target(
    williamson_eviction_census_geocode_script,
    "scripts/data/williamson_evictions_geocode_census.R",
    format = "file"
  ),
  tar_target(
    williamson_eviction_geocodes,
    run_r_script_stage(
      williamson_eviction_census_geocode_script,
      c(
        "output/williamson_eviction_addresses_geocoded.csv",
        "output/williamson_eviction_geocode_qa.csv"
      ),
      dependencies = list(
        prepared_williamson_evictions,
        williamson_eviction_local_geocodes,
        williamson_eviction_coa_geocodes,
        hex_grid,
        current_austin_jurisdiction_boundary
      ),
      environment = "WILLIAMSON_EVICTION_CENSUS_NETWORK=false"
    ),
    format = "file"
  ),
  tar_target(
    williamson_eviction_arcgis_geocode_script,
    "scripts/data/williamson_evictions_geocode_arcgis.R",
    format = "file"
  ),
  tar_target(
    williamson_eviction_arcgis_geocodes,
    run_r_script_stage(
      williamson_eviction_arcgis_geocode_script,
      c(
        "output/williamson_eviction_addresses_geocoded_arcgis.csv",
        "output/williamson_eviction_addresses_geocoded_with_arcgis.csv",
        "output/williamson_eviction_geocode_arcgis_qa.csv",
        "output/williamson_eviction_geocode_arcgis_review.csv"
      ),
      dependencies = list(
        prepared_williamson_evictions,
        williamson_eviction_coa_geocodes,
        williamson_eviction_geocodes,
        hex_grid,
        current_austin_jurisdiction_boundary
      ),
      environment = "WILLIAMSON_EVICTION_ARCGIS_NETWORK=false"
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
        "output/eviction_filings_by_hex_year.csv",
        "output/eviction_filings_full_geocoded_hex.rds"
      ),
      dependencies = list(
        hex_grid,
        prepared_evictions,
        eviction_geocode_input,
        eviction_input_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    requests_311_type_config,
    "config/311_smoke_signal_types.csv",
    format = "file"
  ),
  tar_target(
    requests_311_script,
    "scripts/data/austin_311.R",
    format = "file"
  ),
  tar_target(
    requests_311,
    run_r_script_stage(
      requests_311_script,
      c(
        "output/311_requests_by_hex_summary.rds",
        "output/311_requests_by_hex_year.csv",
        "output/311_service_request_selection.csv"
      ),
      dependencies = list(
        hex_grid,
        requests_311_type_config,
        analysis_config
      )
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
        landlord_mapper_manifest,
        analysis_config
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
    amenity_helper_scripts,
    c("R/amenity_classification.R", "R/amenity_scoring.R"),
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
        amenity_helper_scripts,
        amenity_input_manifest,
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
        amenity_helper_scripts,
        hex_grid,
        amenity_source_audit,
        amenity_input_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),

  # Shared feature layer for the baseline vintage.
  # Audited reconstructions are read-only inputs, not the paired eligible mask.
  tar_target(
    current_measurement_inputs,
    build_file_manifest(c(
      "output/part2/acs/acs_run_manifest.json",
      "output/part2/acs/acs_rent_source_candidates.rds",
      "output/part2/acs/2026-04-01/acs_demographics_by_hex.rds",
      "output/part2/ownership/parcel_ownership_snapshots.rds",
      "output/part2/ownership_index/ownership_index_run_manifest.json",
      "output/part2/311/311_features_paired.rds",
      "output/part2/311/311_run_manifest.json",
      "output/part2/demolitions/demolition_features_paired.rds",
      "output/part2/demolitions/demolition_run_manifest.json",
      "output/part2/evictions/eviction_features_paired.rds",
      "output/part2/evictions/eviction_run_manifest.json",
      "output/part2/amenities/amenity_features_paired.rds",
      "output/part2/amenities/amenity_run_manifest.json",
      "R/current_measurement.R", "R/acs_snapshot_scoring.R",
      "R/part2_index_scoring.R", "R/part2_event_scoring.R",
      "R/part2_feature_matrix.R", "R/amenity_scoring.R", "R/ownership_snapshots.R"
    ), require_all = TRUE, hash_files = TRUE),
    cue = tar_cue(mode = "always")
  ),
  tar_target(current_measurement_script, "scripts/features/build_current_measurement.R", format = "file"),
  tar_target(
    current_measurement,
    run_r_script_stage(current_measurement_script,
      c("output/part1/measurement/current_measurement.rds",
        "output/part1/measurement/current_component_scaling.rds",
        "output/part1/measurement/current_eligibility.csv",
        "output/part1/measurement/current_exclusions.csv",
        "output/part1/measurement/current_index_readiness.csv",
        "output/part1/measurement/rent_source_selection.csv",
        "output/part1/measurement/current_measurement_manifest.json"),
      dependencies = list(current_measurement_inputs, hex_grid, corporate_features, analysis_config)),
    format = "file"
  ),
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
        current_measurement,
        demolition_input_manifest,
        analysis_config
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
      dependencies = list(
        current_features,
        feature_dictionary,
        analysis_config
      )
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
        "output/amenity_cluster_metrics.csv",
        "output/amenity_cluster_gap_statistics.csv",
        "output/amenity_cluster_stability.csv",
        "output/amenity_cluster_agreement.csv",
        "output/amenity_cluster_assignments.csv",
        "output/amenity_cluster_recommendations.csv",
        "output/amenity_cluster_profiles.csv",
        "output/amenity_cluster_crosswalk.csv",
        "output/amenity_cluster_selected_crosswalk.csv",
        "output/amenity_cluster_selected_label_mapping.csv",
        "output/amenity_cluster_population_coverage.csv",
        "figures/03d_amenity_cluster_diagnostics.png",
        "figures/03d_amenity_cluster_selected_maps.png",
        "figures/03d_amenity_cluster_selected_profiles.png"
      ),
      dependencies = list(
        current_features,
        feature_audit,
        feature_dictionary,
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
      cluster_labels
      cluster_interpretation
      freeze_baseline_cluster_model(
        feature_file = "output/hex_features.rds",
        cluster_results_file = "output/amenity_cluster_sensitivity.rds",
        label_file = "config/amenity_cluster_labels.csv",
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
        "figures/03e_amenity_clusters_interactive.html",
        "site/index.html"
      ),
      dependencies = list(
        current_features,
        part1_cluster_analysis,
        cluster_labels,
        cluster_interpretation,
        map_orientation_reference,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    part1_validation_script,
    "scripts/audits/part1.R",
    format = "file"
  ),
  tar_target(
    part1_validation,
    run_r_script_stage(
      part1_validation_script,
      c(
        "output/part1/baseline_cluster_validation.csv",
        "output/part1/baseline_cluster_summary.csv",
        "output/part1/baseline_cluster_assignments.csv",
        "output/part1/baseline_cluster_lock.csv"
      ),
      dependencies = list(
        current_features,
        part1_cluster_analysis,
        part1_baseline_model,
        part1_visualizations,
        feature_dictionary,
        cluster_labels,
        pipeline_code_files,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    part1_neighborhood_script,
    "scripts/part1/summarize_neighborhood_clusters.R",
    format = "file"
  ),
  tar_target(
    part1_neighborhood_summary,
    run_r_script_stage(
      part1_neighborhood_script,
      c(
        "output/part1/neighborhood_cluster_composition.csv",
        "output/part1/neighborhood_cluster_summary.csv",
        "output/part1/neighborhood_cluster_coverage.csv",
        "output/part1/neighborhood_cluster_summary.rds",
        "figures/03g_neighborhood_cluster_plurality.png"
      ),
      dependencies = list(
        neighborhood_reference_files,
        current_features,
        part1_validation,
        cluster_labels,
        map_orientation_reference,
        analysis_config
      )
    ),
    format = "file"
  ),

  # Part 2 ownership snapshots are isolated from the canonical Part 1 features.
  tar_target(
    ownership_snapshot_spec,
    "config/ownership_snapshot_spec.json",
    format = "file"
  ),
  tar_target(
    ownership_snapshot_code,
    c("R/ownership_snapshots.R", "scripts/data/build_other_county_ownership.py",
      "scripts/data/prepare_williamson_txgio.R", "scripts/data/williamson_ownership_reconciliation.py",
      "scripts/part2/build_ownership_snapshots.R"),
    format = "file"
  ),
  tar_target(
    ownership_snapshot_inputs,
    {
      ownership_snapshot_code
      spec <- jsonlite::read_json(ownership_snapshot_spec, simplifyVector = TRUE)
      county_paths <- jsonlite::fromJSON(paste(system2(
        Sys.getenv("EWS_PYTHON", unset = "python3"),
        c("-B", "scripts/data/build_other_county_ownership.py", "--list-inputs"),
        stdout = TRUE), collapse = "\n"))
      rbind(build_file_manifest(c(
        file.path(spec$upstream_repository, spec$classifier_path),
        file.path(spec$upstream_repository, "output/historical_ownership/travis_owner_snapshots_2024_2025.csv"),
        file.path(spec$upstream_repository, "output/historical_ownership/travis_owner_snapshot_manifest.json"),
        "data/residential_parcels_for_hex.csv"
      ), require_all = TRUE, hash_files = TRUE),
      # An absent annual county source is an explicit NA vintage; appearance
      # of a verified replacement changes this manifest and rebuilds the stage.
      build_file_manifest(county_paths, require_all = FALSE, hash_files = TRUE))
    },
    cue = tar_cue(mode = "always")
  ),
  tar_target(
    part2_ownership_snapshots,
    run_r_script_stage(
      "scripts/part2/build_ownership_snapshots.R",
      c(
        "output/part2/ownership/ownership_target_parcels.csv",
        "output/part2/ownership/other_county_owner_snapshots_2024_2025.csv",
        "output/part2/ownership/other_county_sources_manifest.json",
        "output/part2/ownership/williamson_txgio_2024_evidence.csv",
        "output/part2/ownership/williamson_txgio_2024_manifest.json",
        "output/part2/ownership/williamson_2024_certified_only_snapshots.csv",
        "output/part2/ownership/williamson_2024_source_reconciliation.csv",
        "output/part2/ownership/williamson_2024_source_reconciliation_qa.csv",
        "output/part2/ownership/williamson_2024_source_conflict_review.csv",
        "output/part2/ownership/williamson_txgio_2025_evidence.csv",
        "output/part2/ownership/williamson_txgio_2025_manifest.json",
        "output/part2/ownership/williamson_2025_certified_only_snapshots.csv",
        "output/part2/ownership/williamson_2025_source_reconciliation.csv",
        "output/part2/ownership/williamson_2025_source_reconciliation_qa.csv",
        "output/part2/ownership/williamson_2025_source_conflict_review.csv",
        "output/part2/ownership/parcel_ownership_snapshots.rds",
        "output/part2/ownership/ownership_features_by_hex_year.rds",
        "output/part2/ownership/ownership_features_by_hex_year.csv",
        "output/part2/ownership/ownership_common_support_by_hex_year.rds",
        "output/part2/ownership/ownership_common_support_by_hex_year.csv",
        "output/part2/ownership/ownership_hex_change.csv",
        "output/part2/ownership/ownership_certified_only_hex_change.csv",
        "output/part2/ownership/ownership_source_agreement_hex_change.csv",
        "output/part2/ownership/ownership_2024_certified_only_hex_change.csv",
        "output/part2/ownership/ownership_source_variant_county_qa.csv",
        "output/part2/ownership/ownership_source_variant_summary.csv",
        "output/part2/ownership/ownership_county_qa.csv",
        "output/part2/ownership/ownership_support_qa.csv",
        "output/part2/ownership/ownership_classification_status_qa.csv",
        "output/part2/ownership/ownership_2025_parity_qa.csv",
        "output/part2/ownership/ownership_2025_parity_differences.csv",
        "output/part2/ownership/ownership_common_support_county_qa.csv",
        "output/part2/ownership/ownership_common_support_transitions.csv",
        "output/part2/ownership/ownership_transition_evidence_qa.csv",
        "output/part2/ownership/ownership_transition_review.csv",
        "output/part2/ownership/ownership_large_hex_changes.csv",
        "output/part2/ownership/ownership_unknown_review.csv",
        "output/part2/ownership/ownership_snapshot_manifest.json"
      ),
      dependencies = list(ownership_snapshot_spec, ownership_snapshot_code,
        ownership_snapshot_inputs, promoted_unit_surface, hex_grid, corporate_features)
    ),
    format = "file"
  ),

  # Part 2: retain the baseline self-reassignment artifact as the template for
  # future vintages after the stricter Part 1 lock audit has passed.
  tar_target(
    part2_baseline_assignment,
    {
      current_features
      part1_cluster_analysis
      part1_validation
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

  # Part 3: build complete outcome panels and forward labels for the initial
  # eviction/demolition pilot. Historical predictor construction follows as a
  # separate implementation slice.
  tar_target(
    forecast_outcome_spec,
    "config/forecast_outcomes.csv",
    format = "file"
  ),
  tar_target(
    hex_county_assignment_reference,
    c(
      "config/hex_county_assignment_2024.csv",
      "config/hex_county_assignment_2024_metadata.csv",
      "config/williamson_jp_hex_assignment.csv",
      "config/williamson_jp_hex_assignment_metadata.csv"
    ),
    format = "file"
  ),
  tar_target(
    eviction_panel_helpers,
    "R/eviction_panel.R",
    format = "file"
  ),
  tar_target(
    eviction_coverage_helpers,
    "R/eviction_coverage.R",
    format = "file"
  ),
  tar_target(
    eviction_outcome_panel_script,
    "scripts/part3/build_eviction_outcome_panel.R",
    format = "file"
  ),
  tar_target(
    eviction_outcome_panel,
    run_r_script_stage(
      eviction_outcome_panel_script,
      c(
        "output/eviction_filings_complete_by_hex_year.csv",
        "output/part3/eviction_case_assignment_issues.csv",
        "output/part3/eviction_case_assignment_summary.csv",
        "output/part3/eviction_complete_panel_qa.csv",
        "output/part3/eviction_source_coverage_qa.csv",
        "output/part3/eviction_source_geography_qa.csv",
        "output/part3/eviction_panel_source_manifest.csv"
      ),
      dependencies = list(
        eviction_panel_helpers,
        eviction_coverage_helpers,
        prepared_evictions,
        prepared_williamson_evictions,
        eviction_geocode_input,
        williamson_eviction_local_geocodes,
        williamson_eviction_coa_geocodes,
        williamson_eviction_geocodes,
        williamson_eviction_arcgis_geocodes,
        hex_grid,
        current_austin_jurisdiction_boundary,
        hex_county_assignment_reference,
        eviction_source_config,
        eviction_input_manifest,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    demolition_panel_helpers,
    "R/demolition_panel.R",
    format = "file"
  ),
  tar_target(
    demolition_coverage_helpers,
    "R/demolition_coverage_history.R",
    format = "file"
  ),
  tar_target(
    demolition_outcome_panel_script,
    "scripts/data/demolitions_panel.R",
    format = "file"
  ),
  tar_target(
    demolition_outcome_panel,
    run_r_script_stage(
      demolition_outcome_panel_script,
      c(
        "output/demolition_permits_by_hex_year.csv",
        "output/demolition_permits_source_qa.csv",
        "output/demolition_permits_annual_qa.csv",
        "output/demolition_permits_unmatched_qa.csv",
        "output/part3/demolition_panel_source_manifest.csv",
        "output/part3/demolition_historical_coverage_by_hex_year.csv",
        "output/part3/demolition_coverage_current_snapshot_qa.csv",
        "output/part3/demolition_coverage_current_snapshot_summary.csv"
      ),
      dependencies = list(
        demolition_panel_helpers,
        demolition_coverage_helpers,
        demolition_input_manifest,
        hex_grid,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    forecast_label_helpers,
    "R/forecast_labels.R",
    format = "file"
  ),
  tar_target(
    part3_forecast_label_script,
    "scripts/part3/build_forecast_labels.R",
    format = "file"
  ),
  tar_target(
    part3_forecast_labels,
    run_r_script_stage(
      part3_forecast_label_script,
      c(
        "output/part3/eviction_demolition_forecast_labels_long.rds",
        "output/part3/eviction_demolition_forecast_labels_long.csv",
        "output/part3/eviction_demolition_forecast_labels_wide.csv",
        "output/part3/forecast_label_task_contract.csv",
        "output/part3/forecast_label_qa.csv",
        "output/part3/forecast_label_common_support_qa.csv",
        "output/part3/forecast_label_run_manifest.csv"
      ),
      dependencies = list(
        forecast_label_helpers,
        forecast_outcome_spec,
        eviction_outcome_panel,
        demolition_outcome_panel,
        analysis_config
      )
    ),
    format = "file"
  ),
  tar_target(
    part3_forecast_readiness,
    {
      part3_forecast_labels
      build_forecast_readiness(
        outcome_spec_file = forecast_outcome_spec,
        output_file = "output/part3/forecast_readiness.csv",
        config = analysis_config,
        source_files = c(
          eviction_filings =
            "output/eviction_filings_complete_by_hex_year.csv",
          residential_demolitions =
            "output/demolition_permits_by_hex_year.csv"
        ),
        label_file =
          "output/part3/eviction_demolition_forecast_labels_long.rds",
        predictor_panel_file =
          "output/part3/historical_predictor_panel.rds"
      )
    },
    format = "file"
  )
)
