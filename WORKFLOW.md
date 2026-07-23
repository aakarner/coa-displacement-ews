# Displacement Early Warning System - Workflow

## Quick Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_analysis.R                               │
│              (Master Pipeline Orchestrator)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Create Hexagonal Grid                                 │
│  01_create_hex_grid.R                                          │
│  ├─ Fetch Austin, TX boundary                                  │
│  ├─ Generate H3 hexagons (resolution 9)                        │
│  ├─ ~7,000 hexagons covering Austin                            │
│  └─ Output: hex_grid.rds + visualizations                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Process Data                                           │
│  02_process_data.R                                              │
│  ├─ Build calibrated residential parcel support               │
│  ├─ Download/cache Census block and ACS block-group data       │
│  ├─ Allocate counts with block/parcel dasymetric weights       │
│  ├─ Assign dominant-BG medians with tract fallback             │
│  ├─ Process demolitions (optional)                             │
│  ├─ Process rent prices (optional)                             │
│  ├─ Spatial join to hexagons                                   │
│  └─ Output: hex_data_processed.rds                             │
│  02i_process_appraisal_history.R                                │
│  ├─ Normalize 2021-2025 county certified appraisal values      │
│  ├─ Preserve explicit missing parcel-years and source QA       │
│  └─ Output: parcel-year, hex-year, and hex trend panels         │
│  02j_process_appraisal_adjusted_trends.R                        │
│  ├─ Estimate full-county annual appraisal baselines             │
│  ├─ Remove common county-year shifts from parcel changes        │
│  └─ Output: adjusted parcel/hex trends and sensitivity QA       │
│  02k_audit_ownership_transactions.R                             │
│  ├─ Audit annual owner identities and deed/sales histories      │
│  ├─ Measure source coverage against the analysis parcel universe│
│  └─ Output: ownership/transaction source and interval QA        │
│  02l_process_ownership_transactions.R                           │
│  ├─ Build equal-window transaction pressure measures            │
│  ├─ Build direct or documented proxy corporate-entry measures   │
│  └─ Output: parcel/hex features plus source and event QA         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Engineer Features                                      │
│  03_feature_engineering.R                                       │
│  ├─ Temporal: Rent changes, acceleration, volatility (6)       │
│  ├─ Demolitions: Density, trends (4)                           │
│  ├─ Vulnerability: Income, poverty, education (6)              │
│  ├─ Spatial lags: Neighborhood effects (4)                     │
│  ├─ Interactions: Combined risk factors (4)                    │
│  └─ Output: hex_features.rds (24+ features)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Train Models                                           │
│  04_train_models.R                                              │
│  ├─ Split: 70% train, 30% test                                 │
│  ├─ Random Forest (500 trees, tuned mtry)                      │
│  ├─ XGBoost (tuned depth, eta, nrounds)                        │
│  ├─ Elastic Net (tuned alpha, lambda)                          │
│  ├─ 5-fold cross-validation                                    │
│  └─ Output: trained_models.rds                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Validate Models                                        │
│  05_validate_models.R                                           │
│  ├─ Residual diagnostics                                       │
│  ├─ Predicted vs. actual plots                                 │
│  ├─ Feature importance comparison                              │
│  ├─ Spatial cross-validation                                   │
│  ├─ Error analysis by risk level                               │
│  └─ Output: validation_results.rds + plots                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Generate Risk Scores                                   │
│  06_predict_risk_scores.R                                       │
│  ├─ Predict with all 3 models                                  │
│  ├─ Create weighted ensemble                                   │
│  ├─ Scale to 0-100 risk scores                                 │
│  ├─ Categorize: Low/Moderate/High/Very High                    │
│  ├─ Identify contributing factors                              │
│  └─ Output: displacement_risk_scores.rds/.csv                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Visualize Results                                      │
│  07_visualize_results.R                                         │
│  ├─ Interactive map (Leaflet)                                  │
│  ├─ Static risk map                                            │
│  ├─ Categorical risk map                                       │
│  ├─ Model comparison maps                                      │
│  ├─ Feature importance plots                                   │
│  ├─ Distribution plots                                         │
│  ├─ Summary dashboard                                          │
│  └─ Output: 8+ visualizations in figures/                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Analysis Done!  │
                    └─────────────────┘
```

## Data Flow

```
Census API ──┐
             │
Demolitions ─┼─► 02_process_data ─► hex_data ─► 03_feature_eng ─► features
             │                                                         │
Rent data ───┘                                                         │
                                                                       ▼
                                                           04_train_models
                                                                  │
                                                                  ├─► Random Forest
                                                                  ├─► XGBoost
                                                                  └─► Elastic Net
                                                                       │
                                                                       ▼
                                                           05_validate_models
                                                                       │
                                                                       ▼
                                                        06_predict_risk_scores
                                                                       │
                                                                       ▼
                                                         07_visualize_results
```

## File Dependencies

```
00_requirements.R (install missing packages first)
    │
    └─► packages.R
            │
            └─► run_analysis.R
                    │
                    ├─► 01_create_hex_grid.R
                    │       └─► output/hex_grid.rds
                    │
                    ├─► 02_process_data.R
                    │       ├─ requires: hex_grid.rds
                    │       ├─ runs: 02d → 02e → 02c → 02f
                    │       ├─► output/acs_dasymetric_allocation_qa.csv
                    │       └─► output/hex_data_processed.rds
                    │
                    ├─► 02h_process_acs_rent_history.R
                    │       ├─ requires: residential parcel support, ACS cache
                    │       └─► output/acs_rent_trends_by_hex.rds
                    │
                    ├─► 02i_process_appraisal_history.R
                    │       ├─ requires: calibrated residential parcels, hex_grid.rds
                    │       ├─► output/appraisal_values_by_parcel_year.rds
                    │       ├─► output/appraisal_values_by_hex_year.rds
                    │       └─► output/appraisal_value_trends_by_hex.rds
                    │
                    ├─► 02j_process_appraisal_adjusted_trends.R
                    │       ├─ requires: county and target appraisal panels
                    │       ├─► output/appraisal_county_year_baselines.csv
                    │       └─► output/appraisal_adjusted_trends_by_hex.rds
                    │
                    ├─► 02k_audit_ownership_transactions.R
                    │       └─► output/ownership_transaction_source_audit.csv
                    │
                    ├─► 02l_process_ownership_transactions.R
                    │       ├─ requires: parcel universe, owner snapshots, deed/sales histories
                    │       ├─► output/ownership_transaction_features_by_parcel.rds
                    │       ├─► output/ownership_transaction_features_by_hex.rds/.csv
                    │       └─► output/ownership_transaction_source_qa.csv
                    │
                    ├─► 02m_audit_amenity_sources.R
                    │       ├─ downloads/caches state establishment histories
                    │       └─► output/amenity_source_candidates.rds and QA tables
                    │
                    ├─► 02n_process_amenity_change.R
                    │       ├─ Census batch geocodes core opening events
                    │       └─► output/amenity_change_features_by_hex.rds/.csv
                    │
                    ├─► 03_feature_engineering.R
                    │       ├─ requires: hex_data_processed.rds
                    │       └─► output/hex_features.rds
                    │
                    ├─► 03a_feature_audit.R
                    │       ├─ requires: hex_features.rds
                    │       └─► output/feature_coverage_audit.csv
                    │
                    ├─► 03b_cluster_analysis.R (NEW)
                    │       ├─ requires: hex_features.rds
                    │       ├─► output/hex_features_with_clusters.rds
                    │       ├─► output/cluster_analysis_results.rds
                    │       ├─► output/cluster_profiles.csv
                    │       └─► figures/03b_*.png
                    │
                    ├─► 04_train_models.R
                    │       ├─ requires: hex_features_with_clusters.rds
                    │       ├─ requires: cluster_analysis_results.rds
                    │       └─► output/trained_models.rds
                    │
                    ├─► 05_validate_models.R
                    │       ├─ requires: trained_models.rds, hex_features_with_clusters.rds
                    │       └─► output/validation_results.rds + figures/
                    │
                    ├─► 06_predict_risk_scores.R
                    │       ├─ requires: trained_models.rds, hex_features_with_clusters.rds
                    │       ├─ requires: cluster_analysis_results.rds
                    │       └─► output/displacement_risk_scores.rds/.csv
                    │
                    └─► 07_visualize_results.R
                            ├─ requires: displacement_risk_scores.rds, validation_results.rds
                            └─► figures/*.png, figures/*.html
```

## Key Outputs

### Primary Outputs
- **displacement_risk_scores.rds** - Spatial risk scores (main result)
- **displacement_risk_scores.csv** - Tabular risk scores with cluster predictions
- **07_interactive_risk_map.html** - Interactive map (main visualization)

### Clustering Outputs (NEW)
- **cluster_analysis_results.rds** - Complete clustering analysis
- **cluster_profiles.csv** - Cluster characterizations
- **hex_features_with_clusters.rds** - Features + cluster assignments
- **figures/03b_*.png** - Cluster visualizations (elbow, silhouette, PCA, map, profiles)

### Supporting Outputs
- **trained_models.rds** - All 3 trained ML models
- **validation_results.rds** - Model performance metrics
- **hex_grid.rds** - Hexagonal grid geometry
- **appraisal_values_by_parcel_year.rds** - Complete current parcel universe by tax year
- **appraisal_values_by_hex_year.rds** - Inflation-adjusted annual appraisal values and coverage
- **appraisal_value_trends_by_hex.rds** - Land/total value levels, growth, acceleration, and reliability
- **appraisal_adjusted_trends_by_hex.rds** - County-relative land-value level, growth, acceleration, and reliability
- **appraisal_county_year_baselines.csv** - Full-county annual appraisal shifts used for adjustment
- **ownership_transaction_features_by_hex.rds/.csv** - Transaction pressure and ownership-change measures with coverage fields
- **ownership_transaction_source_qa.csv** - County source completeness and ownership-direction methods
- **figures/*.png** - All static visualizations

## Runtime Estimates

| Step | Typical Runtime |
|------|----------------|
| 01 - Create Grid | 1-2 minutes |
| 02 - Process Data | 3-5 minutes |
| 03 - Feature Engineering | 2-4 minutes |
| 03b - Cluster Analysis | 3-5 minutes |
| 04 - Train Models | 15-30 minutes |
| 05 - Validate Models | 5-10 minutes |
| 06 - Risk Scores | 2-3 minutes |
| 07 - Visualize | 3-5 minutes |
| **TOTAL** | **35-65 minutes** |

*Times vary based on hardware, data size, and model parameters*

## Customization Points

| Component | Customization | File |
|-----------|--------------|------|
| Grid Resolution | H3_RESOLUTION | 01_create_hex_grid.R |
| Clustering Variables | clustering_vars | 03b_cluster_analysis.R |
| Optimal Clusters | optimal_k | 03b_cluster_analysis.R |
| Cluster Risk Mapping | cluster_risk_mapping | 06_predict_risk_scores.R |
| Model Parameters | Grid searches | 04_train_models.R |
| Feature Selection | predictor_vars | 04_train_models.R |
| Risk Thresholds | categorize_risk() | R/utils.R |
| Color Schemes | viridis options | 07_visualize_results.R |

## See Also

- **README.md** - Complete documentation
- **QUICKSTART.md** - Quick setup guide
- **VERIFICATION.md** - Requirements checklist
- **data/README.md** - Data format specs
