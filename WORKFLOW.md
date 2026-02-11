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
│  ├─ Generate H3 hexagons (resolution 8)                        │
│  ├─ ~500-1000 hexagons covering Austin                         │
│  └─ Output: hex_grid.rds + visualizations                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Process Data                                           │
│  02_process_data.R                                              │
│  ├─ Download Census/ACS data (14 variables)                    │
│  ├─ Process demolitions (optional)                             │
│  ├─ Process rent prices (optional)                             │
│  ├─ Spatial join to hexagons                                   │
│  └─ Output: hex_data_processed.rds                             │
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
packages.R (load first)
    │
    └─► run_analysis.R
            │
            ├─► 01_create_hex_grid.R
            │       └─► output/hex_grid.rds
            │
            ├─► 02_process_data.R
            │       ├─ requires: hex_grid.rds
            │       └─► output/hex_data_processed.rds
            │
            ├─► 03_feature_engineering.R
            │       ├─ requires: hex_data_processed.rds
            │       └─► output/hex_features.rds
            │
            ├─► 04_train_models.R
            │       ├─ requires: hex_features.rds
            │       └─► output/trained_models.rds
            │
            ├─► 05_validate_models.R
            │       ├─ requires: trained_models.rds, hex_features.rds
            │       └─► output/validation_results.rds + figures/
            │
            ├─► 06_predict_risk_scores.R
            │       ├─ requires: trained_models.rds, hex_features.rds
            │       └─► output/displacement_risk_scores.rds/.csv
            │
            └─► 07_visualize_results.R
                    ├─ requires: displacement_risk_scores.rds, validation_results.rds
                    └─► figures/*.png, figures/*.html
```

## Key Outputs

### Primary Outputs
- **displacement_risk_scores.rds** - Spatial risk scores (main result)
- **displacement_risk_scores.csv** - Tabular risk scores
- **07_interactive_risk_map.html** - Interactive map (main visualization)

### Supporting Outputs
- **trained_models.rds** - All 3 trained ML models
- **validation_results.rds** - Model performance metrics
- **hex_grid.rds** - Hexagonal grid geometry
- **figures/*.png** - All static visualizations

## Runtime Estimates

| Step | Typical Runtime |
|------|----------------|
| 01 - Create Grid | 1-2 minutes |
| 02 - Process Data | 3-5 minutes |
| 03 - Feature Engineering | 2-4 minutes |
| 04 - Train Models | 15-30 minutes |
| 05 - Validate Models | 5-10 minutes |
| 06 - Risk Scores | 2-3 minutes |
| 07 - Visualize | 3-5 minutes |
| **TOTAL** | **30-60 minutes** |

*Times vary based on hardware, data size, and model parameters*

## Customization Points

| Component | Customization | File |
|-----------|--------------|------|
| Grid Resolution | H3_RESOLUTION | 01_create_hex_grid.R |
| Risk Weights | displacement_risk formula | 06_predict_risk_scores.R |
| Model Parameters | Grid searches | 04_train_models.R |
| Feature Selection | predictor_vars | 04_train_models.R |
| Risk Thresholds | categorize_risk() | R/utils.R |
| Color Schemes | viridis options | 07_visualize_results.R |

## See Also

- **README.md** - Complete documentation
- **QUICKSTART.md** - Quick setup guide
- **VERIFICATION.md** - Requirements checklist
- **data/README.md** - Data format specs
