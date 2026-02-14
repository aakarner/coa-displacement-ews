# Changelog

All notable changes to the Displacement Early Warning System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-02-14

### Fixed

Comprehensive fix for 15 issues across the R analysis pipeline ([PR #3](https://github.com/aakarner/coa-displacement-ews/pull/3)).

#### Critical Bugs

- **Duplicate spatial computation** (`01_create_hex_grid.R`): Removed redundant `polygon_to_cells()` call that executed twice and incorrectly used `length()` on data frame
- **Broken demolition aggregation** (`02_process_data.R`): Fixed `sum(!is.na(.))` which counted non-NA cells across all columns instead of actual demolition records; now uses `sum(!is.na(calendar_year_issued))`
- **Variable scoping error** (`02_process_data.R`): Fixed `tryCatch` block where error handler assigned `acs_data` in wrong scope, causing "object not found" on API failure; restructured to `acs_data <- tryCatch({...})` pattern
- **Missing fallback object** (`02_process_data.R`): Added empty `st_sf` object when demolitions CSV is missing to prevent downstream failures
- **Unloaded dependencies** (`packages.R`): Added `spdep`, `blockCV`, `gridExtra`, `htmlwidgets`; removed unused `calculate_spatial_lag()` requiring unloaded `nngeo`

#### Data Inconsistencies

- **Column name collision** (`02_process_data.R`): Prevented `median_rent.x`/`.y` suffix columns by dropping duplicate before join
- **Configuration mismatch** (`run_analysis.R`): Corrected H3 resolution from 8 to 9 to match actual value used (~0.1 km² cells)
- **Step numbering** (`run_analysis.R`): Fixed inconsistent step headers (1/7, 2/7, 3/8... → 1/8, 2/8, 3/8...)
- **Misleading header** (`05_validate_models.R`): Removed claim of temporal CV implementation (only spatial CV implemented)

#### Documentation & Cleanup

- Added INPUTS/OUTPUTS/DEPENDENCIES sections to all 8 pipeline scripts
- Documented `set.seed()` and `source()` calls for standalone execution pattern
- Removed tracked `.DS_Store`, added to `.gitignore`
- Clarified WHY THIS MATTERS sections for spatial CV and feature engineering approaches
- 13 files modified, +201/-68 lines

## [1.1.0] - 2026-02-12

### Changed

Replaced circular synthetic outcome variable with cluster-based classification ([PR #2](https://github.com/aakarner/coa-displacement-ews/pull/2)).

#### Added

- **Unsupervised Clustering Phase** (`03b_cluster_analysis.R`)
  - K-means, hierarchical, and DBSCAN to discover displacement patterns empirically
  - Silhouette analysis and elbow plots for optimal k selection
  - Cluster profiling: characterize patterns by rent pressure, demolition activity, vulnerability
  - Outputs: cluster assignments, profiles, validation metrics, visualizations

- **New Dependencies**: `cluster`, `factoextra`, `dbscan`, `Rtsne`

#### Modified

- **Model Training** (`04_train_models.R`): Classification instead of regression — models now predict `cluster_class` (factor) instead of synthetic `displacement_risk` (numeric); metric changed from RMSE to Accuracy
- **Model Validation** (`05_validate_models.R`): Classification metrics — confusion matrices, F1-scores, per-cluster accuracy; replaced residual diagnostics with prediction agreement analysis
- **Risk Score Generation** (`06_predict_risk_scores.R`): Cluster probability → risk score conversion using empirical profiles; probability-weighted ensemble scoring
- **Visualization** (`07_visualize_results.R`): Cluster visualization overlays on risk maps
- **Documentation**: Updated `README.md` with clustering methodology; updated `WORKFLOW.md` with Step 3b

#### Interpretation Shift

- **Before**: "Risk score of 75/100" (arbitrary composite)
- **After**: "75% probability of high-risk cluster (high rent growth + high vulnerability)"
- Backward compatible: maintains 0-100 risk scores for existing visualizations

## [1.0.0] - 2026-02-11

### Added - Initial Release

#### Core Functionality
- **Hexagonal Grid System** (`01_create_hex_grid.R`)
  - H3 spatial indexing at resolution 9 (~0.1 km² cells)
  - Coverage of Austin, TX city boundaries
  - Static and interactive grid visualizations

- **Data Processing Pipeline** (`02_process_data.R`)
  - Automatic Census/ACS data retrieval via `tidycensus`
  - Demographic and socioeconomic indicators (14 variables)
  - Building demolition data processing (with synthetic fallback)
  - Rent price time series handling (with synthetic fallback)
  - Placeholder structure for future data sources (evictions, land values, corporate ownership)
  - Spatial aggregation to hexagonal grid

- **Feature Engineering** (`03_feature_engineering.R`)
  - Temporal rent features (6 features)
  - Demolition metrics (4 features)
  - Socioeconomic vulnerability indices (6 features)
  - Spatial lag features (4 features)
  - Interaction terms (4 features)
  - Missing data handling strategy

- **Machine Learning Models** (`04_train_models.R`)
  - Random Forest implementation with tuning
  - XGBoost (gradient boosting) with extensive parameter search
  - Elastic Net regularized regression
  - 5-fold cross-validation
  - Feature importance extraction
  - Comprehensive educational comments for traditional statisticians

- **Model Validation** (`05_validate_models.R`)
  - Residual diagnostics
  - Predicted vs. actual plots
  - Feature importance comparison across models
  - Spatial cross-validation framework
  - Error analysis by risk level
  - Distribution comparisons

- **Risk Score Generation** (`06_predict_risk_scores.R`)
  - Ensemble predictions (weighted average of models)
  - Risk scores on 0-100 scale
  - Four-level risk categorization (Low/Moderate/High/Very High)
  - Contributing factor identification
  - Spatial and tabular output formats

- **Visualization Suite** (`07_visualize_results.R`)
  - Interactive Leaflet maps with popups
  - Static publication-quality maps
  - Model comparison visualizations
  - Feature importance plots
  - Risk distribution analysis
  - Summary dashboard

- **Pipeline Orchestration** (`run_analysis.R`)
  - Master script to run complete analysis
  - Configuration management
  - Error handling and progress tracking
  - Runtime estimation and summary statistics

#### Supporting Infrastructure
- **Utility Functions** (`R/utils.R`)
  - Spatial lag calculation
  - Normalization and categorization helpers
  - Progress reporting functions
  - I/O convenience functions

- **Package Management** (`packages.R`)
  - Automatic installation of missing packages
  - Organized by category (spatial, ML, data, viz, census)
  - Global options configuration
  - Reproducibility setup

#### Documentation
- **Comprehensive README.md**
  - Project overview and motivation
  - Complete installation instructions
  - Usage guide (quick start and step-by-step)
  - Data requirements and format specifications
  - ML concepts explained for beginners
  - Customization options
  - Troubleshooting guide
  - Academic references

- **Quick Start Guide** (`QUICKSTART.md`)
  - 5-minute setup instructions
  - Common issues and solutions
  - Visual workflow diagram

- **Data Documentation** (`data/README.md`)
  - Format specifications for optional data sources
  - Integration instructions
  - Privacy considerations
  - Geocoding tips

- **Contributing Guide** (`CONTRIBUTING.md`)
  - Contribution workflow
  - Coding standards
  - Priority areas for contributions
  - Testing requirements

### Technical Specifications

- **Languages**: R (4.0+)
- **Key Dependencies**: 
  - Spatial: sf, h3jsr, tigris, lwgeom
  - ML: caret, randomForest, xgboost, glmnet
  - Data: tidyverse, data.table, lubridate
  - Viz: leaflet, mapview, ggplot2, viridis
  - Census: tidycensus

- **Data Sources**:
  - Primary: U.S. Census Bureau ACS 5-year estimates
  - Optional: Local demolition records, rent prices, evictions, land values

- **Model Performance** (on synthetic data):
  - Ensemble approach combining three algorithms
  - Cross-validated hyperparameter tuning
  - Spatial awareness in validation

### Known Limitations

- Synthetic data used as fallback when real data unavailable
- Spatial cross-validation implementation is basic
- No temporal forecasting (current snapshot only)
- Limited to Austin, TX (adaptable to other cities)
- Requires Census API key setup
- Memory intensive with large hexagonal grids

### Future Enhancements (Planned)

See [CONTRIBUTING.md](CONTRIBUTING.md) for priority areas:
- Additional data source integrations
- Temporal forecasting capabilities
- Deep learning models
- Interactive Shiny dashboard
- Multi-city comparative analysis
- Model interpretability tools (SHAP values)

---

## Version History

### Version Numbering

- **Major version** (1.x.x): Significant architectural changes, breaking changes
- **Minor version** (x.1.x): New features, enhancements, non-breaking changes
- **Patch version** (x.x.1): Bug fixes, documentation updates

### Release Dates

- **1.2.1** - February 14, 2026: Bug fixes, data inconsistency corrections, documentation improvements
- **1.1.0** - February 12, 2026: Cluster-based classification replacing synthetic outcome
- **1.0.0** - February 11, 2026: Initial release

---

For detailed commit history, see the [GitHub repository](https://github.com/aakarner/coa-displacement-ews).