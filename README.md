# Displacement Early Warning System for Austin, TX

A machine learning-based early warning system for predicting residential displacement risk in Austin, Texas. This system uses hexagonal spatial grids, multiple data sources, and ensemble machine learning models to identify areas at high risk of displacement.

## Overview

This project implements a prototype displacement early warning system that:
- Analyzes displacement risk at the neighborhood level using hexagonal grids
- Combines multiple data sources (Census demographics, rent prices, building demolitions)
- **Uses a two-phase machine learning approach** with unsupervised clustering followed by supervised classification
- Employs three machine learning algorithms (Random Forest, XGBoost, Elastic Net)
- Generates interpretable risk scores and visualizations for policy action

**Target Users**: Urban planners, housing policy analysts, community organizations, and researchers working on displacement prevention.

## Methodology: Two-Phase Machine Learning Approach

This system uses a scientifically rigorous two-phase approach to avoid circular reasoning:

### Phase 1: Unsupervised Clustering (Pattern Discovery)
Instead of creating a synthetic "displacement_risk" variable from the same features used for prediction (which creates circular reasoning), we:
1. **Identify natural displacement patterns** using unsupervised clustering (K-means, Hierarchical, DBSCAN)
2. **Characterize clusters** based on rent pressure, demolition activity, and socioeconomic vulnerability
3. **Label clusters** as displacement risk types (e.g., "High Rent Growth + High Vulnerability")

### Phase 2: Supervised Classification (Risk Prediction)
We then train models to predict which cluster (displacement pattern) a neighborhood belongs to:
1. **Train classifiers** to predict cluster membership based on observable features
2. **Generate risk scores** by converting cluster probabilities to continuous risk scores
3. **Interpret results** as "This area resembles other high-risk displacement clusters"

**Key Advantages:**
- ✅ **Non-circular methodology**: Outcome is derived independently from predictors
- ✅ **Empirically-grounded**: Displacement patterns emerge from data, not assumptions
- ✅ **Interpretable**: Results explain which displacement type an area resembles
- ✅ **Discovery-oriented**: Can identify unexpected displacement patterns

## Key Features

- 🗺️ **Hexagonal Grid Analysis**: Uses H3 spatial indexing for consistent, efficient spatial analysis
- 🔬 **Cluster-Based Risk Assessment**: Unsupervised learning identifies displacement patterns
- 🤖 **Multiple ML Models**: Trains and compares Random Forest, XGBoost, and Elastic Net classifiers
- 📊 **Rich Visualizations**: Interactive maps, cluster profiles, static plots, and summary dashboards
- 📈 **Feature Importance**: Identifies key drivers of displacement risk
- 🎯 **Interpretable Predictions**: "This area resembles Cluster 2: High Rent Growth + High Vulnerability"
- 🔄 **Extensible Design**: Easy to add new data sources (evictions, land values, corporate ownership)
- 📚 **Educational**: Extensive comments explaining ML concepts for traditional statisticians

## Project Structure

```
coa-displacement-ews/
├── README.md                     # This file
├── packages.R                    # Package installation and loading
├── run_analysis.R               # Master pipeline script
│
├── 01_create_hex_grid.R         # Create hexagonal grid
├── 02_process_data.R            # Process and aggregate data
├── 03_feature_engineering.R     # Engineer features for ML
├── 03b_cluster_analysis.R       # **NEW: Unsupervised clustering**
├── 04_train_models.R            # Train ML models (cluster-based)
├── 05_validate_models.R         # Validate and diagnose models
├── 06_predict_risk_scores.R     # Generate risk scores
├── 07_visualize_results.R       # Create visualizations
│
├── R/
│   └── utils.R                  # Utility functions
│
├── data/                        # Input data (user-provided)
│   ├── demolitions.csv          # Building demolition records (optional)
│   ├── rent_prices.csv          # Rent price time series (optional)
│   └── README.md                # Data format specifications
│
├── output/                      # Generated data files
│   ├── hex_grid.rds             # Hexagonal grid
│   ├── hex_data_processed.rds   # Processed data
│   ├── hex_features.rds         # Engineered features
│   ├── cluster_analysis_results.rds  # **NEW: Clustering results**
│   ├── hex_features_with_clusters.rds # **NEW: Features + clusters**
│   ├── cluster_profiles.csv     # **NEW: Cluster characterizations**
│   ├── trained_models.rds       # Trained ML models
│   ├── validation_results.rds   # Model validation results
│   └── displacement_risk_scores.rds  # Final risk scores
│
└── figures/                     # Generated visualizations
    ├── 01_hex_grid_static.png
    ├── 03b_elbow_plot.png       # **NEW: Cluster optimization**
    ├── 03b_silhouette_plot.png  # **NEW: Cluster validation**
    ├── 03b_pca_clusters.png     # **NEW: Cluster visualization**
    ├── 03b_cluster_map.png      # **NEW: Geographic clusters**
    ├── 03b_cluster_profiles.png # **NEW: Cluster characteristics**
    ├── 07_interactive_risk_map.html
    ├── 07_summary_dashboard.png
    └── ...
```

## Installation

### Prerequisites

- **R**: Version 4.0 or higher
- **RStudio**: Recommended but not required
- **System Dependencies**: 
  - GDAL (for spatial operations)
  - GEOS (for spatial operations)
  - PROJ (for coordinate transformations)

On Ubuntu/Debian:
```bash
sudo apt-get install libgdal-dev libgeos-dev libproj-dev
```

On macOS (with Homebrew):
```bash
brew install gdal geos proj
```

### R Package Installation

1. Clone this repository:
```bash
git clone https://github.com/aakarner/coa-displacement-ews.git
cd coa-displacement-ews
```

2. Open R or RStudio and install required packages:
```r
source("packages.R")
```

This will automatically install all required packages including:
- Spatial: `sf`, `h3jsr`, `tigris`, `lwgeom`
- ML: `caret`, `randomForest`, `xgboost`, `glmnet`
- Clustering: `cluster`, `factoextra`, `dbscan`, `Rtsne`
- Data: `tidyverse`, `data.table`, `lubridate`
- Visualization: `leaflet`, `mapview`, `ggplot2`, `viridis`
- Census: `tidycensus`

### Census API Key Setup

The system uses Census ACS data. You need a free API key:

1. Get a key at: https://api.census.gov/data/key_signup.html
2. In R, run:
```r
library(tidycensus)
census_api_key("YOUR_KEY_HERE", install = TRUE)
```

## Usage

### Quick Start

Run the entire pipeline with one command:

```r
source("run_analysis.R")
```

This will:
1. Create hexagonal grid for Austin
2. Process Census and other data sources
3. Engineer features for machine learning
4. Train three ML models
5. Validate models
6. Generate risk scores
7. Create visualizations

**Estimated runtime**: 30-60 minutes (depending on hardware)

### Step-by-Step Execution

You can also run individual steps:

```r
# Load packages first
source("packages.R")

# Then run steps individually
source("01_create_hex_grid.R")
source("02_process_data.R")
source("03_feature_engineering.R")
source("03b_cluster_analysis.R")    # NEW: Clustering step
source("04_train_models.R")
source("05_validate_models.R")
source("06_predict_risk_scores.R")
source("07_visualize_results.R")
```

## Data Requirements

### Required Data

The system requires Census/ACS data, which is automatically downloaded via the `tidycensus` package. No manual data collection needed for basic functionality.

### Optional Data Sources

To enhance predictions, you can add:

#### 1. Building Demolitions (`data/demolitions.csv`)

Format:
```csv
demo_id,latitude,longitude,demo_date,building_type
1,30.267,-97.743,2021-06-15,Single Family
2,30.268,-97.744,2021-08-22,Multi-Family
```

#### 2. Rent Prices (`data/rent_prices.csv`)

Format:
```csv
hex_id,date,median_rent
1,2021-01-01,1200
1,2021-04-01,1250
```

#### 3. Future Data Sources (Placeholders Included)

The system has placeholder columns for:
- **Eviction filings**: Add to `02_process_data.R`
- **Land values**: Add to `02_process_data.R`
- **Corporate ownership**: Add to `02_process_data.R`

See `02_process_data.R` for integration points.

## Understanding the Models

This system uses three complementary machine learning approaches:

### 1. **Random Forest**
- **What it is**: Ensemble of decision trees voting together
- **Strengths**: Captures non-linear relationships, automatic interaction detection
- **Best for**: Robust predictions with minimal tuning
- **Interpretability**: Moderate (via feature importance)

### 2. **XGBoost (Gradient Boosting)**
- **What it is**: Sequential trees learning from previous errors
- **Strengths**: Often highest predictive performance
- **Best for**: Maximizing accuracy
- **Interpretability**: Moderate (via feature importance and SHAP values)

### 3. **Elastic Net**
- **What it is**: Regularized linear regression (L1 + L2 penalties)
- **Strengths**: Most similar to traditional regression, performs variable selection
- **Best for**: When interpretability is paramount
- **Interpretability**: High (coefficients directly interpretable)

### Ensemble Approach

The final risk scores combine all three models using a weighted average based on their validation performance (RMSE). This typically provides more robust predictions than any single model.

## Interpreting Results

### Risk Scores

- **Scale**: 0-100 (higher = more displacement risk)
- **Components**: 
  - 40% Rent increases (rapid housing cost growth)
  - 30% Demolitions (direct displacement events)
  - 30% Vulnerability (community susceptibility)

### Risk Categories

- **Low (0-25)**: Minimal displacement pressure
- **Moderate (25-50)**: Some risk factors present
- **High (50-75)**: Multiple risk factors converging
- **Very High (75-100)**: Acute displacement risk

### Key Outputs

1. **Interactive Map**: `figures/07_interactive_risk_map.html`
   - Click on hexagons to see details
   - Shows risk score and contributing factors

2. **Risk Scores CSV**: `output/displacement_risk_scores.csv`
   - Hex-level risk scores for further analysis
   - Includes coordinates for GIS integration

3. **Summary Dashboard**: `figures/07_summary_dashboard.png`
   - Overview of risk distribution
   - Key statistics and visualizations

## Adding New Data Sources

### Step-by-Step Guide

1. **Prepare your data** in CSV format with coordinates
2. **Add to `02_process_data.R`**:
   ```r
   # Load your data
   new_data <- read_csv("data/your_data.csv") %>%
     st_as_sf(coords = c("longitude", "latitude"), crs = 4326)
   
   # Aggregate to hexagons
   hex_with_new_data <- hex_grid %>%
     st_join(new_data) %>%
     group_by(hex_id) %>%
     summarise(new_metric = mean(value, na.rm = TRUE))
   ```
3. **Create features in `03_feature_engineering.R`**:
   ```r
   hex_features <- hex_features %>%
     mutate(new_feature = some_transformation(new_metric))
   ```
4. **Add to predictor variables in `04_train_models.R`**:
   ```r
   predictor_vars <- c(predictor_vars, "new_feature")
   ```

## Machine Learning Concepts for Beginners

### Training vs. Testing

- **Training Set (70%)**: Data used to build the model
- **Testing Set (30%)**: Held-out data to evaluate performance
- Like studying vs. taking an exam

### Cross-Validation

- Repeatedly split training data into smaller train/validate sets
- Helps tune hyperparameters without overfitting
- 5-fold CV = divide into 5 parts, train on 4, validate on 1, repeat 5 times

### Hyperparameters

Settings that control how the model learns:
- **Random Forest**: `mtry` (variables per split), `ntree` (number of trees)
- **XGBoost**: `eta` (learning rate), `max_depth` (tree depth), `nrounds` (iterations)
- **Elastic Net**: `alpha` (L1/L2 mix), `lambda` (penalty strength)

### Performance Metrics

- **RMSE** (Root Mean Squared Error): Average prediction error (lower = better)
- **MAE** (Mean Absolute Error): Typical absolute error (lower = better)
- **R²**: Proportion of variance explained (0-1, higher = better)

### Feature Importance

Shows which variables matter most for predictions:
- Similar to p-values or t-statistics in regression
- But based on predictive improvement, not statistical significance
- Higher importance = more influence on predictions

## Customization

### Change Grid Resolution

In `01_create_hex_grid.R`:
```r
H3_RESOLUTION <- 9  # Smaller cells (~0.1 km²)
```

### Adjust Risk Weights

In `06_predict_risk_scores.R`:
```r
displacement_risk = (
  0.5 * rent_risk +      # Increase rent weight to 50%
  0.3 * demo_risk +      # Keep demolitions at 30%
  0.2 * vuln_risk        # Reduce vulnerability to 20%
)
```

### Modify Model Parameters

In `04_train_models.R`, adjust the parameter grids for each model.

## Troubleshooting

### Common Issues

**Issue**: "Census API key not found"
- **Solution**: Run `tidycensus::census_api_key("YOUR_KEY", install = TRUE)`

**Issue**: Spatial packages won't install
- **Solution**: Install system dependencies (GDAL, GEOS, PROJ) first

**Issue**: Out of memory errors
- **Solution**: Increase H3 resolution (larger cells), reduce number of trees, or use a machine with more RAM

**Issue**: Models take too long to train
- **Solution**: Reduce the parameter grid size in `04_train_models.R` or use fewer CV folds

## References

### Academic Background

- Chapple, K. et al. (2017). "Developing a New Methodology for Analyzing Potential Displacement"
- Ding, L. et al. (2016). "Gentrification and Residential Mobility in Philadelphia"
- Freeman, L. (2005). "Displacement or Succession? Residential Mobility in Gentrifying Neighborhoods"

### Technical Resources

- **H3 Spatial Index**: https://h3geo.org/
- **Random Forests**: Breiman, L. (2001). Random Forests. Machine Learning.
- **XGBoost**: Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System
- **Elastic Net**: Zou & Hastie (2005). Regularization and Variable Selection via the Elastic Net

### Related Projects

- Urban Displacement Project (UC Berkeley)
- Eviction Lab (Princeton University)
- MAPC Displacement Risk Model (Boston)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [project contact]

## Acknowledgments

- City of Austin for open data
- U.S. Census Bureau for demographic data
- Urban Displacement Project for methodological guidance
- R community for excellent spatial and ML packages

---

**Last Updated**: 2026-02-11