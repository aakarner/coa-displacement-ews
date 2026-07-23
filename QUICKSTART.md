# Quick Start Guide

Get up and running with the Displacement Early Warning System in 5 minutes.

## Prerequisites Check

Before starting, ensure you have:
- [ ] R version 4.0+ installed
- [ ] RStudio (recommended) or R command line
- [ ] At least 8GB RAM
- [ ] Internet connection (for downloading Census data)

## Step 1: Clone Repository

```bash
git clone https://github.com/aakarner/coa-displacement-ews.git
cd coa-displacement-ews
```

## Step 2: Install R Packages

Open R or RStudio and run:

```r
source("00_requirements.R")
```

This installs missing packages into a version-specific `.r-library` directory
inside the project. The project `.Rprofile` activates that library automatically
for later R sessions. **Time**: ~5-10 minutes on the first run.

## Step 3: Set Up Census API Key

1. Get a free API key at: https://api.census.gov/data/key_signup.html
2. In R, run:

```r
library(tidycensus)
census_api_key("YOUR_KEY_HERE", install = TRUE)
```

## Step 4: Run the Analysis

```r
source("run_analysis.R")
```

**Time**: ~30-60 minutes depending on your hardware

The script will:
1. ✓ Create hexagonal grid for Austin
2. ✓ Download and process Census data
3. ✓ Build the 2021-2025 county appraisal panels
4. ✓ Build ownership-change and transaction-pressure measures
5. ✓ Engineer features
6. ✓ Train three ML models
7. ✓ Validate models
8. ✓ Generate risk scores and visualizations

## Running Individual Scripts

If you want to run scripts individually (rather than the full `run_analysis.R`), you'll need to source the utility functions first:

```r
source("R/utils.R")
```

Then you can run individual scripts:

```r
source("01_create_hex_grid.R")
source("02_process_data.R")
source("02i_process_appraisal_history.R")
source("02j_process_appraisal_adjusted_trends.R")
source("02k_audit_ownership_transactions.R")
source("02l_process_ownership_transactions.R")
source("02m_audit_amenity_sources.R")
source("02n_process_amenity_change.R")
source("03_feature_engineering.R")
source("03b_cluster_analysis.R")
source("03c_cluster_sensitivity_analysis.R")
source("03d_amenity_cluster_sensitivity.R")
source("03e_visualize_amenity_clusters.R")
# etc.
```

**Note**: The utility functions in `utils.R` (like `print_header()` and `print_progress()`) are used throughout the analysis scripts.
The first appraisal run downloads and caches large certified archives under
`data/raw_parcels/appraisal_history/`; later runs use the normalized caches.
Run `02j_process_appraisal_adjusted_trends.R` after `02i` to remove county-wide
appraisal-year shifts before using land-value trends in feature engineering.
The ownership/transaction scripts also require the cached Travis deed extract
from the sibling `landlord-mapper` repository and the local county owner/sales
files documented in `data/README.md`.
The amenity scripts download and cache public Texas Comptroller and City of
Austin API extracts under `data/raw_amenities/`. An optional Socrata app token
can be supplied through `SOCRATA_APP_TOKEN`; no token is required for cached
reruns.

`02_process_data.R` builds the calibrated residential parcel support before
running `02f_process_acs_demographics.R`. ACS additive counts are allocated
from block groups through 2020 Census blocks and residential parcel floor-area
support; suppressed block-group medians fall back to dominant tracts. Census
downloads are cached under `data/raw_acs/`, so later runs are local.

`03d_amenity_cluster_sensitivity.R` is a focused, noncanonical comparison of
the six-domain baseline against the same matrix plus amenity change. It reports
silhouette, gap-statistic, repeated-subsample stability, cluster agreement, and
profiles without replacing the primary `03b` assignments. Its bootstrap counts
can be changed with `CLUSTER_GAP_BOOTSTRAPS` and
`CLUSTER_STABILITY_REPLICATES`.
The `03e` script maps the selected amenity solution using the tentative names,
concern levels, and colors in `config/amenity_cluster_labels_k6.csv`. It creates
both a publication-ready PNG and a zoomable HTML map with per-hex profiles.

## Step 5: View Results

### Interactive Map
Open in your browser:
```
figures/07_interactive_risk_map.html
```

### Summary Dashboard
View:
```
figures/07_summary_dashboard.png
```

### Risk Scores Data
Import into Excel/GIS:
```
output/displacement_risk_scores.csv
```

## Common Issues

### "Census API key not found"
**Solution**: Did you run Step 3? Make sure to run both lines.

### "Package installation failed"
**Solution**: You may need to install system dependencies first:
- **Ubuntu/Debian**: `sudo apt-get install libgdal-dev libgeos-dev libproj-dev`
- **macOS**: `brew install gdal geos proj`

### "Out of memory"
**Solution**: Close other applications or use a machine with more RAM.

## Next Steps

1. **Explore the interactive map** - Click on hexagons to see risk factors
2. **Review model performance** - Check `figures/05_*.png` for validation plots
3. **Add your own data** - See `data/README.md` for format specifications
4. **Customize the analysis** - Edit parameters in `run_analysis.R`

## Getting Help

- **Full documentation**: See [README.md](README.md)
- **Data format help**: See [data/README.md](data/README.md)
- **Issues**: Open an issue on GitHub

## What's Happening Behind the Scenes

```
00_requirements.R   → Install missing packages in the project library
packages.R          → Load all required R packages
01_create_hex_grid → Create H3 hexagonal grid over Austin
02_process_data    → Download Census data, aggregate to hexagons
03_feature_eng...  → Create 20+ features for ML models
04_train_models    → Train Random Forest, XGBoost, Elastic Net
05_validate_models → Cross-validate and diagnose models
06_predict_risk... → Generate 0-100 risk scores for each hex
07_visualize_...   → Create maps and plots
```
