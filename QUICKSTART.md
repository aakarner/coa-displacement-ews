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
source("packages.R")
```

This will install all required packages. **Time**: ~5-10 minutes

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
3. ✓ Engineer features
4. ✓ Train three ML models
5. ✓ Validate models
6. ✓ Generate risk scores
7. ✓ Create visualizations

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
packages.R          → Load all required R packages
01_create_hex_grid → Create H3 hexagonal grid over Austin
02_process_data    → Download Census data, aggregate to hexagons
03_feature_eng...  → Create 20+ features for ML models
04_train_models    → Train Random Forest, XGBoost, Elastic Net
05_validate_models → Cross-validate and diagnose models
06_predict_risk... → Generate 0-100 risk scores for each hex
07_visualize_...   → Create maps and plots
```

## Quick Reference

| File | Purpose |
|------|---------|
| `packages.R` | Install/load packages |
| `run_analysis.R` | Run full pipeline |
| `01-07_*.R` | Individual analysis steps |
| `output/*.rds` | Intermediate data files |
| `output/*.csv` | Export-ready data |
| `figures/*.png` | Static visualizations |
| `figures/*.html` | Interactive maps |

---

**Questions?** See the full [README.md](README.md) or open an issue.
