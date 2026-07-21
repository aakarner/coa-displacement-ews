################################################################################
# Package Installation and Loading for Displacement Early Warning System
################################################################################
# 
# This script verifies and loads all required packages for the displacement
# early warning system. Package installation is handled by 00_requirements.R.
#
# Author: COA Displacement EWS Team
# Date: 2026-02-11
################################################################################

if (!exists("EWS_REQUIRED_PACKAGES", inherits = FALSE)) {
  source(file.path(getwd(), "00_requirements.R"), local = FALSE)
}

# Load all packages
message("\nLoading packages...")
suppressPackageStartupMessages({
  # Spatial
  library(sf)
  library(h3jsr)
  library(tigris)
  library(lwgeom)
  library(spdep)
  library(tidygeocoder)
  library(arcgisgeocode)
  library(arcgisutils)
  
  # ML
  library(caret)
  library(randomForest)
  library(xgboost)
  library(glmnet)
  
  # Model validation
  library(blockCV)
  
  # Clustering
  library(cluster)
  library(factoextra)
  library(dbscan)
  library(Rtsne)
  
  # Data manipulation
  library(tidyverse)
  library(data.table)
  library(lubridate)
  library(readxl)
  
  # Visualization
  library(leaflet)
  library(mapview)
  library(viridis)
  library(scales)
  library(patchwork)
  library(gridExtra)
  library(htmlwidgets)
  library(ggthemes)
  library(classInt)
  library(ggspatial)
  library(rosm)
  
  # Census
  library(tidycensus)
  
  # Utils
  library(here)
  library(janitor)
  library(tictoc)
})

# Set global options
options(tigris_use_cache = TRUE)  # Cache census geography downloads
options(scipen = 999)              # Avoid scientific notation
sf_use_s2(FALSE)                   # Disable s2 for simpler spatial operations

# Set seed for reproducibility
# NOTE: This seed is set here for standalone execution of packages.R
# When running the full pipeline via run_analysis.R, that script also sets
# the seed, ensuring reproducibility across the entire analysis
set.seed(42)

message("\n✓ All packages loaded successfully!")
message("✓ Random seed set to 42 for reproducibility")
message("\nReady to run displacement early warning system analysis.")
