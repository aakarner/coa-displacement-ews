################################################################################
# Package Installation and Loading for Displacement Early Warning System
################################################################################
# 
# This script installs and loads all required packages for the displacement
# early warning system. Run this once before running the analysis pipeline.
#
# Author: COA Displacement EWS Team
# Date: 2026-02-11
################################################################################

# Function to install packages if not already installed
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages) > 0) {
    message("Installing missing packages: ", paste(new_packages, collapse = ", "))
    install.packages(new_packages, dependencies = TRUE, repos = "https://cran.rstudio.com/")
  } else {
    message("All required packages are already installed.")
  }
}

# Define required packages by category
packages <- list(
  # Spatial analysis and mapping
  spatial = c("sf", "h3jsr", "tigris", "lwgeom", "spdep"),
  
  # Machine Learning
  ml = c("caret", "randomForest", "xgboost", "glmnet"),
  
  # Model validation and cross-validation
  validation = c("blockCV"),
  
  # Clustering and dimensionality reduction
  clustering = c("cluster", "factoextra", "dbscan", "Rtsne"),
  
  # Data manipulation and processing
  data = c("tidyverse", "data.table", "lubridate"),
  
  # Visualization
  viz = c("leaflet", "mapview", "ggplot2", "viridis", "scales", "patchwork", "gridExtra", "htmlwidgets"),
  
  # Census data
  census = c("tidycensus"),
  
  # Additional utilities
  utils = c("here", "janitor", "tictoc")
)

# Flatten the list
all_packages <- unlist(packages, use.names = FALSE)

# Install missing packages
message("Checking and installing required packages...")
install_if_missing(all_packages)

# Load all packages
message("\nLoading packages...")
suppressPackageStartupMessages({
  # Spatial
  library(sf)
  library(h3jsr)
  library(tigris)
  library(lwgeom)
  library(spdep)
  
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
  
  # Visualization
  library(leaflet)
  library(mapview)
  library(viridis)
  library(scales)
  library(patchwork)
  library(gridExtra)
  library(htmlwidgets)
  
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
