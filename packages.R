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
  spatial = c("sf", "h3jsr", "tigris", "lwgeom"),
  
  # Machine Learning
  ml = c("caret", "randomForest", "xgboost", "glmnet"),
  
  # Data manipulation and processing
  data = c("tidyverse", "data.table", "lubridate"),
  
  # Visualization
  viz = c("leaflet", "mapview", "ggplot2", "viridis", "scales", "patchwork"),
  
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
  
  # ML
  library(caret)
  library(randomForest)
  library(xgboost)
  library(glmnet)
  
  # Data manipulation
  library(tidyverse)
  library(data.table)
  library(lubridate)
  
  # Visualization
  library(leaflet)
  library(mapview)
  library(ggplot2)
  library(viridis)
  library(scales)
  library(patchwork)
  
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
set.seed(42)

message("\n✓ All packages loaded successfully!")
message("✓ Random seed set to 42 for reproducibility")
message("\nReady to run displacement early warning system analysis.")
