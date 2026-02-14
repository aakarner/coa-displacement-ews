################################################################################
# Main Analysis Pipeline - Displacement Early Warning System
################################################################################
#
# This is the master script that runs the complete displacement risk analysis
# pipeline from start to finish. It sources all component scripts in order.
#
# USAGE:
#   source("run_analysis.R")
#
# Or from command line:
#   Rscript run_analysis.R
#
# REQUIREMENTS:
#   - Run packages.R first to install required packages
#   - Census API key configured (see README for instructions)
#   - At least 8GB RAM recommended
#   - Estimated runtime: 30-60 minutes depending on hardware
#
################################################################################

# Clear workspace
rm(list = ls())

# Record start time
start_time <- Sys.time()

cat("\n")
cat("================================================================================\n")
cat("  DISPLACEMENT EARLY WARNING SYSTEM - FULL ANALYSIS PIPELINE\n")
cat("  Austin, TX\n")
cat(paste0("  Started: ", start_time, "\n"))
cat("================================================================================\n")
cat("\n")

################################################################################
# CONFIGURATION SECTION
################################################################################

# File paths (relative to project root)
CONFIG <- list(
  # Directories
  output_dir = "output",
  data_dir = "data", 
  figures_dir = "figures",
  
  # Hexagonal grid parameters
  # NOTE: Child scripts may override this with their own constants
  # (e.g., 01_create_hex_grid.R uses H3_RESOLUTION <- 9)
  h3_resolution = 9,  # Resolution 9 ≈ 0.1 km² cells
  
  # Census data
  acs_year = 2021,
  
  # Model training parameters
  train_test_split = 0.7,  # 70% training, 30% testing
  cv_folds = 5,            # 5-fold cross-validation
  random_seed = 42,        # For reproducibility
  
  # Risk score thresholds
  risk_thresholds = c(25, 50, 75),  # Low | Moderate | High | Very High
  
  # Runtime options
  verbose = TRUE,
  save_intermediate = TRUE
)

# Set global seed for reproducibility
set.seed(CONFIG$random_seed)

################################################################################
# CHECK PREREQUISITES
################################################################################

cat("Checking prerequisites...\n")

# Check if packages are loaded
if(!require(here, quietly = TRUE)) {
  stop("Package 'here' not found. Please run packages.R first.")
}

# Check if packages.R has been run
required_packages <- c("tidyverse", "sf", "h3jsr", "caret", "randomForest", "xgboost")
missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if(length(missing_packages) > 0) {
  cat("\nERROR: Missing required packages.\n")
  cat("Please run packages.R first to install all dependencies:\n")
  cat("  source('packages.R')\n\n")
  stop("Missing packages: ", paste(missing_packages, collapse = ", "))
}

cat("✓ All required packages available\n\n")

################################################################################
# LOAD PACKAGES
################################################################################

cat("Loading packages...\n")
source(here::here("packages.R"))
cat("\n")

################################################################################
# RUN ANALYSIS PIPELINE
################################################################################

# Step 1: Create hexagonal grid
cat("\n")
cat("################################################################################\n")
cat("# STEP 1/8: Creating hexagonal grid\n")
cat("################################################################################\n\n")

tryCatch({
  source(here::here("01_create_hex_grid.R"))
  cat("\n✓ Step 1 completed successfully\n")
}, error = function(e) {
  cat("\n✗ ERROR in Step 1:\n")
  cat(e$message, "\n")
  stop("Pipeline halted at Step 1")
})

# Step 2: Process data
cat("\n")
cat("################################################################################\n")
cat("# STEP 2/8: Processing and aggregating data\n")
cat("################################################################################\n\n")

tryCatch({
  source(here::here("02_process_data.R"))
  cat("\n✓ Step 2 completed successfully\n")
}, error = function(e) {
  cat("\n✗ ERROR in Step 2:\n")
  cat(e$message, "\n")
  stop("Pipeline halted at Step 2")
})

# Step 3: Feature engineering
cat("\n")
cat("################################################################################\n")
cat("# STEP 3/8: Engineering features\n")
cat("################################################################################\n\n")

tryCatch({
  source(here::here("03_feature_engineering.R"))
  cat("\n✓ Step 3 completed successfully\n")
}, error = function(e) {
  cat("\n✗ ERROR in Step 3:\n")
  cat(e$message, "\n")
  stop("Pipeline halted at Step 3")
})

# Step 3b: Cluster analysis
cat("\n")
cat("################################################################################\n")
cat("# STEP 3b/8: Cluster analysis for displacement patterns\n")
cat("################################################################################\n\n")

tryCatch({
  source(here::here("03b_cluster_analysis.R"))
  cat("\n✓ Step 3b completed successfully\n")
}, error = function(e) {
  cat("\n✗ ERROR in Step 3b:\n")
  cat(e$message, "\n")
  stop("Pipeline halted at Step 3b")
})

# Step 4: Train models
cat("\n")
cat("################################################################################\n")
cat("# STEP 4/8: Training machine learning models\n")
cat("################################################################################\n\n")

tryCatch({
  source(here::here("04_train_models.R"))
  cat("\n✓ Step 4 completed successfully\n")
}, error = function(e) {
  cat("\n✗ ERROR in Step 4:\n")
  cat(e$message, "\n")
  stop("Pipeline halted at Step 4")
})

# Step 5: Validate models
cat("\n")
cat("################################################################################\n")
cat("# STEP 5/8: Validating models\n")
cat("################################################################################\n\n")

tryCatch({
  source(here::here("05_validate_models.R"))
  cat("\n✓ Step 5 completed successfully\n")
}, error = function(e) {
  cat("\n✗ ERROR in Step 5:\n")
  cat(e$message, "\n")
  stop("Pipeline halted at Step 5")
})

# Step 6: Generate risk scores
cat("\n")
cat("################################################################################\n")
cat("# STEP 6/8: Generating displacement risk scores\n")
cat("################################################################################\n\n")

tryCatch({
  source(here::here("06_predict_risk_scores.R"))
  cat("\n✓ Step 6 completed successfully\n")
}, error = function(e) {
  cat("\n✗ ERROR in Step 6:\n")
  cat(e$message, "\n")
  stop("Pipeline halted at Step 6")
})

# Step 7: Visualize results
cat("\n")
cat("################################################################################\n")
cat("# STEP 7/8: Creating visualizations\n")
cat("################################################################################\n\n")

tryCatch({
  source(here::here("07_visualize_results.R"))
  cat("\n✓ Step 7 completed successfully\n")
}, error = function(e) {
  cat("\n✗ ERROR in Step 7:\n")
  cat(e$message, "\n")
  stop("Pipeline halted at Step 7")
})

################################################################################
# FINAL SUMMARY
################################################################################

end_time <- Sys.time()
elapsed_time <- difftime(end_time, start_time, units = "mins")

cat("\n\n")
cat("================================================================================\n")
cat("  PIPELINE COMPLETED SUCCESSFULLY!\n")
cat("================================================================================\n\n")

cat("Analysis Summary:\n")
cat(paste0("  - Started:  ", format(start_time, "%Y-%m-%d %H:%M:%S"), "\n"))
cat(paste0("  - Finished: ", format(end_time, "%Y-%m-%d %H:%M:%S"), "\n"))
cat(paste0("  - Runtime:  ", round(elapsed_time, 1), " minutes\n\n"))

cat("Output Files:\n")
cat("  - Hexagonal grid:        output/hex_grid.rds\n")
cat("  - Processed data:        output/hex_data_processed.rds\n")
cat("  - Engineered features:   output/hex_features.rds\n")
cat("  - Cluster analysis:      output/cluster_analysis_results.rds\n")
cat("  - Features w/ clusters:  output/hex_features_with_clusters.rds\n")
cat("  - Cluster profiles:      output/cluster_profiles.csv\n")
cat("  - Trained models:        output/trained_models.rds\n")
cat("  - Validation results:    output/validation_results.rds\n")
cat("  - Risk scores (spatial): output/displacement_risk_scores.rds\n")
cat("  - Risk scores (CSV):     output/displacement_risk_scores.csv\n\n")

cat("Visualizations:\n")
cat("  - All maps and plots saved to: figures/\n")
cat("  - Interactive map:       figures/07_interactive_risk_map.html\n")
cat("  - Summary dashboard:     figures/07_summary_dashboard.png\n")
cat("  - See visualization_index.csv for complete list\n\n")

cat("Next Steps:\n")
cat("  1. Review the interactive map in your web browser\n")
cat("  2. Examine model performance plots in figures/\n")
cat("  3. Check risk_scores.csv for detailed results\n")
cat("  4. Customize analysis by modifying configuration in this script\n")
cat("  5. Add new data sources to 02_process_data.R as they become available\n\n")

cat("For questions or issues, consult the README.md file.\n\n")

cat("================================================================================\n")
