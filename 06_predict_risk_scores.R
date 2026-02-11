################################################################################
# 06 - Generate Displacement Risk Scores and Classifications
################################################################################
#
# This script uses the trained models to generate displacement risk scores
# for all hexagonal cells in Austin. It produces:
# - Risk scores (0-100 scale)
# - Risk categories (Low, Moderate, High, Very High)
# - Contributing factors for each cell
# - Ensemble predictions combining multiple models
#
################################################################################

print_header("06 - GENERATING DISPLACEMENT RISK SCORES")

# Source utilities
source(here::here("R/utils.R"))

# Configuration
OUTPUT_DIR <- here::here("output")
set.seed(42)

################################################################################
# Step 1: Load models and data
################################################################################

print_progress("Loading trained models and feature data...")

models_list <- load_output(
  file.path(OUTPUT_DIR, "trained_models.rds"),
  "trained models"
)

hex_features <- load_output(
  file.path(OUTPUT_DIR, "hex_features.rds"),
  "engineered features"
)

# Extract components
model_elastic <- models_list$elastic_net
model_rf <- models_list$random_forest
model_xgb <- models_list$xgboost
predictor_vars <- models_list$predictor_vars

################################################################################
# Step 2: Prepare data for prediction
################################################################################

print_progress("Preparing data for prediction...")

# Create prediction dataset
# For cells with missing values, we'll use model-specific approaches
prediction_data <- hex_features %>%
  st_drop_geometry() %>%
  select(hex_id, h3_index, longitude, latitude, all_of(predictor_vars))

# Count how many cells have complete data
complete_cases <- sum(complete.cases(prediction_data[, predictor_vars]))
total_cases <- nrow(prediction_data)

cat(paste0("\nData completeness:\n"))
cat(paste0("  - Cells with complete data: ", complete_cases, 
          " (", round(complete_cases/total_cases*100, 1), "%)\n"))
cat(paste0("  - Cells with missing values: ", total_cases - complete_cases,
          " (", round((total_cases - complete_cases)/total_cases*100, 1), "%)\n"))

################################################################################
# Step 3: Generate predictions from each model
################################################################################

print_header("GENERATING PREDICTIONS")

print_progress("Generating predictions from Elastic Net model...")
# Elastic Net requires complete cases (no missing values)
pred_elastic <- rep(NA_real_, nrow(prediction_data))
complete_idx <- complete.cases(prediction_data[, predictor_vars])
pred_elastic[complete_idx] <- predict(
  model_elastic,
  newdata = prediction_data[complete_idx, predictor_vars]
)

print_progress("Generating predictions from Random Forest model...")
# Random Forest can handle some missing values
pred_rf <- predict(
  model_rf,
  newdata = prediction_data[, predictor_vars]
)

print_progress("Generating predictions from XGBoost model...")
# XGBoost can handle missing values
pred_xgb <- predict(
  model_xgb,
  newdata = prediction_data[, predictor_vars]
)

cat("\nPrediction coverage:\n")
cat(paste0("  - Elastic Net: ", sum(!is.na(pred_elastic)), " cells\n"))
cat(paste0("  - Random Forest: ", sum(!is.na(pred_rf)), " cells\n"))
cat(paste0("  - XGBoost: ", sum(!is.na(pred_xgb)), " cells\n"))

################################################################################
# Step 4: Create ensemble prediction
################################################################################

print_header("CREATING ENSEMBLE PREDICTIONS")

# CONCEPT: Ensemble = combining multiple models
# Often more robust than any single model
# Simple approach: Average predictions
# More sophisticated: Weighted average based on validation performance

print_progress("Combining models into ensemble prediction...")

# Get model performance weights from validation
performance <- models_list$performance
weights <- 1 / performance$RMSE  # Lower RMSE = higher weight
weights <- weights / sum(weights)  # Normalize to sum to 1

cat("\nEnsemble weights (based on test RMSE):\n")
cat(paste0("  - Elastic Net: ", round(weights[1], 3), "\n"))
cat(paste0("  - Random Forest: ", round(weights[2], 3), "\n"))
cat(paste0("  - XGBoost: ", round(weights[3], 3), "\n"))

# Create weighted ensemble
# Use simple average where elastic net is missing
risk_scores <- prediction_data %>%
  mutate(
    pred_elastic_net = pred_elastic,
    pred_random_forest = pred_rf,
    pred_xgboost = pred_xgb,
    
    # Weighted ensemble (when all models available)
    risk_score_ensemble = if_else(
      !is.na(pred_elastic_net),
      weights[1] * pred_elastic_net + weights[2] * pred_random_forest + weights[3] * pred_xgboost,
      (pred_random_forest + pred_xgboost) / 2  # Simple average if elastic net missing
    ),
    
    # Constrain to 0-100 range
    risk_score_ensemble = pmax(0, pmin(100, risk_score_ensemble)),
    
    # Also keep individual model scores
    risk_score_rf = pmax(0, pmin(100, pred_random_forest)),
    risk_score_xgb = pmax(0, pmin(100, pred_xgboost))
  )

################################################################################
# Step 5: Classify into risk categories
################################################################################

print_progress("Classifying cells into risk categories...")

risk_scores <- risk_scores %>%
  mutate(
    # Primary classification (using ensemble)
    risk_category = categorize_risk(risk_score_ensemble),
    
    # Also classify by individual models for comparison
    risk_category_rf = categorize_risk(risk_score_rf),
    risk_category_xgb = categorize_risk(risk_score_xgb)
  )

# Summary of risk categories
cat("\nRisk category distribution (Ensemble):\n")
print(table(risk_scores$risk_category))

cat("\nRisk category percentages:\n")
print(round(prop.table(table(risk_scores$risk_category)) * 100, 1))

################################################################################
# Step 6: Identify key contributing factors
################################################################################

print_header("IDENTIFYING KEY CONTRIBUTING FACTORS")

print_progress("Determining top contributing factors for each cell...")

# Get feature importance (average across models)
validation_results <- readRDS(file.path(OUTPUT_DIR, "validation_results.rds"))
feature_importance <- validation_results$feature_importance %>%
  arrange(desc(importance_avg))

# For each cell, identify which risk factors are elevated
# We'll flag features in the top 75th percentile for that feature

identify_contributing_factors <- function(data, top_n = 3) {
  factors <- character(nrow(data))
  
  # Get top important features to check
  top_features <- head(feature_importance$feature, 10)
  
  for(i in 1:nrow(data)) {
    cell_factors <- c()
    
    # Check each top feature
    for(feat in top_features) {
      if(feat %in% names(data)) {
        value <- data[[feat]][i]
        if(!is.na(value)) {
          # Get percentile of this value
          percentile <- ecdf(data[[feat]])(value)
          
          # If in top 25% for this risk factor, add it
          if(percentile >= 0.75) {
            cell_factors <- c(cell_factors, feat)
          }
        }
      }
    }
    
    # Take top N factors
    if(length(cell_factors) > 0) {
      factors[i] <- paste(head(cell_factors, top_n), collapse = "; ")
    } else {
      factors[i] <- "Multiple factors"
    }
  }
  
  return(factors)
}

# Get original feature data for contributing factor analysis
original_features <- hex_features %>%
  st_drop_geometry() %>%
  select(hex_id, any_of(predictor_vars))

risk_scores_with_factors <- risk_scores %>%
  left_join(original_features, by = "hex_id") %>%
  mutate(
    contributing_factors = identify_contributing_factors(., top_n = 3)
  )

# Clean up by removing the extra feature columns
risk_scores <- risk_scores_with_factors %>%
  select(hex_id, h3_index, longitude, latitude,
         pred_elastic_net, pred_random_forest, pred_xgboost,
         risk_score_ensemble, risk_score_rf, risk_score_xgb,
         risk_category, risk_category_rf, risk_category_xgb,
         contributing_factors)

################################################################################
# Step 7: Add spatial geometry back
################################################################################

print_progress("Adding spatial geometry to risk scores...")

# Join back to hex grid
hex_grid <- readRDS(file.path(OUTPUT_DIR, "hex_grid.rds"))

risk_scores_spatial <- hex_grid %>%
  select(hex_id, h3_index, geometry) %>%
  left_join(
    st_drop_geometry(risk_scores),
    by = c("hex_id", "h3_index")
  )

################################################################################
# Step 8: Generate summary statistics
################################################################################

print_header("RISK SCORE SUMMARY STATISTICS")

cat("\nEnsemble Risk Score Statistics:\n")
print(summary(risk_scores$risk_score_ensemble))

cat("\nRisk Categories:\n")
risk_category_summary <- risk_scores %>%
  count(risk_category) %>%
  mutate(percentage = round(n / sum(n) * 100, 1))
print(risk_category_summary)

cat("\nGeographic Distribution:\n")
cat(paste0("  - Mean risk score: ", round(mean(risk_scores$risk_score_ensemble, na.rm = TRUE), 2), "\n"))
cat(paste0("  - Median risk score: ", round(median(risk_scores$risk_score_ensemble, na.rm = TRUE), 2), "\n"))
cat(paste0("  - Standard deviation: ", round(sd(risk_scores$risk_score_ensemble, na.rm = TRUE), 2), "\n"))

# Top 10 highest risk cells
cat("\nTop 10 Highest Risk Cells:\n")
top_risk <- risk_scores %>%
  arrange(desc(risk_score_ensemble)) %>%
  head(10) %>%
  select(hex_id, longitude, latitude, risk_score_ensemble, risk_category, contributing_factors)
print(top_risk)

################################################################################
# Step 9: Save risk scores
################################################################################

print_progress("Saving risk scores...")

# Save as RDS (with spatial geometry)
save_output(
  risk_scores_spatial,
  file.path(OUTPUT_DIR, "displacement_risk_scores.rds"),
  "displacement risk scores (spatial)"
)

# Save as CSV (without geometry for easy inspection)
risk_scores_csv <- risk_scores_spatial %>%
  st_drop_geometry() %>%
  select(hex_id, h3_index, longitude, latitude,
         risk_score = risk_score_ensemble,
         risk_category,
         contributing_factors,
         risk_score_rf, risk_score_xgb)

write_csv(
  risk_scores_csv,
  file.path(OUTPUT_DIR, "displacement_risk_scores.csv")
)

print_progress("Also saved CSV version to: output/displacement_risk_scores.csv")

# Save summary statistics
summary_stats <- list(
  risk_category_counts = risk_category_summary,
  summary_statistics = data.frame(
    metric = c("Mean", "Median", "SD", "Min", "Max"),
    value = c(
      mean(risk_scores$risk_score_ensemble, na.rm = TRUE),
      median(risk_scores$risk_score_ensemble, na.rm = TRUE),
      sd(risk_scores$risk_score_ensemble, na.rm = TRUE),
      min(risk_scores$risk_score_ensemble, na.rm = TRUE),
      max(risk_scores$risk_score_ensemble, na.rm = TRUE)
    )
  ),
  top_10_highest_risk = top_risk
)

save_output(
  summary_stats,
  file.path(OUTPUT_DIR, "risk_score_summary.rds"),
  "risk score summary statistics"
)

################################################################################
# Summary
################################################################################

print_header("STEP 06 COMPLETE")
cat("✓ Predictions generated from all three models\n")
cat("✓ Ensemble predictions created using weighted average\n")
cat("✓ Risk scores calculated for all cells (0-100 scale)\n")
cat("✓ Cells classified into risk categories\n")
cat("✓ Contributing factors identified for each cell\n")
cat(paste0("✓ Spatial risk scores saved to: ", 
          file.path(OUTPUT_DIR, "displacement_risk_scores.rds"), "\n"))
cat(paste0("✓ CSV export saved to: ",
          file.path(OUTPUT_DIR, "displacement_risk_scores.csv"), "\n"))
