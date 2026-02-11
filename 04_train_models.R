################################################################################
# 04 - Train Machine Learning Models for Displacement Risk
################################################################################
#
# EDUCATIONAL OVERVIEW FOR TRADITIONAL STATISTICIANS
# ====================================================
# This script trains three different machine learning approaches and compares
# them to traditional statistical methods you may be familiar with:
#
# 1. RANDOM FOREST
#    - Think of it as: Many decision trees voting together
#    - Similar to: Multiple regression trees, but more robust
#    - Advantage: Automatically captures non-linear relationships and interactions
#    - No assumptions about linearity, normality, or homoscedasticity
#
# 2. GRADIENT BOOSTING (XGBoost)
#    - Think of it as: Sequential trees that learn from previous mistakes
#    - Similar to: Iterative regression, but with trees
#    - Advantage: Often highest predictive performance
#    - Like iteratively reweighted regression, focusing on hard-to-predict cases
#
# 3. ELASTIC NET REGRESSION
#    - Think of it as: Traditional linear regression with penalties
#    - Similar to: Multiple regression, but with L1 (Lasso) + L2 (Ridge) penalties
#    - Advantage: Prevents overfitting, performs variable selection
#    - Most similar to traditional approaches
#
# KEY MACHINE LEARNING CONCEPTS:
# - Training/Testing Split: Evaluate on unseen data (like out-of-sample validation)
# - Cross-Validation: K-fold approach to avoid overfitting (similar to jackknife)
# - Hyperparameters: Model settings we tune (unlike traditional fixed methods)
# - Feature Importance: Which variables matter most (similar to t-statistics/p-values)
# - RMSE/MAE: Prediction error metrics (similar to residual standard error)
#
################################################################################

print_header("04 - TRAINING MACHINE LEARNING MODELS")

# Source utilities
source(here::here("R/utils.R"))

# Configuration
OUTPUT_DIR <- here::here("output")
set.seed(42)  # For reproducibility - ensures same random splits each time

################################################################################
# Step 1: Load engineered features
################################################################################

print_progress("Loading engineered features...")
hex_features <- load_output(
  file.path(OUTPUT_DIR, "hex_features.rds"),
  "engineered features"
)

################################################################################
# Step 2: Define outcome variable (target for prediction)
################################################################################

print_header("DEFINING OUTCOME VARIABLE")

# IMPORTANT CONCEPT: In supervised learning, we need a "target" or "outcome" 
# variable to predict. This is like your dependent variable in regression.
#
# For displacement risk, we create a COMPOSITE OUTCOME combining:
# - Rent increases (market pressure)
# - Demolitions (direct displacement)
# - Vulnerability (community susceptibility)

print_progress("Creating composite displacement risk outcome...")

model_data <- hex_features %>%
  st_drop_geometry() %>%  # Remove spatial geometry for modeling
  filter(sufficient_data) %>%  # Only use observations with enough data
  mutate(
    # Standardize components to 0-100 scale
    rent_risk = normalize_to_100(rent_change_total),
    demo_risk = normalize_to_100(demo_density),
    vuln_risk = normalize_to_100(vulnerability_index),
    
    # Composite outcome: weighted average of risk factors
    # You can adjust these weights based on domain knowledge
    displacement_risk = (
      0.4 * rent_risk +      # 40% weight on rent increases
      0.3 * demo_risk +      # 30% weight on demolitions
      0.3 * vuln_risk        # 30% weight on vulnerability
    ),
    
    # Also create binary outcome for classification (high risk vs. not)
    high_risk = if_else(displacement_risk > 60, 1, 0)
  )

cat("\nOutcome variable summary:\n")
cat("  - displacement_risk: Continuous 0-100 score\n")
cat("  - high_risk: Binary indicator (>60 threshold)\n")
print(summary(model_data$displacement_risk))

################################################################################
# Step 3: Prepare feature matrix (predictors)
################################################################################

print_header("PREPARING PREDICTOR VARIABLES")

# Select predictor variables (features) for modeling
# Exclude the outcome and derivative variables
predictor_vars <- c(
  # Rent features
  "rent_change_recent", "rent_acceleration", "rent_volatility",
  "rent_level_ratio",
  
  # Demolition features  
  "demo_density", "demo_recent", "has_recent_demos",
  
  # Socioeconomic
  "median_income", "pct_renter", "pct_college", "poverty_rate",
  "pct_poc", "rent_burden_proxy",
  
  # Spatial lags
  "rent_change_total_lag", "demo_density_lag", "median_income_lag",
  
  # Interactions
  "rent_vuln_interaction", "neighborhood_rent_pressure"
)

print_progress(paste0("Using ", length(predictor_vars), " predictor variables"))

# Create clean modeling dataset
model_data_clean <- model_data %>%
  select(hex_id, displacement_risk, high_risk, all_of(predictor_vars)) %>%
  drop_na()  # Remove any remaining rows with missing values

cat(paste0("\nFinal modeling dataset: ", nrow(model_data_clean), " observations\n"))
cat(paste0("Removed ", nrow(model_data) - nrow(model_data_clean), 
          " observations due to missing values\n"))

################################################################################
# Step 4: Train/Test Split
################################################################################

print_header("TRAIN/TEST SPLIT")

# CONCEPT: We split data into training (build model) and testing (evaluate model)
# This is like out-of-sample validation in traditional statistics
# Typical split: 70-80% training, 20-30% testing

print_progress("Splitting data into training (70%) and testing (30%) sets...")

train_index <- createDataPartition(
  model_data_clean$displacement_risk, 
  p = 0.7,           # 70% for training
  list = FALSE       # Return as matrix, not list
)

train_data <- model_data_clean[train_index, ]
test_data <- model_data_clean[-train_index, ]

cat(paste0("  - Training set: ", nrow(train_data), " observations\n"))
cat(paste0("  - Testing set: ", nrow(test_data), " observations\n"))

################################################################################
# Step 5: Set up cross-validation
################################################################################

print_header("CROSS-VALIDATION SETUP")

# CONCEPT: Cross-validation repeatedly splits training data to tune models
# K-Fold: Split training data into K parts, train on K-1, validate on 1
# Repeat K times with different validation part each time
# This is similar to jackknife or bootstrap validation

print_progress("Setting up 5-fold cross-validation...")

train_control <- trainControl(
  method = "cv",           # Cross-validation method
  number = 5,              # 5 folds (common choice: 5 or 10)
  verboseIter = TRUE,      # Show progress
  savePredictions = "final"  # Save predictions for analysis
)

################################################################################
# Step 6: Train Elastic Net Model
################################################################################

print_header("MODEL 1: ELASTIC NET REGRESSION")

# EDUCATIONAL NOTE:
# Elastic Net is MOST SIMILAR to traditional multiple regression, but with
# important differences:
# 
# Traditional Regression: y = β₀ + β₁x₁ + β₂x₂ + ... + ε
#   - Minimizes: Sum of squared residuals (SSR)
#   - Problem: Can overfit with many predictors
#
# Elastic Net: Same equation, but minimizes:
#   SSR + λ * (α * |β| + (1-α) * β²)
#   
#   Where:
#   - λ (lambda): Overall penalty strength (higher = more shrinkage)
#   - α (alpha): Mix of L1 (Lasso) and L2 (Ridge) penalties
#     - α = 1: Pure Lasso (some coefficients go to exactly 0, variable selection)
#     - α = 0: Pure Ridge (all coefficients shrunk but non-zero)
#     - α = 0.5: Equal mix of both
#
# Benefits:
# - Prevents overfitting by penalizing large coefficients
# - Performs automatic variable selection (like stepwise, but better)
# - Handles multicollinearity well

print_progress("Training Elastic Net model with cross-validation...")

# Define parameter grid to search
# caret will try all combinations and pick the best
elastic_grid <- expand.grid(
  alpha = c(0, 0.5, 1),      # Try Ridge, Elastic Net, Lasso
  lambda = seq(0.001, 1, length.out = 20)  # Try 20 penalty values
)

print_progress("Searching across parameter grid...")
print_progress(paste0("  - Testing ", nrow(elastic_grid), " parameter combinations"))

tic("Elastic Net training")
model_elastic <- train(
  x = train_data[, predictor_vars],
  y = train_data$displacement_risk,
  method = "glmnet",           # Elastic Net method
  trControl = train_control,
  tuneGrid = elastic_grid,
  preProcess = c("center", "scale"),  # Standardize predictors (important for penalties)
  metric = "RMSE"              # Optimize for Root Mean Squared Error
)
toc()

# Show best parameters
cat("\nBest Elastic Net parameters:\n")
print(model_elastic$bestTune)

# Extract and display feature importance
# In Elastic Net, this is based on absolute coefficient values
elastic_importance <- varImp(model_elastic)
cat("\nTop 10 most important features (Elastic Net):\n")
print(head(elastic_importance$importance[order(-elastic_importance$importance$Overall), , drop = FALSE], 10))

################################################################################
# Step 7: Train Random Forest Model
################################################################################

print_header("MODEL 2: RANDOM FOREST")

# EDUCATIONAL NOTE:
# Random Forest is quite different from traditional regression:
#
# How it works:
# 1. Create many (e.g., 500) decision trees
# 2. Each tree is trained on a random sample of data (with replacement)
# 3. At each split, only consider random subset of predictors
# 4. Final prediction: Average across all trees
#
# Think of it as: "Wisdom of crowds" approach
# - Each tree is somewhat different (due to randomness)
# - Trees vote or average their predictions
# - Reduces overfitting compared to single tree
#
# Key hyperparameters:
# - mtry: Number of variables to consider at each split
#   - For regression, typical default is p/3 (where p = number of predictors)
#   - Smaller mtry = more randomness, less correlation between trees
# - ntree: Number of trees (more is usually better, but slower)
# - nodesize: Minimum observations in leaf nodes (like minimum cell size)
#
# Advantages:
# - No linearity assumptions
# - Automatically captures interactions
# - Handles missing data well
# - Resistant to outliers
# - Provides feature importance rankings
#
# Disadvantages:
# - Less interpretable than regression
# - Can be slow with many trees
# - Predictions don't extrapolate beyond training data range

print_progress("Training Random Forest model...")

# Define parameter grid
# We'll tune mtry (variables at each split)
rf_grid <- expand.grid(
  mtry = c(3, 5, 7, 10)  # Try different numbers of variables per split
)

print_progress(paste0("  - Testing ", nrow(rf_grid), " values of mtry"))

tic("Random Forest training")
model_rf <- train(
  x = train_data[, predictor_vars],
  y = train_data$displacement_risk,
  method = "rf",               # Random Forest
  trControl = train_control,
  tuneGrid = rf_grid,
  ntree = 500,                 # Number of trees (fixed at 500)
  importance = TRUE,           # Calculate feature importance
  metric = "RMSE"
)
toc()

# Show best parameters
cat("\nBest Random Forest parameters:\n")
print(model_rf$bestTune)

# Feature importance
# In Random Forest, importance based on how much each variable 
# improves predictions across all trees
rf_importance <- varImp(model_rf)
cat("\nTop 10 most important features (Random Forest):\n")
print(head(rf_importance$importance[order(-rf_importance$importance$Overall), , drop = FALSE], 10))

################################################################################
# Step 8: Train XGBoost Model
################################################################################

print_header("MODEL 3: GRADIENT BOOSTING (XGBoost)")

# EDUCATIONAL NOTE:
# XGBoost (eXtreme Gradient Boosting) is often the highest-performing ML method:
#
# How it works:
# 1. Start with a simple prediction (e.g., mean of outcome)
# 2. Build a tree to predict the RESIDUALS (errors)
# 3. Add this tree to the model with a learning rate (shrinkage)
# 4. Repeat: Each new tree tries to correct remaining errors
#
# This is somewhat like iteratively reweighted regression, where you
# keep focusing on the cases you're getting wrong.
#
# Key differences from Random Forest:
# - Sequential (one tree at a time) vs. Parallel (all trees independent)
# - Each tree learns from previous trees' mistakes
# - Typically uses shallower trees (depth 3-6 vs. full depth)
# - Requires more careful tuning
#
# Key hyperparameters:
# - nrounds: Number of trees (boosting iterations)
# - max_depth: How deep each tree can be (like interaction depth)
# - eta: Learning rate (shrinkage). Smaller = more conservative
#   - Range: 0.01-0.3 (smaller is better but needs more trees)
# - colsample_bytree: Fraction of features used per tree (like RF's mtry)
# - subsample: Fraction of data used per tree (introduces randomness)
# - gamma: Minimum loss reduction to make a split (regularization)
#
# Advantages:
# - Often best predictive performance
# - Handles missing data
# - Fast training (with GPU support)
# - Good with interactions and non-linearity
#
# Disadvantages:
# - Can overfit if not tuned carefully
# - More hyperparameters to tune
# - Less interpretable

print_progress("Training XGBoost model...")

# Define parameter grid for tuning
# We'll tune several key parameters
xgb_grid <- expand.grid(
  nrounds = c(100, 200, 300),           # Number of trees
  max_depth = c(3, 4, 6),                # Tree depth
  eta = c(0.01, 0.05, 0.1),             # Learning rate
  gamma = c(0, 0.1),                     # Minimum loss reduction
  colsample_bytree = c(0.6, 0.8),       # Column sampling
  min_child_weight = 1,                  # Minimum observations in leaf
  subsample = c(0.7, 0.8)               # Row sampling
)

print_progress(paste0("  - Testing ", nrow(xgb_grid), " parameter combinations"))
print_progress("  - This may take several minutes...")

tic("XGBoost training")
model_xgb <- train(
  x = train_data[, predictor_vars],
  y = train_data$displacement_risk,
  method = "xgbTree",          # XGBoost with trees
  trControl = train_control,
  tuneGrid = xgb_grid,
  metric = "RMSE",
  verbosity = 0                # Reduce XGBoost output
)
toc()

# Show best parameters
cat("\nBest XGBoost parameters:\n")
print(model_xgb$bestTune)

# Feature importance
# In XGBoost, importance based on gain (improvement in accuracy)
xgb_importance <- varImp(model_xgb)
cat("\nTop 10 most important features (XGBoost):\n")
print(head(xgb_importance$importance[order(-xgb_importance$importance$Overall), , drop = FALSE], 10))

################################################################################
# Step 9: Compare model performance on test set
################################################################################

print_header("MODEL PERFORMANCE COMPARISON")

print_progress("Evaluating all models on held-out test set...")

# Make predictions on test set
pred_elastic <- predict(model_elastic, newdata = test_data[, predictor_vars])
pred_rf <- predict(model_rf, newdata = test_data[, predictor_vars])
pred_xgb <- predict(model_xgb, newdata = test_data[, predictor_vars])

# Calculate performance metrics
# RMSE: Root Mean Squared Error (average prediction error, same units as outcome)
# MAE: Mean Absolute Error (average absolute error, more robust to outliers)
# R²: R-squared (proportion of variance explained, 0-1, higher is better)

calc_metrics <- function(pred, actual) {
  data.frame(
    RMSE = sqrt(mean((pred - actual)^2)),
    MAE = mean(abs(pred - actual)),
    R2 = cor(pred, actual)^2
  )
}

performance <- rbind(
  Elastic_Net = calc_metrics(pred_elastic, test_data$displacement_risk),
  Random_Forest = calc_metrics(pred_rf, test_data$displacement_risk),
  XGBoost = calc_metrics(pred_xgb, test_data$displacement_risk)
)

cat("\nTest Set Performance Metrics:\n")
print(round(performance, 3))

# Interpretation guide
cat("\nInterpretation:\n")
cat("  - RMSE: Lower is better (average prediction error in risk score points)\n")
cat("  - MAE: Lower is better (typical absolute error)\n")
cat("  - R²: Higher is better (1.0 = perfect predictions, 0 = no better than mean)\n")

################################################################################
# Step 10: Save trained models
################################################################################

print_progress("Saving trained models...")

models_list <- list(
  elastic_net = model_elastic,
  random_forest = model_rf,
  xgboost = model_xgb,
  predictor_vars = predictor_vars,
  performance = performance,
  train_data_summary = summary(train_data$displacement_risk),
  test_predictions = data.frame(
    hex_id = test_data$hex_id,
    actual = test_data$displacement_risk,
    pred_elastic = pred_elastic,
    pred_rf = pred_rf,
    pred_xgb = pred_xgb
  )
)

save_output(models_list, file.path(OUTPUT_DIR, "trained_models.rds"), "trained models")

################################################################################
# Summary
################################################################################

print_header("STEP 04 COMPLETE")
cat("✓ Three ML models trained and evaluated:\n")
cat("  - Elastic Net (regularized regression)\n")
cat("  - Random Forest (ensemble of trees)\n")
cat("  - XGBoost (gradient boosting)\n")
cat("✓ Cross-validation used for hyperparameter tuning\n")
cat("✓ Models evaluated on held-out test set\n")
cat("✓ Feature importance extracted from all models\n")
cat(paste0("✓ Models saved to: ", file.path(OUTPUT_DIR, "trained_models.rds"), "\n"))
