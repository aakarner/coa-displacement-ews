################################################################################
# 03 - Feature Engineering for Displacement Risk Prediction
################################################################################
#
# This script creates features from the processed data for use in machine
# learning models. Features include:
# - Temporal features from rent price changes
# - Spatial lag features (neighborhood effects)
# - Interaction terms
# - Derived risk indicators
#
# Each feature is explained with comments on why it matters for displacement.
#
################################################################################

print_header("03 - FEATURE ENGINEERING")

# Source utilities (enables standalone execution; also sourced by run_analysis.R)
source(here::here("R/utils.R"))

# Load required packages for spatial operations
library(spdep)

# Configuration
OUTPUT_DIR <- here::here("output")

################################################################################
# Step 1: Load processed data
################################################################################

print_progress("Loading processed data...")
hex_data <- load_output(
  file.path(OUTPUT_DIR, "hex_data_processed.rds"),
  "processed hexagonal data"
)

################################################################################
# Step 2: Create temporal features from rent data
################################################################################

print_header("TEMPORAL RENT FEATURES")
print_progress("Creating temporal features from rent price time series...")

# WHY THIS MATTERS: Rapid rent increases are a key indicator of displacement
# pressure. Areas with accelerating rent growth may be experiencing gentrification
# and increased housing cost burden for existing residents.

hex_features <- hex_data %>%
  mutate(
    # 1. Overall rate of rent increase (2019 Q1 to 2022 Q4)
    # Percentage change over the entire period
    rent_change_total = if_else(
      !is.na(rent_2019_q1) & !is.na(rent_2022_q4) & rent_2019_q1 > 0,
      ((rent_2022_q4 - rent_2019_q1) / rent_2019_q1) * 100,
      NA_real_
    ),
    
    # 2. Recent rent change (2021 Q4 to 2022 Q4)
    # More recent changes may better predict near-term displacement risk
    rent_change_recent = if_else(
      !is.na(rent_2021_q4) & !is.na(rent_2022_q4) & rent_2021_q4 > 0,
      ((rent_2022_q4 - rent_2021_q4) / rent_2021_q4) * 100,
      NA_real_
    ),
    
    # 3. Acceleration in rent increases
    # Calculate if rent increases are speeding up or slowing down
    rent_acceleration = {
      early_change <- (rent_2020_q4 - rent_2019_q1) / 4  # Avg quarterly change early
      late_change <- (rent_2022_q4 - rent_2021_q1) / 4   # Avg quarterly change late
      if_else(
        !is.na(early_change) & !is.na(late_change) & early_change > 0,
        ((late_change - early_change) / early_change) * 100,
        NA_real_
      )
    },
    
    # 4. Rent volatility (coefficient of variation)
    # High volatility may indicate market instability
    rent_volatility = {
      rent_values <- select(., starts_with("rent_20")) %>% 
        st_drop_geometry() %>%
        as.matrix()
      apply(rent_values, 1, function(x) {
        x <- x[!is.na(x)]
        if(length(x) >= 3 && mean(x) > 0) {
          sd(x) / mean(x)
        } else {
          NA_real_
        }
      })
    },
    
    # 5. Current rent level relative to median
    # Higher than median rents may indicate already-gentrified areas
    rent_current = rent_2022_q4,
    rent_level_ratio = if_else(
      !is.na(median_rent) & median_rent > 0,
      rent_2022_q4 / median_rent,
      NA_real_
    )
  )

print_progress("Created temporal rent features:")
cat("  - rent_change_total: Overall % change 2019-2022\n")
cat("  - rent_change_recent: Recent % change 2021-2022\n")
cat("  - rent_acceleration: Acceleration of rent increases\n")
cat("  - rent_volatility: Coefficient of variation in rents\n")
cat("  - rent_level_ratio: Current rent vs. area median\n")

################################################################################
# Step 3: Create demolition-based features
################################################################################

print_header("DEMOLITION FEATURES")
print_progress("Creating features from building demolitions...")

# WHY THIS MATTERS: Building demolitions, especially of affordable housing,
# directly displace residents and can signal neighborhood change.

hex_features <- hex_features %>%
  mutate(
    # 1. Demolition density (per km²)
    demo_density = demo_count_total / area_km2,
    
    # 2. Recent demolition activity (2021-2022)
    demo_recent = demo_count_2021 + demo_count_2022,
    
    # 3. Trend in demolitions (is it increasing?)
    demo_trend = if_else(
      demo_count_2020 > 0,
      ((demo_count_2021 + demo_count_2022) / 2 - demo_count_2020) / demo_count_2020,
      NA_real_
    ),
    
    # 4. Binary indicator for any recent demolitions
    has_recent_demos = if_else(demo_recent > 0, 1, 0)
  )

print_progress("Created demolition features:")
cat("  - demo_density: Demolitions per km²\n")
cat("  - demo_recent: Count of recent demolitions\n")
cat("  - demo_trend: Trend in demolition activity\n")
cat("  - has_recent_demos: Binary indicator\n")

################################################################################
# Step 4: Create vulnerability features
################################################################################

print_header("VULNERABILITY FEATURES")
print_progress("Creating socioeconomic vulnerability features...")

# WHY THIS MATTERS: Communities with lower incomes, higher renter percentages,
# and higher poverty rates are more vulnerable to displacement when faced with
# rising housing costs.

hex_features <- hex_features %>%
  mutate(
    # 1. Rent burden proxy (rent to income ratio)
    # Higher values = less affordable
    rent_burden_proxy = if_else(
      !is.na(median_income) & median_income > 0,
      (median_rent * 12) / median_income,
      NA_real_
    ),
    
    # 2. Composite vulnerability index
    # Standardize and combine multiple vulnerability indicators
    vuln_low_income = normalize_to_100(-median_income),  # Lower income = higher vulnerability
    vuln_high_rent = normalize_to_100(pct_renter),       # More renters = higher vulnerability
    vuln_poverty = normalize_to_100(poverty_rate),       # Higher poverty = higher vulnerability
    vuln_low_edu = normalize_to_100(-pct_college),       # Lower education = higher vulnerability
    
    vulnerability_index = (vuln_low_income + vuln_high_rent + 
                          vuln_poverty + vuln_low_edu) / 4,
    
    # 3. Demographic change potential (proxy)
    # Areas with more people of color may be more susceptible to gentrification
    # in some contexts - this is a sensitive metric that requires careful interpretation
    pct_poc_vulnerable = pct_poc
  )

print_progress("Created vulnerability features:")
cat("  - rent_burden_proxy: Annual rent / median income\n")
cat("  - vulnerability_index: Composite of income, rent, poverty, education\n")
cat("  - Component scores: vuln_low_income, vuln_high_rent, vuln_poverty, vuln_low_edu\n")

################################################################################
# Step 5: Create spatial lag features (neighborhood effects)
################################################################################

print_header("SPATIAL LAG FEATURES")
print_progress("Creating spatial lag features for neighborhood effects...")

# WHY THIS MATTERS: Displacement risk in one area affects neighboring areas.
# Spatial lags capture spillover effects and broader neighborhood trends.

# Function to calculate spatial lag safely
safe_spatial_lag <- function(data, var_name, k = 6) {
  tryCatch({
    # Get centroids
    centroids <- st_centroid(data)
    
    # Create spatial weights using k-nearest neighbors
    coords <- st_coordinates(centroids)
    knn <- knearneigh(coords, k = k)
    nb <- knn2nb(knn)
    weights <- nb2listw(nb, style = "W", zero.policy = TRUE)
    
    # Calculate spatial lag
    values <- data[[var_name]]
    lag_values <- lag.listw(weights, values, zero.policy = TRUE)
    
    return(lag_values)
  }, error = function(e) {
    warning(paste("Error calculating spatial lag for", var_name, ":", e$message))
    return(rep(NA_real_, nrow(data)))
  })
}

# Calculate spatial lags for key variables
print_progress("Calculating spatial lags (this may take a moment)...")

hex_features <- hex_features %>%
  mutate(
    # Spatial lag of rent change
    rent_change_total_lag = safe_spatial_lag(., "rent_change_total"),
    
    # Spatial lag of demolitions
    demo_density_lag = safe_spatial_lag(., "demo_density"),
    
    # Spatial lag of median income
    median_income_lag = safe_spatial_lag(., "median_income"),
    
    # Spatial lag of vulnerability
    vulnerability_index_lag = safe_spatial_lag(., "vulnerability_index")
  )

print_progress("Created spatial lag features:")
cat("  - rent_change_total_lag: Average rent change in neighboring cells\n")
cat("  - demo_density_lag: Average demolition density in neighboring cells\n")
cat("  - median_income_lag: Average median income in neighboring cells\n")
cat("  - vulnerability_index_lag: Average vulnerability in neighboring cells\n")

################################################################################
# Step 6: Create interaction terms
################################################################################

print_header("INTERACTION FEATURES")
print_progress("Creating interaction terms between key predictors...")

# WHY THIS MATTERS: Displacement risk often emerges from combinations of factors.
# For example, rapid rent increases in vulnerable communities create higher
# displacement risk than either factor alone.

hex_features <- hex_features %>%
  mutate(
    # 1. Rent increase × Vulnerability
    # Rapid rent growth is more concerning in vulnerable communities
    rent_vuln_interaction = rent_change_total * vulnerability_index,
    
    # 2. Demolitions × High rent areas
    # Demolitions in expensive areas may signal new development
    demo_rent_interaction = demo_density * rent_level_ratio,
    
    # 3. Low income × High rent burden
    # Income-rent mismatch
    income_burden_interaction = vuln_low_income * rent_burden_proxy,
    
    # 4. Neighborhood rent pressure
    # Combine local and neighboring rent changes
    neighborhood_rent_pressure = (rent_change_total + rent_change_total_lag) / 2
  )

print_progress("Created interaction features:")
cat("  - rent_vuln_interaction: Rent change × Vulnerability\n")
cat("  - demo_rent_interaction: Demolitions × Rent level\n")
cat("  - income_burden_interaction: Low income × Rent burden\n")
cat("  - neighborhood_rent_pressure: Combined local and neighbor rent changes\n")

################################################################################
# Step 7: Handle missing data
################################################################################

print_header("MISSING DATA HANDLING")
print_progress("Analyzing and handling missing data...")

# Count missing values in key features
feature_cols <- hex_features %>%
  st_drop_geometry() %>%
  select(
    starts_with("rent_"),
    starts_with("demo_"),
    starts_with("vuln_"),
    starts_with("pct_"),
    vulnerability_index,
    median_income,
    poverty_rate,
    ends_with("_lag"),
    ends_with("_interaction")
  ) %>%
  names()

missing_counts <- hex_features %>%
  st_drop_geometry() %>%
  select(all_of(feature_cols)) %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "missing_count") %>%
  mutate(missing_pct = (missing_count / nrow(hex_features)) * 100) %>%
  arrange(desc(missing_pct))

cat("\nTop 10 features with missing data:\n")
print(head(missing_counts, 10))

# Strategy: For now, keep NAs - they will be handled during model training
# Different models handle missing data differently:
# - Random Forest: can use surrogate splits
# - XGBoost: has built-in handling
# - Elastic Net: we'll need to impute or exclude

# Flag observations with excessive missing data
hex_features <- hex_features %>%
  mutate(
    missing_feature_count = rowSums(is.na(select(., all_of(feature_cols)))),
    missing_feature_pct = (missing_feature_count / length(feature_cols)) * 100,
    sufficient_data = missing_feature_pct < 50  # Flag if less than 50% missing
  )

print_progress(paste0("Flagged ", sum(!hex_features$sufficient_data), 
                     " hexagons with >50% missing features"))

################################################################################
# Step 8: Create summary of engineered features
################################################################################

print_header("FEATURE SUMMARY")

cat("\nFeature categories created:\n")
cat("  1. Temporal rent features: 6 features\n")
cat("  2. Demolition features: 4 features\n")
cat("  3. Vulnerability features: 6 features\n")
cat("  4. Spatial lag features: 4 features\n")
cat("  5. Interaction features: 4 features\n")
cat("  TOTAL: ~24 engineered features\n\n")

# Quick summary statistics of key features
summary_features <- c(
  "rent_change_total", "rent_change_recent", "demo_density",
  "vulnerability_index", "rent_burden_proxy", "neighborhood_rent_pressure"
)

cat("Summary statistics for key features:\n")
print(
  hex_features %>%
    st_drop_geometry() %>%
    select(all_of(summary_features)) %>%
    summary()
)

################################################################################
# Step 9: Save engineered features
################################################################################

output_file <- file.path(OUTPUT_DIR, "hex_features.rds")
save_output(hex_features, output_file, "engineered features")

# Also save feature list for reference
feature_list <- data.frame(
  feature_name = feature_cols,
  category = case_when(
    str_starts(feature_cols, "rent_") & !str_ends(feature_cols, "_lag") ~ "Temporal Rent",
    str_starts(feature_cols, "demo_") & !str_ends(feature_cols, "_lag") ~ "Demolitions",
    str_starts(feature_cols, "vuln_") | feature_cols == "vulnerability_index" ~ "Vulnerability",
    str_ends(feature_cols, "_lag") ~ "Spatial Lag",
    str_ends(feature_cols, "_interaction") ~ "Interactions",
    TRUE ~ "Other"
  )
)

write_csv(feature_list, file.path(OUTPUT_DIR, "feature_list.csv"))
print_progress("Saved feature list to: output/feature_list.csv")

################################################################################
# Summary
################################################################################

print_header("STEP 03 COMPLETE")
cat("✓ Temporal features from rent data created\n")
cat("✓ Demolition features created\n")
cat("✓ Vulnerability indices calculated\n")
cat("✓ Spatial lag features computed\n")
cat("✓ Interaction terms generated\n")
cat("✓ Missing data analyzed\n")
cat(paste0("✓ Features saved to: ", output_file, "\n"))
