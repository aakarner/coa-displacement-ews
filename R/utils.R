################################################################################
# Utility Functions for Displacement Early Warning System
################################################################################
#
# This file contains reusable utility functions used throughout the analysis.
#
################################################################################

#' Print section header for console output
#' @param text The header text to display
print_header <- function(text) {
  cat("\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat(paste0("  ", text, "\n"))
  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat("\n")
}

#' Print progress message
#' @param text The message to display
print_progress <- function(text) {
  cat(paste0(">>> ", text, "\n"))
}

#' Calculate spatial lag (average of neighboring cells)
#' @param sf_data An sf object with the data
#' @param value_col Name of the column to calculate lag for
#' @param k Number of nearest neighbors to consider
#' @return Vector of spatial lag values
calculate_spatial_lag <- function(sf_data, value_col, k = 6) {
  print_progress(paste0("Calculating spatial lag for ", value_col, "..."))
  
  # Get centroids
  centroids <- st_centroid(sf_data)
  
  # Find k nearest neighbors
  neighbors <- st_nn(centroids, centroids, k = k + 1)  # +1 to exclude self
  
  # Calculate lag
  lag_values <- sapply(1:nrow(sf_data), function(i) {
    neighbor_indices <- neighbors[[i]][-1]  # Remove self (first element)
    mean(sf_data[[value_col]][neighbor_indices], na.rm = TRUE)
  })
  
  return(lag_values)
}

#' Normalize values to 0-100 scale
#' @param x Vector of values to normalize
#' @return Normalized vector
normalize_to_100 <- function(x) {
  if(all(is.na(x))) return(x)
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)) * 100
}

#' Categorize risk scores into levels
#' @param scores Vector of risk scores (0-100)
#' @return Factor with risk categories
categorize_risk <- function(scores) {
  cut(scores,
      breaks = c(-Inf, 25, 50, 75, Inf),
      labels = c("Low", "Moderate", "High", "Very High"),
      include.lowest = TRUE)
}

#' Calculate rate of change
#' @param x Vector of values
#' @param time_diff Time difference between observations
#' @return Rate of change
calculate_rate_of_change <- function(x, time_diff = 1) {
  if(length(x) < 2) return(NA)
  (x[length(x)] - x[1]) / (time_diff * x[1]) * 100
}

#' Calculate acceleration (second derivative)
#' @param x Vector of values over time
#' @return Acceleration value
calculate_acceleration <- function(x) {
  if(length(x) < 3) return(NA)
  # Simple finite difference approximation
  first_diff <- diff(x)
  second_diff <- diff(first_diff)
  mean(second_diff, na.rm = TRUE)
}

#' Calculate volatility (standard deviation / mean)
#' @param x Vector of values
#' @return Coefficient of variation
calculate_volatility <- function(x) {
  if(length(x) < 2 || mean(x, na.rm = TRUE) == 0) return(NA)
  sd(x, na.rm = TRUE) / mean(x, na.rm = TRUE)
}

#' Save object with informative message
#' @param object Object to save
#' @param filepath Path where to save
#' @param description Description of the object
save_output <- function(object, filepath, description = NULL) {
  saveRDS(object, filepath)
  if(!is.null(description)) {
    print_progress(paste0("Saved ", description, " to: ", filepath))
  } else {
    print_progress(paste0("Saved to: ", filepath))
  }
}

#' Load object with informative message
#' @param filepath Path to load from
#' @param description Description of the object
#' @return Loaded object
load_output <- function(filepath, description = NULL) {
  if(!file.exists(filepath)) {
    stop(paste0("File not found: ", filepath))
  }
  object <- readRDS(filepath)
  if(!is.null(description)) {
    print_progress(paste0("Loaded ", description, " from: ", filepath))
  } else {
    print_progress(paste0("Loaded from: ", filepath))
  }
  return(object)
}
