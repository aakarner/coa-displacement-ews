# Exercise the explicit boundary exception and fail-closed geographic guards.
suppressPackageStartupMessages({library(dplyr); library(sf)})
source("R/reviewed_unit_properties.R")
r <- read_reviewed_unit_properties()
r$projects <- Filter(function(x) grepl("^batch2_", x$review_id), r$projects)
before <- readRDS("output/residential_property_batch2/before/output/residential_parcels_unit_promoted.rds")
ids <- unlist(lapply(r$projects, `[[`, "parcel_ids"))
before <- before[before$parcel_id %in% ids, ]
grid <- readRDS("output/hex_grid.rds")
x <- apply_reviewed_unit_properties(before, r, grid)$parcels
stopifnot(nrow(x) == 4L, sum(x$promoted_units) == 1204,
          sum(x$unit_review_added_account) == 1L)
expect_error <- function(fun, expected) {
  err <- tryCatch({fun(); NULL}, error = identity)
  stopifnot(inherits(err, "error"), grepl(expected, conditionMessage(err), fixed = TRUE))
}
bad <- r; bad$projects[[1]]$geometry$minimum_city_overlap_fraction <- 1
expect_error(function() apply_reviewed_unit_properties(before, bad, grid), "City scope")
bad <- r; bad$projects[[1]]$geometry$expected_hex_id <- 3244L
expect_error(function() apply_reviewed_unit_properties(before, bad, grid), "expected grid cell")
bad <- before; bad$lon[bad$parcel_id == "911866"] <- -97.76
expect_error(function() apply_reviewed_unit_properties(bad, r, grid), "original geometry changed")
bad <- r; bad$projects[[1]]$geometry$polygon_parcel_id <- "911866"
expect_error(function() apply_reviewed_unit_properties(before, bad, grid), "verified footprint")
cat("Reviewed boundary references reject City-scope, cell, original-coordinate and parcel drift.\n")
