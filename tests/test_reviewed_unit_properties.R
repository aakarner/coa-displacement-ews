suppressPackageStartupMessages({library(dplyr); library(sf)})
source("R/reviewed_unit_properties.R")
review <- read_reviewed_unit_properties()
before <- readRDS("output/reviewed_units_20261007/before/output/residential_parcels_unit_promoted.rds")
grid <- readRDS("output/hex_grid.rds")
result <- apply_reviewed_unit_properties(before, review, grid)
p <- result$parcels
ids <- unlist(lapply(review$projects, `[[`, "parcel_ids"))
unchanged <- !before$parcel_id %in% ids
j <- match(before$parcel_id[unchanged], p$parcel_id)
stopifnot(nrow(p) == nrow(before) + length(reviewed_unit_supplement_ids(review)),
  identical(before$promoted_units[unchanged], p$promoted_units[j]),
  identical(before$lon[unchanged], p$lon[j]), identical(before$lat[unchanged], p$lat[j]),
  identical(before$owner_names[unchanged], p$owner_names[j]),
  p$promoted_units[p$parcel_id == "533185"] == 0,
  p$promoted_units[p$parcel_id == "975264"] == 330,
  p$promoted_units[p$parcel_id == "774412"] == 26,
  p$promoted_units[p$parcel_id == "737158"] == before$promoted_units[before$parcel_id == "737158"],
  sum(p$promoted_units[p$parcel_id %in% c(as.character(774341:774344), "774412")]) == 438,
  sum(p$unit_review_added_account) == length(reviewed_unit_supplement_ids(review)),
  p$promoted_units[p$parcel_id == "291453"] == 170,
  p$unit_model_selection_method[p$parcel_id == "291453"] == "reviewed_assumed_project_total",
  p$unit_estimation_confidence_targeted[p$parcel_id == "291453"] == "low",
  p$unit_review_count_status[p$parcel_id == "291453"] == "provisional_assumption",
  abs(sum(p$promoted_units) - sum(before$promoted_units) - sum(result$audit$delta)) < 1e-6)
expect_error <- function(f, text) {
  err <- tryCatch({f(); NULL}, error = identity)
  stopifnot(inherits(err, "error"), grepl(text, conditionMessage(err), fixed = TRUE))
}
bad <- review; bad$projects[[4]]$parcel_ids <- as.list(as.character(774341:774343))
expect_error(function() apply_reviewed_unit_properties(before, bad, grid), "membership changed")
bad <- before; bad$source_county[bad$parcel_id == "878332"] <- "Williamson"
expect_error(function() apply_reviewed_unit_properties(bad, review, grid), "membership changed")
bad <- before; bad$lon[bad$parcel_id == "774341"] <- bad$lon[bad$parcel_id == "774341"] + .01
expect_error(function() apply_reviewed_unit_properties(bad, review, grid), "original geometry changed")
bad <- review; bad$projects[[4]]$geometry$expected_hex_id <- 6381L
expect_error(function() apply_reviewed_unit_properties(before, bad, grid), "expected grid cell")
bad <- review; bad$projects[[4]]$geometry$polygon_parcel_id <- "737155"
expect_error(function() apply_reviewed_unit_properties(before, bad, grid), "outside its verified footprint")
expect_error(function() apply_reviewed_unit_properties(p, review, grid), "original geometry changed")
bad <- before; bad$county_unit_exclude_from_unit_universe[bad$parcel_id == "878332"] <- TRUE
expect_error(function() apply_reviewed_unit_properties(bad, review, grid), "excluded appraisal account")
bad <- review; bad$projects <- list(review$projects[[5]])
expect_error(function() apply_reviewed_unit_properties(p, bad, grid), "overlaps the upstream inventory")
temp <- tempfile(fileext = ".json")
bad <- jsonlite::read_json("config/residential_unit_property_reviews.json")
bad$evidence[[1]]$sha256 <- paste(rep("0", 64), collapse = "")
jsonlite::write_json(bad, temp, auto_unbox = TRUE)
expect_error(function() read_reviewed_unit_properties(temp), "changed unit-review evidence")
unlink(temp)
cat("Reviewed totals, geometry, named supplements, explicit provisional provenance, unchanged other properties, and drift rejection pass.\n")
