suppressPackageStartupMessages({library(dplyr); library(sf); library(testthat)})
source("R/eviction_property_geography.R")

fixture <- function() {
  cases <- data.frame(case_number = LETTERS[1:4], file_date = as.Date("2025-06-01"),
    outcome_year = 2025L, source_county = "Travis", source_jp_district = "JP3",
    assignment_status = c("assigned_unique_hex", "excluded_multiple_hexes", rep("assigned_unique_hex", 2)),
    assigned_hex_key = c("1", NA, "1", "1"))
  evidence <- data.frame(case_number = c("A", "B", "C", "D", "D"), source_county = "Travis",
    address_for_geocoding = c("shared apartment", "excluded", "shared apartment", "unit 11103", "unit 11130"))
  geography <- data.frame(source_county = "Travis", address_for_geocoding = unique(evidence$address_for_geocoding),
    property_id = "project:old", property_hex_id = 1L, property_link_status = "verified", reference_distance_m = 0)
  reviews <- data.frame(case_number = c("A", "B", "D", "D"), source_county = "Travis",
    address_for_geocoding = c("shared apartment", "excluded", "unit 11103", "unit 11130"),
    expected_file_date = "2025-06-01", expected_source_jp = "JP3", property_id = c("project:a", "project:a", "project:d", "project:d"),
    property_hex_id = c(2L, 2L, 3L, 3L), property_link_status = "verified", reference_distance_m = 90,
    property_review_id = c("review:A", "review:B", "review:D", "review:D"), property_review_basis = "court_property",
    property_apartment_conflict = c(FALSE, FALSE, TRUE, TRUE))
  attr(geography, "case_reviews") <- reviews
  list(resolved = list(cases = cases, assigned_cases = filter(cases, assignment_status == "assigned_unique_hex")),
    evidence = evidence, geography = geography,
    coverage = data.frame(hex_id = 1:3, outcome_year = 2025L, source_covered = TRUE, coverage_jp_district = "JP3"),
    hex_counties = data.frame(hex_id = 1:3, source_county = "Travis"), city_hexes = 1:3)
}

test_that("reviews are case specific, preserve exclusions, and retain apartment uncertainty", {
  f <- fixture(); x <- do.call(apply_eviction_property_geography, f)
  expect_identical(x$cases$assigned_hex_key, c("2", NA, "1", "3"))
  expect_identical(x$cases$assignment_status, f$resolved$cases$assignment_status)
  expect_identical(x$cases$original_assigned_hex_key, f$resolved$cases$assigned_hex_key)
  expect_identical(x$cases$property_apartment_conflict, c(FALSE, FALSE, FALSE, TRUE))
  expect_true(all(x$cases$all_addresses_verified[c(1, 4)])) # Property, not exact apartment.
  expect_true(is.na(x$cases$property_review_id[3])) # Same address, different case.
  expect_identical(x$assigned_cases$assigned_hex_key, c("2", "1", "3"))
  expect_identical(f$evidence$address_for_geocoding[4:5], c("unit 11103", "unit 11130"))
})

test_that("new, missing or changed reliable addresses and case identity cannot use old reviews", {
  f <- fixture(); f$evidence <- rbind(f$evidence, data.frame(case_number = "A", source_county = "Travis", address_for_geocoding = "new location"))
  expect_error(do.call(apply_eviction_property_geography, f), "re-review required")
  f <- fixture(); f$evidence <- f$evidence[-5, ]
  expect_error(do.call(apply_eviction_property_geography, f), "re-review required")
  for (field in c("file_date", "source_county", "source_jp_district")) {
    f <- fixture()
    f$resolved$cases[1, field] <- switch(field, file_date = as.Date("2025-06-02"), source_county = "Williamson", source_jp_district = "JP4")
    expect_error(do.call(apply_eviction_property_geography, f), "re-review required")
  }
})

test_that("manual reviews cannot bypass City, county or covered-court limits", {
  f <- fixture(); f$city_hexes <- c(1L, 3L)
  x <- do.call(apply_eviction_property_geography, f)
  expect_equal(x$cases$assigned_hex_key[1], "1")
  expect_match(x$cases$property_assignment_status[1], "outside_city")
  f <- fixture(); f$hex_counties$source_county[2] <- "Williamson"
  x <- do.call(apply_eviction_property_geography, f)
  expect_equal(x$cases$assigned_hex_key[1], "1")
  expect_match(x$cases$property_assignment_status[1], "county_conflict")
  f <- fixture()
  f$resolved$cases$source_county[1] <- "Williamson"
  f$evidence$source_county[1] <- "Williamson"
  r <- attr(f$geography, "case_reviews"); r$source_county[r$case_number == "A"] <- "Williamson"
  attr(f$geography, "case_reviews") <- r
  f$hex_counties$source_county[2] <- "Williamson"; f$coverage$coverage_jp_district[2] <- "JP4"
  x <- do.call(apply_eviction_property_geography, f)
  expect_match(x$cases$property_assignment_status[1], "court_conflict")
})

test_that("compiled reviews require original evidence, geocodes and a single occupied project cell", {
  dir <- tempfile(); dir.create(dir); on.exit(unlink(dir, recursive = TRUE))
  evidence <- file.path(dir, "evidence.txt"); writeLines("reviewed source", evidence)
  entry <- list(case_number = "A", source_county = "Travis", source_jp_district = "JP3", file_date = "2025-06-01",
    parcel_id = "land", project_id = "project:land", expected_unit_hex_id = 2L, review_basis = "court_property",
    review_id = "review:A", apartment_conflict = FALSE,
    addresses = list(list(address_for_geocoding = "shared apartment", longitude = -97.79, latitude = 30.15)))
  bundle <- file.path(dir, "bundle.json"); config <- file.path(dir, "config.json")
  jsonlite::write_json(list(schema_version = 1L, batch_id = "test", cases = list(entry),
    evidence = list(list(path = evidence, sha256 = digest::digest(file = evidence, algo = "sha256")))), bundle, auto_unbox = TRUE)
  jsonlite::write_json(list(schema_version = 1L, batches = list(list(batch_id = "test", path = bundle,
    sha256 = digest::digest(file = bundle, algo = "sha256"), case_count = 1L))), config, auto_unbox = TRUE)
  p <- data.frame(parcel_id = c("land", "improvement"), project_id = "project:land", operational_units = c(0, 330),
    source_county = "Travis", unit_hex_id = 2L, lon = -97.79, lat = 30.15)
  g <- data.frame(source_county = "Travis", address_for_geocoding = "shared apartment", longitude = -97.79, latitude = 30.15)
  x <- compile_property_case_reviews(p, g, config)
  expect_equal(x$rows$property_hex_id, 2L) # Land + improvement counted as a project.
  moved <- g; moved$longitude <- -97.8
  expect_error(compile_property_case_reviews(p, moved, config), "geocode changed")
  split <- rbind(p, transform(p[2, ], parcel_id = "another", unit_hex_id = 3L))
  expect_error(compile_property_case_reviews(split, g, config), "supported unit reference")
  writeLines("altered evidence", evidence)
  expect_error(compile_property_case_reviews(p, g, config), "changed property review evidence")
})

test_that("the generated crosswalk cannot be edited after its manifest was written", {
  dir <- tempfile(); dir.create(dir); on.exit(unlink(dir, recursive = TRUE))
  path <- file.path(dir, "eviction_address_properties.rds")
  input <- file.path(dir, "input.txt"); writeLines("pinned input", input)
  x <- fixture()$geography; saveRDS(x, path)
  entry <- function(p) data.frame(path = p, sha256 = digest::digest(file = p, algo = "sha256"))
  jsonlite::write_json(list(rule = eviction_property_geography_version(), inputs = entry(input),
    outputs = entry(path)), file.path(dir, "property_geography_manifest.json"), auto_unbox = TRUE)
  expect_equal(read_eviction_property_geography(path), x)
  x$property_hex_id[1] <- 999L; saveRDS(x, path)
  expect_error(read_eviction_property_geography(path), "changed property review evidence")
})
