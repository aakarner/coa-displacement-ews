# Synthetic ACS source-coherence and missing-count tests; no downloads required.
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(sf)
})
source("R/acs_dasymetric.R")

expect_equal <- function(actual, expected) {
  stopifnot(isTRUE(all.equal(actual, expected, check.attributes = FALSE)))
}

# Estimates, MOEs, and geographic provenance must come from one source.
primary <- tibble(hex_id = c("a", "b", "c", "d"),
  median_income = c(100, NA, NA, 0), median_income_moe = c(NA, 88, 44, 0),
  acs_source_geoid = paste0("bg", 1:4), acs_source_geography = "block_group",
  acs_source_residential_share = .8, acs_source_assignment_method = "fixture_bg")
fallback <- tibble(hex_id = c("a", "b", "c", "d", "e"),
  median_income = c(200, 300, NA, 400, 500), median_income_moe = c(20, 30, 50, 40, 60),
  acs_source_geoid = paste0("tract", 1:5), acs_source_geography = "tract",
  acs_source_residential_share = .9, acs_source_assignment_method = "fixture_tract")
result <- combine_acs_median_sources(primary, fallback, "median_income")
expect_equal(result$median_income, c(100, 300, NA, 0, 500))
expect_equal(result$median_income_moe, c(NA, 30, NA, 0, 60))
expect_equal(result$median_income_source_geoid, c("bg1", "tract2", NA, "bg4", "tract5"))
expect_equal(result$median_income_source_geography, c("block_group", "tract", NA, "block_group", "tract"))
expect_equal(result$median_income_source_residential_share, c(.8, .9, NA, .8, .9))
expect_equal(result$median_income_source_assignment_method,
             c("fixture_bg", "fixture_tract", NA, "fixture_bg", "fixture_tract"))

# A missing fallback MOE is not replaced by a primary MOE from a missing estimate.
fallback$median_income_moe[2] <- NA
stopifnot(is.na(combine_acs_median_sources(primary, fallback, "median_income")$median_income_moe[2]))

# Positive-weight unavailable source contributions mask a paired count, while
# legacy defaults still sum available contributions. Observed zero stays zero.
sources <- tibble(GEOID = c("s1", "s2", "s3", "s4"), variable = "people",
                   estimate = c(100, NA, 0, 40), moe = c(10, NA, 0, NA))
crosswalk <- tibble(
  hex_id = c("mixed", "mixed", "known", "zero", "missing", "moe_only", "irrelevant", "irrelevant"),
  source_geoid = c("s1", "s2", "s1", "s3", "s2", "s4", "s1", "s2"),
  population_allocation_weight = c(.25, .3, .5, 1, .7, 1, .25, 0),
  housing_allocation_weight = c(.25, .3, .5, 1, .7, 1, .25, 0),
  population_allocation_basis = "fixture_population",
  housing_allocation_basis = "fixture_housing"
)
legacy <- allocate_acs_count_variables(sources, crosswalk, "people", character())
strict <- allocate_acs_count_variables(sources, crosswalk, "people", character(), preserve_missing = TRUE)
value <- function(result, id, field) result$values[[field]][result$values$hex_id == id]
expect_equal(value(legacy, "mixed", "people"), 25)
stopifnot(is.na(value(strict, "mixed", "people")),
          is.na(value(strict, "mixed", "people_moe")))
expect_equal(value(strict, "known", "people"), 50)
expect_equal(value(strict, "known", "people_moe"), 5)
expect_equal(value(strict, "zero", "people"), 0)
expect_equal(value(strict, "zero", "people_moe"), 0)
stopifnot(is.na(value(strict, "missing", "people")))
expect_equal(value(strict, "moe_only", "people"), 40)
stopifnot(is.na(value(strict, "moe_only", "people_moe")))
expect_equal(value(strict, "irrelevant", "people"), 25)
expect_equal(value(strict, "irrelevant", "people_moe"), 2.5)
expect_equal(strict$conservation_qa$incomplete_estimate_hexes, 2)
expect_equal(strict$conservation_qa$incomplete_moe_hexes, 3)
expect_equal(strict$conservation_qa$allocated_project_estimate, 140)
expect_equal(strict$conservation_qa$emitted_hex_estimate_total, 115)
expect_equal(strict$conservation_qa$conservation_difference, 0)

cat("ACS median-source coherence and paired missing-count tests passed.\n")
