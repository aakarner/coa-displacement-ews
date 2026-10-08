suppressPackageStartupMessages({library(dplyr); library(testthat)})
source("R/eviction_panel.R")

test_that("ambiguous cases stay unassigned while both candidate cells retain valid counts", {
  filings <- data.frame(case_number=c("A","B","X","X"), file_date=as.Date("2025-06-01"),
    source_county="Travis", jp_district="JP1")
  mapped <- transform(filings, hex_id=c(1L,2L,1L,2L))
  r <- resolve_eviction_case_hexes(filings,mapped,unique(filings$case_number))
  expect_equal(r$cases$assignment_status[r$cases$case_number=="X"],"excluded_multiple_hexes")
  expect_true(is.na(r$cases$assigned_hex_key[r$cases$case_number=="X"]))
  counties <- data.frame(hex_id=1:4,source_county=c("Travis","Travis","Travis","Hays"))
  uncertain <- bind_rows(r$uncertain_hex_years,
    data.frame(assigned_hex_key=c("3","4","1"),outcome_year=c(2025L,2025L,2026L),unresolved_candidate_cases=1L))
  panel <- build_complete_eviction_panel(counties,r$assigned_cases,as.Date("2025-01-01"),
    as.Date("2026-04-01"),uncertain_hex_years=uncertain)
  complete <- filter(panel,outcome_year==2025)
  expect_equal(complete$eviction_cases,c(1L,1L,0L,NA_integer_))
  expect_true(all(complete$has_unassigned_ambiguous_cases))
  expect_equal(complete$count_observed,c(TRUE,TRUE,TRUE,FALSE))
  expect_true(all(is.na(panel$eviction_cases[panel$outcome_year==2026])))
  expect_false(any(panel$all_filing_locations_complete))
  expect_equal(sum(complete$eviction_cases,na.rm=TRUE),nrow(r$assigned_cases))
  invalid <- panel; invalid$has_unassigned_ambiguous_cases[1] <- FALSE
  expect_error(validate_complete_eviction_panel(invalid),"ambiguity audit flags")
  qa <- summarize_complete_eviction_panel(panel,r)
  expect_equal(qa$hexes_with_ambiguous_case_location,c(4L,1L))
})
