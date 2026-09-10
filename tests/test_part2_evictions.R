suppressPackageStartupMessages({library(dplyr); library(sf); library(testthat)})
source("R/eviction_panel.R"); source("R/eviction_coverage.R")
source("R/part2_evictions.R"); source("R/part2_index_scoring.R")

test_that("rolling windows include April2 and history is expanding", {
  a <- part2_eviction_windows(as.Date("2025-04-01")); b <- part2_eviction_windows(as.Date("2026-04-01"))
  expect_equal(a$recent_start, as.Date("2024-04-02")); expect_equal(a$previous_start, as.Date("2023-04-02"))
  expect_equal(b$recent_start, as.Date("2025-04-02")); expect_equal(b$previous_start, a$recent_start)
  expect_equal(c(a$recent_days, a$previous_days, b$recent_days), c(365L, 366L, 365L))
  expect_equal(b$history_days - a$history_days, 365L)
  expect_error(part2_eviction_windows(as.Date("2025-04-01"), as.Date("2024-01-01")))
})

test_that("prepared namespaces preserve invalid case IDs as audit-only rows", {
  p <- tempfile(fileext = ".csv"); withr::defer(unlink(p))
  readr::write_csv(data.frame(case_number = c("ABC", "BAD"), case_uid = c("Williamson:JP1:ABC", NA),
    file_date = "2024-06-01", jp_district = "JP1", address_for_geocoding = c("fixture1", "fixture2"),
    geocoding_candidate = TRUE), p)
  x <- part2_eviction_read_filings(p, "Williamson", "fixture")
  expect_equal(x$case_number[1], "WILLIAMSON:JP1:ABC")
  expect_equal(x$case_identity_valid, c(TRUE, FALSE))
  expect_match(x$case_number[2], "UNIDENTIFIED_SOURCE_ROW")
})

test_that("source coverage requires exactly the scored window, not pre-window history", {
  counties <- data.frame(hex_id = 1:4, source_county = c("Travis", "Williamson", "Williamson", "Hays"))
  sources <- data.frame(source_id = paste0("s", 1:6), source_county = c(rep("Travis", 5), "Williamson"),
    jp_district = c(paste0("JP", 1:5), "JP1"), source_period_start = as.Date("2022-01-01"), source_period_end = as.Date("2026-04-01"))
  jp <- data.frame(hex_id = 2:3, effective_start_date = as.Date("2022-01-01"), effective_end_date = as.Date(NA),
    jp_district = c("JP1", "JP3"), boundary_vintage = "2022-present", assignment_status = "assigned")
  x <- part2_eviction_source_coverage(counties, sources, jp, as.Date("2026-04-01"))
  expect_equal(x$coverage$eviction_scored_window_source_covered, c(TRUE, TRUE, FALSE, FALSE))
  expect_equal(min(x$segments$requested_start), as.Date("2024-04-02"))
  expect_true(all(x$segments$requested_end[x$segments$outcome_year == 2026L] == as.Date("2026-04-01")))
  sources$source_period_start[6] <- as.Date("2024-04-02")
  expect_true(part2_eviction_source_coverage(counties, sources, jp, as.Date("2026-04-01"))$coverage$eviction_scored_window_source_covered[2])
  expect_false(part2_eviction_source_coverage(counties, sources, jp, as.Date("2025-04-01"))$coverage$eviction_scored_window_source_covered[2])
  sources$source_period_start[6] <- as.Date("2024-04-03")
  expect_false(part2_eviction_source_coverage(counties, sources, jp, as.Date("2026-04-01"))$coverage$eviction_scored_window_source_covered[2])
  sources$source_period_start[6] <- as.Date("2022-01-01"); sources$source_period_end[1] <- as.Date("2026-03-31")
  expect_false(part2_eviction_source_coverage(counties, sources, jp, as.Date("2026-04-01"))$coverage$eviction_scored_window_source_covered[1])
})

test_that("case resolution deduplicates and masks all dates/candidatehexes conservatively", {
  source <- data.frame(case_number = c("A", "A", "B", "C", "D", "D", "E", "E", "F", "G", "H", "I"),
    file_date = as.Date(c("2024-04-02", "2024-04-02", "2024-04-01", "2022-01-01", "2025-01-01", "2026-06-01",
      "2024-06-01", "2024-06-01", "2025-04-01", "2025-04-02", "2023-04-02", "2023-04-01")),
    source_county = "Travis", jp_district = "JP1")
  mapped <- source %>% mutate(hex_id = c(1L, 1L, 1L, 1L, 2L, 2L, 3L, 4L, 1L, 1L, 1L, 1L))
  r <- resolve_eviction_case_hexes(source, mapped, unique(source$case_number))
  resolved <- list(source = source, cases = r$cases, candidates = distinct(mapped, case_number, hex_id))
  expect_equal(nrow(r$assigned_cases), 7L)
  uncertainty <- part2_eviction_uncertainty(r$cases, resolved$candidates, source, as.Date("2025-04-01"), as.Date("2022-01-01"))
  expect_setequal(uncertainty$hex_id, 2:4)
  expect_equal(nrow(filter(uncertainty, case_number == "D")), 1L)
  support <- data.frame(hex_id = 1:7, area_km2 = 1, residential_units = c(20, 20, 20, 20, 19, 20, 20),
    eviction_inside_current_city = c(rep(TRUE, 6), FALSE))
  coverage <- data.frame(hex_id = 1:7, eviction_scored_window_source_covered = c(rep(TRUE, 5), FALSE, TRUE))
  a <- part2_eviction_snapshot(resolved, support, coverage, as.Date("2025-04-01"))
  b <- part2_eviction_snapshot(resolved, support, coverage, as.Date("2026-04-01"))
  expect_equal(a$features$eviction_cases_total[1], 6L)
  expect_equal(a$features$eviction_cases_latest_12mo[1], 2L)
  expect_equal(a$features$eviction_cases_previous_12mo[1], 2L)
  expect_equal(a$features$eviction_latest_12mo_per_100_units[1], 10)
  expect_equal(a$features$eviction_latest_12mo_rate_change_per_100_units[1], 0)
  expect_equal(a$features$eviction_cases_latest_12mo_change_pct[1], 0)
  expect_equal(a$features$eviction_recent_share[1], 2/6)
  expect_equal(b$features$eviction_cases_latest_12mo[1], 1L)
  expect_equal(b$features$eviction_cases_previous_12mo[1], 2L)
  expect_equal(b$features$eviction_latest_12mo_rate_change_per_100_units[1], -5)
  expect_equal(b$features$eviction_cases_latest_12mo_change_pct[1], -50)
  expect_true(all(is.na(a$features$eviction_cases_latest_12mo[c(2:4, 6:7)])))
  expect_true(a$features$eviction_valid_zero_recent[5])
  expect_true(is.na(a$features$eviction_latest_12mo_per_100_units[5]))
  expect_true(is.na(a$features$eviction_latest_12mo_rate_change_per_100_units[5]))
  expect_true(is.na(a$features$eviction_cases_latest_12mo_change_pct[5]))
  expect_true(is.na(a$features$eviction_recent_share[5]))
  expect_true(is.integer(a$features$hex_id))
  # 0->0, 0->1 and 1->0 are observed differences, not undefined percentages.
  for (counts in list(c(0L,0L), c(0L,1L), c(1L,0L))) {
    keys <- c(if (counts[1]) "H", if (counts[2]) "F")
    small <- list(source = source[source$case_number %in% keys, ],
      cases = r$cases[r$cases$case_number %in% keys, ],
      candidates = resolved$candidates[resolved$candidates$case_number %in% keys, ])
    z <- part2_eviction_snapshot(small, support, coverage, as.Date("2025-04-01"))$features
    expect_equal(z$eviction_latest_12mo_rate_change_per_100_units[1], 5*(counts[2]-counts[1]))
  }
})

test_that("only potentially in-window ambiguities suppress a cell", {
  # Each case has one physical candidate. Date conflicts retain all source
  # dates; neither an older nor a later row can hide an in-window possibility.
  source <- data.frame(case_number=c("OLD","START","END","FUTURE","MISSING",
    "CROSS_START","CROSS_START","CROSS_END","CROSS_END"),
    file_date=as.Date(c("2024-04-01","2024-04-02","2026-04-01","2026-04-02",NA,
      "2022-06-01","2025-06-01","2026-04-01","2027-04-01")))
  cases <- data.frame(case_number=unique(source$case_number),
    assignment_status="excluded_inconsistent_filing_dates", file_date=as.Date(NA),assigned_hex_key=NA_character_)
  candidates <- data.frame(case_number=cases$case_number,hex_id=seq_len(nrow(cases)))
  support <- data.frame(hex_id=1:7,area_km2=1,residential_units=20,eviction_inside_current_city=TRUE)
  coverage <- data.frame(hex_id=1:7,eviction_scored_window_source_covered=TRUE)
  result <- part2_eviction_snapshot(list(source=source,cases=cases,candidates=candidates),support,coverage,
    as.Date("2026-04-01"))
  expect_equal(result$features$eviction_count_observed,c(TRUE,FALSE,FALSE,TRUE,FALSE,FALSE,FALSE))
  expect_equal(result$features$eviction_latest_12mo_per_100_units[c(1,4)],c(0,0))
  expect_true(all(result$features$eviction_eligibility_window_start==as.Date("2024-04-02")))
  # OLD is recent enough for the earlier snapshot: paired eligibility still
  # requires both independent windows, not just the latest one.
  earlier <- part2_eviction_snapshot(list(source=source,cases=cases,candidates=candidates),support,coverage,
    as.Date("2025-04-01"))
  expect_false(earlier$features$eviction_count_observed[1])
})

test_that("event points require fixedcity county and court, with boundaryties retained", {
  # Production uses projected GEOS predicates; this compact longitude fixture
  # disables S2's semi-open polygon model to exercise the same closed edges.
  old_s2 <- sf_use_s2(FALSE); withr::defer(sf_use_s2(old_s2))
  square <- function(x) st_polygon(list(matrix(c(x,30, x+.01,30, x+.01,30.01, x,30.01, x,30), ncol=2, byrow=TRUE)))
  grid <- st_sf(hex_id = 1:3, geometry = st_sfc(lapply(c(-98, -97.99, -97.98), square), crs = 4326))
  city <- st_sf(geometry = st_union(grid[1:2, ]))
  counties <- data.frame(hex_id = 1:3, source_county = c("Travis", "Williamson", "Travis"))
  cityref <- part2_eviction_city_reference(grid, city, 4326)
  source <- data.frame(case_number = c("A", "B", "C", "D", "E", "F", "G", "H"), file_date = as.Date("2025-01-01"),
    source_county = c("Travis", "Williamson", "Travis", "Williamson", "Travis", "Travis", "Travis", "Travis"),
    jp_district = c("JP1", "JP1", "JP1", "JP2", "JP1", "JP1", "JP1", "JP1"),
    address_for_geocoding = letters[1:8], case_identity_valid = c(rep(TRUE,7),FALSE), geocode_registry = "fixture")
  geo <- data.frame(address_for_geocoding = letters[1:8], geocode_registry = "fixture", status = "M",
    score = c(100,100,100,100,89,100,100,100), longitude = c(-97.995,-97.985,-97.975,-97.985,-97.995,-97.99,-97.985,-97.995), latitude=30.005)
  annual <- data.frame(hex_id = 1:3, outcome_year=2025L, source_covered=TRUE, coverage_jp_district=c("ALL","JP1","ALL"))
  r <- suppressWarnings(part2_eviction_resolve(source, geo, grid, counties, city, cityref, annual, crs=4326))
  expect_setequal(r$assigned$case_number, c("A", "B"))
  expect_equal(r$cases$assignment_status[match("E", r$cases$case_number)], "excluded_no_reliable_location")
  expect_true(any(r$evidence$point_hex_matches > 1L))
  expect_equal(r$cases$assignment_status[match("F", r$cases$case_number)], "excluded_mixed_inside_outside_study_geography")
  expect_equal(r$cases$assignment_status[match("H", r$cases$case_number)], "excluded_missing_valid_case_identifier")
  u <- part2_eviction_uncertainty(r$cases,r$candidates,r$source,as.Date("2025-04-01"),as.Date("2022-01-01"))
  expect_equal(u$hex_id[u$case_number == "H"],1L)
})

test_that("eviction score bounds freeze earlier and preserve missingness", {
  a <- data.frame(analysis_as_of_date = as.Date("2025-04-01"), eviction_latest_12mo_per_100_units=c(0,10,20,NA),
    eviction_latest_12mo_rate_change_per_100_units=c(-20,0,20,NA))
  scale <- part2_fit_index_scaling(a, part2_eviction_components(), "eviction_pressure_index", as.Date("2025-04-01"))
  b <- a; b$analysis_as_of_date <- as.Date("2026-04-01"); b[1,part2_eviction_components()] <- list(100,1000)
  result <- part2_apply_index_scaling(b, scale)$features
  expect_equal(result$eviction_pressure_index[1],100)
  expect_true(is.na(result$eviction_pressure_index[4]))
  expect_equal(result$eviction_pressure_index_components_available[4],0)
  expect_true(all(result$eviction_pressure_index_scaling_reference_as_of_date == as.Date("2025-04-01")))
})

test_that("court-window QA separates identified cases from unidentified audit rows", {
  r <- list(source = data.frame(case_number=c("A","B","C","D","D"), source_county="Travis", jp_district="JP1",
    file_date=as.Date(c("2024-06-01","2024-06-01","2024-06-01","2025-04-01","2025-04-02"))),
    cases=data.frame(case_number=c("A","B","C","D"), has_reliable_geocode=c(TRUE,FALSE,TRUE,TRUE),
      assignment_status=c("assigned_unique_hex","excluded_no_reliable_location","excluded_missing_valid_case_identifier","excluded_inconsistent_filing_dates")))
  q <- part2_eviction_court_window_qa(r,as.Date("2025-04-01")) %>% filter(window=="recent")
  expect_equal(q$identified_unique_cases,3L)
  expect_equal(q$unidentified_source_rows,1L)
  expect_equal(q$potentially_in_window_case_or_row_units,4L)
  expect_equal(q$conflicting_or_missing_date_cases,1L)
  expect_equal(q$no_reliable_location_share,1/3)
  expect_equal(q$reliable_mapped_share,1/3)
})
cat("Part2 eviction synthetic tests complete.\n")
