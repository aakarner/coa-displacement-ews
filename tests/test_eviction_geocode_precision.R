suppressPackageStartupMessages({library(dplyr); library(sf); library(testthat)})
source("R/eviction_panel.R")
source("R/eviction_coverage.R")
source("R/part2_evictions.R")

test_that("match confidence cannot substitute for address precision", {
  x <- data.frame(status="M", score=100, longitude=-97.7, latitude=30.3,
    addr_type=c("Postal", "StreetName", "POI", NA, "unknown", "PointAddress", "Subaddress", "APT", "UNIT", "StreetAddress", "StreetAddressExt"))
  z <- assess_eviction_geocodes(x)
  expect_equal(z$geocode_location_usable, c(rep(FALSE,5),rep(TRUE,6)))
  expect_equal(tail(z$geocode_location_quality,2), rep("street_segment",2))
  expect_error(assess_eviction_geocodes(select(x,-addr_type)), "addr_type")
  x$score[6] <- 89; x$longitude[7] <- 190; x$status[8] <- "U"
  expect_false(any(assess_eviction_geocodes(x)$geocode_location_usable[6:8]))
})

test_that("coarse candidates do not assign or suppress cells, precise evidence still resolves", {
  square <- function(a,b) st_polygon(list(matrix(c(a,30,a+.01,30,a+.01,30.01,a,30.01,a,30),ncol=2,byrow=TRUE)))
  grid <- st_sf(hex_id=1:2,geometry=st_sfc(square(-98,0),square(-97.99,0),crs=4326))
  city <- st_sf(geometry=st_union(grid))
  county <- data.frame(hex_id=1:2,source_county="Travis")
  cityref <- part2_eviction_city_reference(grid,city,4326)
  filings <- data.frame(case_number=c("POSTAL","STREET","GOOD","MIXED","MIXED","CONFLICT","CONFLICT"),
    address_for_geocoding=letters[1:7], geocode_registry="test", file_date=as.Date("2025-06-01"),
    source_county="Travis",jp_district="JP1",case_identity_valid=TRUE)
  geo <- data.frame(address_for_geocoding=letters[1:7],geocode_registry="test",status="M",score=100,
    longitude=c(-97.995,-97.995,-97.995,-97.995,-97.985,-97.995,-97.985),latitude=30.005,
    addr_type=c("Postal","StreetName","PointAddress","Subaddress","Postal","PointAddress","PointAddress"))
  coverage <- data.frame(hex_id=1:2,outcome_year=2025L,source_covered=TRUE,coverage_jp_district="ALL")
  z <- suppressWarnings(part2_eviction_resolve(filings,geo,grid,county,city,cityref,coverage,crs=4326))
  get <- function(id) z$cases[z$cases$case_number==id,]
  expect_equal(get("POSTAL")$assignment_status,"excluded_insufficient_geocode_precision")
  expect_equal(get("STREET")$assignment_status,"excluded_insufficient_geocode_precision")
  expect_true(is.na(get("POSTAL")$assigned_hex_key))
  expect_equal(get("MIXED")$assigned_hex_key,"1")
  expect_true(get("MIXED")$has_rejected_imprecise_geocode)
  expect_equal(get("CONFLICT")$assignment_status,"excluded_multiple_hexes")
  expect_false(any(z$candidates$case_number %in% c("POSTAL","STREET")))
  support <- data.frame(hex_id=1:2,area_km2=1,residential_units=100,eviction_inside_current_city=TRUE)
  scored <- part2_eviction_snapshot(z,support,data.frame(hex_id=1:2,eviction_scored_window_source_covered=TRUE),as.Date("2026-04-01"))$features
  expect_equal(scored$eviction_cases_latest_12mo,c(2L,0L))
  expect_true(all(scored$eviction_count_observed))
})
