suppressPackageStartupMessages({library(dplyr);library(sf);library(testthat)})
source("R/wcad_residential_evidence.R")
source("R/wcad_unit_eligibility.R")
source("R/eviction_property_geography.R")
test_that("coded commercial accounts require independent residential evidence", {
  x <- wcad_corroborated_multifamily(c("C3","C5","C5","C3","Residential","C3"),
    c(100000,120000,3000,0,2000,100000),
    c("city_exact_parcel_apartment_condo","reviewed_housing_inventory_and_county_geometry",
      "reviewed_non_unit_companion","city_exact_parcel_apartment_condo",
      "city_exact_parcel_apartment_condo",NA))
  expect_identical(x,c(TRUE,TRUE,FALSE,FALSE,FALSE,FALSE))
  expect_false(wcad_corroborated_multifamily("C3",100000,"city_exact_parcel_apartment_condo",TRUE))
})
test_that("verified relocation preserves accepted totals and original exclusions", {
  cases <- data.frame(case_number=LETTERS[1:5],file_date=as.Date("2025-06-01"),outcome_year=2025L,
    source_county=c("Travis","Travis","Travis","Williamson","Travis"),
    source_jp_district=c("JP1","JP1","JP1","JP1","JP1"),
    assignment_status=c("assigned_unique_hex","excluded_multiple_hexes",rep("assigned_unique_hex",3)),
    assigned_hex_key=c("1",NA,"1","3","1"))
  r <- list(cases=cases,assigned_cases=filter(cases,assignment_status=="assigned_unique_hex"))
  e <- data.frame(case_number=c("A","B","B","C","D","E","E"),
    source_county=c(rep("Travis",4),"Williamson","Travis","Travis"),
    address_for_geocoding=c("a","b","bb","c","d","e","missing"))
  g <- data.frame(source_county=c(rep("Travis",4),"Williamson","Travis"),
    address_for_geocoding=c("a","b","bb","c","d","e"),property_id=c("p","p","p",NA,"w","p"),
    property_hex_id=c(2,2,2,NA,4,2),property_link_status=c(rep("verified",3),"review",rep("verified",2)),
    reference_distance_m=c(90,90,100,NA,90,90))
  counties <- data.frame(hex_id=1:4,source_county=c("Travis","Travis","Williamson","Williamson"))
  coverage <- data.frame(hex_id=1:4,outcome_year=2025L,source_covered=TRUE,coverage_jp_district=c("JP1","JP1","JP1","JP2"))
  out <- apply_eviction_property_geography(r,e,g,coverage,counties,1:4)
  expect_identical(out$cases$assigned_hex_key,c("2",NA,"1","3","1"))
  expect_identical(out$cases$original_assigned_hex_key,c("1",NA,"1","3","1"))
  expect_identical(out$cases$assignment_status,cases$assignment_status)
  expect_equal(nrow(out$assigned_cases),4)
  expect_match(out$cases$property_assignment_status[4],"court_conflict")
  expect_match(out$cases$property_assignment_status[5],"unverified")
})
test_that("references do not become extra unit-bearing accounts", {
  f <- tempfile(fileext=".csv"); on.exit(unlink(f))
  write.csv(data.frame(geometry_source_parcel_id=character(),certified_quick_ref_id=character(),evidence=character()),f,row.names=FALSE)
  x <- data.frame(QuickRefID=c("R1","R2","R3"),LegalDescription=c(
    "REFERENCE ONLY - BLOCK D {R9/NON-REF}","REFERENCE ONLY - BLOCK K {R9/NON-REF}","BLOCK C"))
  y <- wcad_nonreference_links(x,f)
  expect_equal(nrow(y),2)
  expect_setequal(y$geometry_source_parcel_id,c("R1","R2"))
  expect_identical(unique(y$certified_quick_ref_id),"R9")
})
