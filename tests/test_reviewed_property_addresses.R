suppressPackageStartupMessages({library(testthat);library(dplyr);library(sf)})
source('R/eviction_property_reviews.R')
config <- jsonlite::read_json('config/eviction_property_address_reviews.json')
config$reviews <- config$reviews[1]
config_path <- tempfile(fileext='.json')
jsonlite::write_json(config,config_path,auto_unbox=TRUE,digits=NA)
r <- config$reviews[[1]]
refs <- data.frame(project_id=r$project_id,parcel_id=r$parcel_id,source_county=r$source_county,
 operational_units=100,unit_hex_id=r$expected_unit_hex_id,lon=r$reference_longitude,lat=r$reference_latitude)
addresses <- c('8000 US 290 W, AUSTIN, TX 78736','8000 W HIGHWAY 290 APT 123, AUSTIN, TX 78737',
 '8001 US 290 W, AUSTIN, TX 78736','8000 US 290 E, AUSTIN, TX 78736',
 '8000 US 290 W, DRIPPING SPRINGS, TX 78736','8000 US 290 W, AUSTIN, TX 78748')
g <- data.frame(source_county='Travis',address_for_geocoding=addresses,longitude=-97.89,latitude=30.23,
 property_id=NA_character_,property_hex_id=NA_integer_,property_link_status='unverified',reference_distance_m=NA_real_)
test_that('reviewed alias is narrow and reference drift fails',{
 z <- apply_reviewed_property_addresses(g,refs,config_path)$rows
 expect_equal(which(z$property_link_status=='verified'),1:2)
 expect_true(all(is.na(z$property_id[3:6])))
 bad <- refs;bad$unit_hex_id <- 999L
 expect_error(apply_reviewed_property_addresses(g,bad,config_path))
 bad <- refs;bad$lon <- bad$lon+.001
 expect_error(apply_reviewed_property_addresses(g,bad,config_path))
})
test_that('overlap and evidence drift fail closed',{
 path <- tempfile(fileext='.json');on.exit(unlink(path))
 bad <- config;bad$reviews <- rep(bad$reviews,2)
 jsonlite::write_json(bad,path,auto_unbox=TRUE,digits=NA)
 expect_error(apply_reviewed_property_addresses(g,refs,path),'Overlapping')
 bad <- config;bad$reviews[[1]]$evidence[[1]]$sha256 <- strrep('0',64)
 jsonlite::write_json(bad,path,auto_unbox=TRUE,digits=NA)
 expect_error(apply_reviewed_property_addresses(g,refs,path),'changed property review evidence')
})

test_that('shared property references require consistent county, cell and coordinates',{
 shared <- bind_rows(refs,mutate(refs,parcel_id='another_home'))
 expect_equal(apply_reviewed_property_addresses(g,shared,config_path)$rows$property_hex_id[1],r$expected_unit_hex_id)
 for (field in c('lon','lat','unit_hex_id')) {
  bad <- shared;bad[[field]][2] <- bad[[field]][2]+1
  expect_error(apply_reviewed_property_addresses(g,bad,config_path))
 }
 bad <- shared;bad$source_county[2] <- 'Williamson'
 expect_error(apply_reviewed_property_addresses(g,bad,config_path))
})
unlink(config_path)
