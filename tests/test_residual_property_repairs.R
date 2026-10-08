suppressPackageStartupMessages({library(testthat);library(dplyr);library(sf);library(readr)})
source('R/reviewed_unit_properties.R')
r <- read_reviewed_unit_properties('config/residual_property_reviews.json')
# Geometry guards exercise the preserved input, not the already-corrected output.
p <- readRDS('output/residential_cluster_rebuild_20261007/before/output/residential_parcels_unit_promoted.rds');g <- readRDS('output/hex_grid.rds')
geometry <- r;geometry$projects <- Filter(function(x)identical(x$count_status,'reviewed_geometry_only'),r$projects)
test_that('geometry-only reviews preserve estimates and reject unit drift',{
 z <- apply_reviewed_unit_properties(p,geometry,g)$parcels
 ids <- c('549351','942518');a <- p[match(ids,p$parcel_id),];b <- z[match(ids,z$parcel_id),]
 for(f in c('promoted_units','unit_model_selection_method','unit_model_used','unit_estimation_confidence_targeted','unit_estimation_notes_targeted')) expect_identical(a[[f]],b[[f]])
 expect_true(all(b$unit_review_count_status=='reviewed_geometry_only'))
 bad <- p;bad$promoted_units[bad$parcel_id=='549351'] <- 300
 expect_error(apply_reviewed_unit_properties(bad,geometry,g),'Geometry-only review units changed')
})
test_that('park additions retain unique homes and disclose shared references',{
 homes <- Filter(function(x)!is.null(x$inventory_group),r$projects)
 expect_length(homes,676L)
 expect_equal(sum(vapply(homes,function(x)identical(x$geometry$location_precision,'park_phase'),logical(1))),503L)
 expect_true(all(vapply(homes,function(x)x$units==1,logical(1))))
 inventory <- read_csv('data/reviewed_unit_properties/residual_20261007/home_inventory_review.csv',show_col_types=FALSE)
 expect_equal(sum(!inventory$ready),3L)
 expect_equal(sum(inventory$ready),676L)
})
