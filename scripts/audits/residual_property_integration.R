# Staging-only denominator replay. No canonical or cluster outputs are written.
suppressPackageStartupMessages({library(dplyr);library(readr);library(sf)})
source('R/reviewed_unit_properties.R')
root <- 'tmp/residential_residual_repair_20261007';dir.create(root,recursive=TRUE,showWarnings=FALSE)
reviews <- read_reviewed_unit_properties('config/residual_property_reviews.json')
p <- readRDS('tmp/residential_followup_20261007/oak_staged_promoted.rds')
# Prior Oak replay stores promoted values before the final output-column copy.
oak <- read_reviewed_unit_properties('config/manufactured_home_property_reviews.json')
oakids <- unlist(lapply(oak$projects,`[[`,'parcel_ids'))
oi <- p$parcel_id %in% oakids
p$units_calibrated_targeted[oi] <- p$promoted_units[oi]
p$property_units_targeted[oi] <- p$promoted_units[oi]
p$unit_land_use_validation_excluded[oi] <- FALSE
grid <- readRDS('output/hex_grid.rds')
z <- apply_reviewed_unit_properties(p,reviews,grid)
ids <- unlist(lapply(reviews$projects,`[[`,'parcel_ids'))
# Restore existing review metadata on accounts outside this batch.
k <- match(p$parcel_id[!p$parcel_id %in% ids],z$parcels$parcel_id)
for (field in grep('^unit_review_',names(p),value=TRUE))
 z$parcels[[field]][k] <- p[[field]][!p$parcel_id %in% ids]
i <- z$parcels$parcel_id %in% ids
z$parcels$units_calibrated_targeted[i] <- z$parcels$promoted_units[i]
z$parcels$property_units_targeted[i] <- z$parcels$promoted_units[i]
z$parcels$unit_land_use_validation_excluded[i] <- FALSE
stopifnot(nrow(z$parcels)==nrow(p)+676L,abs(sum(z$audit$delta)-676)<1e-6,
 !anyDuplicated(z$parcels$parcel_id), all(z$parcels$promoted_units[i]>0))
# Geometry-only corrections retain model provenance and counts.
for (id in c('549351','942518')) {
 a <- p[p$parcel_id==id,];b <- z$parcels[z$parcels$parcel_id==id,]
 for (f in c('promoted_units','unit_model_selection_method','unit_model_used','unit_estimation_confidence_targeted')) stopifnot(identical(a[[f]],b[[f]]))
}
unchanged <- !p$parcel_id %in% ids
for(f in c('promoted_units','lon','lat','owner_names')) stopifnot(identical(p[[f]][unchanged],z$parcels[[f]][k]))
# Preserve the unresolved Hudson boundary allocation; no 276-unit supplement.
stopifnot(!'103824' %in% z$parcels$parcel_id)
saveRDS(z$parcels,file.path(root,'staged_promoted.rds'));write_csv(z$audit,file.path(root,'unit_review_audit.csv'))
# Combined source-year ownership import must retain independent home accounts.
allreviews <- read_reviewed_unit_properties();spec <- jsonlite::read_json('config/ownership_snapshot_spec.json')
owners <- read_csv(file.path(spec$upstream_repository,'output/historical_ownership/travis_owner_snapshots_2024_2025.csv'),col_types=cols(.default=col_character()),na=c('','NA'),show_col_types=FALSE) %>% select(-any_of(c('property_units','residential_use_category'))) %>% mutate(tax_year=as.integer(tax_year))
o <- append_reviewed_home_owners(owners,allreviews,spec$classification_rule_version)
homeids <- unlist(lapply(Filter(function(r)!is.null(r$inventory_group),reviews$projects),`[[`,'parcel_ids'))
ho <- filter(o,parcel_id %in% homeids)
stopifnot(nrow(ho)==1352L,!anyDuplicated(o[c('parcel_id','tax_year')]),sum(ho$tax_year==2025 & ho$classification_status=='matched_classified')==676L,sum(ho$tax_year==2024 & ho$classification_status=='source_parcel_not_found')==105L)
write_csv(count(ho,tax_year,classification_status),file.path(root,'ownership_coverage.csv'))
# Diagnostic only: independent geocoder containment of Pecan phase candidates.
cr <- jsonlite::read_json('data/reviewed_eviction_properties/residual_20261007/cases.json')$cases
foot <- st_read('data/reviewed_unit_properties/residual_20261007/footprints.geojson',quiet=TRUE)
h <- read_csv('data/reviewed_unit_properties/residual_20261007/ready_home_locations.csv',col_types=cols(.default=col_character()),show_col_types=FALSE)
checks <- bind_rows(lapply(cr,function(r){
 a <- bind_rows(r$addresses); parent <- h$parent_id[match(r$parcel_id,h$parcel_id)]
 pt <- st_transform(st_as_sf(a,coords=c('longitude','latitude'),crs=4326),st_crs(foot))
 data.frame(case_number=r$case_number,parent_id=parent,geocoder_inside_phase=all(lengths(st_intersects(pt,foot[foot$polygon_parcel_id==parent,]))>0),city_phase_reference_conflict=isTRUE(r$city_phase_reference_conflict))
}))
stopifnot(nrow(checks)==18L,all(checks$geocoder_inside_phase),sum(checks$city_phase_reference_conflict)==6L)
write_csv(checks,file.path(root,'case_phase_checks.csv'));print(checks)
cat('Staged 676 new homes and two geometry-only corrections; ownership checks passed.\n')
