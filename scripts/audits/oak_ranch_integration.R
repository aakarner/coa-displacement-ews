# Focused integration replay only; canonical outputs are never overwritten.
suppressPackageStartupMessages({library(dplyr);library(readr);library(sf)})
source('R/reviewed_unit_properties.R');source('R/eviction_property_reviews.R')
root <- 'tmp/residential_followup_20261007'
paths <- c('output/residential_parcels_unit_promoted.rds','output/part2/ownership/parcel_owner_panel.rds','output/property_geography/eviction_address_properties.rds')
paths <- paths[file.exists(paths)]
hashes <- vapply(paths,digest::digest,character(1),file=TRUE,algo='sha256')
reviews <- read_reviewed_unit_properties("config/manufactured_home_property_reviews.json")
homes <- reviews; homes$projects <- Filter(function(r)!is.null(r$inventory_group),reviews$projects)
p <- readRDS(paths[1]);grid <- readRDS('output/hex_grid.rds')
z <- apply_reviewed_unit_properties(p,homes,grid)
ids <- unlist(lapply(homes$projects,`[[`,'parcel_ids'))
h <- filter(z$parcels,parcel_id %in% ids)
stopifnot(nrow(h)==793L,nrow(z$parcels)==nrow(p)+791L,sum(h$promoted_units)==793,
 sum(z$audit$delta)==791,sum(h$unit_review_added_account)==791,!anyDuplicated(z$parcels$parcel_id),
 all(h$promoted_units==1),all(is.finite(h$lon)),all(is.finite(h$lat)))
unchanged <- !p$parcel_id %in% ids
j <- match(p$parcel_id[unchanged],z$parcels$parcel_id)
stopifnot(identical(p$promoted_units[unchanged],z$parcels$promoted_units[j]),
 identical(p$lon[unchanged],z$parcels$lon[j]),identical(p$lat[unchanged],z$parcels$lat[j]))
# Preserve both original owner classifications and test duplicate rejection.
spec <- jsonlite::read_json('config/ownership_snapshot_spec.json')
owners <- read_csv(file.path(spec$upstream_repository,'output/historical_ownership/travis_owner_snapshots_2024_2025.csv'),
 col_types=cols(.default=col_character()),na=c('','NA'),show_col_types=FALSE) %>%
 select(-any_of(c('property_units','residential_use_category'))) %>% mutate(tax_year=as.integer(tax_year))
o <- append_reviewed_home_owners(owners,reviews,spec$classification_rule_version)
stopifnot(nrow(o)==nrow(owners)+1582L,!anyDuplicated(o[c('parcel_id','tax_year')]))
ho <- filter(o,parcel_id %in% ids)
stopifnot(sum(ho$tax_year==2025 & ho$classification_status=='matched_classified')==793L,
 sum(ho$tax_year==2024 & ho$classification_status=='source_parcel_not_found')==208L)
refs <- read_csv('output/property_geography/residential_unit_references.csv',col_types=cols(parcel_id='c'),show_col_types=FALSE)
newrefs <- h %>% transmute(parcel_id,source_county,project_id=unit_model_project_id,operational_units=promoted_units,lon,lat)
pt <- st_transform(st_as_sf(newrefs,coords=c('lon','lat'),crs=4326),st_crs(grid))
newrefs$unit_hex_id <- grid$hex_id[vapply(st_within(pt,grid),function(x){stopifnot(length(x)==1);x},integer(1))]
refs <- bind_rows(filter(refs,!parcel_id %in% ids),newrefs)
g <- readRDS('output/property_geography/eviction_address_properties.rds')
case_config <- jsonlite::read_json("config/eviction_property_reviews.json")
case_config$batches <- Filter(function(b)b$batch_id != "residual_20261007",case_config$batches)
case_path <- tempfile(fileext=".json")
jsonlite::write_json(case_config,case_path,auto_unbox=TRUE,digits=NA)
cases <- compile_property_case_reviews(refs,g,case_path)
unlink(case_path)
c <- filter(cases$rows,grepl('oak_ranch_case_',property_review_id))
stopifnot(n_distinct(c$case_number)==24,n_distinct(c$property_id)==17)
saveRDS(z$parcels,file.path(root,'oak_staged_promoted.rds'))
write_csv(newrefs,file.path(root,'oak_staged_unit_references.csv'))
write_csv(c,file.path(root,'oak_staged_case_reviews.csv'))
write_csv(count(ho,tax_year,classification_status),file.path(root,'oak_ownership_coverage.csv'))
write_csv(newrefs %>% group_by(unit_hex_id) %>% summarise(units=sum(operational_units),.groups='drop'),file.path(root,'oak_staged_units_by_hex.csv'))
stopifnot(identical(hashes,vapply(paths,digest::digest,character(1),file=TRUE,algo='sha256')))
jsonlite::write_json(list(production_applied=FALSE,homes=793,added_homes=791,existing_homes_relocated=2,reviewed_cases=24,case_home_accounts=17,owner_2025_classified=793,owner_2024_classified=585,owner_2024_missing=208,source_hashes=as.list(hashes)),file.path(root,'oak_integration_validation.json'),auto_unbox=TRUE,pretty=TRUE)
cat('Oak Ranch staged integration passes: 793 homes, +791 units, 24 cases, source-year ownership; production unchanged.\n')
