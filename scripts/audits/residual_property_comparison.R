# Compare this batch with the preceding staged batch, not with stale production.
suppressPackageStartupMessages({library(dplyr);library(readr);library(sf)})
root <- 'tmp/residential_residual_repair_20261007'
a <- readRDS(file.path(root,'recent_assigned_cases.rds'))
b <- readRDS('tmp/residential_followup_20261007/staged_case_ledger.rds') %>% filter(file_date>=as.Date('2025-04-02'),file_date<=as.Date('2026-04-01'),assignment_status=='assigned_unique_hex')
low <- read_csv('tmp/residential_followup_20261007/staged_low_unit_cells.csv',show_col_types=FALSE)
x <- inner_join(a,select(b,case_number,prior_hex=assigned_hex_key,prior_property=property_id),by='case_number') %>% mutate(previous_low=as.integer(prior_hex) %in% low$hex_id,current_low=staged_units<20)
e <- read_csv('tmp/residential_residual_triage_20261007/case_address_evidence.csv',show_col_types=FALSE) %>% distinct(case_number,address_key)
changes <- x %>% filter(previous_low != current_low) %>% left_join(e,by='case_number')
write_csv(changes,file.path(root,'threshold_changes_since_previous_stage.csv'))
summary <- count(changes,address_key,previous_low,current_low,name='filings')
write_csv(summary,file.path(root,'threshold_change_summary.csv'));print(summary,n=30)
withdrawn <- x %>% filter(prior_property %in% 'parcel:549351',is.na(property_id))
write_csv(withdrawn,file.path(root,'withdrawn_olivine_links_recent.csv'))
stopifnot(nrow(a)==nrow(b),setequal(a$case_number,b$case_number),sum(x$previous_low)==161L,sum(x$current_low)==108L,
 !any(!x$previous_low & x$current_low),all(a$assigned_hex_key==b$assigned_hex_key[match(a$case_number,b$case_number)]))
checks <- read_csv(file.path(root,'case_phase_checks.csv'),show_col_types=FALSE)
stopifnot(nrow(checks)==18L,all(checks$geocoder_inside_phase),sum(checks$city_phase_reference_conflict)==6L)
review <- attr(readRDS(file.path(root,'geography/eviction_address_properties.rds')),'case_reviews')
expected <- distinct(review,case_number,property_id,property_hex_id,property_review_id)
stopifnot(nrow(expected)==144L)
v <- a[match(expected$case_number,a$case_number),]
stopifnot(!anyNA(v$case_number),identical(v$property_id,expected$property_id),identical(v$assigned_hex_key,as.character(expected$property_hex_id)),identical(v$property_review_id,expected$property_review_id))
result <- list(previous_low_filings=161,staged_low_filings=108,removed_from_low_unit=53,new_low_unit_cases=0,assigned_filings=11716,changed_assigned_cells=0,
 zero_unit_filings=sum(a$staged_units<1e-8),zero_unit_cells=n_distinct(a$hex_id[a$staged_units<1e-8]),
 positive_low_filings=sum(a$staged_units>=1e-8 & a$staged_units<20),positive_low_cells=n_distinct(a$hex_id[a$staged_units>=1e-8 & a$staged_units<20]),
 recent_false_olivine_links_withdrawn=nrow(withdrawn),case_reviews_verified=144,production_applied=FALSE)
jsonlite::write_json(result,file.path(root,'batch_comparison.json'),pretty=TRUE,auto_unbox=TRUE);print(result)
# Check canonical files remain the exact pre-batch products.
paths <- c(measurement='output/part1/measurement/current_measurement.rds',ledger='output/part2/evictions/eviction_case_ledger.rds',geography='output/property_geography/eviction_address_properties.rds',units='output/residential_parcels_unit_promoted.rds')
previous <- jsonlite::read_json('tmp/residential_followup_20261007/staging_summary.json')$source_hashes
for(n in names(paths))stopifnot(identical(digest::digest(file=paths[[n]],algo='sha256'),previous[[n]]))
