# Staging-only replay: precision and reviewed addresses, never production writes.
suppressPackageStartupMessages({library(dplyr);library(readr);library(sf)})
source('R/eviction_panel.R');source('R/eviction_coverage.R');source('R/part2_evictions.R');source('R/eviction_property_geography.R')
root<-'tmp/residential_followup_20261007';dir.create(root,recursive=TRUE,showWarnings=FALSE)
paths<-c(measurement='output/part1/measurement/current_measurement.rds',ledger='output/part2/evictions/eviction_case_ledger.rds',geography='output/property_geography/eviction_address_properties.rds',units='output/residential_parcels_unit_promoted.rds')
before<-vapply(paths,digest::digest,character(1),file=TRUE,algo='sha256')
refs<-read_csv('output/property_geography/residential_unit_references.csv',col_types=cols(parcel_id='c'),show_col_types=FALSE)
# Optional staged homes are built by oak_ranch_integration.R, never production.
home_path<-file.path(root,'oak_staged_unit_references.csv')
if(file.exists(home_path)) {
 homes<-read_csv(home_path,col_types=cols(parcel_id='c'),show_col_types=FALSE)
 refs<-bind_rows(filter(refs,!parcel_id %in% homes$parcel_id),homes)
}
# Freeze this historical staging checkpoint before the residual repair batch.
case_config<-jsonlite::read_json('config/eviction_property_reviews.json')
case_config$batches<-Filter(function(b)b$batch_id!='residual_20261007',case_config$batches)
case_path<-tempfile(fileext='.json');jsonlite::write_json(case_config,case_path,auto_unbox=TRUE,digits=NA)
address_config<-jsonlite::read_json('config/eviction_property_address_reviews.json')
address_config$reviews<-Filter(function(r)!startsWith(r$review_id,'residual_'),address_config$reviews)
address_path<-tempfile(fileext='.json');jsonlite::write_json(address_config,address_path,auto_unbox=TRUE,digits=NA)
geography<-readRDS(paths[['geography']]);case_reviews<-if(file.exists(home_path)) compile_property_case_reviews(refs,geography,case_path)$rows else attr(geography,'case_reviews')
a<-apply_reviewed_property_addresses(geography,refs,address_path);unlink(c(case_path,address_path)); geography<-a$rows;attr(geography,'case_reviews')<-case_reviews
write_csv(filter(geography,!is.na(property_address_review_id)),file.path(root,'reviewed_address_matches.csv'))
files<-c(Travis='output/eviction_filings_prepared_for_geocoding.csv',Williamson='output/williamson_eviction_filings_prepared_for_geocoding.csv')
registries<-c(Travis='travis_reviewed',Williamson='williamson_geocode_cascade')
geofiles<-c(Travis='output/eviction_addresses_geocoded.csv',Williamson='output/williamson_eviction_addresses_geocoded_with_arcgis.csv')
f<-lapply(names(files),function(n)part2_eviction_read_filings(files[n],n,registries[n]))
g<-bind_rows(lapply(seq_along(f),function(i)part2_eviction_read_geocodes(geofiles[i],registries[i],f[[i]])))
f<-bind_rows(f)
grid<-readRDS('output/hex_grid.rds')
counties<-read_csv('config/hex_county_assignment_2024.csv',show_col_types=FALSE)
jps<-read_csv('config/williamson_jp_hex_assignment.csv',show_col_types=FALSE)
sc<-read_csv('config/eviction_sources.csv',show_col_types=FALSE)
city<-st_read('data/BOUNDARIES_jurisdictions_20260429.geojson',quiet=TRUE) %>% filter(toupper(trimws(city_name))=='CITY OF AUSTIN',toupper(trimws(jurisdiction_type))=='FULL') %>% st_make_valid() %>% st_transform(3083) %>% summarise()
coverage<-build_eviction_hex_year_coverage(select(counties,hex_id,source_county),2022:2026,sc,jps,as.Date('2026-04-01'))
z<-part2_eviction_resolve(f,g,grid,select(counties,hex_id,source_county),city,part2_eviction_city_reference(grid,city),coverage,property_geography=geography)
saveRDS(z$cases,file.path(root,'staged_case_ledger.rds'))
old<-readRDS(paths[['ledger']]);old<-old[match(z$cases$case_number,old$case_number),]
stopifnot(identical(old$case_number,z$cases$case_number))
comparison<-z$cases %>% mutate(previous_status=old$assignment_status,previous_hex=old$assigned_hex_key,
 changed_status=assignment_status!=previous_status,
 changed_hex=coalesce(assigned_hex_key,'NA')!=coalesce(previous_hex,'NA'))
write_csv(filter(comparison,changed_status|changed_hex),file.path(root,'changed_cases.csv'))
recent<-comparison %>% filter(file_date>=as.Date('2025-04-02'),file_date<=as.Date('2026-04-01'))
recovery<-recent %>% filter(previous_status=='assigned_unique_hex',assignment_status=='excluded_insufficient_geocode_precision')
write_csv(recovery,file.path(root,'coarse_geocode_recovery_cases.csv'))
write_csv(inner_join(select(recovery,case_number),z$geocode_quality,by='case_number'),file.path(root,'coarse_geocode_recovery_addresses.csv'))
write_csv(filter(recent,grepl('oak_ranch_case_',property_review_id)),file.path(root,'oak_staged_ledger_cases.csv'))
summary<-recent %>% filter(changed_status|changed_hex) %>% count(previous_status,assignment_status,previous_hex,assigned_hex_key,name='filings')
write_csv(summary,file.path(root,'recent_changes.csv'))
stopifnot(identical(before,vapply(paths,digest::digest,character(1),file=TRUE,algo='sha256')))
jsonlite::write_json(list(production_applied=FALSE,source_hashes=as.list(before),
 precision_rule=eviction_geocode_precision_rule(),recent_changes=summary,
 previous_assigned=sum(recent$previous_status=='assigned_unique_hex'),staged_assigned=sum(recent$assignment_status=='assigned_unique_hex')),
 file.path(root,'staging_summary.json'),pretty=TRUE,auto_unbox=TRUE)
print(summary,n=40,width=160)
if (file.exists(home_path)) {
  m<-st_drop_geometry(readRDS(paths[['measurement']]))
  originals<-read_csv('output/property_geography/residential_unit_references.csv',col_types=cols(parcel_id='c'),show_col_types=FALSE) %>% filter(parcel_id %in% homes$parcel_id)
  delta<-bind_rows(transmute(homes,hex_id=unit_hex_id,delta=operational_units),
    transmute(originals,hex_id=unit_hex_id,delta=-operational_units)) %>%
    group_by(hex_id) %>% summarise(delta=sum(delta),.groups='drop')
  m<-m %>% left_join(delta,by='hex_id') %>% mutate(staged_units=residential_units+coalesce(delta,0))
  cases<-recent %>% filter(assignment_status=='assigned_unique_hex') %>%
    mutate(hex_id=as.integer(assigned_hex_key)) %>% left_join(select(m,hex_id,residential_units,staged_units),by='hex_id')
  stopifnot(!anyNA(cases$staged_units),abs(sum(m$staged_units)-sum(m$residential_units)-791)<1e-6)
  write_csv(cases %>% filter(staged_units<20) %>% count(hex_id,staged_units,name='filings'),file.path(root,'staged_low_unit_cells.csv'))
  diagnostic<-list(production_applied=FALSE,cluster_refit=FALSE,
    assigned_filings=nrow(cases),low_unit_filings=sum(cases$staged_units<20),
    low_unit_cells=n_distinct(cases$hex_id[cases$staged_units<20]),
    low_unit_filings_before_home_integration=sum(cases$residential_units<20),
    previous_total_grid_units=sum(m$residential_units),staged_total_grid_units=sum(m$staged_units),
    oak_cases_verified=sum(grepl('oak_ranch_case_',cases$property_review_id)))
  jsonlite::write_json(diagnostic,file.path(root,'combined_diagnostic.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
  print(diagnostic)
}
