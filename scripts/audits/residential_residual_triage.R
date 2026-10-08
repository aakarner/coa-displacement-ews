# Read-only triage of the staged residual; no production rebuild or overrides.
suppressPackageStartupMessages({library(dplyr);library(readr);library(sf);library(tidyr)})
source('R/eviction_panel.R');source('R/part2_evictions.R');source('R/unit_count_helpers.R')
root<-'tmp/residential_residual_triage_20261007';dir.create(root,recursive=TRUE,showWarnings=FALSE)
low<-read_csv('tmp/residential_followup_20261007/staged_low_unit_cells.csv',show_col_types=FALSE)
cases<-readRDS('tmp/residential_followup_20261007/staged_case_ledger.rds') %>%
 filter(file_date>=as.Date('2025-04-02'),file_date<=as.Date('2026-04-01'),assignment_status=='assigned_unique_hex') %>%
 mutate(hex_id=as.integer(assigned_hex_key)) %>% inner_join(low,by='hex_id')
f<-bind_rows(part2_eviction_read_filings('output/eviction_filings_prepared_for_geocoding.csv','Travis','travis_reviewed'),
 part2_eviction_read_filings('output/williamson_eviction_filings_prepared_for_geocoding.csv','Williamson','williamson_geocode_cascade')) %>%
 semi_join(cases,by='case_number')
registries<-c(Travis='output/eviction_addresses_geocoded.csv',Williamson='output/williamson_eviction_addresses_geocoded_with_arcgis.csv')
g<-bind_rows(lapply(names(registries),function(county)read_csv(registries[county],col_types=cols(.default=col_skip(),address_for_geocoding='c',st_addr='c',addr_type='c',longitude='d',latitude='d',status='c',score='d'),show_col_types=FALSE) %>% mutate(source_county=county)))
stopifnot(!anyDuplicated(g[c('source_county','address_for_geocoding')]))
e<-inner_join(f,g,by=c('source_county','address_for_geocoding'),relationship='many-to-one') %>% assess_eviction_geocodes() %>% filter(geocode_location_usable) %>%
 inner_join(select(cases,case_number,hex_id,staged_units,property_assignment_status,property_id),by='case_number')
key<-function(x)normalize_unit_address(sub(',.*$','',x)) %>% stringr::str_replace_all('\\bI H\\b|\\bI 35\\b','IH 35')
e<-e %>% mutate(address_key=key(st_addr),point_key=paste(longitude,latitude,sep=':'))
a<-e %>% group_by(address_key) %>% summarise(filings=n_distinct(case_number),cells=paste(sort(unique(hex_id)),collapse=';'),
 unit_range=paste(sort(unique(staged_units)),collapse=';'),precision=paste(sort(unique(addr_type)),collapse=';'),.groups='drop') %>% arrange(desc(filings))
write_csv(e,file.path(root,'case_address_evidence.csv'));write_csv(a,file.path(root,'address_groups.csv'))
s<-readRDS('output/residential_unit_source_records.rds') %>% filter(source_name %in% c('costar_current','austin_affordable_housing_inventory','austin_universal_recycling_inventory')) %>% mutate(address_key=key(source_address))
p<-readRDS('tmp/residential_followup_20261007/oak_staged_promoted.rds') %>%
 transmute(parcel_id,project_id=unit_model_project_id,situs_address,owner_names,units=promoted_units,lon,lat,state_code=propertyProf_imprvStateCd,method=unit_model_selection_method,address_key=key(situs_address))
links<-readRDS('output/residential_unit_source_parcel_links.rds')
sm<-inner_join(a,s,by='address_key',relationship='many-to-many') %>% left_join(links,by='source_record_id',relationship='many-to-many') %>%
 left_join(select(p,-address_key),by='parcel_id',relationship='many-to-many')
write_csv(sm,file.path(root,'inventory_matches.csv'))
pm<-inner_join(a,p,by='address_key',relationship='many-to-many');write_csv(pm,file.path(root,'parcel_address_matches.csv'))
polys<-readRDS('output/property_geography/county_polygons.rds')$polygons
loc<-e %>% distinct(point_key,longitude,latitude)
pt<-st_transform(st_as_sf(loc,coords=c('longitude','latitude'),crs=4326),st_crs(polys))
hits<-st_intersects(pt,polys)
pl<-bind_rows(lapply(seq_along(hits),function(i)data.frame(point_key=loc$point_key[i],polygon_row=if(length(hits[[i]]))hits[[i]] else NA_integer_))) %>%
 mutate(parcel_id=polys$polygon_parcel_id[polygon_row]) %>% left_join(select(p,-address_key),by='parcel_id') %>%
 left_join(distinct(e,point_key,address_key,hex_id),by='point_key',relationship='many-to-many')
write_csv(pl,file.path(root,'containing_parcel_matches.csv'))
print(a,n=15,width=130)
cat('Residual cases:',nrow(cases),'zero-unit:',sum(cases$staged_units==0),'positive under20:',sum(cases$staged_units>0),'\n')
print(sm %>% distinct(address_key,filings,source_name,source_project_name,source_unit_count),n=20,width=160)
