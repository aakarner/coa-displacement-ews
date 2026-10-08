# Diagnostic audit only: do not alter filing assignments, units, or clusters.
# Detailed addresses and case identifiers remain in the ignored output directory.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf); library(tidyr)})
source("R/eviction_panel.R")
source("R/part2_evictions.R")
root <- "output/eviction_low_unit_audit"
dir.create(root, recursive=TRUE, showWarnings=FALSE)
cutoff <- as.Date("2026-04-01"); start <- as.Date("2025-04-02")
grid <- st_transform(readRDS("output/hex_grid.rds"),3083)
features <- st_drop_geometry(readRDS("output/hex_features.rds"))
targets <- features %>% filter(eviction_source_covered,residential_units<20,eviction_recent_observed_cases>=10) %>%
  select(hex_id,longitude,latitude,source_county,residential_units,residential_parcels,total_pop,total_housing_units,
    costar_units_current,eviction_recent_observed_cases,eviction_previous_observed_cases) %>%
  arrange(desc(eviction_recent_observed_cases))
stopifnot(nrow(targets)==41L,sum(targets$eviction_recent_observed_cases)==1219L)
write_csv(targets,file.path(root,"target_cells.csv"))
ledger <- readRDS("output/part2/evictions/eviction_case_ledger.rds") %>%
  filter(assignment_status=="assigned_unique_hex", file_date>=start,file_date<=cutoff,
    assigned_hex_key %in% as.character(targets$hex_id)) %>% mutate(hex_id=as.integer(assigned_hex_key))
stopifnot(nrow(ledger)==1219L,!anyDuplicated(ledger$case_number))
source_files <- c(Travis="output/eviction_filings_prepared_for_geocoding.csv",
  Williamson="output/williamson_eviction_filings_prepared_for_geocoding.csv")
geocode_files <- c(Travis="output/eviction_addresses_geocoded.csv",
  Williamson="output/williamson_eviction_addresses_geocoded_with_arcgis.csv")
rows <- bind_rows(lapply(intersect(names(source_files),unique(ledger$source_county)),function(county){
  f <- part2_eviction_read_filings(source_files[county],county,county) %>%
    semi_join(ledger,by="case_number")
  g <- read_csv(geocode_files[county],col_types=cols(.default=col_character(),
    score=col_double(),longitude=col_double(),latitude=col_double(),display_x=col_double(),display_y=col_double(),x=col_double(),y=col_double()),show_col_types=FALSE) %>%
    select(address_for_geocoding,status,score,match_addr,addr_type,loc_name,st_addr,place_name,
      longitude,latitude,display_x,display_y,x,y)
  inner_join(f,g,by="address_for_geocoding") %>%
    filter(status %in% c("M","T"),score>=90,is.finite(longitude),is.finite(latitude)) %>%
    left_join(select(ledger,case_number,hex_id),by="case_number")
}))
stopifnot(n_distinct(rows$case_number)==nrow(ledger))
points <- st_transform(st_as_sf(rows,coords=c("longitude","latitude"),crs=4326,remove=FALSE),3083)
hits <- st_intersects(points,grid)
stopifnot(all(vapply(seq_along(hits),function(i) rows$hex_id[i] %in% grid$hex_id[hits[[i]]],logical(1))))
rows$point_key <- paste(rows$hex_id,format(rows$longitude,digits=14),format(rows$latitude,digits=14),sep=":")
locations <- rows %>% group_by(point_key,hex_id,longitude,latitude) %>% summarise(
  cases=n_distinct(case_number),source_addresses=n_distinct(address_for_geocoding),
  source_address_examples=paste(head(sort(unique(address_for_geocoding)),3),collapse=" | "),
  matched_addresses=paste(sort(unique(match_addr)),collapse=" | "),
  street_addresses=paste(sort(unique(st_addr)),collapse=" | "),
  addr_types=paste(sort(unique(addr_type)),collapse=" | "),min_score=min(score),
  display_x=first(display_x),display_y=first(display_y),.groups="drop") %>% arrange(desc(cases))
write_csv(rows,file.path(root,"case_geocode_evidence.csv"))
write_csv(locations,file.path(root,"geocode_locations.csv"))
write_csv(rows %>% distinct(case_number,hex_id,addr_type) %>% count(addr_type,name="case_type_links"),file.path(root,"geocode_precision_summary.csv"))
parcels <- readRDS("output/residential_parcels_unit_promoted.rds") %>%
  select(parcel_id,situs_address,source_county,lon,lat,coord_source,property_units,promoted_units,units_calibrated_targeted,
    unit_model_project_id,unit_model_selection_method,unit_model_allocation_method,unit_estimation_method,
    direct_costar_units,city_land_use_labels,unit_land_use_validation_excluded,is_multifamily_like,
    improvement_sqft,land_sqft,parcel_address_key) %>% filter(is.finite(lon),is.finite(lat)) %>%
  mutate(property_units_raw=property_units,property_units=if_else(coalesce(unit_land_use_validation_excluded,FALSE),0,coalesce(units_calibrated_targeted,0)))
pp <- st_transform(st_as_sf(parcels,coords=c("lon","lat"),crs=4326,remove=FALSE),3083)
source_grid <- readRDS("output/hex_grid.rds")
ph <- st_within(st_transform(st_as_sf(parcels,coords=c("lon","lat"),crs=4326),st_crs(source_grid)),source_grid)
stopifnot(max(lengths(ph))<=1L)
parcels$unit_hex_id <- grid$hex_id[vapply(ph,function(i) if(length(i))i[1] else NA_integer_,integer(1))]
parcels$point_x <- st_coordinates(pp)[,1];parcels$point_y <- st_coordinates(pp)[,2]
reconcile <- parcels %>% filter(!is.na(unit_hex_id)) %>% group_by(unit_hex_id) %>% summarise(units=sum(property_units,na.rm=TRUE),.groups="drop")
stopifnot(max(abs(coalesce(reconcile$units[match(features$hex_id,reconcile$unit_hex_id)],0)-features$residential_units))<1e-6)
saveRDS(list(targets=targets,ledger=ledger,rows=rows,locations=locations,parcels=parcels),file.path(root,"evidence.rds"))
print(targets %>% count(source_county))
print(rows %>% distinct(case_number,addr_type) %>% count(addr_type))
print(locations %>% select(hex_id,cases,source_addresses,street_addresses,addr_types,min_score),n=60)
cat("Extracted",nrow(locations),"distinct geocode locations for",nrow(ledger),"unique cases.\n")
