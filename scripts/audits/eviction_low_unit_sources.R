# Local address/source corroboration; candidates are not assignment overrides.
suppressPackageStartupMessages({library(dplyr);library(readr);library(tidyr);library(sf)})
source("R/unit_count_helpers.R")
root <- "output/eviction_low_unit_audit"
e <- readRDS(file.path(root,"evidence.rds"))
base_key <- function(x) normalize_unit_address(sub(",.*$","",x)) %>%
  stringr::str_replace_all("\\bI H\\b|\\bI 35\\b","IH 35") %>%
  stringr::str_replace_all("\\bF M\\b","FM")
a <- e$rows %>% transmute(point_key,address_key=base_key(st_addr)) %>% distinct() %>% filter(!is.na(address_key))
s <- readRDS("output/residential_unit_source_records.rds") %>%
  filter(source_name %in% c("costar_current","austin_affordable_housing_inventory","austin_universal_recycling_inventory")) %>%
  mutate(address_key=base_key(source_address))
links <- readRDS("output/residential_unit_source_parcel_links.rds")
matches <- inner_join(a,s,by="address_key",relationship="many-to-many") %>%
  left_join(links,by="source_record_id",relationship="many-to-many") %>%
  left_join(e$parcels %>% select(parcel_id,unit_model_project_id,property_units,unit_hex_id,situs_address,
    unit_model_selection_method,unit_land_use_validation_excluded),by="parcel_id") %>%
  left_join(e$locations %>% select(point_key,hex_id,cases,longitude,latitude),by="point_key")
write_csv(matches,file.path(root,"address_source_matches.csv"))
pm <- inner_join(a,e$parcels %>% mutate(address_key=base_key(situs_address)),by="address_key",relationship="many-to-many") %>%
  left_join(e$locations %>% select(point_key,hex_id,cases,longitude,latitude),by="point_key")
write_csv(pm,file.path(root,"address_parcel_matches.csv"))
write_csv(matches %>% distinct(hex_id,cases,source_name,source_address,source_project_name,source_unit_count,
  parcel_id,property_units,unit_hex_id,unit_model_project_id,match_confidence),file.path(root,"address_source_summary.csv"))
saveRDS(list(source_matches=matches,parcel_matches=pm),file.path(root,"sources.rds"))
located_sources <- s %>% filter(is.finite(source_lon),is.finite(source_lat))
sp <- st_transform(st_as_sf(located_sources,coords=c("source_lon","source_lat"),crs=4326,remove=FALSE),3083)
lp <- st_transform(st_as_sf(e$locations,coords=c("longitude","latitude"),crs=4326,remove=FALSE),3083)
d <- as.matrix(unclass(st_distance(lp,sp)))
near <- bind_rows(lapply(seq_len(nrow(lp)),function(i) {
  j <- head(order(d[i,]),5)
  located_sources[j,] %>% mutate(point_key=e$locations$point_key[i],hex_id=e$locations$hex_id[i],
    cases=e$locations$cases[i],street_address=e$locations$street_addresses[i],distance_m=d[i,j])
})) %>% filter(distance_m<=500)
write_csv(near,file.path(root,"nearby_inventory_candidates.csv"))
print(matches %>% distinct(hex_id,cases,source_name,source_address,source_project_name,source_unit_count,
  parcel_id,property_units,unit_hex_id,match_confidence) %>% arrange(desc(cases)) %>% as_tibble(),n=80,width=190)
