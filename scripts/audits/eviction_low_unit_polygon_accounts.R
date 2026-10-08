# Reconcile tax-account/reference points inside a mapped parcel footprint.
# A polygon identifier is not necessarily the residential tax-account identifier.
suppressPackageStartupMessages({library(dplyr);library(sf);library(readr)})
root <- "output/eviction_low_unit_audit"
e <- readRDS(file.path(root,"evidence.rds")); s <- readRDS(file.path(root,"spatial.rds"))
ids <- unique(c(s$links$polygon_row, match(s$locations$nearest_polygon_id,s$polys$parcel_id)))
poly <- s$polys[na.omit(ids),]
pp <- st_as_sf(e$parcels,coords=c("point_x","point_y"),crs=3083,remove=FALSE)
hit <- st_intersects(poly,pp)
a <- bind_rows(lapply(seq_along(hit),function(i){
  if(!length(hit[[i]]))return(NULL)
  e$parcels[hit[[i]],] %>% mutate(polygon_row=poly$polygon_row[i],polygon_parcel_id=poly$parcel_id[i])
}))
summary <- a %>% group_by(polygon_row,polygon_parcel_id) %>% summarise(
  residential_accounts=n(),polygon_units=sum(property_units),
  account_ids=paste(unique(parcel_id),collapse=" | "),
  account_addresses=paste(unique(situs_address),collapse=" | "),
  unit_hexes=paste(sort(unique(unit_hex_id[property_units>0])),collapse=" | "),
  unit_projects=paste(sort(unique(unit_model_project_id[property_units>0])),collapse=" | "),.groups="drop")
write_csv(a,file.path(root,"polygon_residential_accounts.csv"))
write_csv(summary,file.path(root,"polygon_account_summary.csv"))
saveRDS(list(accounts=a,summary=summary),file.path(root,"polygon_accounts.rds"))
print(left_join(s$links %>% filter(is.na(property_units)) %>% distinct(event_hex_id,cases,polygon_row,parcel_id),
  summary %>% select(-polygon_parcel_id),by="polygon_row") %>% as_tibble(),n=60,width=190)
