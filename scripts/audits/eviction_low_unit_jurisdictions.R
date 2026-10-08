# Compare each filing point with the unit universe's full-purpose city scope.
suppressPackageStartupMessages({library(dplyr);library(sf);library(readr)})
root <- "output/eviction_low_unit_audit"
e <- readRDS(file.path(root,"evidence.rds")); l <- e$locations
lp <- st_transform(st_as_sf(l,coords=c("longitude","latitude"),crs=4326,remove=FALSE),3083)
j <- st_read("data/BOUNDARIES_jurisdictions_20260429.geojson",quiet=TRUE) %>% st_make_valid() %>% st_transform(3083)
h <- st_intersects(lp,j)
l$point_jurisdiction <- vapply(h,function(i)paste(sort(unique(j$jurisdiction_type[i])),collapse=" | "),character(1))
l$point_jurisdiction_label <- vapply(h,function(i)paste(sort(unique(j$jurisdiction_label[i])),collapse=" | "),character(1))
l$point_in_full_city <- vapply(h,function(i)any(j$jurisdiction_type[i]=="FULL"),logical(1))
full <- st_union(j[j$jurisdiction_type=="FULL",])
l$distance_to_full_city_m <- as.numeric(st_distance(lp,full))
write_csv(l,file.path(root,"geocode_jurisdictions.csv"))
saveRDS(l,file.path(root,"jurisdictions.rds"))
print(l %>% select(hex_id,cases,street_addresses,point_jurisdiction,point_in_full_city,distance_to_full_city_m) %>% as_tibble(),n=60,width=170)
print(e$rows %>% select(case_number,point_key,hex_id) %>% distinct() %>% left_join(l %>% select(point_key,point_in_full_city),by="point_key") %>%
  group_by(case_number,hex_id) %>% summarise(in_full=any(point_in_full_city),.groups="drop") %>% count(in_full,wt=NULL))
