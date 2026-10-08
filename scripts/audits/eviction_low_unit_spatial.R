# Continue the local diagnostic audit with cached county parcel polygons.
suppressPackageStartupMessages({library(sf);library(dplyr);library(readr);library(tidyr)})
root <- "output/eviction_low_unit_audit"
e <- readRDS(file.path(root,"evidence.rds")); l <- e$locations; p <- e$parcels
lp <- st_transform(st_as_sf(l,coords=c("longitude","latitude"),crs=4326,remove=FALSE),3083)
grid <- st_transform(readRDS("output/hex_grid.rds"),3083)
area <- st_union(st_buffer(grid[grid$hex_id %in% e$targets$hex_id,],500))
shape_cache <- file.path(root,"nearby_county_parcels.rds")
if(file.exists(shape_cache)) {
  polys <- readRDS(shape_cache)
} else {
  cat("Reading local Travis polygons near the audit cells...\n")
  shape_dir <- file.path(root,"travis_shape")
  if(!file.exists(file.path(shape_dir,"Parcel_poly.shp"))) {
    dir.create(shape_dir,showWarnings=FALSE)
    unzip("data/raw_parcels/travis/Parcel_poly.zip",exdir=shape_dir)
  }
  tc <- st_read(file.path(shape_dir,"Parcel_poly.shp"),
    query="SELECT PROP_ID, SITUS, Multi_PID FROM Parcel_poly",quiet=TRUE)
  cat("Travis polygons loaded; filtering locally...\n")
  tc <- tc[lengths(st_intersects(tc,st_transform(area,st_crs(tc))))>0,] %>% st_make_valid() %>% st_transform(3083) %>%
    transmute(parcel_id=as.character(PROP_ID),polygon_county="Travis",polygon_address=as.character(SITUS),
      polygon_other_ids=as.character(Multi_PID))
  cat("Reading cached Williamson polygons...\n")
  wc <- st_transform(readRDS("data/raw_parcels/williamson/wcad_parcels.rds"),3083)
  wc <- wc[lengths(st_intersects(wc,st_transform(area,st_crs(wc))))>0,] %>% st_make_valid() %>% st_transform(3083) %>%
    transmute(parcel_id=paste0("WILLIAMSON:",trimws(parcelid)),polygon_county="Williamson",polygon_address=siteaddress,
      polygon_other_ids=as.character(propertyid))
  polys <- bind_rows(tc,wc) %>% mutate(polygon_row=row_number())
  saveRDS(polys,shape_cache)
}
cat("Nearby polygons:",nrow(polys),"\n")
hits <- st_intersects(lp,polys)
links <- bind_rows(lapply(seq_along(hits),function(i) data.frame(location_row=i,
  polygon_row=if(length(hits[[i]]))hits[[i]] else NA_integer_))) %>%
  left_join(st_drop_geometry(polys),by="polygon_row") %>%
  left_join(p,by="parcel_id") %>% mutate(point_key=l$point_key[location_row],event_hex_id=l$hex_id[location_row],cases=l$cases[location_row])
# Nearest polygons are evidence to inspect, not automatic case assignments.
nearest <- st_nearest_feature(lp,polys)
near <- st_drop_geometry(polys[nearest,])
l$nearest_polygon_id <- near$parcel_id
l$nearest_polygon_distance_m <- as.numeric(st_distance(lp,polys[nearest,],by_element=TRUE))
l$containing_polygon_count <- lengths(hits)
l$containing_polygon_ids <- vapply(hits,function(i)paste(sort(unique(polys$parcel_id[i])),collapse=" | "),character(1))
# Parcel-reference points and the canonical unit hexes, including neighboring cells.
pp <- st_as_sf(p,coords=c("point_x","point_y"),crs=3083,remove=FALSE)
np <- st_nearest_feature(lp,pp)
l$nearest_unit_parcel_id <- p$parcel_id[np]
l$nearest_unit_parcel_distance_m <- as.numeric(st_distance(lp,pp[np,],by_element=TRUE))
l$nearest_unit_parcel_hex <- p$unit_hex_id[np]
l$nearest_unit_parcel_units <- p$property_units[np]
write_csv(links,file.path(root,"point_containing_parcel_links.csv"))
write_csv(l,file.path(root,"geocode_locations_spatial.csv"))
saveRDS(list(locations=l,links=links,polys=polys),file.path(root,"spatial.rds"))
print(links %>% select(event_hex_id,cases,parcel_id,polygon_address,situs_address,property_units,unit_hex_id,
  unit_model_project_id,unit_model_selection_method) %>% arrange(desc(cases)) %>% head(50) %>% as_tibble(),n=50,width=160)
cat("Locations outside every nearby parcel polygon:",sum(lengths(hits)==0),"\n")
