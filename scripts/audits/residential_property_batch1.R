# Evidence review only; no production assignment or unit inputs are changed.
suppressPackageStartupMessages({library(dplyr);library(readr);library(sf);library(stringr);library(ggplot2)})
root <- "output/residential_property_batch1"
dir.create(root, recursive=TRUE, showWarnings=FALSE)
e <- readRDS("output/eviction_low_unit_audit/evidence.rds")
r <- e$rows %>% filter(hex_id %in% c(6381L,6965L,6260L,3485L))
stopifnot(n_distinct(r$case_number)==162L)
locations <- r %>% group_by(hex_id,st_addr,longitude,latitude) %>% summarise(
  cases=n_distinct(case_number),.groups="drop")
write_csv(locations,file.path(root,"locations.csv"))
write_csv(r %>% select(case_number,hex_id,st_addr,address_for_geocoding,longitude,latitude),
  file.path(root,"case_address_evidence.csv"))
polys <- readRDS("output/eviction_low_unit_audit/nearby_county_parcels.rds")
grid <- st_transform(readRDS("output/hex_grid.rds"),st_crs(polys))
units <- read_csv("output/property_geography/residential_unit_references.csv",show_col_types=FALSE)
up <- st_transform(st_as_sf(units,coords=c("lon","lat"),crs=4326,remove=FALSE),st_crs(polys))
lp <- st_transform(st_as_sf(locations,coords=c("longitude","latitude"),crs=4326,remove=FALSE),st_crs(polys))
profiles <- read_csv("../landlord-mapper/output/property_profile.csv",col_types=cols(.default="c"),show_col_types=FALSE)
chars <- read_csv("../landlord-mapper/output/property_characteristics.csv",col_types=cols(.default="c"),show_col_types=FALSE)
links <- read_csv("../landlord-mapper/output/links.csv",col_types=cols(.default="c"),show_col_types=FALSE)
ids <- as.character(c(291453,513751,533185,576183,975264,878332,887111,774333:774344,774412,866401))
accounts <- profiles %>% filter(propertyProf_pID %in% ids) %>%
  select(parcel_id=propertyProf_pID,propertyProf_landOnly,propertyProf_imprvOnly,propertyProf_imprvStateCd,
    propertyProf_imprvType,propertyProf_imprvMainArea,propertyProf_imprvUnits,propertyProf_imprvActualYearBuilt) %>%
  left_join(chars %>% select(parcel_id=propertyChar_pID,propertyChar_dba,propertyChar_useCd,propertyChar_subType),by="parcel_id") %>%
  left_join(units,by="parcel_id")
write_csv(accounts,file.path(root,"account_evidence.csv"))
write_csv(links %>% filter(link_pID %in% ids | link_linkedPID %in% ids),file.path(root,"county_account_links.csv"))
distances <- bind_rows(lapply(seq_len(nrow(lp)),function(i) {
  d <- as.numeric(st_distance(lp[i,],polys))
  j <- head(order(d),8)
  st_drop_geometry(polys[j,]) %>% mutate(st_addr=lp$st_addr[i],distance_m=d[j])
}))
write_csv(distances,file.path(root,"nearby_parcels.csv"))
for (cell in unique(locations$hex_id)) {
  pts <- lp[lp$hex_id==cell,]
  extent <- st_buffer(st_union(pts),if(cell==3485) 850 else 550)
  pp <- polys[lengths(st_intersects(polys,extent))>0,]
  uu <- up[lengths(st_intersects(up,extent))>0 & up$operational_units>=20,]
  gg <- grid[lengths(st_intersects(grid,extent))>0,]
  bb <- st_bbox(extent)
  fig <- ggplot() + geom_sf(data=gg,fill=NA,color="#c6d4df",linewidth=.6) +
    geom_sf(data=pp,fill="#f4f5f5",color="#b6b6b6",linewidth=.2) +
    geom_sf(data=pp[pp$parcel_id %in% ids,],fill="#d5ebe5",color="#54796d",linewidth=.5) +
    geom_sf_text(data=pp[pp$parcel_id %in% ids,],aes(label=parcel_id),size=3,check_overlap=TRUE) +
    geom_sf(data=uu,aes(shape="Housing unit reference"),color="#13698e",size=3) +
    geom_sf(data=pts,aes(shape="Filing geocode"),color="#bc493f",size=3) +
    geom_sf_text(data=gg,aes(label=paste0("Cell ",hex_id)),size=3,color="#6588a0",check_overlap=TRUE) +
    scale_shape_manual(values=c("Filing geocode"=17,"Housing unit reference"=16),name=NULL) +
    coord_sf(xlim=c(bb["xmin"],bb["xmax"]),ylim=c(bb["ymin"],bb["ymax"]),datum=NA) +
    labs(title=paste("Property evidence around cell",cell),subtitle=paste(pts$st_addr,collapse="; "),
      caption="County polygons and current analytical unit points. Labels are parcel IDs; no assignments changed.") +
    theme_void(base_size=11) + theme(legend.position="bottom",plot.title=element_text(face="bold"),plot.margin=margin(12,12,12,12))
  ggsave(file.path(root,paste0("cell_",cell,".png")),fig,width=9,height=8,dpi=150,bg="white")
}
bell_path <- file.path(root,"bell_site_map.geojson")
if(file.exists(bell_path)) {
  b <- st_read(bell_path,quiet=TRUE) %>% filter(type=="UnitLabel") %>% st_transform(st_crs(polys))
  stopifnot(nrow(b)==949L,!anyDuplicated(b$unit_id))
  h <- st_intersects(b,polys)
  b$parcel_id <- vapply(h,function(j)paste(sort(unique(polys$parcel_id[j])),collapse="|"),character(1))
  br <- r %>% filter(st_addr=="10500 S Interstate 35") %>%
    mutate(unit_string=str_match(address_for_geocoding,"(?:APT\\s*#?\\s*|#)([0-9]+)")[,2],
      unit_key=as.character(as.integer(unit_string)))
  bl <- st_drop_geometry(b) %>% transmute(unit_key=as.character(as.integer(unit_label)),unit_label,unit_id,parcel_id)
  matches <- left_join(br,bl,by="unit_key",relationship="many-to-many")
  write_csv(matches %>% select(case_number,address_for_geocoding,unit_string,unit_label,unit_id,parcel_id),
    file.path(root,"bell_unit_matches.csv"))
  write_csv(st_drop_geometry(b) %>% count(parcel_id,name="mapped_units"),file.path(root,"bell_mapped_unit_counts.csv"))
  decisions <- matches %>% group_by(case_number) %>% summarise(n_parcels=n_distinct(parcel_id,na.rm=TRUE),
    all_addresses_matched=all(!is.na(unit_id)),parcel_ids=paste(sort(unique(na.omit(parcel_id))),collapse="|"),.groups="drop") %>%
    mutate(review_status=case_when(!all_addresses_matched~"unmatched_address_variant",
      n_parcels==1L~"unique_phase_candidate",TRUE~"multiple_phase_candidates"))
  write_csv(decisions,file.path(root,"bell_case_review.csv"))
  stopifnot(sum(decisions$review_status=="unique_phase_candidate")==31L,
    sum(decisions$review_status=="multiple_phase_candidates")==22L,
    sum(decisions$review_status=="unmatched_address_variant")==1L)
  print(decisions %>% count(review_status,n_parcels,parcel_ids,name="cases"))
}
domain_path <- file.path(root,"domain_site_map.geojson")
if(file.exists(domain_path)) {
  d <- st_read(domain_path,quiet=TRUE) %>% filter(type=="UnitLabel") %>% st_transform(st_crs(polys))
  stopifnot(nrow(d)==438L,!anyDuplicated(d$unit_id))
  h <- st_intersects(d,polys)
  d$parcel_id <- vapply(h,function(j)paste(sort(unique(polys$parcel_id[j])),collapse="|"),character(1))
  h <- st_intersects(d,grid)
  d$hex_id <- vapply(h,function(j)if(length(j)==1L)grid$hex_id[j] else NA_integer_,integer(1))
  write_csv(st_drop_geometry(d),file.path(root,"domain_mapped_units.csv"))
  write_csv(st_drop_geometry(d) %>% count(parcel_id,hex_id,name="mapped_units"),file.path(root,"domain_mapped_unit_counts.csv"))
  stopifnot(sum(d$parcel_id=="774333")==412L,sum(d$parcel_id=="737155")==26L)
  wrong <- up %>% filter(parcel_id %in% as.character(774341:774344))
  parent <- polys[polys$parcel_id=="774333",]
  footprint_distance <- as.numeric(st_distance(wrong,parent))
  write_csv(st_drop_geometry(wrong) %>% mutate(distance_to_parent_m=footprint_distance),file.path(root,"domain_existing_units.csv"))
  units_summary <- d %>% group_by(parcel_id) %>% summarise(mapped_units=n(),.groups="drop")
  # These centers describe the independent site-map evidence, not a production allocation rule.
  proposed_centers <- suppressWarnings(st_centroid(units_summary))
  h <- st_intersects(proposed_centers,grid)
  proposed_centers$site_map_center_hex <- vapply(h,function(j)if(length(j)==1L)grid$hex_id[j] else NA_integer_,integer(1))
  write_csv(st_drop_geometry(proposed_centers),file.path(root,"domain_site_map_centers.csv"))
  bb <- st_bbox(st_buffer(st_union(c(st_geometry(d),st_geometry(wrong))),100))
  ext <- st_as_sfc(bb)
  pp <- polys[lengths(st_intersects(polys,ext))>0,]
  gg <- grid[lengths(st_intersects(grid,ext))>0,]
  fig <- ggplot() + geom_sf(data=pp,fill="#f5f6f7",color="#bfc3c5",linewidth=.25) +
    geom_sf(data=gg,fill=NA,color="#b8cbd8",linewidth=.55) +
    geom_sf(data=d,aes(color=parcel_id),size=.8) +
    geom_sf(data=wrong,shape=4,color="#ba403a",size=4,stroke=1.4) +
    annotate("text",x=bb["xmin"]+25,y=st_coordinates(wrong)[1,2]-45,hjust=0,
      label="Four accounts currently here\n405.3 estimated units\nCell 3486",size=3.3,color="#9f3732") +
    geom_sf_text(data=gg,aes(label=paste0("Cell ",hex_id)),color="#47738d",size=3,check_overlap=TRUE) +
    scale_color_manual(values=c("737155"="#e1a32c","774333"="#187d70"),
      labels=c("737155"="26 homes: northern footprint","774333"="412 homes: southern footprint"),name=NULL) +
    coord_sf(xlim=c(bb["xmin"],bb["xmax"]),ylim=c(bb["ymin"],bb["ymax"]),datum=NA) +
    labs(title="Villages at the Domain: housing is present but misplaced",
      subtitle="Operator site map has 438 distinct homes; four existing account points share the red cross",
      caption="Map geometry created April 21, 2025; retrieved October 2, 2026. Review evidence only.") +
    theme_void(base_size=11) + theme(legend.position="bottom",plot.title=element_text(face="bold"),plot.margin=margin(12,12,12,12))
  ggsave(file.path(root,"domain_reconciliation.png"),fig,width=9,height=9,dpi=150,bg="white")
}
print(locations %>% select(hex_id,st_addr,cases))
