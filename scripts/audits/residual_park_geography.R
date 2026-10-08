suppressPackageStartupMessages({library(sf);library(dplyr);library(readr)})
r<-'data/reviewed_unit_properties/residual_20261007'
h<-read_csv(file.path(r,'home_inventory_review.csv'),col_types=cols(parcel_id='c',parent_id='c'),show_col_types=FALSE)
p<-st_read(file.path(r,'footprints.geojson'),quiet=TRUE);g<-readRDS('output/hex_grid.rds')
pt<-st_as_sf(filter(h,ready),coords=c('longitude','latitude'),crs=4326,remove=FALSE)
inside<-st_within(pt,p)
pt$parent_verified<-vapply(seq_len(nrow(pt)),function(i)pt$parent_id[i] %in% p$polygon_parcel_id[inside[[i]]],logical(1))
hits<-st_within(st_transform(pt,st_crs(g)),g)
pt$point_hex_id<-vapply(hits,function(i)if(length(i)==1L)g$hex_id[i] else NA_integer_,integer(1))
base<-readRDS('tmp/residential_followup_20261007/oak_staged_promoted.rds')
pt$already_promoted<-pt$parcel_id %in% base$parcel_id
pt$existing_units<-base$promoted_units[match(pt$parcel_id,base$parcel_id)]
pt$existing_units[!pt$already_promoted]<-0
pt$spatial_ready<-pt$parent_verified & !is.na(pt$point_hex_id)
write_csv(st_drop_geometry(pt),file.path(r,'home_spatial_review.csv'))
write_csv(st_drop_geometry(filter(pt,spatial_ready)),file.path(r,'ready_home_locations.csv'))
print(st_drop_geometry(pt) %>% group_by(park) %>% summarise(candidates=n(),ready=sum(spatial_ready),existing=sum(already_promoted),existing_units=sum(existing_units),.groups='drop'))
