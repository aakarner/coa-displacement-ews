# Independently verify complete boundary coverage and permanent reviewed IDs.
suppressPackageStartupMessages({library(sf);library(dplyr);library(readr)})
source('R/grid_contract.R');m<-grid_contract()
g<-readRDS('output/hex_grid.rds')
b<-readRDS('output/residential_cluster_rebuild_20261007/before/output/hex_grid.rds')
r<-read_csv('config/hex_id_registry.csv',col_types=cols(hex_id='i',h3_index='c'),show_col_types=FALSE)
j<-match(b$hex_id,g$hex_id)
stopifnot(nrow(g)==7950L,nrow(b)==7027L,nrow(r)==nrow(g),!anyDuplicated(g$hex_id),!anyDuplicated(g$h3_index),
 identical(g$hex_id,r$hex_id),identical(g$h3_index,r$h3_index),identical(b$h3_index,g$h3_index[j]),
 identical(st_as_binary(st_geometry(b)),st_as_binary(st_geometry(g[j,]))),
 identical(b$longitude,g$longitude[j]),identical(b$latitude,g$latitude[j]))
c<-st_read(m$boundary_path,quiet=TRUE) %>% filter(toupper(trimws(city_name))=='CITY OF AUSTIN',toupper(trimws(jurisdiction_type))=='FULL') %>% st_transform(3083) %>% st_make_valid() %>% summarise()
gp<-st_transform(g,3083)
missing<-suppressWarnings(st_difference(st_geometry(c),st_union(gp)))
stopifnot(sum(as.numeric(st_area(missing)))<1,
 sum(lengths(st_covered_by(suppressWarnings(st_point_on_surface(gp)),c))>0)==6196L,
 m$city_center_cells==6196L,m$grid_cells==7950L)
old_counties<-read_csv('output/residential_cluster_rebuild_20261007/before/config/hex_county_assignment_2024.csv',show_col_types=FALSE)
counties<-read_csv('config/hex_county_assignment_2024.csv',show_col_types=FALSE)
stopifnot(nrow(counties)==nrow(g),!anyNA(counties$source_county),identical(old_counties$source_county,counties$source_county[match(old_counties$hex_id,counties$hex_id)]))
cat('Expanded H3 grid: 923 additions, original IDs/geometries and county assignments preserved, zero uncovered City area, 6196 center-selected cells.\n')

old_jp<-read_csv('output/residential_cluster_rebuild_20261007/before/config/williamson_jp_hex_assignment.csv',show_col_types=FALSE)
new_jp<-read_csv('config/williamson_jp_hex_assignment.csv',show_col_types=FALSE)
key<-function(x)paste(x$hex_id,x$effective_start_date)
j<-match(key(old_jp),key(new_jp))
stopifnot(!anyNA(j),identical(old_jp$jp_district,new_jp$jp_district[j]),identical(old_jp$h3_index,new_jp$h3_index[j]))
