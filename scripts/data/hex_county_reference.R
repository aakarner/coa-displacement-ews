# Assign the versioned grid using the same county vintage as the original grid.
suppressPackageStartupMessages({library(sf);library(dplyr);library(readr)})
source('R/grid_contract.R');contract<-grid_contract()
p<-'data/raw_geography/counties_2024.zip'
expected<-'da4051717caec55c75e3748c3608c2a3dbde8d1ff401bbaf4f952e3c3fb63ef1'
stopifnot(identical(digest::digest(file=p,algo='sha256'),expected))
c<-st_read(paste0('/vsizip/',normalizePath(p)),quiet=TRUE) %>% filter(STATEFP=='48',NAME %in% c('Travis','Hays','Williamson')) %>% st_transform(3083)
g<-readRDS('output/hex_grid.rds');pts<-suppressWarnings(st_point_on_surface(st_transform(g,3083)))
hits<-st_within(pts,c);stopifnot(all(lengths(hits)==1L))
x<-data.frame(hex_id=g$hex_id,h3_index=g$h3_index,source_county=c$NAME[unlist(hits)])
old<-read_csv('config/hex_county_assignment_2024.csv',show_col_types=FALSE)
j<-match(old$hex_id,x$hex_id);stopifnot(identical(old$source_county,x$source_county[j]),identical(old$h3_index,x$h3_index[j]))
write_csv(x,'config/hex_county_assignment_2024.csv')
meta<-read_csv('config/hex_county_assignment_2024_metadata.csv',show_col_types=FALSE);meta$grid_rows<-nrow(g)
write_csv(meta,'config/hex_county_assignment_2024_metadata.csv');print(count(x,source_county))
