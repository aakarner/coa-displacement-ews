suppressPackageStartupMessages({library(sf);library(dplyr);library(ggplot2)})
root <- "output/eviction_low_unit_audit"
e <- readRDS(file.path(root,"evidence.rds")); s <- readRDS(file.path(root,"spatial.rds"))
grid <- st_transform(readRDS("output/hex_grid.rds"),3083)
g <- list(); polys <- list(); event <- list(); unit <- list(); segments <- list()
for(i in seq_along(c(1806,1220))) {
  id <- c(1806,1220)[i]; pid <- c("896397","956771")[i]; boundary_id <- c("896397","956770")[i]
  title <- c("Orbit: 107 filings in cell 1806", "Alta Blue Goose: 79 filings in cell 1220")[i]
  ep <- st_transform(st_as_sf(e$locations[e$locations$hex_id==id,],coords=c("longitude","latitude"),crs=4326),3083)
  up <- st_as_sf(e$parcels[e$parcels$parcel_id==pid,],coords=c("point_x","point_y"),crs=3083)
  pg <- s$polys[s$polys$parcel_id==boundary_id,]
  gg <- grid[lengths(st_intersects(grid,st_buffer(pg,80)))>0,]
  gg$panel<-title; gg$role<-ifelse(gg$hex_id==id,"Filing cell",ifelse(gg$hex_id==up$unit_hex_id,"Unit cell","Other cell"))
  pg$panel<-title;ep$panel<-title;up$panel<-title
  ep$label<-paste0("Filings: ",sum(e$targets$eviction_recent_observed_cases[e$targets$hex_id==id]))
  up$label<-paste0("Units: ",round(up$property_units))
  seg <- st_sf(panel=title,geometry=st_sfc(st_linestring(rbind(st_coordinates(ep)[1,],st_coordinates(up)[1,])),crs=3083))
  g[[i]]<-gg;polys[[i]]<-pg;event[[i]]<-ep;unit[[i]]<-up;segments[[i]]<-seg
}
g<-bind_rows(g);polys<-bind_rows(polys);event<-bind_rows(event);unit<-bind_rows(unit);segments<-bind_rows(segments)
# Translate each panel to a local origin for a shared metric scale.
shift <- function(x) {
  for(n in unique(x$panel)) {
    origin <- st_coordinates(unit[unit$panel==n,])[1,]
    idx<-which(x$panel==n);st_geometry(x)[idx]<-st_geometry(x)[idx]-origin
  }; st_crs(x)<-st_crs(3083);x
}
g<-shift(g);polys<-shift(polys);event<-shift(event);segments<-shift(segments);unit<-shift(unit)
plot<-ggplot()+geom_sf(data=g,aes(fill=role),color="#7B8794",linewidth=.35)+
  geom_sf(data=polys,fill=NA,color="#2B3A44",linewidth=1)+
  geom_sf(data=segments,color="#475569",linetype="dashed")+
  geom_sf_text(data=g,aes(label=hex_id),size=3,color="#53606B")+
  geom_sf(data=event,color="#BD3C2D",size=3.5)+geom_sf(data=unit,color="#166B8F",shape=17,size=4)+
  geom_sf_text(data=event,aes(label=label),nudge_y=-35,size=3.5,fontface="bold",color="#BD3C2D")+
  geom_sf_text(data=unit,aes(label=label),nudge_y=35,size=3.5,fontface="bold",color="#166B8F")+
  facet_wrap(~panel,nrow=1)+scale_fill_manual(values=c("Filing cell"="#F6DDD6","Unit cell"="#D7EAF2","Other cell"="#F4F5F6"))+
  coord_sf(datum=NA)+labs(title="The same property contributes filings and units to different cells",
    subtitle="Dark outline: county parcel boundary. Red point: filing geocode. Blue triangle: unit reference point.",
    caption="April 2, 2025–April 1, 2026 filings. Units are current operational estimates; no values were changed.",fill=NULL)+
  theme_void(base_size=12)+theme(legend.position="bottom",plot.title=element_text(face="bold",size=16),
    plot.subtitle=element_text(size=11,margin=margin(b=12)),strip.text=element_text(face="bold",size=12),
    plot.caption=element_text(size=10,hjust=0),plot.margin=margin(15,15,15,15))
ggsave(file.path(root,"filings_units_different_cells.png"),plot,width=12,height=6,dpi=160,bg="white")
