# Independent read-only audit of local paired 311 artifacts.
suppressPackageStartupMessages({library(dplyr); library(sf)})
source("R/part2_311.R")
eq <- function(a,b) stopifnot(isTRUE(all.equal(a,b,check.attributes=FALSE,tolerance=1e-9)))
root <- "output/part2/311"
summary <- read.csv(file.path(root,"311_snapshot_summary.csv"))
stopifnot(all(c("usable_rate_change_hexes","usable_legacy_percent_change_hexes") %in% names(summary)),
  !"usable_change_hexes" %in% names(summary))
manifest <- jsonlite::read_json(file.path(root,"311_run_manifest.json"),simplifyVector=FALSE)
stopifnot(manifest$status=="paired_311_features_complete_v2",manifest$part1_outputs_unchanged,
          identical(manifest$all_requests_coverage_verified,FALSE))
n_hash <- 0L
for (entry in c(manifest$inputs,manifest$outputs)) {
  stopifnot(identical(digest::digest(file=entry$path,algo="sha256"),entry$sha256));n_hash<-n_hash+1L
}
grid<-readRDS("output/hex_grid.rds")
units<-st_drop_geometry(readRDS("output/corporate_ownership_by_hex.rds"))
raw<-readRDS("data/raw_311/austin_311_selected_20200101_20260401.rds")$data
ledger<-readRDS(file.path(root,"311_event_ledger.rds"))
stopifnot(!anyDuplicated(ledger$sr_number),setequal(ledger$sr_number,raw$sr_number))
original<-raw[match(ledger$sr_number,raw$sr_number),]
eq(ledger$sr_created_date,as.Date(substr(original$sr_created_date,1,10)))
ok<-which(ledger$coordinate_valid)
points<-st_transform(st_as_sf(ledger[ok,],coords=c("longitude","latitude"),crs=4326),3083)
hits<-st_within(points,st_transform(grid,3083))
one<-which(lengths(hits)==1L)
eq(ledger$hex_id[ok[one]],grid$hex_id[vapply(hits[one],`[`,integer(1),1L)])
ref<-readRDS(file.path(root,"311_scaling.rds"))
stopifnot(ref$reference_date==as.Date("2025-04-01"))
sets<-list()
for(i in 1:2) {
  cutoff<-as.Date(c("2025-04-01","2026-04-01")[i])
  recent_start<-as.Date(c("2024-04-02","2025-04-02")[i])
  previous_start<-as.Date(c("2023-04-02","2024-04-02")[i])
  directory<-file.path(root,as.character(cutoff))
  f<-readRDS(file.path(directory,"311_features_by_hex.rds"));sets[[i]]<-f
  stopifnot(identical(f$hex_id,grid$hex_id),is.integer(f$hex_id),nrow(f)==7027L,
    all(f$analysis_as_of_date==cutoff),sum(f$sr_311_in_current_city_scope)==6060L,
    all(!f$sr_311_all_requests_coverage_verified))
  eq(f$area_km2,grid$area_km2)
  eq(f$residential_units,units$residential_units[match(grid$hex_id,units$hex_id)])
  good<-ledger$event_date_valid & !is.na(ledger$hex_id) & ledger$event_inside_current_city %in% TRUE
  recent<-good & ledger$sr_created_date>=recent_start & ledger$sr_created_date<=cutoff
  previous<-good & ledger$sr_created_date>=previous_start & ledger$sr_created_date<recent_start
  count<-function(which) as.integer(table(factor(ledger$hex_id[which],levels=grid$hex_id)))
  nr<-count(recent);np<-count(previous)
  eq(f$sr_311_recent_observed_count,nr);eq(f$sr_311_previous_observed_count,np)
  covered<-f$sr_311_poc_coverage_usable
  stopifnot(all(!covered | (f$sr_311_in_current_city_scope & f$sr_311_geography_source_covered %in% TRUE)),
    all(is.na(f$sr_311_smoke_signal_latest_12mo[!covered])),
    all(is.na(f$sr_311_pressure_index[!covered])),all(!f$sr_311_valid_zero_latest[!covered]))
  eq(f$sr_311_smoke_signal_latest_12mo[covered],nr[covered])
  rate<-rep(NA_real_,nrow(f));take<-covered & is.finite(f$residential_units) & f$residential_units>=20
  rate[take]<-100*nr[take]/f$residential_units[take]
  eq(f$sr_311_smoke_signal_latest_12mo_per_100_units,rate)
  rate_change <- rep(NA_real_,nrow(f)); rate_change[take] <- 100*(nr[take]-np[take])/f$residential_units[take]
  eq(f$sr_311_smoke_signal_latest_12mo_rate_change_per_100_units,rate_change)
  density<-ifelse(covered,nr/f$area_km2,NA_real_)
  change<-rep(NA_real_,nrow(f));take<-covered & np>0
  change[take]<-100*(nr[take]/np[take]-1)
  eq(f$sr_311_smoke_signal_latest_12mo_density,density)
  eq(f$sr_311_smoke_signal_latest_12mo_change_pct,change)
  eq(f$sr_311_valid_zero_latest,covered & nr==0L)
  comp<-part2_311_components();scores<-matrix(NA_real_,nrow(f),length(comp))
  for(j in seq_along(comp)) {
    x<-f[[comp[j]]];b<-ref$bounds[ref$bounds$component==comp[j],]
    signed <- grepl("rate_change_per_100_units$",comp[j])
    if(i==1L) eq(c(b$lower_bound,b$upper_bound), if(signed) c(-1,1)*as.numeric(quantile(abs(x),.99,na.rm=TRUE,type=7)) else as.numeric(quantile(x,c(.01,.99),na.rm=TRUE,type=7)))
    expected<-if(b$degenerate_range) ifelse(is.na(x),NA_real_,if(signed) 50 else 0) else
      100*(pmin(pmax(x,b$lower_bound),b$upper_bound)-b$lower_bound)/(b$upper_bound-b$lower_bound)
    eq(f[[paste0(comp[j],"_score")]],expected);scores[,j]<-expected
  }
  n<-rowSums(!is.na(scores));index<-rowMeans(scores,na.rm=FALSE);index[n!=length(comp)]<-NA_real_
  eq(f$sr_311_pressure_index_components_complete,n==length(comp))
  stopifnot(all(f$sr_311_pressure_index_components_required==length(comp)))
  eq(f$sr_311_pressure_index,index);eq(f$sr_311_pressure_index_components_available,n)
  membership<-read.csv(file.path(directory,"311_event_window_membership.csv"))
  stopifnot(!anyDuplicated(membership$sr_number))
  membership<-membership[match(ledger$sr_number,membership$sr_number),]
  expected_window<-ifelse(ledger$event_date_valid & ledger$sr_created_date>=recent_start & ledger$sr_created_date<=cutoff,"recent",
    ifelse(ledger$event_date_valid & ledger$sr_created_date>=previous_start & ledger$sr_created_date<recent_start,"previous","outside"))
  eq(membership$event_window,expected_window)
}
eq(readRDS(file.path(root,"311_features_paired.rds")),bind_rows(sets))
changes<-read.csv(file.path(root,"311_feature_changes_by_hex.csv"))
eq(changes$delta_sr_311_pressure_index,sets[[2]]$sr_311_pressure_index-sets[[1]]$sr_311_pressure_index)
before<-read.csv(file.path(root,"part1_preservation_before.csv"));after<-read.csv(file.path(root,"part1_preservation_after.csv"))
eq(before[c("path","sha256")],after[c("path","sha256")])
cat("Part2 311 output audit passed:",n_hash,"hash checks; independent counts, windows, geometry, denominators, frozen scores.\n")
