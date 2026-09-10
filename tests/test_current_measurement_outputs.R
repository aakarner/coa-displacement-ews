# Independent reconstruction of current selected inputs; no production helpers.
suppressPackageStartupMessages({library(sf); library(readr)})
eq <- function(a,b) stopifnot(isTRUE(all.equal(a,b,check.attributes=FALSE,tolerance=1e-9)))
root <- "output/part1/measurement"
x <- readRDS(file.path(root,"current_measurement.rds"))
manifest <- jsonlite::fromJSON(file.path(root,"current_measurement_manifest.json"))
stopifnot(manifest$status=="current_measurement_complete_v2", nrow(x)==7027L,
  !anyDuplicated(x$hex_id), is.integer(x$hex_id), all(x$analysis_as_of_date==as.Date("2026-04-01")),
  all(x$measurement_scope=="single_cutoff_current_sample"))
for (entries in list(manifest$inputs,manifest$outputs)) {
  stopifnot(!anyDuplicated(entries$path),all(file.exists(entries$path)))
  eq(unname(vapply(entries$path,digest::digest,character(1),file=TRUE,algo="sha256")),entries$sha256)
}
canonical <- st_drop_geometry(readRDS("output/hex_features.rds"))
canonical <- canonical[match(x$hex_id,canonical$hex_id),]
for(n in names(x)) eq(canonical[[n]],x[[n]])
units <- st_drop_geometry(readRDS("output/corporate_ownership_by_hex.rds"))
eq(x$residential_units,units$residential_units[match(x$hex_id,units$hex_id)])

# Reliability depends on current triplet only. Each vintage uses its own MOE.
candidates <- readRDS("output/part2/acs/acs_rent_source_candidates.rds")
years <- c(2014L,2019L,2024L)
keys <- paste(candidates$hex_id,candidates$acs_year,candidates$source_geography)
sets <- lapply(c("block_group","tract"),function(g) lapply(years,function(y)
  candidates[match(paste(x$hex_id,y,g),keys),]))
reliable <- lapply(sets,function(s) Reduce(`&`,lapply(s,function(r)
  is.finite(r$estimate)&r$estimate>0&is.finite(r$moe)&r$moe>=0&r$moe/r$estimate<=.30)))
chosen <- ifelse(reliable[[1]],"block_group",ifelse(reliable[[2]],"tract",NA_character_))
eq(x$acs_rent_source_geography,chosen)
selected <- lapply(years,function(y) candidates[match(paste(x$hex_id,y,chosen),keys),])
cpi <- c(`2014`=236.736,`2019`=255.657,`2024`=313.689)
real <- lapply(seq_along(years),function(i) selected[[i]]$estimate*313.689/cpi[as.character(years[i])])
eq(x$acs_rent_current_real,real[[3]])
growth <- 100*log(real[[3]]/real[[2]])/5
acceleration <- growth-100*log(real[[2]]/real[[1]])/5
eq(x$acs_rent_growth_recent_annualized_pct,growth);eq(x$acs_rent_acceleration_pp,acceleration)
for(i in 1:3) eq(x[[paste0("acs_rent_",c("earliest","previous","current")[i],"_source_geoid")]],selected[[i]]$source_geoid)
demo <- st_drop_geometry(readRDS("output/part2/acs/2026-04-01/acs_demographics_by_hex.rds"))
demo <- demo[match(x$hex_id,demo$hex_id),]
for(n in c("median_income_real","pct_renter","poverty_rate","pct_rent_burden_30plus","pct_college"))eq(x[[n]],demo[[n]])

# Independently sum current-year known parcels. Never use common_owner_support.
p <- readRDS("output/part2/ownership/parcel_ownership_snapshots.rds")
p <- p[p$tax_year==2025L & !is.na(p$hex_id),]
known <- !is.na(p$is_corporate_owned)&!is.na(p$has_financialized_owner)
sum_by_hex <- function(value) {
  out<-numeric(nrow(x)); totals<-rowsum(value,p$hex_id,reorder=FALSE)
  out[match(as.integer(rownames(totals)),x$hex_id)]<-totals[,1];out
}
parcels <- sum_by_hex(rep(1,nrow(p))); observed_parcels <- sum_by_hex(as.integer(known))
observed_units <- sum_by_hex(ifelse(known,p$residential_units,0))
corp_units <- sum_by_hex(ifelse(known & p$is_corporate_owned %in% TRUE,p$residential_units,0))
fin_parcels <- sum_by_hex(as.integer(known & p$has_financialized_owner %in% TRUE))
coverage <- observed_units>=20 & observed_units/x$residential_units>=.95 & observed_parcels/parcels>=.95
coverage[is.na(coverage)]<-FALSE
eq(x$ownership_current_usable,coverage);eq(x$ownership_observed_units,observed_units)
eq(x$ownership_observed_parcels,observed_parcels)
own_values <- list(pct_corporate_units=100*corp_units/observed_units,
  corporate_owned_units_per_km2=corp_units/x$area_km2,
  pct_financialized_owner_parcels=100*fin_parcels/observed_parcels)
for(n in names(own_values)){z<-own_values[[n]];z[!coverage]<-NA_real_;eq(x[[n]],z)}

for(spec in list(c("311/311","sr_311_smoke_signal_latest_12mo","sr_311_smoke_signal_previous_12mo","sr_311_smoke_signal_latest_12mo_rate_change_per_100_units"),
                c("evictions/eviction","eviction_cases_latest_12mo","eviction_cases_previous_12mo","eviction_latest_12mo_rate_change_per_100_units"))){
  s<-readRDS(paste0("output/part2/",spec[1],"_features_paired.rds"));s<-s[s$analysis_as_of_date==as.Date("2026-04-01"),];s<-s[match(x$hex_id,s$hex_id),]
  eq(x[[spec[2]]],s[[spec[2]]]);eq(x[[spec[3]]],s[[spec[3]]])
  difference<-100*(s[[spec[2]]]-s[[spec[3]]])/x$residential_units
  difference[!is.finite(difference)|x$residential_units<20]<-NA_real_
  eq(x[[spec[4]]],difference)
}

# Fresh current-reference clipping and complete fixed weights, independently.
normalize <- function(z,signed=FALSE){
  b<-if(signed){v<-as.numeric(quantile(abs(z),.99,na.rm=TRUE,type=7));c(-v,v)}else as.numeric(quantile(z,c(.01,.99),na.rm=TRUE,type=7))
  if(b[1]==b[2])return(ifelse(is.na(z),NA_real_,if(signed)50 else 0))
  100*(pmin(pmax(z,b[1]),b[2])-b[1])/(b[2]-b[1])
}
specs <- list(rent_pressure_citywide_index=c("acs_rent_current_real","acs_rent_growth_recent_annualized_pct","acs_rent_acceleration_pp"),
  demographic_vulnerability_index=c("median_income_real","pct_renter","poverty_rate","pct_rent_burden_30plus","pct_college"),
  demolition_pressure_index=c("demo_recent_density","demo_trend_positive","demo_total_recent_density"),
  eviction_pressure_index=c("eviction_latest_12mo_per_100_units","eviction_latest_12mo_rate_change_per_100_units"),
  sr_311_pressure_index=c("sr_311_smoke_signal_latest_12mo_per_100_units","sr_311_smoke_signal_latest_12mo_density","sr_311_smoke_signal_latest_12mo_rate_change_per_100_units"),
  ownership_pressure_index=names(own_values))
for(index in names(specs)){
  columns<-specs[[index]]
  scores<-sapply(columns,function(n){z<-x[[n]];if(n %in% c("median_income_real","pct_college"))z<- -z
    normalize(z,grepl("rate_change_per_100_units$",n))})
  eq(x[[index]],rowMeans(scores));eq(x[[paste0(index,"_terms_available")]],rowSums(is.finite(scores)))
}
amenity_scores<-sapply(c("cafe","full_service_restaurant","drinking_place"),function(g)
  (normalize(x[[paste0(g,"_recent")]])+normalize(pmax(x[[paste0(g,"_recent")]]-x[[paste0(g,"_previous")]],0)))/2)
eq(x$amenity_change_index,rowMeans(amenity_scores))
indices<-c(names(specs),"amenity_change_index")
complete<-rowSums(is.finite(as.matrix(x[indices])))==7
gates<-x$in_current_city_scope & x$minimum_unit_support & coverage & x$sr_311_poc_coverage_usable &
  x$demolition_comparison_ready & x$eviction_count_observed & x$amenity_retrospective_usable & complete
eq(x$primary_cluster_eligible,gates);stopifnot(!anyNA(gates),sum(gates)==manifest$eligible_cells)
paired<-readRDS("output/part2/matrix/part2_eligibility_by_hex.rds")
stopifnot(all(paired$hex_id[paired$common_comparison_ready] %in% x$hex_id[gates]),
  all(x$all_required_components_available[gates]))
cat("Independent current measurement audit passed:",sum(gates),"eligible cells; current-only source support, all seven recipes, raw events, canonical promotion and source hashes verified.\n")
