# Compare the preserved pre-repair baseline, inventory-only counterfactual and
# completed repair. All detailed evidence stays in ignored local output.
suppressPackageStartupMessages({library(dplyr);library(readr);library(sf)})
root <- "output/residential_geography_repair"
before <- file.path(root,"before")
read_before <- function(path) readRDS(file.path(before,path))
stages <- list(before=read_before("output/part1/measurement/current_measurement.rds"),
  inventory_only=readRDS(file.path(root,"inventory_only/output/part1/measurement/current_measurement.rds")),
  repaired=readRDS("output/part1/measurement/current_measurement.rds"))
summary <- bind_rows(lapply(names(stages),function(stage) {
  x<-stages[[stage]]; eligible<-x$primary_cluster_eligible
  supported<-x$eviction_source_covered
  data.frame(stage,eligible_cells=sum(eligible),residential_units=sum(x$residential_units),
    eligible_filings=sum(x$eviction_recent_observed_cases[eligible]),
    covered_filings=sum(x$eviction_recent_observed_cases[supported]),
    low_unit_filing_cells=sum(supported & x$residential_units<20 & x$eviction_recent_observed_cases>0),
    low_unit_filings=sum(x$eviction_recent_observed_cases[supported & x$residential_units<20]))
}))
write_csv(summary,file.path(root,"stage_summary.csv"));print(summary)
old<-read_before("output/part2/evictions/eviction_case_ledger.rds")
new<-readRDS("output/part2/evictions/eviction_case_ledger.rds")
old<-old[match(new$case_number,old$case_number),]
stopifnot(identical(new$case_number,old$case_number),identical(new$assignment_status,old$assignment_status),
  identical(new$original_assigned_hex_key,old$assigned_hex_key))
recent<-new%>%filter(file_date>=as.Date("2025-04-02"),file_date<=as.Date("2026-04-01"))
write_csv(count(recent,source_county,property_assignment_status,name="filings"),file.path(root,"recent_assignment_summary.csv"))
u<-stages$repaired%>%select(hex_id,residential_units)
audited<-readRDS("output/eviction_low_unit_audit/evidence.rds")
cases<-recent%>%filter(case_number%in%audited$ledger$case_number)%>%
  mutate(hex_id=as.integer(assigned_hex_key))%>%left_join(u,by="hex_id")
stopifnot(nrow(cases)==nrow(audited$ledger))
write_csv(cases,file.path(root,"audited_case_outcomes.csv"))
write_csv(cases%>%count(property_assignment_status,supported=residential_units>=20,name="filings"),
  file.path(root,"audited_case_summary.csv"))
remaining<-cases%>%filter(residential_units<20)%>%
  left_join(readRDS("output/eviction_low_unit_audit/summary.rds")$cases%>%select(case_number,category),by="case_number")
write_csv(remaining%>%count(hex_id,category,name="filings"),file.path(root,"remaining_audited_cells.csv"))
cells<-audited$targets%>%select(hex_id,before_units=residential_units,before_filings=eviction_recent_observed_cases)%>%
  left_join(stages$repaired%>%select(hex_id,after_units=residential_units,after_filings=eviction_recent_observed_cases,
    primary_cluster_eligible),by="hex_id")
write_csv(cells,file.path(root,"all_41_cells_after.csv"))
p0<-read_before("output/residential_parcels_unit_promoted.rds")
p1<-readRDS("output/residential_parcels_unit_promoted.rds")
recovered<-p1%>%filter(!parcel_id%in%p0$parcel_id)%>%select(parcel_id,situs_address,lon,lat,
  units_calibrated_targeted,unit_model_selection_method,unit_land_use_validation_excluded)
write_csv(recovered,file.path(root,"recovered_accounts.csv"))
# Hold the pre-repair classifier fixed to isolate input changes from refitting.
m<-read_before("output/part1/baseline_cluster_model.rds")
fixed<-lapply(stages,function(x) {
  good<-x$primary_cluster_eligible; z<-scale(as.matrix(x[good,m$features]),
    center=m$preprocessing$center,scale=m$preprocessing$scale)
  d<-sapply(seq_len(m$k),function(k)rowSums(sweep(z,2,m$centroids[k,],"-")^2))
  data.frame(hex_id=x$hex_id[good],cluster=max.col(-d,ties.method="first"))
})
fixed_summary<-bind_rows(lapply(c("inventory_only","repaired"),function(stage) {
  j<-inner_join(fixed$before,fixed[[stage]],by="hex_id",suffix=c("_before","_after"))
  data.frame(stage,common_cells=nrow(j),changed_fixed_classifier=sum(j$cluster_before!=j$cluster_after))
}))
write_csv(fixed_summary,file.path(root,"fixed_classifier_changes.csv"));print(fixed_summary)
old_assign<-read_csv(file.path(before,"output/part1/baseline_cluster_assignments.csv"),show_col_types=FALSE)
new_assign<-read_csv("output/part1/baseline_cluster_assignments.csv",show_col_types=FALSE)
j<-inner_join(old_assign%>%select(hex_id,before=tentative_name),
  new_assign%>%select(hex_id,after=tentative_name),by="hex_id")
write_csv(count(j,before,after,name="cells"),file.path(root,"refitted_profile_transitions.csv"))
write_csv(data.frame(common_cells=nrow(j),same_profile=sum(j$before==j$after),
  changed_profile=sum(j$before!=j$after),new_cells=sum(!new_assign$hex_id%in%old_assign$hex_id),
  lost_cells=sum(!old_assign$hex_id%in%new_assign$hex_id)),file.path(root,"refitted_change_summary.csv"))
