# One-off decision audit: compare compact pre-change masks with current outputs.
# Usage: Rscript scripts/audits/eviction_window_change.R /path/to/before.rds
# The temporary input contains only masks/counts, not an archived model run.
suppressPackageStartupMessages({library(dplyr); library(readr)})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L,file.exists(args[1]))
b <- readRDS(args[1])
p <- readRDS("output/part2/evictions/eviction_features_paired.rds")
x <- readRDS("output/part1/measurement/current_measurement.rds")
e <- readRDS("output/part2/matrix/part2_eligibility_by_hex.rds")
stopifnot(identical(b$evictions$hex_id,p$hex_id),
  identical(b$evictions$analysis_as_of_date,p$analysis_as_of_date),
  identical(b$evictions$eviction_source_covered,p$eviction_source_covered),
  identical(b$part1$hex_id,x$hex_id),identical(b$part2$hex_id,e$hex_id),
  identical(b$evictions$eviction_recent_observed_cases,p$eviction_recent_observed_cases),
  identical(b$evictions$eviction_previous_observed_cases,p$eviction_previous_observed_cases),
  !any(b$evictions$eviction_count_observed & !p$eviction_count_observed),
  !any(b$part1$primary_cluster_eligible & !x$primary_cluster_eligible),
  !any(b$part2$common_comparison_ready & !e$common_comparison_ready))
events <- data.frame(hex_id=p$hex_id,analysis_as_of_date=p$analysis_as_of_date,
  source_county=p$source_county,window_start=p$eviction_eligibility_window_start,
  old_source_covered=b$evictions$eviction_source_covered,new_source_covered=p$eviction_source_covered,
  old_ambiguity_cases=b$evictions$eviction_unresolved_candidate_cases,
  new_ambiguity_cases=p$eviction_unresolved_candidate_cases,
  old_counts_usable=b$evictions$eviction_count_observed,new_counts_usable=p$eviction_count_observed,
  old_score_usable=is.finite(b$evictions$eviction_pressure_index),new_score_usable=is.finite(p$eviction_pressure_index),
  recent_mapped_cases=p$eviction_recent_observed_cases,previous_mapped_cases=p$eviction_previous_observed_cases,
  raw_window_counts_unchanged=TRUE)
cells <- data.frame(hex_id=x$hex_id,source_county=x$source_county,
  part1_before=b$part1$primary_cluster_eligible,part1_after=x$primary_cluster_eligible,
  part1_exclusion_before=b$part1$primary_exclusion,part1_exclusion_after=x$primary_exclusion,
  part2_before=b$part2$common_comparison_ready,part2_after=e$common_comparison_ready)
row <- function(measure,before,after,date=NA_character_) data.frame(measure=measure,date=date,
  before=sum(before),after=sum(after),regained=sum(!before & after),lost=sum(before & !after))
summary <- bind_rows(lapply(unique(events$analysis_as_of_date),function(date){
  y<-events[events$analysis_as_of_date==date,]
  bind_rows(row("Usable mapped eviction counts",y$old_counts_usable,y$new_counts_usable,as.character(date)),
    row("Usable eviction index",y$old_score_usable,y$new_score_usable,as.character(date)))
}),row("Part 1 complete current sample",cells$part1_before,cells$part1_after,"2026-04-01"),
  row("Part 2 complete paired sample",cells$part2_before,cells$part2_after,"2025/2026"))
root<-"output/part1"
write_csv(events,file.path(root,"eviction_window_change_events.csv"))
write_csv(cells,file.path(root,"eviction_window_change_eligibility.csv"))
write_csv(summary,file.path(root,"eviction_window_change_summary.csv"))
print(summary)
cat("Window-change audit passed: raw counts unchanged; no eligibility losses.\n")
