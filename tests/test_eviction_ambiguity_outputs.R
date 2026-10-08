source("R/grid_contract.R")
# Independent checks of the annual mapped-filing proxy and its forward labels.
suppressPackageStartupMessages({library(dplyr); library(readr)})
p <- read_csv("output/eviction_filings_complete_by_hex_year.csv",show_col_types=FALSE)
stopifnot(!anyDuplicated(p[c("hex_id","outcome_year")]),
  nrow(p)==grid_contract()$grid_cells*length(unique(p$outcome_year)),
  all(p$eviction_ambiguity_rule=="flag_unassigned_cases_keep_cells_v1"),
  all(p$measurement_complete),!any(p$all_filing_locations_complete),
  identical(p$has_unassigned_ambiguous_cases,p$unresolved_candidate_cases>0L),
  identical(p$count_observed,p$source_covered & p$period_complete),
  identical(is.na(p$eviction_cases),!p$count_observed),
  all(is.na(p$eviction_cases[p$outcome_year==2026L])),
  all(p$eviction_cases[p$count_observed]==p$eviction_cases_observed_to_date[p$count_observed]))
qa <- read_csv("output/part3/eviction_complete_panel_qa.csv",show_col_types=FALSE)
stopifnot(all(qa$assigned_count_reconciles),
  all(qa$panel_eviction_cases[qa$period_complete]==qa$assigned_inside_coverage_cases[qa$period_complete]))
l <- readRDS("output/part3/eviction_demolition_forecast_labels_long.rds")
e <- filter(l,outcome_id=="eviction_filings")
# Explicit year expansion and count sums, independent of label-building helpers.
years <- bind_rows(lapply(seq_len(max(e$horizon_years)),function(offset) {
  z <- e[e$horizon_years>=offset,c("hex_id","forecast_origin_year","horizon_years")]
  z$outcome_year <- z$forecast_origin_year+offset; z
})) %>% left_join(select(p,hex_id,outcome_year,count_observed,eviction_cases),by=c("hex_id","outcome_year")) %>%
  group_by(hex_id,forecast_origin_year,horizon_years) %>% summarise(
    observed=all(count_observed %in% TRUE),expected=sum(eviction_cases),.groups="drop")
k <- function(x) paste(x$hex_id,x$forecast_origin_year,x$horizon_years)
years <- years[match(k(e),k(years)),]
stopifnot(identical(e$label_observed,years$observed))
# The value column is checked explicitly below to preserve the label contract.
stopifnot(identical(is.na(e$outcome_count),!years$observed),
  all(e$outcome_count[years$observed]==years$expected[years$observed]))
cat("Annual ambiguity policy and forward-label count audit passed.\n")
