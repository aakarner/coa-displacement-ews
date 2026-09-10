# Synthetic tests: no downloads or production writes.
suppressPackageStartupMessages({library(dplyr); library(sf)})
source("R/demolition_coverage_history.R")
source("R/part2_311.R")
source("R/part2_event_scoring.R")
fail <- function(expr) stopifnot(inherits(tryCatch({force(expr); NULL}, error = identity), "error"))
types <- data.frame(sr_type_desc = c("A", "B"))
polygon <- function(x1, x2, y1 = 30.2, y2 = 30.3) st_polygon(list(rbind(c(x1,y1),c(x2,y1),c(x2,y2),c(x1,y2),c(x1,y1))))
grid <- st_sf(hex_id = 1:4, geometry = st_sfc(polygon(-97.80,-97.78),polygon(-97.78,-97.76),
  polygon(-97.76,-97.74),polygon(-97.74,-97.72), crs=4326))
city <- st_sf(geometry = st_sfc(polygon(-97.80,-97.74),crs=4326))
dates <- c("2023-04-01", "2023-04-02", "2024-04-01", "2024-04-02", "2025-04-01", "2025-04-02")
raw <- data.frame(sr_number=as.character(1:6),sr_type_desc="A",sr_created_date=paste0(dates,"T00:00:00.000"),
  sr_location_lat=30.25,sr_location_long=-97.79)
cache <- list(schema_version=1L,complete=TRUE,start_date=as.Date("2020-01-01"),analysis_as_of_date=as.Date("2026-04-01"),
  selected_type_descriptions=c("A","B"),data=raw)
part2_311_validate_cache(cache,types)
bad<-cache;bad$complete<-FALSE;fail(part2_311_validate_cache(bad,types))
bad<-cache;bad$data$sr_type_desc[1]<-"C";fail(part2_311_validate_cache(bad,types))
prepared<-part2_311_prepare_events(cache,types,grid,city)
stopifnot(all(prepared$events$hex_id==1L),all(prepared$events$event_inside_current_city))
duplicate<-cache;duplicate$data<-rbind(raw,raw[1,]);dedup<-part2_311_prepare_events(duplicate,types,grid,city)
stopifnot(nrow(dedup$events)==6L,dedup$qa$exact_duplicate_rows_removed==1L)
duplicate$data$sr_location_long[7]<--97.75;fail(part2_311_prepare_events(duplicate,types,grid,city))
bad<-cache;bad$data$sr_location_lat[1]<-0;bad$data$sr_location_long[2]<--97.73
q<-part2_311_prepare_events(bad,types,grid,city)
stopifnot(!q$events$coordinate_valid[1],is.na(q$events$hex_id[1]),q$events$hex_id[2]==4L,
          !q$events$event_inside_current_city[2])
# Duplicate polygon geometry creates an ambiguity; no arbitrary hex assignment.
overlap<-grid;st_geometry(overlap)[2]<-st_geometry(grid)[1]
q<-part2_311_prepare_events(cache,types,overlap,city)
stopifnot(all(q$events$spatial_match_count==2L),all(is.na(q$events$hex_id)))
support<-data.frame(hex_id=1:4,area_km2=1,residential_units=c(20,19,40,50))
geo<-data.frame(hex_id=1:4,sr_311_in_current_city_scope=c(TRUE,TRUE,TRUE,FALSE),
  sr_311_geography_source_covered=c(TRUE,TRUE,NA,TRUE))
w<-part2_311_windows("2025-04-01")
stopifnot(w$latest_12mo_start==as.Date("2024-04-02"),w$previous_12mo_start==as.Date("2023-04-02"))
x<-part2_311_snapshot(prepared$events,support,geo,as.Date("2025-04-01"),cache$start_date,cache$analysis_as_of_date)
f<-x$features
stopifnot(identical(x$membership$event_window,c("outside","previous","previous","recent","recent","outside")),
  f$sr_311_smoke_signal_latest_12mo[1]==2L,f$sr_311_smoke_signal_previous_12mo[1]==2L,
  f$sr_311_smoke_signal_latest_12mo_per_100_units[1]==10,
  f$sr_311_smoke_signal_latest_12mo_rate_change_per_100_units[1]==0,
  f$sr_311_smoke_signal_latest_12mo_density[1]==2,
  f$sr_311_smoke_signal_latest_12mo_change_pct[1]==0,
  f$sr_311_valid_zero_latest[2],is.na(f$sr_311_smoke_signal_latest_12mo_per_100_units[2]),
  is.na(f$sr_311_smoke_signal_latest_12mo_change_pct[2]),
  is.na(f$sr_311_smoke_signal_latest_12mo_rate_change_per_100_units[2]),
  all(is.na(f$sr_311_smoke_signal_latest_12mo[3:4])),all(!f$sr_311_valid_zero_latest[3:4]))
short<-part2_311_snapshot(prepared$events,support,geo,as.Date("2025-04-01"),as.Date("2024-01-01"),cache$analysis_as_of_date)
stopifnot(all(is.na(short$features$sr_311_smoke_signal_latest_12mo)))
ambiguous<-part2_311_snapshot(q$events,support,geo,as.Date("2025-04-01"),cache$start_date,cache$analysis_as_of_date)
stopifnot(all(ambiguous$features$sr_311_ambiguous_assignment_affects_window[1:2]),
          all(!ambiguous$features$sr_311_poc_coverage_usable[1:2]))
# FULL-only historical screen rejects LTD and partial-period annexation, without
# invoking the demolition builder's broader supported-authority list.
baseline<-st_sf(OBJECTID=1L,JURISDICTION_TYPE="FULL",JURISDICTION_DATE="2020-01-01",geometry=st_geometry(city))
actions<-st_sf(OBJECTID=1L,JURISDICTION_CASE_NUMBER="case",ORDINANCE_NUMBER="ord",JURISDICTION_DESCRIPTION="test",
  EFFECTIVE_DATE=as.Date("2024-07-01"),JURISDICTION_TYPE="LTD",GLOBALID="id",geometry=st_geometry(grid)[1])
coverage<-part2_311_geography(grid,city,baseline,actions,as.Date("2025-04-01"))
stopifnot(!coverage$sr_311_geography_source_covered[1],coverage$sr_311_geography_source_covered[2],
          is.na(coverage$sr_311_geography_source_covered[4]))
ref<-part2_fit_event_scaling(f,part2_311_components(),"sr_311_pressure_index",as.Date("2025-04-01"))
scored<-part2_apply_event_scaling(f,ref)$features
stopifnot(all(is.na(scored$sr_311_pressure_index[2:4])))
# Zero-safe absolute differences: 0->0, 0->1, 1->0 all remain observed.
for (counts in list(c(0L,0L),c(0L,1L),c(1L,0L))) {
  events <- prepared$events[0, ]
  if (counts[1]) events <- rbind(events, prepared$events[2, ])
  if (counts[2]) events <- rbind(events, prepared$events[4, ])
  z <- part2_311_snapshot(events,support,geo,as.Date("2025-04-01"),cache$start_date,cache$analysis_as_of_date)$features
  stopifnot(z$sr_311_smoke_signal_latest_12mo_rate_change_per_100_units[1] == 5*(counts[2]-counts[1]))
}
cat("Part 2 311 synthetic tests passed.\n")
