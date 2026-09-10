# Synthetic unit tests; no private source data, downloads, or output mutations.
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(sf)})
source("R/demolition_panel.R")
source("R/demolition_coverage_history.R")
source("R/part2_demolitions.R")
source("R/part2_event_scoring.R")
expect_equal <- function(actual, expected) stopifnot(isTRUE(all.equal(actual, expected, check.attributes = FALSE)))
expect_error <- function(expr, pattern) {
  error <- tryCatch(force(expr), error = identity)
  stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error)))
}

windows <- part2_demolition_windows("2025-04-01")
expect_equal(windows$window_start, as.Date(c("2021-04-02", "2023-04-02")))
expect_equal(windows$window_end, as.Date(c("2023-04-01", "2025-04-01")))
expect_equal(part2_demolition_windows("2026-04-01")$window_start,
             as.Date(c("2022-04-02", "2024-04-02")))
expect_error(part2_demolition_windows(NA), "one cutoff")

raw <- tibble::tibble(`Permit Num` = paste0("p", 1:6), `Permit Class Mapped` = "Residential",
  `Work Class` = "Demolition", `Issued Date` = c("2022-01-01", "2024-01-01", "2024-01-02", "2024-06-01", "2027-01-01", "2024-01-03"),
  Latitude = 30.205, Longitude = -97.795, Jurisdiction = "FULL",
  Description = c("Total demolition of home", "total DEMO", "Partial demo", "total demo", "total demo", NA))
raw$`Permit Class Mapped`[4] <- "Commercial"
prepared <- part2_prepare_demolitions(raw, as.Date("2026-04-01"))
expect_equal(prepared$events$permit_id, c("P1", "P2", "P3", "P6"))
expect_equal(prepared$events$is_total_demolition, c(TRUE, TRUE, FALSE, FALSE))
stopifnot(nrow(prepared$classification_disagreements) == 0L)
duplicate <- part2_prepare_demolitions(bind_rows(raw, raw[1, ]), as.Date("2026-04-01"))
expect_equal(duplicate$events$permit_id, prepared$events$permit_id)
stopifnot(sum(duplicate$source_audit$source_disposition == "duplicate_permit_row") == 1L)
bad <- raw[1, ]; bad$`Issued Date` <- "2023-01-01"
expect_error(part2_prepare_demolitions(bind_rows(raw, bad), as.Date("2026-04-01")), "Conflicting duplicate")
bad <- raw[1, ]; bad$Description <- "Partial demo"
expect_error(part2_prepare_demolitions(bind_rows(raw, bad), as.Date("2026-04-01")), "Conflicting duplicate")
changed_taxonomy <- raw; changed_taxonomy$`Permit Class Mapped`[4] <- "Nonresidential"
stopifnot(nrow(part2_prepare_demolitions(changed_taxonomy, as.Date("2026-04-01"))$classification_disagreements) == 1L)

rectangle <- function(x0, x1, y0 = 30.2, y1 = 30.21) {
  st_polygon(list(matrix(c(x0,y0,x1,y0,x1,y1,x0,y1,x0,y0), ncol = 2, byrow = TRUE)))
}
grid <- st_sf(hex_id = 1:3, area_km2 = c(1, 2, 1), geometry = st_sfc(
  rectangle(-97.8, -97.79), rectangle(-97.79, -97.78), rectangle(-97.78, -97.77), crs = 4326))
boundary <- st_sf(city_name = "CITY OF AUSTIN", jurisdiction_type = "FULL",
  geometry = st_sfc(rectangle(-97.801, -97.783, 30.199, 30.211), crs = 4326))
city <- part2_demolition_city_reference(grid, boundary)
expect_equal(city$reference$hex_center_inside_current_austin_full, c(TRUE, TRUE, FALSE))
expect_equal(city$reference$hex_straddles_current_austin_full, c(FALSE, TRUE, FALSE))

# A cell that is FULL at cutoff but ETJ for any part of its window is unavailable.
baseline <- st_sf(OBJECTID = 1:2, JURISDICTION_TYPE = c("FULL", "ETJ"), JURISDICTION_DATE = "1960-01-01",
  geometry = st_geometry(grid)[1:2])
actions <- st_sf(OBJECTID = 1L, JURISDICTION_CASE_NUMBER = "fixture", ORDINANCE_NUMBER = "fixture",
  JURISDICTION_DESCRIPTION = "Fixture annexation", EFFECTIVE_DATE = "2024-01-01",
  JURISDICTION_TYPE = "FULL", GLOBALID = "fixture", geometry = st_geometry(grid)[2])
context <- part2_demolition_coverage_context(grid, baseline, actions)
coverage <- part2_demolition_window_coverage(grid, context, windows, city$reference)
expect_equal(coverage$window_usable[coverage$event_window == "recent"], c(TRUE, FALSE, FALSE))
partial <- coverage %>% filter(hex_id == "2", event_window == "recent")
stopifnot(isTRUE(partial$end_state_source_covered), !partial$window_usable,
          partial$period_effective_change_date_count == 1L,
          partial$period_first_uncovered_date == as.Date("2023-04-02"))
stopifnot(all(is.na(coverage$historical_source_covered[coverage$hex_id == "3"])))
later <- part2_demolition_window_coverage(grid, context, part2_demolition_windows("2026-04-01"), city$reference)
stopifnot(later$window_usable[later$hex_id == "2" & later$event_window == "recent"])
short_source <- part2_demolition_window_coverage(grid, context, windows, city$reference,
  source_start = as.Date("2022-01-01"))
stopifnot(!any(short_source$window_usable[short_source$event_window == "previous"]))

# Exact City point masking excludes a point outside City even in a selected
# straddling hex. Missing coordinates remain unknown, not known outsiders.
spatial_events <- prepared$events
spatial_events$longitude <- c(-97.795, -97.786, -97.782, NA)
spatial_events$valid_coordinates <- c(TRUE, TRUE, TRUE, FALSE)
assigned_spatial <- part2_assign_demolitions(spatial_events, grid, city$reference, city$boundary)
expect_equal(assigned_spatial$events$spatial_status,
  c("mapped_current_city_study_hex", "mapped_current_city_study_hex", "outside_current_austin_full_purpose", "invalid_coordinates"))
stopifnot(is.na(assigned_spatial$events$point_inside_current_austin_full[4]),
          assigned_spatial$events$hex_id[3] == "2")
overlap <- rbind(grid[1, ], grid[1, ]); overlap$hex_id <- c(1L, 4L)
overlap_city <- part2_demolition_city_reference(overlap, boundary)
overlap_assigned <- part2_assign_demolitions(spatial_events[1, ], overlap,
  overlap_city$reference, overlap_city$boundary)
stopifnot(overlap_assigned$events$spatial_status == "multiple_hex_matches",
          is.na(overlap_assigned$events$hex_id),
          setequal(overlap_assigned$ambiguous_hex_links$hex_id, c("1", "4")))

# Counts use inclusive exact endpoints, covered zero differs from unavailable,
# and the positive trend is a log-count difference, not percentage change.
events <- tibble::tibble(permit_id = paste0("event", 1:7), hex_id = "1",
  issue_date = as.Date(c("2021-04-01", "2021-04-02", "2023-04-01", "2023-04-02", "2025-04-01", "2025-04-02", "2024-02-01")),
  spatial_status = "mapped_current_city_study_hex", is_total_demolition = c(TRUE, TRUE, FALSE, TRUE, FALSE, TRUE, TRUE))
assigned <- list(events = events,
  ambiguous_hex_links = tibble::tibble(permit_id = character(), hex_id = character(), issue_date = as.Date(character())))
features <- part2_demolition_features(grid, assigned, coverage, windows)
expect_equal(features$demo_latest_24mo, c(3L, NA, NA))
expect_equal(features$demo_previous_24mo, c(2L, NA, NA))
expect_equal(features$demo_recent_density, c(3, NA, NA))
expect_equal(features$demo_trend_positive, c(log1p(3) - log1p(2), NA, NA))
expect_equal(features$demo_total_recent_density, c(2, NA, NA))
expect_equal(features$demolition_comparison_ready, c(TRUE, FALSE, FALSE))
empty <- assigned; empty$events <- empty$events[FALSE, ]
zero <- part2_demolition_features(grid, empty, coverage, windows)
expect_equal(zero$demo_latest_24mo, c(0L, NA, NA))
expect_equal(zero$demo_recent_density, c(0, NA, NA))
ambiguous <- assigned
ambiguous$ambiguous_hex_links <- tibble::tibble(permit_id = "uncertain", hex_id = "1", issue_date = as.Date("2024-06-01"))
masked <- part2_demolition_features(grid, ambiguous, coverage, windows)
stopifnot(all(is.na(masked$demo_recent_density)), all(is.na(masked$demo_trend_positive)),
          all(is.na(masked$demo_total_recent_density)))
scaling <- part2_fit_event_scaling(features, part2_event_components("demolition_pressure_index"),
  "demolition_pressure_index", as.Date("2025-04-01"))
scored <- part2_apply_event_scaling(features, scaling)$features
stopifnot(is.finite(scored$demolition_pressure_index[1]),
          all(is.na(scored$demolition_pressure_index[2:3])))
cat("Part 2 demolition window, source, spatial, coverage, and scoring tests passed.\n")
