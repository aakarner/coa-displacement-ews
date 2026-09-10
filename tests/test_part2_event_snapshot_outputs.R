# Read-only cross-stream audit after the two standalone event builds.
suppressPackageStartupMessages({library(dplyr); library(sf)})
source("R/part2_event_scoring.R")
source("R/utils.R")
eq <- function(x, y) stopifnot(isTRUE(all.equal(x, y, check.attributes = FALSE, tolerance = 1e-9)))
grid <- readRDS("output/hex_grid.rds")
ids <- grid$hex_id
cutoffs <- as.Date(c("2025-04-01", "2026-04-01"))
hash_checks <- 0L
for (domain in c("311", "demolitions")) {
  prefix <- if (domain == "311") "311" else "demolition"
  root <- file.path("output/part2", domain)
  manifest <- jsonlite::read_json(file.path(root, paste0(prefix, "_run_manifest.json")), simplifyVector = FALSE)
  stopifnot(identical(manifest$status, paste0("paired_", prefix, "_features_complete_v2")))
  for (entry in c(manifest$inputs, manifest$outputs)) {
    stopifnot(file.exists(entry$path), identical(digest::digest(file = entry$path, algo = "sha256"), entry$sha256))
    hash_checks <- hash_checks + 1L
  }
  index <- if (domain == "311") "sr_311_pressure_index" else "demolition_pressure_index"
  components <- part2_event_components(index)
  paired <- readRDS(file.path(root, paste0(prefix, "_features_paired.rds")))
  stopifnot(nrow(paired) == 2L * nrow(grid), identical(typeof(paired$hex_id), typeof(ids)),
    all(c("hex_id", "analysis_as_of_date", index, components,
          paste0(index, "_components_available")) %in% names(paired)),
    !anyDuplicated(paste(paired$hex_id, paired$analysis_as_of_date)),
    setequal(unique(paired$analysis_as_of_date), cutoffs))
  sets <- lapply(cutoffs, function(date) paired[paired$analysis_as_of_date == date, ])
  reference_path <- if (domain == "311") file.path(root, "311_scaling.rds") else
    file.path(root, as.character(cutoffs[1]), "demolition_scaling.rds")
  reference <- readRDS(reference_path)
  eq(reference, part2_fit_event_scaling(sets[[1]], components, index, cutoffs[1], preserved_scaling = reference))
  city_column <- if (domain == "311") "sr_311_in_current_city_scope" else "hex_center_inside_current_austin_full"
  usable_column <- if (domain == "311") "sr_311_poc_coverage_usable" else "demolition_comparison_ready"
  for (i in 1:2) {
    x <- sets[[i]]
    stopifnot(identical(x$hex_id, ids), all(c(city_column, usable_column, "area_km2") %in% names(x)),
      is.logical(x[[usable_column]]), !anyNA(x[[usable_column]]),
      sum(x[[city_column]]) == 6060L, all(!x[[usable_column]] | x[[city_column]]))
    eq(x$area_km2, grid$area_km2)
    scored <- part2_apply_event_scaling(x, reference)$features
    eq(x[c(index, paste0(components, "_score"), paste0(index, "_components_available"))],
       scored[c(index, paste0(components, "_score"), paste0(index, "_components_available"))])
    stopifnot(all(is.na(as.matrix(x[!x[[usable_column]], c(components, index)]))),
      identical(is.finite(x[[index]]), x[[paste0(index, "_components_complete")]]),
      all(x[[paste0(index, "_components_required")]] == length(components)),
      all(x[[index]] >= 0, na.rm = TRUE), all(x[[index]] <= 100, na.rm = TRUE))
    # Independently reproduce the earlier robust scoring, not just helper reuse.
    if (i == 1L) {
      scores <- lapply(components, function(component) {
        b <- reference$bounds[match(component, reference$bounds$component), ]; value <- x[[component]]
        if(b$degenerate_range) ifelse(is.na(value), NA_real_, if(grepl("rate_change_per_100_units$", component)) 50 else 0) else
          100*(pmax(pmin(value,b$upper_bound),b$lower_bound)-b$lower_bound)/(b$upper_bound-b$lower_bound)
      })
      expected <- rowMeans(as.data.frame(scores), na.rm = FALSE)
      eq(x[[index]], expected)
    }
    eq(x, readRDS(file.path(root, as.character(cutoffs[i]), paste0(prefix, "_features_by_hex.rds"))))
  }
  eq(sets[[1]][[city_column]], sets[[2]][[city_column]])
  changes <- read.csv(file.path(root, paste0(prefix, "_feature_changes_by_hex.csv")))
  eq(changes$hex_id, ids)
  eq(changes[[paste0("delta_", index)]], sets[[2]][[index]] - sets[[1]][[index]])
  if (domain == "311") {
    sr_sets <- sets
    units <- st_drop_geometry(readRDS("output/corporate_ownership_by_hex.rds"))
    units <- units$residential_units[match(ids, units$hex_id)]
    ledger <- readRDS(file.path(root, "311_event_ledger.rds"))
    stopifnot(!anyDuplicated(ledger$sr_number), !anyNA(ledger$sr_number),
              identical(manifest$all_requests_coverage_verified, FALSE))
    for (i in 1:2) {
      x <- sets[[i]]
      eq(x$residential_units, units)
      eq(x$sr_311_rate_units_denominator, ifelse(is.finite(units) & units >= 20, units, NA_real_))
      eq(x$sr_311_smoke_signal_latest_12mo_per_100_units,
         100 * x$sr_311_smoke_signal_latest_12mo / x$sr_311_rate_units_denominator)
      eq(x$sr_311_smoke_signal_latest_12mo_density, x$sr_311_smoke_signal_latest_12mo / x$area_km2)
      previous <- x$sr_311_smoke_signal_previous_12mo
      recent <- x$sr_311_smoke_signal_latest_12mo
      eq(x$sr_311_smoke_signal_latest_12mo_change_pct,
         ifelse(!is.na(previous) & previous > 0, 100 * (recent / previous - 1), NA_real_))
      for (window in c("previous", "recent")) {
        start <- as.Date(if (window == "recent") c("2024-04-02", "2025-04-02")[i] else c("2023-04-02", "2024-04-02")[i])
        end <- as.Date(if (window == "recent") as.character(cutoffs[i]) else c("2024-04-01", "2025-04-01")[i])
        keep <- !is.na(ledger$hex_id) & ledger$event_inside_current_city %in% TRUE & ledger$event_date_valid %in% TRUE &
          ledger$sr_created_date >= start & ledger$sr_created_date <= end
        expected <- as.integer(table(factor(ledger$hex_id[which(keep)], levels = ids)))
        eq(x[[paste0("sr_311_", window, "_observed_count")]], expected)
      }
    }
    common <- sets[[1]]$sr_311_poc_coverage_usable & sets[[2]]$sr_311_poc_coverage_usable
    eq(sets[[1]]$sr_311_smoke_signal_latest_12mo[common], sets[[2]]$sr_311_smoke_signal_previous_12mo[common])
  } else {
    demo_sets <- sets
    events <- readRDS(file.path(root, "demolition_event_locations.rds"))
    stopifnot(!anyDuplicated(events$permit_id), !anyNA(events$permit_id))
    for (i in 1:2) {
      x <- sets[[i]]
      for (window in c("previous", "recent")) {
        start <- as.Date(if (window == "recent") c("2023-04-02", "2024-04-02")[i] else c("2021-04-02", "2022-04-02")[i])
        end <- as.Date(if (window == "recent") as.character(cutoffs[i]) else c("2023-04-01", "2024-04-01")[i])
        keep <- events$spatial_status == "mapped_current_city_study_hex" & events$issue_date >= start & events$issue_date <= end
        expected <- as.integer(table(factor(events$hex_id[which(keep)], levels = ids)))
        totals <- as.integer(table(factor(events$hex_id[which(keep & events$is_total_demolition)], levels = ids)))
        eq(x[[paste0("demo_", window, "_observed_permits")]], expected)
        eq(x[[paste0("demo_", window, "_observed_total_permits")]], totals)
      }
      usable <- x$demolition_comparison_ready
      eq(x$demo_recent_density, ifelse(usable, x$demo_latest_24mo / x$area_km2, NA_real_))
      eq(x$demo_total_recent_density, ifelse(usable, x$demo_total_latest_24mo / x$area_km2, NA_real_))
      eq(x$demo_trend_positive, ifelse(usable, pmax(log1p(x$demo_latest_24mo) - log1p(x$demo_previous_24mo), 0), NA_real_))
    }
    eq(sets[[1]][[city_column]], sr_sets[[1]]$sr_311_in_current_city_scope)
  }
}
preserved <- readRDS("output/part2/events/existing_outputs_before.rds")
# The user authorized overwriting provisional Part2 outputs; Part1/Part3 remain protected.
preserved <- preserved[!grepl("(^|/)output/part2/", preserved$path), ]
source("tests/current_preservation_policy.R")
preserved <- current_preservation_entries(preserved)
for (i in seq_len(nrow(preserved))) stopifnot(identical(
  digest::digest(file = preserved$path[i], algo = "sha256"), preserved$sha256[i]))
cat("Paired event audit passed:", hash_checks, "manifest SHA-256 checks;",
    nrow(preserved), "existing outputs unchanged;", length(ids), "hexes per date/stream.\n")
