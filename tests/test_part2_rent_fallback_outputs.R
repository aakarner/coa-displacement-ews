# Independent read-only validation of PROMOTED corrected rent, not the obsolete
# sensitivity audit. No production fallback helper or runner is sourced here.
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(sf)})
root <- "output/part2/acs"
eq <- function(x, y) stopifnot(isTRUE(all.equal(x, y, check.attributes = FALSE, tolerance = 1e-9)))
grid <- readRDS("output/hex_grid.rds"); ids <- grid$hex_id
dates <- as.Date(c("2025-04-01", "2026-04-01"))
years <- c(2013L, 2014L, 2018L, 2019L, 2023L, 2024L)
geographies <- c("block_group", "tract")
cpi <- c(`2013` = 232.957, `2014` = 236.736, `2018` = 251.107,
  `2019` = 255.657, `2023` = 304.702, `2024` = 313.689)
manifest <- jsonlite::read_json(file.path(root, "acs_run_manifest.json"), simplifyVector = TRUE)
stopifnot(identical(manifest$status, "paired_acs_features_complete_v2"),
  manifest$relative_moe_limit == .30, manifest$dollar_base_year == 2024L,
  manifest$raw_caches_and_crosswalks_unchanged, manifest$part1_outputs_unchanged,
  manifest$required_components$rent == 3L, manifest$required_components$vulnerability == 5L)
eq(unlist(manifest$cpi_u)[names(cpi)], cpi)
for (entries in list(manifest$inputs, manifest$outputs)) {
  actual <- vapply(entries$path, digest::digest, character(1), file = TRUE, algo = "sha256")
  stopifnot(identical(unname(actual), entries$sha256))
}
candidates <- readRDS(file.path(root, "acs_rent_source_candidates.rds"))
selection <- readRDS(file.path(root, "acs_rent_source_selection.rds"))
series <- readRDS(file.path(root, "acs_rent_fixed_series_features.rds"))
paired <- readRDS(file.path(root, "acs_features_paired.rds"))
scaling <- readRDS(file.path(root, "acs_scaling.rds"))
stopifnot(identical(digest::digest(scaling$bounds, algo = "sha256"),
  "045e6d84ddc67880dee5fe7528c2279cdbde35adfc1f33d1866c539c7c96df94"))
key <- function(x) paste(x$hex_id, x$acs_year, x$source_geography, sep = ":")
original <- bind_rows(lapply(dates, function(date) read_csv(file.path(root, as.character(date),
  "acs_rent_dominant_sources_by_hex_vintage.csv"), show_col_types = FALSE,
  col_types = cols(acs_year = col_integer(), source_geography = col_character(), hex_id = col_integer(),
    dominant_source_geoid = col_character(), dominant_source_share = col_double(), dominant_source_method = col_character()))))
original$estimate <- NA_real_; original$moe <- NA_real_
for (year in years) for (geography in geographies) {
  raw <- st_drop_geometry(readRDS(file.path("data/raw_acs", paste0("acs_", year, "_acs5_", geography, "_median_rent.rds"))))
  stopifnot(!anyDuplicated(raw$GEOID), all(raw$variable == "median_rent"))
  rows <- which(original$acs_year == year & original$source_geography == geography)
  match_row <- match(original$dominant_source_geoid[rows], raw$GEOID)
  stopifnot(!anyNA(match_row))
  original$estimate[rows] <- raw$estimate[match_row]; original$moe[rows] <- raw$moe[match_row]
}
original$relative_moe <- ifelse(is.finite(original$estimate) & original$estimate > 0 &
  is.finite(original$moe) & original$moe >= 0, original$moe / original$estimate, NA_real_)
original$reliable <- is.finite(original$relative_moe) & original$relative_moe <= .30
stopifnot(nrow(candidates) == 12L * length(ids), !anyDuplicated(key(candidates)),
  !anyDuplicated(key(original)), setequal(key(candidates), key(original)))
c <- candidates[match(key(original), key(candidates)), ]
for (column in c("estimate", "moe", "relative_moe", "reliable")) eq(c[[column]], original[[column]])
eq(c$source_geoid, original$dominant_source_geoid)
eq(c$source_residential_share, original$dominant_source_share)
eq(c$source_assignment_method, original$dominant_source_method)
stopifnot(all(c$candidate_present), !any(c$source_geoid_missing))
reliable_series <- original %>% group_by(hex_id, source_geography) %>%
  summarise(reliable = n() == 6L && n_distinct(acs_year) == 6L && setequal(acs_year, years) && all(reliable), .groups = "drop") %>%
  pivot_wider(names_from = source_geography, values_from = reliable)
reliable_series <- reliable_series[match(ids, reliable_series$hex_id), ]
expected_geography <- ifelse(reliable_series$block_group, "block_group", ifelse(reliable_series$tract, "tract", NA_character_))
selection <- selection[match(ids, selection$hex_id), ]
eq(selection$bg_reliable_both, reliable_series$block_group); eq(selection$tract_reliable_both, reliable_series$tract)
eq(selection$selected_geography, expected_geography)
stopifnot(sum(expected_geography == "block_group", na.rm = TRUE) == 3018L,
  sum(expected_geography == "tract", na.rm = TRUE) == 1605L, sum(is.na(expected_geography)) == 2404L)
supported <- !is.na(expected_geography)
rent_components <- c("rent_level", "rent_growth", "rent_acceleration")
for (i in seq_along(dates)) {
  current_year <- 2022L + i; profile_years <- current_year - c(10L, 5L, 0L)
  f <- paired[paired$analysis_as_of_date == dates[i], ]; f <- f[match(ids, f$hex_id), ]
  s <- series[series$analysis_as_of_date == dates[i], ]; s <- s[match(ids, s$hex_id), ]
  vintage <- st_drop_geometry(readRDS(file.path(root, as.character(dates[i]), "acs_rent_by_hex_vintage.rds")))
  trend <- st_drop_geometry(readRDS(file.path(root, as.character(dates[i]), "acs_rent_trends_by_hex.rds")))
  trend <- trend[match(ids, trend$hex_id), ]
  stopifnot(nrow(f) == length(ids), is.integer(f$hex_id), nrow(vintage) == 3L * length(ids),
    !anyDuplicated(vintage[c("hex_id", "acs_year")]), setequal(vintage$acs_year, profile_years))
  eq(f$acs_rent_source_geography, expected_geography); eq(s$selected_geography, expected_geography)
  eq(f$acs_rent_series_supported, supported); eq(f$acs_rent_trend_reliable, supported)
  expected_values <- list()
  for (role in c("current", "previous", "earliest")) {
    year <- current_year - c(current = 0L, previous = 5L, earliest = 10L)[[role]]
    source_row <- match(paste(ids, year, expected_geography, sep = ":"), key(original))
    nominal <- original$estimate[source_row]; moe <- original$moe[source_row]
    multiplier <- unname(cpi["2024"] / cpi[as.character(year)])
    expected_values[[role]] <- nominal * multiplier
    v <- vintage[vintage$acs_year == year, ]; v <- v[match(ids, v$hex_id), ]
    eq(v$median_rent, nominal); eq(v$median_rent_moe, moe)
    eq(v$median_rent_source_geography, expected_geography)
    eq(v$median_rent_source_geoid, original$dominant_source_geoid[source_row])
    eq(v$median_rent_real, nominal * multiplier); eq(v$median_rent_moe_real, moe * multiplier)
    eq(v$median_rent_relative_moe, original$relative_moe[source_row]); eq(v$median_rent_reliable, supported)
    eq(v$median_rent_source_residential_share, original$dominant_source_share[source_row])
    eq(v$median_rent_source_assignment_method, original$dominant_source_method[source_row])
    eq(s[[paste0("source_geoid_", role)]], original$dominant_source_geoid[source_row])
    eq(f[[paste0("acs_rent_", role, "_source_geoid")]], original$dominant_source_geoid[source_row])
    for (suffix in c("nominal", "moe_nominal", "real", "moe_real")) {
      expected <- switch(suffix, nominal = nominal, moe_nominal = moe, real = nominal * multiplier, moe_real = moe * multiplier)
      eq(s[[paste0("rent_", role, "_", suffix)]], expected)
      eq(f[[paste0("acs_rent_", role, "_", suffix)]], expected)
      eq(trend[[paste0("acs_rent_", role, "_", suffix)]], expected)
    }
  }
  growth <- 20 * log(expected_values$current / expected_values$previous)
  acceleration <- growth - 20 * log(expected_values$previous / expected_values$earliest)
  expected <- data.frame(rent_level = expected_values$current, rent_growth = growth, rent_acceleration = acceleration)
  eq(s[rent_components], expected)
  eq(f$acs_rent_current_real, expected$rent_level); eq(f$acs_rent_growth_recent_annualized_pct, growth)
  eq(f$acs_rent_acceleration_pp, acceleration)
  expected_scores <- expected
  for (component in rent_components) {
    bound <- scaling$bounds[scaling$bounds$component == component, ]
    expected_scores[[component]] <- 100 * (pmax(bound$lower_bound, pmin(expected[[component]], bound$upper_bound)) - bound$lower_bound) /
      (bound$upper_bound - bound$lower_bound)
    eq(f[[paste0("acs_score_", component)]], expected_scores[[component]])
  }
  eq(f$rent_pressure_citywide_index, rowMeans(expected_scores, na.rm = FALSE))
  stopifnot(all(is.finite(f$rent_pressure_citywide_index) == supported),
    all(f$acs_rent_components_available == ifelse(supported, 3L, 0L)),
    all(is.na(as.matrix(f[!supported, c("acs_rent_current_real", "acs_rent_growth_recent_annualized_pct", "acs_rent_acceleration_pp",
      paste0("acs_score_", rent_components), "rent_pressure_citywide_index")]))))
}
cat("Promoted rent fallback output audit passed: twelve caches, six vintages, fixed source level, all three components, original bounds;",
  sum(supported), "supported and", sum(!supported), "unknown cells at both dates.\n")
