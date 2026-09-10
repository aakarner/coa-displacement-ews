# Synthetic, file-free audit of fixed-geography historical rent fallback.
source("R/part2_rent_fallback.R")
expect_error <- function(expr, pattern = NULL) {
  result <- tryCatch(force(expr), error = identity)
  stopifnot(inherits(result, "error"))
  if (!is.null(pattern)) stopifnot(grepl(pattern, conditionMessage(result)))
}
expect_equal <- function(actual, expected) stopifnot(isTRUE(all.equal(actual, expected, check.attributes = FALSE)))
years <- part2_rent_required_years()
fixture <- expand.grid(hex_id = 1:4, acs_year = years, source_geography = c("block_group", "tract"), stringsAsFactors = FALSE)
fixture$source_geoid <- paste(fixture$source_geography, fixture$hex_id, fixture$acs_year, sep = "-")
fixture$estimate <- ifelse(fixture$source_geography == "block_group", 100, 200)
fixture$moe <- fixture$estimate * .30
fixture$optional_audit_note <- "preserved"
# Hex 1: exactly 30% in both sources -> BG. Hex 2: one BG term at
# 30.01% -> tract at BOTH dates. Hex 3: mixed reliable vintages across sources
# but neither full series passes. Hex 4: an entirely missing BG vintage.
fixture$moe[fixture$hex_id == 2 & fixture$source_geography == "block_group" & fixture$acs_year == 2024] <- 30.01
fixture$moe[fixture$hex_id == 3 & fixture$source_geography == "block_group" & fixture$acs_year == 2024] <- NA_real_
fixture$estimate[fixture$hex_id == 3 & fixture$source_geography == "tract" & fixture$acs_year == 2013] <- NA_real_
fixture <- fixture[!(fixture$hex_id == 4 & fixture$source_geography == "block_group" & fixture$acs_year == 2018), ]
selected <- part2_select_rent_series(fixture)
s <- selected$selection; c <- selected$candidates
stopifnot(identical(s$hex_id, 1:4), identical(s$selected_geography, c("block_group", "tract", NA_character_, "tract")),
  identical(s$bg_reliable_both, c(TRUE, FALSE, FALSE, FALSE)), all(s$tract_reliable_both == c(TRUE, TRUE, FALSE, TRUE)),
  s$bg_reliable_2025[2], !s$bg_reliable_2026[2], !s$tract_reliable_2025[3], s$tract_reliable_2026[3],
  s$bg_failed_years[4] == "2018", nrow(c) == 48L,
  all(c$optional_audit_note[c$candidate_present] == "preserved"), all(s$relative_moe_limit == .30))
missing <- c[c$hex_id == 4 & c$source_geography == "block_group" & c$acs_year == 2018, ]
stopifnot(!missing$candidate_present, !missing$reliable, missing$failure_reason == "missing_vintage",
  is.na(missing$estimate), is.na(missing$relative_moe))
boundary <- c[c$hex_id == 1, ]
stopifnot(all(boundary$relative_moe == .30), all(boundary$reliable), all(boundary$failure_reason == "reliable"))
over <- c[c$hex_id == 2 & c$source_geography == "block_group" & c$acs_year == 2024, ]
expect_equal(over$relative_moe, .3001)
stopifnot(!over$reliable, over$failure_reason == "relative_moe_above_limit")
stopifnot(part2_select_rent_series(fixture, .301)$selection$selected_geography[2] == "block_group")
expect_equal(part2_select_rent_series(fixture[nrow(fixture):1, ]), selected)
expect_equal(part2_select_rent_series(selected$candidates), selected)
# Independent small-table scans confirm the grouped implementation's row layout
# and all per-date flags, including filled-in missing-vintage rows.
for (hex in s$hex_id) for (geography in c("block_group", "tract")) {
  prefix <- if (geography == "block_group") "bg" else "tract"
  rows <- c[c$hex_id == hex & c$source_geography == geography, ]
  selection_row <- which(s$hex_id == hex)
  for (date in c("2025", "2026")) {
    date_years <- if (date == "2025") c(2013L, 2018L, 2023L) else c(2014L, 2019L, 2024L)
    stopifnot(s[[paste0(prefix, "_reliable_", date)]][selection_row] == all(rows$reliable[rows$acs_year %in% date_years]))
  }
  stopifnot(s[[paste0(prefix, "_reliable_both")]][selection_row] == all(rows$reliable),
    s[[paste0(prefix, "_failed_years")]][selection_row] == paste(rows$acs_year[!rows$reliable], collapse = ";"))
}

invalid_values <- list(
  list(column = "moe", value = NA_real_, reason = "moe_missing"),
  list(column = "moe", value = Inf, reason = "moe_nonfinite"),
  list(column = "moe", value = -1, reason = "moe_negative"),
  list(column = "estimate", value = NA_real_, reason = "estimate_missing"),
  list(column = "estimate", value = NaN, reason = "estimate_missing"),
  list(column = "estimate", value = Inf, reason = "estimate_nonfinite"),
  list(column = "estimate", value = 0, reason = "estimate_nonpositive"),
  list(column = "estimate", value = -1, reason = "estimate_nonpositive"))
one <- fixture[fixture$hex_id == 1, ]
for (case in invalid_values) {
  bad <- one; row <- which(bad$source_geography == "block_group" & bad$acs_year == 2023)
  bad[[case$column]][row] <- case$value
  checked <- part2_select_rent_series(bad)
  audit <- checked$candidates[checked$candidates$source_geography == "block_group" & checked$candidates$acs_year == 2023, ]
  stopifnot(!audit$reliable, audit$failure_reason == case$reason,
    checked$selection$selected_geography == "tract")
}
zero_moe <- one; zero_moe$moe <- 0
stopifnot(all(part2_select_rent_series(zero_moe, 0)$candidates$reliable))
missing_provenance <- one; missing_provenance$source_geoid[1] <- NA_character_
checked <- part2_select_rent_series(missing_provenance)
stopifnot(sum(checked$candidates$source_geoid_missing) == 1L, all(checked$candidates$reliable))
no_tract <- one[one$source_geography == "block_group", ]
checked <- part2_select_rent_series(no_tract)
stopifnot(checked$selection$selected_geography == "block_group", !checked$selection$tract_reliable_both,
  sum(!checked$candidates$candidate_present) == 6L)
expect_error(part2_select_rent_series(rbind(one, one[1, ])), "Duplicate")
bad <- one; bad$source_geoid[1] <- "different-geoid"
expect_error(part2_select_rent_series(rbind(one, bad[1, ])), "Duplicate")
bad <- one; bad$source_geography[1] <- "county"
expect_error(part2_select_rent_series(bad), "schema")
bad <- one; bad$acs_year[1] <- 2022L
expect_error(part2_select_rent_series(bad), "schema")
bad <- one; bad$estimate <- as.character(bad$estimate)
expect_error(part2_select_rent_series(bad), "schema")
expect_error(part2_select_rent_series(one[setdiff(names(one), "moe")]), "schema")
for (limit in list(-.1, NA_real_, Inf, c(.3, .4))) expect_error(part2_select_rent_series(one, limit), "limit")

cpi <- c(`2013` = 100, `2014` = 110, `2018` = 125, `2019` = 137.5, `2023` = 150, `2024` = 165)
features <- part2_rent_series_features(fixture, selected$selection, cpi)
stopifnot(nrow(features) == 8L, is.integer(features$hex_id),
  identical(features$analysis_as_of_date, rep(as.Date(c("2025-04-01", "2026-04-01")), each = 4)),
  identical(features$selected_geography, rep(s$selected_geography, 2)),
  all(is.na(as.matrix(features[features$hex_id == 3, c("rent_level", "rent_growth", "rent_acceleration")]))),
  all(is.na(features$source_geoid_current[features$hex_id == 3])),
  all(features$rent_current_nominal[features$hex_id == 2] == 200),
  all(features$rent_previous_nominal[features$hex_id == 2] == 200),
  all(features$rent_earliest_nominal[features$hex_id == 2] == 200),
  all(features$rent_dollar_base_year == 2024L))
expect_equal(part2_rent_series_features(fixture[nrow(fixture):1, ], s[4:1, ], cpi), features)
expect_equal(part2_rent_series_features(selected$candidates, s, cpi), features)
stopifnot(features$source_geoid_current[2] == "tract-2-2023", features$source_geoid_current[6] == "tract-2-2024",
  features$source_geoid_previous[2] == "tract-2-2018", features$source_geoid_earliest[6] == "tract-2-2014")
expect_equal(features$rent_level[1], 110)
expect_equal(features$rent_current_moe_real[1], 33)
expect_equal(features$rent_growth[1], 100 * log(125 / 150) / 5)
expect_equal(features$rent_acceleration[1], 100 * log(125 / 150) / 5 - 100 * log(100 / 125) / 5)

# Known real-dollar sequences isolate inflation adjustment and both five-year
# growth intervals. No per-vintage or per-date source mixing is permitted.
real_rents <- c(`2013` = 100, `2014` = 200, `2018` = 120, `2019` = 220, `2023` = 180, `2024` = 264)
growth <- one
growth$estimate <- real_rents[as.character(growth$acs_year)] * cpi[as.character(growth$acs_year)] / cpi[["2024"]]
growth$moe <- growth$estimate * .1
growth_selection <- part2_select_rent_series(growth)
g <- part2_rent_series_features(growth, growth_selection$selection, cpi)
expect_equal(g$rent_level, c(180, 264))
expect_equal(g$rent_growth, 100 * log(c(180 / 120, 264 / 220)) / 5)
expect_equal(g$rent_acceleration, 100 * log(c(180 / 120, 264 / 220)) / 5 - 100 * log(c(120 / 100, 220 / 200)) / 5)
expect_equal(g$rent_current_moe_real, c(18, 26.4))
older_base <- part2_rent_series_features(growth, growth_selection$selection, cpi, base_year = 2023L)
expect_equal(older_base$rent_level, g$rent_level * 150 / 165)
expect_equal(older_base$rent_growth, g$rent_growth)
expect_equal(older_base$rent_acceleration, g$rent_acceleration)
inflation_only <- one
inflation_only$estimate <- 10 * cpi[as.character(inflation_only$acs_year)]
inflation_only$moe <- inflation_only$estimate * .1
flat <- part2_rent_series_features(inflation_only, part2_select_rent_series(inflation_only)$selection, cpi)
expect_equal(flat$rent_level, c(1650, 1650)); expect_equal(flat$rent_growth, c(0, 0)); expect_equal(flat$rent_acceleration, c(0, 0))

bad_selection <- s; bad_selection$selected_geography[2] <- "block_group"
expect_error(part2_rent_series_features(fixture, bad_selection, cpi), "fixed six-vintage rule")
expect_error(part2_rent_series_features(fixture, s[-1, ], cpi), "fixed six-vintage rule")
expect_error(part2_rent_series_features(fixture, rbind(s, s[1, ]), cpi), "selection")
expect_error(part2_rent_series_features(fixture, s, cpi[-1]), "CPI")
bad_cpi <- cpi; bad_cpi[1] <- 0
expect_error(part2_rent_series_features(fixture, s, bad_cpi), "CPI")
expect_error(part2_rent_series_features(fixture, s, c(cpi, `2013` = 100)), "CPI")
# The promoted adapter keeps all source years and uncertainty paired to the
# selected geography, including unsupported rows with no fabricated rent level.
source("R/part2_acs_rent_products.R")
selected$candidates$source_residential_share <- 1
selected$candidates$source_assignment_method <- "synthetic_fixed_support"
products <- part2_acs_rent_products(selected$candidates, s, features, data.frame(hex_id = 1:4), cpi)
for (i in 1:2) {
  p <- products[[i]]
  stopifnot(nrow(p$vintage) == 12L, nrow(p$trends) == 4L,
    identical(p$trends$acs_rent_source_geography, s$selected_geography),
    all(is.na(p$vintage$median_rent[p$vintage$hex_id == 3])),
    all(is.na(p$vintage$median_rent_moe[p$vintage$hex_id == 3])),
    is.na(p$trends$acs_rent_current_real[3]), !p$trends$acs_rent_trend_reliable[3])
  expect_equal(p$trends$acs_rent_growth_recent_annualized_pct, features$rent_growth[features$analysis_as_of_date == unique(p$trends$analysis_as_of_date)])
  expect_equal(p$trends$acs_rent_acceleration_pp, features$rent_acceleration[features$analysis_as_of_date == unique(p$trends$analysis_as_of_date)])
}
cat("Historical fixed-geography rent fallback tests passed.\n")
