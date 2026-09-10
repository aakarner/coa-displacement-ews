# Pure current-snapshot rules; no output mutation or network.
suppressPackageStartupMessages(library(dplyr))
source("R/ownership_snapshots.R")
source("R/part2_index_scoring.R")
source("R/current_measurement.R")
expect_error <- function(x) stopifnot(inherits(tryCatch({force(x); NULL}, error = identity), "error"))
years <- c(2014L, 2019L, 2024L)
candidates <- expand.grid(hex_id = 1:4, acs_year = c(2013L, 2014L, 2018L, 2019L, 2023L, 2024L),
  source_geography = c("block_group", "tract"), stringsAsFactors = FALSE)
candidates$estimate <- 1000
candidates$moe <- 300
candidates$source_geoid <- "source"
# Earlier-only absence must not exclude a current snapshot.
candidates$moe[candidates$acs_year == 2013L] <- NA_real_
# One bad BG year: entire CURRENT triplet must use tract, not per-year mixing.
candidates$moe[candidates$hex_id == 2 & candidates$source_geography == "block_group" & candidates$acs_year == 2019] <- 301
# Neither series works; all three components stay unknown.
candidates$moe[candidates$hex_id == 3 & candidates$acs_year == 2024] <- NA_real_
candidates$estimate[candidates$hex_id == 4 & candidates$source_geography == "block_group" & candidates$acs_year == 2014] <- 0
cpi <- setNames(rep(1, 6), as.character(sort(unique(candidates$acs_year))))
x <- current_rent_series(candidates, 1:4, years, cpi)
stopifnot(identical(x$acs_rent_source_geography, c("block_group", "tract", NA_character_, "tract")),
  all(x$acs_rent_growth_recent_annualized_pct[c(1,2,4)] == 0),
  all(is.na(x[3, c("acs_rent_current_real", "acs_rent_growth_recent_annualized_pct", "acs_rent_acceleration_pp")])) )
other <- candidates; other$estimate[!other$acs_year %in% years] <- 9000
stopifnot(identical(x, current_rent_series(other, 1:4, years, cpi)))
expect_error(current_rent_series(rbind(candidates, candidates[candidates$acs_year == 2014, ][1, ]), 1:4, years, cpi))
missing <- candidates[!(candidates$hex_id == 1 & candidates$source_geography == "block_group" & candidates$acs_year == 2019), ]
stopifnot(current_rent_series(missing, 1:4, years, cpi)$acs_rent_source_geography[1] == "tract")

# Current known evidence can pass even when every prior-year flag is unknown.
p <- data.frame(parcel_id = as.character(1:20), hex_id = 1L, tax_year = 2025L,
  residential_units = 2, classification_status = "matched_classified",
  is_corporate_owned = rep(c(TRUE,FALSE),10), has_financialized_owner = TRUE)
prior <- p; prior$tax_year <- 2024L; prior$is_corporate_owned <- NA; prior$has_financialized_owner <- NA
support <- data.frame(hex_id = 1:2, area_km2 = 1, residential_units = c(40,0))
o <- current_ownership_features(rbind(p,prior), support)
stopifnot(identical(o$ownership_current_usable, c(TRUE,FALSE)), o$pct_corporate_units[1] == 50,
  o$corporate_owned_units_per_km2[1] == 20, is.na(o$pct_corporate_units[2]))
p$is_corporate_owned[1] <- NA; p$has_financialized_owner[1] <- NA
o <- current_ownership_features(p, support)
stopifnot(o$ownership_current_usable[1], o$ownership_unit_coverage[1] == .95,
  abs(o$pct_corporate_units[1] - 100 * 18/38) < 1e-10)
p$is_corporate_owned[2] <- NA
o <- current_ownership_features(p, support)
stopifnot(!o$ownership_current_usable[1], is.na(o$pct_corporate_units[1]))
expect_error(current_ownership_features(rbind(p,p[1,]), support))
cat("Current measurement synthetic checks passed: three-vintage coherent rent, prior-year independence, observed ownership denominators and coverage.\n")
