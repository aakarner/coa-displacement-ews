# Unit tests extract pure function definitions without running the data pipeline.
suppressPackageStartupMessages(library(dplyr))
script <- parse("scripts/data/acs_rent_history.R")
function_names <- c("validate_acs_rent_profile", "annualized_log_change", "trend_from_vintages")
for (expression in script) {
  if (is.call(expression) && identical(expression[[1]], as.name("<-")) &&
      as.character(expression[[2]]) %in% function_names) eval(expression)
}
expect_error <- function(expression) {
  result <- tryCatch({force(expression); FALSE}, error = function(e) TRUE)
  stopifnot(result)
}
cpi <- c(`2013` = 232.957, `2018` = 251.107, `2023` = 304.702, `2024` = 313.689)
EWS_CONFIG <- list(acs_years = c(2013L, 2018L, 2023L), acs_current_year = 2023L,
                   acs_dollar_base_year = 2024L, acs_cpi_u = cpi,
                   acs_rent_relative_moe_limit = .3)
validate_acs_rent_profile(EWS_CONFIG$acs_years, 2023L, 2024L, cpi)
expect_error(validate_acs_rent_profile(c(2013L, 2018L, 2023L, 2024L), 2024L, 2024L, cpi))
expect_error(validate_acs_rent_profile(c(2013L, 2019L, 2023L), 2023L, 2024L, cpi))
expect_error(validate_acs_rent_profile(c(2013L, 2018L, 2023L), 2024L, 2024L, cpi))
expect_error(validate_acs_rent_profile(c(2013L, 2018L, 2023L), 2023L, 2025L, cpi))
stopifnot(is.na(annualized_log_change(Inf, 100, 2023, 2018)))
stopifnot(is.na(annualized_log_change(100, 0, 2023, 2018)))
stopifnot(abs(annualized_log_change(110, 100, 2023, 2018) - 100 * log(1.1) / 5) < 1e-12)
data <- tibble(acs_year = c(2013L, 2018L, 2023L), median_rent = c(100, 110, 120),
  median_rent_real = median_rent, median_rent_relative_moe = c(.1, .2, .3),
  median_rent_source_geoid = "test", median_rent_source_geography = "block_group",
  median_rent_source_residential_share = 1, median_rent_source_assignment_method = "test")
result <- trend_from_vintages(data)
stopifnot(result$acs_rent_trend_reliable, result$acs_rent_previous_year == 2018L,
          result$acs_rent_earliest_year == 2013L, result$acs_rent_dollar_base_year == 2024L,
          result$acs_rent_recent_interval_years == 5L, result$acs_rent_prior_interval_years == 5L)
data$median_rent_relative_moe[1] <- NA_real_
result <- trend_from_vintages(data)
stopifnot(!result$acs_rent_trend_reliable, result$acs_rent_moe_vintages_available == 2L)
data$median_rent_relative_moe[1] <- -0.1
stopifnot(!trend_from_vintages(data)$acs_rent_trend_reliable)
data$median_rent_relative_moe[1] <- .1
data$median_rent[1] <- Inf
stopifnot(!trend_from_vintages(data)$acs_rent_trend_reliable)
data$median_rent[1] <- 100
data$median_rent_relative_moe[1] <- .31
stopifnot(!trend_from_vintages(data)$acs_rent_trend_reliable)
expect_error(trend_from_vintages(data[-1, ]))
cat("ACS rent profile, growth, metadata, and missing-MOE reliability checks passed.\n")
