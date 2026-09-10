################################################################################
# Build Part 3 Eviction and Demolition Forecast Labels
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})

source(here::here("R/utils.R"))
source(here::here("R/analysis_config.R"))
source(here::here("R/forecast_spec.R"))
source(here::here("R/forecast_labels.R"))

print_header("PART 3 - BUILD FORWARD OUTCOME LABELS")

OUTPUT_DIR <- here::here("output", "part3")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

eviction_panel_file <- here::here(
  "output",
  "eviction_filings_complete_by_hex_year.csv"
)
demolition_panel_file <- here::here(
  "output",
  "demolition_permits_by_hex_year.csv"
)
outcome_spec_file <- here::here("config", "forecast_outcomes.csv")

required_files <- c(
  eviction_panel_file,
  demolition_panel_file,
  outcome_spec_file
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Part 3 label inputs are missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

print_progress("Loading and validating complete outcome panels...")

eviction_panel <- read_csv(
  eviction_panel_file,
  col_select = all_of(c(
    "hex_id", "outcome_year", "eviction_cases", "source_covered",
    "period_complete", "measurement_complete", "count_observed",
    "coverage_reason", "analysis_as_of_date"
  )),
  col_types = cols(
    hex_id = col_integer(),
    outcome_year = col_integer(),
    eviction_cases = col_integer(),
    source_covered = col_logical(),
    period_complete = col_logical(),
    measurement_complete = col_logical(),
    count_observed = col_logical(),
    analysis_as_of_date = col_date(),
    coverage_reason = col_character()
  ),
  show_col_types = FALSE
)

demolition_panel <- read_csv(
  demolition_panel_file,
  col_select = all_of(c(
    "hex_id", "outcome_year", "residential_demolition_permits",
    "source_covered", "period_complete", "measurement_complete",
    "count_observed", "coverage_reason", "analysis_as_of_date"
  )),
  col_types = cols(
    hex_id = col_integer(),
    outcome_year = col_integer(),
    residential_demolition_permits = col_integer(),
    source_covered = col_logical(),
    period_complete = col_logical(),
    measurement_complete = col_logical(),
    count_observed = col_logical(),
    analysis_as_of_date = col_date(),
    coverage_reason = col_character()
  ),
  show_col_types = FALSE
)

outcome_spec <- read_forecast_outcomes(
  outcome_spec_file,
  EWS_CONFIG$forecast_horizons_years
)
panel_analysis_dates <- c(
  unique(eviction_panel$analysis_as_of_date),
  unique(demolition_panel$analysis_as_of_date)
)
if (
  length(panel_analysis_dates) != 2L ||
    anyNA(panel_analysis_dates) ||
    any(panel_analysis_dates != EWS_CONFIG$analysis_as_of_date)
) {
  stop(
    "Outcome panels were not built for the configured analysis_as_of_date. ",
    "Rebuild both panels before labels.",
    call. = FALSE
  )
}
pilot_outcomes <- outcome_spec$proxy_id[outcome_spec$pilot_scope == "pilot"]

if (!setequal(
  pilot_outcomes,
  c("eviction_filings", "residential_demolitions")
)) {
  stop(
    "The implemented Part 3 label builder requires the eviction/demolition ",
    "pilot contract.",
    call. = FALSE
  )
}

expected_label_domain <- bind_rows(
  build_expected_forecast_label_domain(
    eviction_panel,
    count_column = "eviction_cases",
    outcome_id = "eviction_filings",
    horizons = EWS_CONFIG$forecast_horizons_years
  ),
  build_expected_forecast_label_domain(
    demolition_panel,
    count_column = "residential_demolition_permits",
    outcome_id = "residential_demolitions",
    horizons = EWS_CONFIG$forecast_horizons_years
  )
)

eviction_labels <- build_forward_count_labels(
  eviction_panel,
  count_column = "eviction_cases",
  outcome_id = "eviction_filings",
  horizons = EWS_CONFIG$forecast_horizons_years
)
demolition_labels <- build_forward_count_labels(
  demolition_panel,
  count_column = "residential_demolition_permits",
  outcome_id = "residential_demolitions",
  horizons = EWS_CONFIG$forecast_horizons_years
)

forecast_labels <- bind_rows(eviction_labels, demolition_labels) |>
  arrange(
    .data$forecast_origin_year,
    .data$hex_id,
    .data$outcome_id,
    .data$horizon_years
  )

task_contract_qa <- validate_pilot_label_contract(
  forecast_labels,
  outcome_spec,
  EWS_CONFIG$forecast_horizons_years,
  expected_analysis_as_of_date = EWS_CONFIG$analysis_as_of_date,
  expected_domain = expected_label_domain
)
label_qa <- build_forecast_label_qa(forecast_labels)
joint_support_qa <- build_joint_common_support_qa(
  forecast_labels,
  pilot_outcomes
)
forecast_labels_wide <- build_forecast_labels_wide(forecast_labels)

if (any(label_qa$check_status != "pass")) {
  stop("At least one forecast label QA row failed.", call. = FALSE)
}
if (any(joint_support_qa$check_status != "pass")) {
  stop("At least one common-support QA row failed.", call. = FALSE)
}
joint_horizon_availability <- joint_support_qa |>
  group_by(.data$horizon_years) |>
  summarise(
    origins_with_joint_labels = sum(.data$joint_labels_observed > 0L),
    .groups = "drop"
  )
if (any(joint_horizon_availability$origins_with_joint_labels == 0L)) {
  stop(
    "At least one configured horizon has no origin with joint labels.",
    call. = FALSE
  )
}

print_progress("Writing Part 3 labels and QA outputs...")

saveRDS(
  forecast_labels,
  file.path(OUTPUT_DIR, "eviction_demolition_forecast_labels_long.rds")
)
write_csv(
  forecast_labels,
  file.path(OUTPUT_DIR, "eviction_demolition_forecast_labels_long.csv")
)
write_csv(
  forecast_labels_wide,
  file.path(OUTPUT_DIR, "eviction_demolition_forecast_labels_wide.csv")
)
write_csv(
  task_contract_qa,
  file.path(OUTPUT_DIR, "forecast_label_task_contract.csv")
)
write_csv(
  label_qa,
  file.path(OUTPUT_DIR, "forecast_label_qa.csv")
)
write_csv(
  joint_support_qa,
  file.path(OUTPUT_DIR, "forecast_label_common_support_qa.csv")
)

if (!requireNamespace("digest", quietly = TRUE)) {
  stop("Package 'digest' is required for the Part 3 run manifest.", call. = FALSE)
}
manifest_inputs <- c(
  eviction_panel_file,
  demolition_panel_file,
  outcome_spec_file
)
run_manifest <- tibble::tibble(
  artifact_role = c(
    "eviction_outcome_panel",
    "demolition_outcome_panel",
    "forecast_outcome_contract"
  ),
  path = manifest_inputs,
  sha256 = vapply(
    manifest_inputs,
    digest::digest,
    character(1),
    file = TRUE,
    algo = "sha256"
  ),
  bytes = as.numeric(file.info(manifest_inputs)$size),
  analysis_as_of_date = EWS_CONFIG$analysis_as_of_date,
  label_rows = nrow(forecast_labels),
  r_version = R.version.string
)
write_csv(
  run_manifest,
  file.path(OUTPUT_DIR, "forecast_label_run_manifest.csv")
)

print_progress(
  paste0(
    "Built ",
    format(nrow(forecast_labels), big.mark = ","),
    " long-form label rows across ",
    nrow(task_contract_qa),
    " configured tasks."
  )
)
print(task_contract_qa)

print_header("PART 3 FORWARD OUTCOME LABELS COMPLETE")
