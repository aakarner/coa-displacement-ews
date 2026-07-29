################################################################################
# Part 3 Forecast Specification and Readiness
################################################################################

read_forecast_outcomes <- function(path, configured_horizons) {
  outcomes <- readr::read_csv(path, show_col_types = FALSE)
  required <- c(
    "proxy_id",
    "outcome_family",
    "preferred_measure",
    "preferred_denominator",
    "source",
    "horizons_years",
    "status"
  )
  missing <- setdiff(required, names(outcomes))
  if (length(missing) > 0L) {
    stop(
      "Forecast outcome specification is missing: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  parsed_horizons <- lapply(
    outcomes$horizons_years,
    function(value) {
      as.integer(strsplit(as.character(value), "|", fixed = TRUE)[[1]])
    }
  )
  invalid <- vapply(
    parsed_horizons,
    function(value) !setequal(value, configured_horizons),
    logical(1)
  )
  if (any(invalid)) {
    stop(
      "Every Part 3 proxy must use the configured 1-, 3-, and 5-year ",
      "horizons.",
      call. = FALSE
    )
  }
  outcomes
}

build_forecast_readiness <- function(
  outcome_spec_file,
  output_file,
  config,
  source_files
) {
  outcomes <- read_forecast_outcomes(
    outcome_spec_file,
    config$forecast_horizons_years
  )

  source_status <- data.frame(
    proxy_id = c(
      "eviction_filings",
      "residential_demolitions",
      "rent_growth",
      "land_value_growth"
    ),
    historical_artifact = unname(source_files),
    artifact_exists = file.exists(unname(source_files)),
    stringsAsFactors = FALSE
  )
  readiness <- merge(
    outcomes,
    source_status,
    by = "proxy_id",
    all.x = TRUE,
    sort = FALSE
  )
  readiness$ready_for_modeling <-
    readiness$artifact_exists &
    readiness$status == "ready"
  readiness$next_requirement <- ifelse(
    !readiness$artifact_exists,
    "build or restore the historical hex-level outcome artifact",
    ifelse(
      readiness$status != "ready",
      "construct and validate the complete hex-year outcome panel",
      "ready"
    )
  )

  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(readiness, output_file)
  output_file
}
