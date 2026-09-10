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
    "pilot_scope",
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
  invalid_values <- vapply(
    parsed_horizons,
    function(value) {
      length(value) == 0L ||
        any(is.na(value)) ||
        any(value < 1L) ||
        anyDuplicated(value)
    },
    logical(1)
  )
  if (any(invalid_values)) {
    stop(
      "Part 3 horizons must be unique positive integers separated by '|'.",
      call. = FALSE
    )
  }
  invalid <- vapply(
    parsed_horizons,
    function(value) !setequal(value, configured_horizons),
    logical(1)
  )
  if (any(invalid)) {
    stop(
      "Every Part 3 proxy must use the configured horizon set: ",
      paste(sort(configured_horizons), collapse = ", "),
      " year(s).",
      call. = FALSE
    )
  }

  allowed_scopes <- c("pilot", "deferred")
  if (any(!outcomes$pilot_scope %in% allowed_scopes)) {
    stop(
      "pilot_scope must be either 'pilot' or 'deferred'.",
      call. = FALSE
    )
  }
  if (any(outcomes$pilot_scope == "pilot" & outcomes$status != "pilot_active")) {
    stop("Pilot outcomes must use status 'pilot_active'.", call. = FALSE)
  }
  if (any(outcomes$pilot_scope == "deferred" & outcomes$status != "deferred")) {
    stop("Deferred outcomes must use status 'deferred'.", call. = FALSE)
  }
  if (anyDuplicated(outcomes$proxy_id)) {
    stop("Forecast proxy_id values must be unique.", call. = FALSE)
  }
  outcomes
}

validate_forecast_panel_artifact <- function(
  path,
  proxy_id,
  expected_analysis_as_of_date = NULL
) {
  count_columns <- c(
    eviction_filings = "eviction_cases",
    residential_demolitions = "residential_demolition_permits"
  )
  count_column <- unname(count_columns[[proxy_id]])
  if (is.null(count_column)) {
    return(list(
      valid = NA,
      detail = "no complete-panel validator is configured for this outcome"
    ))
  }
  if (!file.exists(path)) {
    return(list(valid = FALSE, detail = "historical panel artifact is missing"))
  }
  if (!exists("validate_complete_outcome_panel", mode = "function")) {
    stop(
      "validate_complete_outcome_panel() must be sourced before readiness ",
      "validation.",
      call. = FALSE
    )
  }

  tryCatch(
    {
      # Base CSV type inference scans the full column. readr's sampled guessing
      # can classify a sparse count column as logical when its early values are
      # only 0/1 and then drop later counts greater than one.
      panel <- utils::read.csv(
        path,
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      summary <- validate_complete_outcome_panel(
        panel,
        count_column,
        paste0(proxy_id, " panel")
      )
      if (
        !is.null(expected_analysis_as_of_date) &&
          summary$analysis_as_of_date != as.Date(expected_analysis_as_of_date)
      ) {
        stop(
          "panel was built for ",
          summary$analysis_as_of_date,
          ", not ",
          as.Date(expected_analysis_as_of_date)
        )
      }
      study_geography_detail <- ""
      if ("hex_center_inside_current_austin_full" %in% names(panel)) {
        study_flag <- panel$hex_center_inside_current_austin_full
        if (!is.logical(study_flag) || anyNA(study_flag)) {
          stop(
            "fixed City study-geography flag must be complete and logical"
          )
        }
        study_reference <- unique(panel[c(
          "hex_id",
          "hex_center_inside_current_austin_full"
        )])
        if (nrow(study_reference) != summary$hexes) {
          stop("fixed City study-geography flag changes across years")
        }
        study_geography_detail <- paste0(
          " (",
          format(
            sum(study_reference$hex_center_inside_current_austin_full),
            big.mark = ","
          ),
          " fixed current-FULL study hexes)"
        )
      }
      list(
        valid = TRUE,
        detail = paste0(
          "valid complete computational panel: ",
          format(summary$hexes, big.mark = ","),
          " hexes",
          study_geography_detail,
          ", ",
          summary$first_year,
          "-",
          summary$last_year
        )
      )
    },
    error = function(error) {
      list(valid = FALSE, detail = conditionMessage(error))
    }
  )
}

validate_forecast_label_artifact <- function(
  path,
  outcomes,
  horizons,
  expected_analysis_as_of_date = NULL,
  source_files = NULL
) {
  if (!file.exists(path)) {
    return(list(valid = FALSE, detail = "forecast label artifact is missing"))
  }
  if (!exists("validate_pilot_label_contract", mode = "function")) {
    stop(
      "validate_pilot_label_contract() must be sourced before readiness ",
      "validation.",
      call. = FALSE
    )
  }
  if (!exists("build_expected_forecast_label_domain", mode = "function")) {
    stop(
      "build_expected_forecast_label_domain() must be sourced before ",
      "readiness validation.",
      call. = FALSE
    )
  }

  tryCatch(
    {
      pilot_outcomes <- outcomes$proxy_id[outcomes$pilot_scope == "pilot"]
      if (
        is.null(source_files) ||
          is.null(names(source_files)) ||
          any(!nzchar(names(source_files))) ||
          !all(pilot_outcomes %in% names(source_files))
      ) {
        stop(
          "source_files must name a source panel for every pilot outcome."
        )
      }
      count_columns <- c(
        eviction_filings = "eviction_cases",
        residential_demolitions = "residential_demolition_permits"
      )
      expected_domain <- lapply(
        pilot_outcomes,
        function(outcome_id) {
          panel_file <- unname(source_files[[outcome_id]])
          count_column <- unname(count_columns[[outcome_id]])
          if (is.null(count_column)) {
            stop(
              "No complete-panel validator is configured for ",
              outcome_id,
              "."
            )
          }
          if (!file.exists(panel_file)) {
            stop(outcome_id, " source panel artifact is missing.")
          }
          panel <- utils::read.csv(
            panel_file,
            stringsAsFactors = FALSE,
            check.names = FALSE
          )
          domain <- build_expected_forecast_label_domain(
            panel,
            count_column,
            outcome_id,
            horizons
          )
          panel_analysis_as_of_date <- unique(
            as.Date(panel$analysis_as_of_date)
          )[[1]]
          if (
            !is.null(expected_analysis_as_of_date) &&
              panel_analysis_as_of_date != as.Date(
                expected_analysis_as_of_date
              )
          ) {
            stop(
              outcome_id,
              " panel was built for ",
              panel_analysis_as_of_date,
              ", not ",
              as.Date(expected_analysis_as_of_date),
              "."
            )
          }
          domain
        }
      ) |>
        dplyr::bind_rows()

      labels <- if (grepl("[.]rds$", path, ignore.case = TRUE)) {
        readRDS(path)
      } else {
        readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
      }
      task_status <- validate_pilot_label_contract(
        labels,
        outcomes,
        horizons,
        expected_analysis_as_of_date = expected_analysis_as_of_date,
        expected_domain = expected_domain
      )
      list(
        valid = all(task_status$check_status == "pass"),
        detail = paste0(
          nrow(task_status),
          " pilot outcome/horizon tasks passed"
        )
      )
    },
    error = function(error) {
      list(valid = FALSE, detail = conditionMessage(error))
    }
  )
}

build_forecast_readiness <- function(
  outcome_spec_file,
  output_file,
  config,
  source_files,
  label_file = NULL,
  predictor_panel_file = NULL
) {
  outcomes <- read_forecast_outcomes(
    outcome_spec_file,
    config$forecast_horizons_years
  )

  pilot_outcomes <- outcomes$proxy_id[outcomes$pilot_scope == "pilot"]
  if (is.null(names(source_files)) || any(!nzchar(names(source_files)))) {
    stop("source_files must be a named character vector.", call. = FALSE)
  }
  if (
    any(!names(source_files) %in% outcomes$proxy_id) ||
      !all(pilot_outcomes %in% names(source_files))
  ) {
    stop(
      "source_files must name every pilot outcome and no unknown outcome.",
      call. = FALSE
    )
  }
  source_status <- data.frame(
    proxy_id = names(source_files),
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
  readiness <- readiness[match(outcomes$proxy_id, readiness$proxy_id), ]
  readiness$pilot_included <- readiness$pilot_scope == "pilot"
  readiness$artifact_exists[!readiness$pilot_included] <- NA

  panel_validation <- lapply(
    seq_len(nrow(readiness)),
    function(index) {
      if (!readiness$pilot_included[[index]]) {
        return(list(valid = NA, detail = "outcome deferred from pilot"))
      }
      validate_forecast_panel_artifact(
        readiness$historical_artifact[[index]],
        readiness$proxy_id[[index]],
        expected_analysis_as_of_date = config$analysis_as_of_date
      )
    }
  )
  readiness$panel_contract_valid <- vapply(
    panel_validation,
    function(result) result$valid,
    logical(1)
  )
  readiness$panel_validation_detail <- vapply(
    panel_validation,
    function(result) result$detail,
    character(1)
  )

  label_validation <- if (is.null(label_file)) {
    list(valid = FALSE, detail = "forecast label artifact was not configured")
  } else {
    validate_forecast_label_artifact(
      label_file,
      outcomes,
      config$forecast_horizons_years,
      expected_analysis_as_of_date = config$analysis_as_of_date,
      source_files = source_files
    )
  }
  readiness$label_artifact <- if (is.null(label_file)) NA_character_ else label_file
  readiness$labels_valid <- ifelse(
    readiness$pilot_included,
    label_validation$valid,
    NA
  )
  readiness$label_validation_detail <- ifelse(
    readiness$pilot_included,
    label_validation$detail,
    "outcome deferred from pilot"
  )

  predictor_exists <- !is.null(predictor_panel_file) &&
    file.exists(predictor_panel_file)
  readiness$predictor_panel_artifact <- if (is.null(predictor_panel_file)) {
    NA_character_
  } else {
    predictor_panel_file
  }
  readiness$predictor_panel_exists <- ifelse(
    readiness$pilot_included,
    predictor_exists,
    NA
  )
  readiness$ready_for_predictor_panel <-
    readiness$pilot_included &
    readiness$panel_contract_valid %in% TRUE &
    readiness$labels_valid %in% TRUE
  readiness$ready_for_modeling <-
    readiness$ready_for_predictor_panel &
    readiness$predictor_panel_exists %in% TRUE

  readiness$next_requirement <- vapply(
    seq_len(nrow(readiness)),
    function(index) {
      if (!readiness$pilot_included[[index]]) {
        return("deferred until the eviction/demolition pilot is evaluated")
      }
      if (!readiness$artifact_exists[[index]]) {
        return("build the complete hex-year outcome panel")
      }
      if (!isTRUE(readiness$panel_contract_valid[[index]])) {
        return("repair the complete-panel contract or QA failure")
      }
      if (!isTRUE(readiness$labels_valid[[index]])) {
        return("build and validate the 1- and 3-year forecast labels")
      }
      if (!isTRUE(readiness$predictor_panel_exists[[index]])) {
        return("construct and validate the leakage-safe historical predictor panel")
      }
      "ready"
    },
    character(1)
  )

  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(readiness, output_file)
  output_file
}
