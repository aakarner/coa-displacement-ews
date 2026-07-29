################################################################################
# Pipeline Orchestration Helpers
################################################################################

build_file_manifest <- function(paths, recursive = FALSE) {
  expanded <- unlist(
    lapply(
      paths,
      function(path) {
        if (dir.exists(path)) {
          list.files(
            path,
            recursive = recursive,
            full.names = TRUE,
            all.files = FALSE,
            no.. = TRUE
          )
        } else if (file.exists(path)) {
          path
        } else {
          character()
        }
      }
    ),
    use.names = FALSE
  )
  expanded <- sort(unique(expanded[file.exists(expanded)]))
  if (length(expanded) == 0L) {
    return(data.frame(
      path = character(),
      size = numeric(),
      modified = character()
    ))
  }

  info <- file.info(expanded)
  data.frame(
    path = normalizePath(expanded, winslash = "/", mustWork = TRUE),
    size = as.numeric(info$size),
    modified = format(info$mtime, "%Y-%m-%dT%H:%M:%OS6%z"),
    stringsAsFactors = FALSE
  )
}

run_r_script_stage <- function(
  script,
  outputs,
  dependencies = NULL,
  environment = character()
) {
  force(dependencies)
  if (length(script) != 1L || !file.exists(script)) {
    stop("Pipeline stage script is missing: ", script, call. = FALSE)
  }

  adopt_existing <- tolower(
    Sys.getenv("EWS_TARGETS_ADOPT_EXISTING", unset = "false")
  ) %in% c("1", "true", "t", "yes", "y")
  if (adopt_existing && all(file.exists(outputs))) {
    message("Adopting existing outputs for ", script, ".")
    return(outputs)
  }

  output_dir <- unique(dirname(outputs))
  invisible(
    lapply(
      output_dir,
      dir.create,
      recursive = TRUE,
      showWarnings = FALSE
    )
  )
  command <- file.path(R.home("bin"), "Rscript")
  status <- system2(
    command,
    args = c("--vanilla", script),
    stdout = "",
    stderr = "",
    env = environment
  )
  if (!identical(status, 0L)) {
    stop(
      "Pipeline stage failed with exit status ",
      status,
      ": ",
      script,
      call. = FALSE
    )
  }

  missing_outputs <- outputs[!file.exists(outputs)]
  if (length(missing_outputs) > 0L) {
    stop(
      "Pipeline stage did not create expected output(s): ",
      paste(missing_outputs, collapse = ", "),
      call. = FALSE
    )
  }
  outputs
}

run_cached_api_stage <- function(
  script,
  outputs,
  dependencies = NULL,
  required_environment,
  expected_as_of = NULL,
  as_of_field = NULL
) {
  missing_environment <- required_environment[
    !nzchar(Sys.getenv(required_environment, unset = ""))
  ]
  if (length(missing_environment) > 0L) {
    if (all(file.exists(outputs))) {
      if (!is.null(expected_as_of) && !is.null(as_of_field)) {
        cached <- readRDS(outputs[[1]])
        if (!as_of_field %in% names(cached)) {
          stop(
            "Cached API output does not record ",
            as_of_field,
            ": ",
            outputs[[1]],
            call. = FALSE
          )
        }
        cached_as_of <- unique(as.Date(cached[[as_of_field]]))
        if (
          length(cached_as_of) != 1L ||
            is.na(cached_as_of) ||
            cached_as_of != as.Date(expected_as_of)
        ) {
          stop(
            "Cached API output uses cutoff ",
            paste(cached_as_of, collapse = ", "),
            " but the pipeline requires ",
            as.Date(expected_as_of),
            ". Set the required API credentials and rerun.",
            call. = FALSE
          )
        }
      }
      message(
        "Using cached outputs for ",
        script,
        " because API credentials are not set."
      )
      return(outputs)
    }
    stop(
      "Missing API environment variable(s): ",
      paste(missing_environment, collapse = ", "),
      call. = FALSE
    )
  }

  run_r_script_stage(
    script = script,
    outputs = outputs,
    dependencies = dependencies
  )
}
