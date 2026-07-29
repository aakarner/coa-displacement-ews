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
    args = script,
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
