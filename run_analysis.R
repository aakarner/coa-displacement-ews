################################################################################
# Run the Displacement Early Warning System Pipeline
################################################################################

if (!requireNamespace("targets", quietly = TRUE)) {
  stop(
    "The targets package is required. Run `Rscript 00_requirements.R` first.",
    call. = FALSE
  )
}

requested_targets <- commandArgs(trailingOnly = TRUE)

if (length(requested_targets) == 0L) {
  targets::tar_make()
} else {
  available_targets <- targets::tar_manifest(fields = name)$name
  unknown_targets <- setdiff(requested_targets, available_targets)
  if (length(unknown_targets) > 0L) {
    stop(
      "Unknown target(s): ",
      paste(unknown_targets, collapse = ", "),
      call. = FALSE
    )
  }
  requested_targets <- unique(requested_targets)
  make_call <- rlang::expr(
    targets::tar_make(
      names = tidyselect::all_of(!!requested_targets)
    )
  )
  eval(make_call)
}
