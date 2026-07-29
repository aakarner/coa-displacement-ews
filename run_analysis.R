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
  make_call <- rlang::expr(
    targets::tar_make(
      names = tidyselect::any_of(!!requested_targets)
    )
  )
  eval(make_call)
}
