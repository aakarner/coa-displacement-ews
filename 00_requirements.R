################################################################################
# 00 - Project Package Requirements
################################################################################
#
# Run this script once after cloning the repository or changing R versions:
#
#   Rscript 00_requirements.R
#
# Missing packages are installed into a version-specific project library under
# .r-library/. The project .Rprofile activates that library automatically when R
# is started from the repository root.
################################################################################

find_project_root <- function(start = getwd()) {
  current <- normalizePath(start, winslash = "/", mustWork = TRUE)

  repeat {
    if (file.exists(file.path(current, "run_analysis.R"))) {
      return(current)
    }

    parent <- dirname(current)
    if (identical(parent, current)) {
      stop("Could not locate the project root containing run_analysis.R.", call. = FALSE)
    }
    current <- parent
  }
}

EWS_PROJECT_ROOT <- find_project_root()
r_minor <- strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1]
EWS_PROJECT_LIBRARY <- file.path(
  EWS_PROJECT_ROOT,
  ".r-library",
  paste0("R-", R.version$major, ".", r_minor)
)

dir.create(EWS_PROJECT_LIBRARY, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(EWS_PROJECT_LIBRARY, .libPaths()))

EWS_PACKAGE_GROUPS <- list(
  data = c(
    "tidyverse", "data.table", "dplyr", "forcats", "janitor", "lubridate",
    "purrr", "readr", "readxl", "stringr", "tibble", "tidyr"
  ),
  spatial = c(
    "sf", "units", "h3jsr", "tigris", "lwgeom", "spdep", "terra",
    "tidygeocoder", "arcgisgeocode", "arcgisutils"
  ),
  census_and_api = c("tidycensus", "httr", "jsonlite"),
  clustering = c("cluster", "factoextra", "dbscan", "Rtsne"),
  modeling = c(
    "caret", "randomForest", "xgboost", "glmnet", "blockCV", "mgcv"
  ),
  visualization = c(
    "ggplot2", "leaflet", "mapview", "viridis", "scales", "patchwork",
    "gridExtra", "htmlwidgets", "ggthemes", "classInt", "ggspatial", "rosm"
  ),
  pipeline = c("targets"),
  utilities = c("here", "tictoc", "digest")
)

EWS_REQUIRED_PACKAGES <- unique(unlist(EWS_PACKAGE_GROUPS, use.names = FALSE))
missing_packages <- EWS_REQUIRED_PACKAGES[
  !vapply(EWS_REQUIRED_PACKAGES, requireNamespace, logical(1), quietly = TRUE)
]

repos <- getOption("repos")
if (is.null(repos) || length(repos) == 0 ||
    is.na(repos["CRAN"]) || identical(unname(repos["CRAN"]), "@CRAN@")) {
  repos <- c(CRAN = "https://cloud.r-project.org")
}
options(repos = repos)

cat("Project library: ", EWS_PROJECT_LIBRARY, "\n", sep = "")
cat("R version: ", R.version.string, "\n", sep = "")

if (length(missing_packages) > 0) {
  cat(
    "Installing missing packages: ",
    paste(missing_packages, collapse = ", "),
    "\n",
    sep = ""
  )

  install.packages(
    missing_packages,
    lib = EWS_PROJECT_LIBRARY,
    dependencies = c("Depends", "Imports", "LinkingTo")
  )
}

unavailable_packages <- EWS_REQUIRED_PACKAGES[
  !vapply(EWS_REQUIRED_PACKAGES, requireNamespace, logical(1), quietly = TRUE)
]

if (length(unavailable_packages) > 0) {
  stop(
    "Required package installation failed: ",
    paste(unavailable_packages, collapse = ", "),
    call. = FALSE
  )
}

cat(
  "All ",
  length(EWS_REQUIRED_PACKAGES),
  " required packages are available.\n",
  sep = ""
)
