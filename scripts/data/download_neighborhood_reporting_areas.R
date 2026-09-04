################################################################################
# Refresh City of Austin Neighborhood Reporting Areas
################################################################################
#
# Downloads the ignored raw geography and its Socrata metadata. This refresh is
# intentionally separate from {targets} so a routine analysis never changes an
# external source snapshot.
################################################################################

source(here::here("R", "utils.R"))

suppressPackageStartupMessages({
  library(jsonlite)
  library(sf)
})

print_header("REFRESH NEIGHBORHOOD REPORTING AREAS")

DATASET_ID <- "a7ap-j2yt"
DATA_DIR <- here::here("data")
GEOMETRY_FILE <- file.path(DATA_DIR, "neighborhood_reporting_areas.geojson")
METADATA_FILE <- file.path(
  DATA_DIR,
  "neighborhood_reporting_areas_metadata.json"
)

sources <- data.frame(
  description = c("reporting-area geometry", "Socrata metadata"),
  url = c(
    paste0(
      "https://data.austintexas.gov/resource/", DATASET_ID,
      ".geojson?$limit=5000"
    ),
    paste0("https://data.austintexas.gov/api/views/", DATASET_ID)
  ),
  destination = c(GEOMETRY_FILE, METADATA_FILE),
  stringsAsFactors = FALSE
)

dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)

for (index in seq_len(nrow(sources))) {
  print_progress(paste0("Downloading ", sources$description[[index]], "..."))
  temporary_file <- tempfile("ews_neighborhood_refresh_")
  status <- tryCatch(
    utils::download.file(
      sources$url[[index]],
      temporary_file,
      mode = "wb",
      method = "libcurl",
      quiet = FALSE
    ),
    error = function(error) error
  )
  if (inherits(status, "error") || !file.exists(temporary_file)) {
    failure_message <- if (inherits(status, "error")) {
      conditionMessage(status)
    } else {
      "temporary download file was not created"
    }
    stop(
      "Download failed for ", sources$description[[index]], ": ",
      failure_message,
      call. = FALSE
    )
  }
  if (file.info(temporary_file)$size <= 0) {
    stop("Downloaded an empty file for ", sources$description[[index]], call. = FALSE)
  }
  if (!file.copy(temporary_file, sources$destination[[index]], overwrite = TRUE)) {
    stop("Could not replace ", sources$destination[[index]], call. = FALSE)
  }
  unlink(temporary_file)
}

reporting_areas <- st_read(GEOMETRY_FILE, quiet = TRUE)
metadata <- fromJSON(METADATA_FILE)

if (!identical(metadata$id, DATASET_ID)) {
  stop("Downloaded metadata does not match dataset ", DATASET_ID, call. = FALSE)
}
if (
  nrow(reporting_areas) == 0L ||
    !"neighname" %in% names(reporting_areas) ||
    any(is.na(reporting_areas$neighname)) ||
    anyDuplicated(reporting_areas$neighname)
) {
  stop("Neighborhood Reporting Area download failed validation.", call. = FALSE)
}

source_updated <- as.POSIXct(
  metadata$rowsUpdatedAt,
  origin = "1970-01-01",
  tz = "UTC"
)
print_progress(
  paste0(
    "Saved ", nrow(reporting_areas), " unique reporting areas; source updated ",
    format(source_updated, "%Y-%m-%d", tz = "UTC"), "."
  )
)
cat("Run: Rscript run_analysis.R part1_neighborhood_summary\n")
