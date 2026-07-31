################################################################################
# Refresh City Land Use Inventory Inputs for the Unit Classification Audit
################################################################################
#
# This script downloads ignored raw snapshots. It is intentionally separate
# from {targets} so a routine pipeline run never changes external source data.
################################################################################

source(here::here("R", "utils.R"))

print_header("REFRESH CITY LAND USE INVENTORY AUDIT INPUTS")

DATA_DIR <- here::here("data")
ACS_DIR <- file.path(DATA_DIR, "raw_acs")
dir.create(ACS_DIR, recursive = TRUE, showWarnings = FALSE)

sources <- tibble::tribble(
  ~description, ~url, ~destination,
  "City Land Use Inventory attributes",
  paste0(
    "https://data.austintexas.gov/resource/7vsm-dvxg.csv?",
    "$select=objectid%2Cland_use_id%2Cland_use%2Cgeneral_land_use%2C",
    "parcel_id_10%2Cproperty_id%2Ccreated_date%2Cmodified_date&",
    "$limit=500000"
  ),
  file.path(DATA_DIR, "austin_land_use_inventory_202607.csv"),
  "City Land Use Inventory geometry",
  paste0(
    "https://data.austintexas.gov/resource/7vsm-dvxg.geojson?",
    "$select=the_geom%2Cobjectid%2Cland_use_id%2Cland_use%2C",
    "general_land_use%2Cparcel_id_10%2Cproperty_id&$limit=500000"
  ),
  file.path(DATA_DIR, "austin_land_use_inventory_202607.geojson"),
  "2024 ACS one-year B25024 table",
  paste0(
    "https://www2.census.gov/programs-surveys/acs/summary_file/2024/",
    "table-based-SF/data/1YRData/acsdt1y2024-b25024.dat"
  ),
  file.path(ACS_DIR, "acsdt1y2024-b25024.dat")
)

for (index in seq_len(nrow(sources))) {
  print_progress(paste0("Downloading ", sources$description[[index]], "..."))
  temporary_file <- tempfile("ews_land_use_refresh_")
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
      "Download failed for ",
      sources$description[[index]],
      ": ",
      failure_message,
      call. = FALSE
    )
  }
  if (file.info(temporary_file)$size <= 0) {
    stop("Downloaded an empty file for ", sources$description[[index]], call. = FALSE)
  }
  if (!file.copy(
    temporary_file,
    sources$destination[[index]],
    overwrite = TRUE
  )) {
    stop("Could not replace ", sources$destination[[index]], call. = FALSE)
  }
  unlink(temporary_file)
}

print_progress("Refresh complete. Run:")
cat("Rscript run_analysis.R land_use_unit_classification_audit\n")
