# Refresh parcel-dependent crosswalks before assembling the paired ACS products.
# All Census extracts are local; the reviewed source/normalization rules persist.
for (date in c("2025-04-01", "2026-04-01")) {
  earlier <- date == "2025-04-01"
  env <- c(paste0("EWS_ANALYSIS_AS_OF_DATE=", date),
    paste0("EWS_ACS_YEARS=", if (earlier) "2013,2018,2023" else "2014,2019,2024"),
    paste0("EWS_ACS_CURRENT_YEAR=", if (earlier) "2023" else "2024"),
    "EWS_ACS_DOLLAR_BASE_YEAR=2024", "EWS_ACS_PRESERVE_MISSING=true",
    paste0("EWS_ACS_OUTPUT_DIR=output/part2/acs/", date))
  status <- system2(file.path(R.home("bin"), "Rscript"),
    "scripts/data/acs_rent_history.R", env = env)
  if (status != 0L) stop("ACS rent support refresh failed: ", date)
}
status <- system2(file.path(R.home("bin"), "Rscript"), "scripts/part2/build_acs_snapshots.R")
if (status != 0L) stop("Paired ACS support refresh failed.")
