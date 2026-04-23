################################################################################
# Corporate Ownership Analysis - Identify Likely Corporate-Owned Parcels
################################################################################
#
# This standalone script identifies likely corporate-owned parcels within the
# City of Austin using raw TCAD (Travis Central Appraisal District) data.
# Parcels are filtered to those lying within the official City of Austin
# boundaries, retrieved via the tigris package.
#
# INPUTS:
#   - data/tcad_properties.csv: TCAD property data export. Expected columns:
#       prop_id      - Unique property identifier
#       owner_name   - Name of the property owner
#       latitude     - Latitude in decimal degrees (WGS84)
#       longitude    - Longitude in decimal degrees (WGS84)
#     Optional columns (used when present):
#       prop_type_cd    - Property type code (e.g. "R" = Real property)
#       appraised_val   - Appraised total value
#       land_val        - Land-only appraised value
#       improvement_val - Improvement (structure) appraised value
#       legal_area      - Parcel area in square feet
#   - City of Austin boundary (downloaded automatically via tigris)
#
# OUTPUTS:
#   - output/corporate_parcels.rds: Likely corporate-owned parcels as sf object
#   - output/corporate_parcels.csv: Same data in flat CSV (no geometry)
#
# DEPENDENCIES:
#   - tidyverse, sf, tigris packages
#   - R/utils.R for helper functions (print_header, print_progress, save_output)
#
################################################################################

# Source utilities (enables standalone execution; also sourced by run_analysis.R)
source(here::here("R/utils.R"))

print_header("CORPORATE OWNERSHIP ANALYSIS")

# Configuration
DATA_DIR   <- here::here("data")
OUTPUT_DIR <- here::here("output")

# Create output directory if it doesn't exist
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

################################################################################
# Step 1: Load raw TCAD data and extract latitude/longitude
################################################################################

print_progress("Loading raw TCAD property data...")

tcad_file <- file.path(DATA_DIR, "tcad_properties.csv")

if (!file.exists(tcad_file)) {
  stop(
    "TCAD data file not found: ", tcad_file, "\n",
    "Download the property export from https://www.traviscad.org/datadownloads/ ",
    "and save it as data/tcad_properties.csv"
  )
}

tcad_raw <- read_csv(tcad_file, show_col_types = FALSE)

print_progress(paste0("Loaded ", nrow(tcad_raw), " raw TCAD records"))

# Pull latitude and longitude into a dedicated data frame for inspection and
# downstream spatial conversion.  Rows missing either coordinate are dropped
# here so that st_as_sf() never receives NA geometry values.
tcad_coords <- tcad_raw %>%
  select(prop_id, owner_name, latitude, longitude,
         any_of(c("prop_type_cd", "appraised_val", "land_val",
                   "improvement_val", "legal_area"))) %>%
  filter(!is.na(latitude), !is.na(longitude))

print_progress(paste0(
  nrow(tcad_coords), " of ", nrow(tcad_raw),
  " records retained after removing missing coordinates"
))

# Convert to sf spatial object (WGS84 / EPSG:4326)
# remove = FALSE keeps the latitude/longitude columns alongside the geometry
tcad_sf <- tcad_coords %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)

################################################################################
# Step 2: Fetch City of Austin boundaries via tigris
################################################################################

print_progress("Fetching City of Austin boundaries from Census TIGER/Line (tigris)...")

austin_boundary <- tigris::places(state = "TX", year = 2021) %>%
  filter(NAME == "Austin") %>%
  st_transform(4326)

print_progress(paste0(
  "Austin boundary loaded. Area: ",
  round(as.numeric(st_area(austin_boundary)) / 1e6, 2), " km²"
))

################################################################################
# Step 3: Filter parcels within Austin city limits
################################################################################

print_progress("Filtering parcels to those within Austin city limits...")

# st_filter with st_within keeps only points whose geometry falls entirely
# inside (not merely touching) the Austin boundary polygon.
tcad_austin <- tcad_sf %>%
  st_filter(austin_boundary, .predicate = st_within)

print_progress(paste0(
  "Retained ", nrow(tcad_austin), " of ", nrow(tcad_sf),
  " parcels that fall within the City of Austin"
))

################################################################################
# Step 4: Identify likely corporate-owned parcels
################################################################################

print_progress("Identifying likely corporate-owned parcels...")

# Patterns commonly associated with non-individual (corporate/institutional)
# ownership.  Matching is performed on an upper-cased copy of the owner name
# so that the patterns are case-insensitive.
corporate_patterns <- paste(c(
  "\\bLLC\\b", "\\bL\\.L\\.C\\b",
  "\\bLP\\b",  "\\bL\\.P\\b",
  "\\bINC\\b", "\\bINCORPORATED\\b",
  "\\bCORP\\b", "\\bCORPORATION\\b",
  "\\bTRUST\\b", "\\bTRUSTEE\\b",
  "\\bPARTNERS\\b", "\\bPARTNERSHIP\\b",
  "\\bHOLDINGS\\b",
  "\\bINVESTMENT\\b", "\\bINVESTMENTS\\b",
  "\\bCOMPANY\\b",
  "\\bMANAGEMENT\\b", "\\bMGMT\\b",
  "\\bPROPERTIES\\b", "\\bPROPERTY\\b",
  "\\bREALTY\\b", "\\bREAL ESTATE\\b",
  "\\bENTERPRISES\\b", "\\bENTERPRISE\\b"
), collapse = "|")

corporate_parcels <- tcad_austin %>%
  mutate(
    owner_name_upper  = toupper(owner_name),
    likely_corporate  = str_detect(owner_name_upper, corporate_patterns)
  ) %>%
  filter(likely_corporate) %>%
  select(-owner_name_upper)

print_progress(paste0(
  "Identified ", nrow(corporate_parcels),
  " likely corporate-owned parcels within Austin city limits"
))

################################################################################
# Step 5: Summary statistics
################################################################################

cat("\nSummary of likely corporate-owned parcels:\n")
cat(paste0("  Total parcels within Austin:          ", nrow(tcad_austin), "\n"))
cat(paste0("  Likely corporate-owned:               ", nrow(corporate_parcels), "\n"))
cat(paste0("  Share corporate-owned:                ",
           round(nrow(corporate_parcels) / nrow(tcad_austin) * 100, 1), "%\n"))

if ("appraised_val" %in% names(corporate_parcels)) {
  cat(paste0("  Median appraised value (corporate):   $",
             format(round(median(corporate_parcels$appraised_val, na.rm = TRUE)),
                    big.mark = ","), "\n"))
}

################################################################################
# Step 6: Save output
################################################################################

print_progress("Saving results...")

save_output(
  corporate_parcels,
  file.path(OUTPUT_DIR, "corporate_parcels.rds"),
  "likely corporate-owned parcels (sf)"
)

corporate_parcels %>%
  st_drop_geometry() %>%
  write_csv(file.path(OUTPUT_DIR, "corporate_parcels.csv"))

print_progress(paste0(
  "CSV saved to: ", file.path(OUTPUT_DIR, "corporate_parcels.csv")
))

################################################################################
# Summary
################################################################################

print_header("CORPORATE OWNERSHIP ANALYSIS COMPLETE")
cat("✓ Raw TCAD data loaded and latitude/longitude extracted\n")
cat("✓ City of Austin boundary retrieved via tigris\n")
cat("✓ Parcels filtered to those within Austin city limits\n")
cat("✓ Likely corporate-owned parcels identified by owner name patterns\n")
cat(paste0("✓ Results saved to: ", OUTPUT_DIR, "/corporate_parcels.*\n"))
