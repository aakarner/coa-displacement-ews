# Explicit promotion of the October 7 second batch; no raw-source edits.
suppressPackageStartupMessages({library(sf); library(dplyr); library(data.table)})
root <- "data/reviewed_unit_properties/batch2_20261007"
config <- "config/residential_unit_property_reviews.json"
review <- jsonlite::read_json(config)
if (any(vapply(review$projects, function(x) grepl("batch2", x$review_id), logical(1))))
  stop("Batch 2 already prepared; do not overwrite pinned evidence.")
dir.create(root, recursive = TRUE, showWarnings = FALSE)
grid <- readRDS("output/hex_grid.rds") |> st_transform(3083)
city <- st_read("data/BOUNDARIES_jurisdictions_20260429.geojson", quiet = TRUE) |>
  filter(city_name == "CITY OF AUSTIN", jurisdiction_type == "FULL") |>
  st_transform(3083) |> st_make_valid() |> summarise()
polys <- readRDS("output/property_geography/county_polygons.rds")$polygons |>
  filter(polygon_parcel_id %in% c("WILLIAMSON:R500219", "911866", "498141", "859326")) |>
  st_transform(3083)
stopifnot(nrow(polys) == 3L,
  setequal(polys$polygon_parcel_id, c("WILLIAMSON:R500219", "911866", "498141")))
footprint_file <- file.path(root, "reviewed_footprints.geojson")
city_file <- file.path(root, "reviewed_city_boundary.geojson")
st_write(st_transform(polys, 4326), footprint_file, quiet = TRUE)
st_write(st_transform(city, 4326), city_file, quiet = TRUE)

# Choose the largest in-grid property portion, independent of filing counts.
# This point represents the property; it does not identify an individual home.
boundary_point <- function(id, expected_hex) {
  p <- polys |> filter(polygon_parcel_id == id)
  city_share <- as.numeric(sum(st_area(st_intersection(p, city))) / st_area(p))
  stopifnot(city_share >= 0.999)
  pieces <- suppressWarnings(st_intersection(grid, st_geometry(p))) |>
    mutate(overlap_area = as.numeric(st_area(geometry))) |>
    arrange(desc(overlap_area), hex_id)
  stopifnot(pieces$hex_id[[1]] == expected_hex)
  within_city <- st_intersection(st_geometry(pieces[1, ]), st_geometry(city))
  point <- st_point_on_surface(within_city) |> st_transform(4326)
  xy <- st_coordinates(point)
  list(lon = xy[1, 1], lat = xy[1, 2], polygon_parcel_id = id,
    expected_hex_id = expected_hex, evidence_path = footprint_file,
    method = "reviewed_largest_in_grid_property_portion_point_on_surface",
    coord_source = "reviewed_boundary_property_reference",
    city_boundary_evidence_path = city_file, minimum_city_overlap_fraction = 0.999)
}
caliza_point <- boundary_point("WILLIAMSON:R500219", 3261L)
nexus_point <- boundary_point("911866", 6929L)
refs <- fread("output/property_geography/residential_unit_references.csv", colClasses = "character")
nexus_ref <- refs[parcel_id == "911866"]
stopifnot(nrow(nexus_ref) == 1L, as.numeric(nexus_ref$operational_units) == 294)
nexus_point$expected_original_lon <- as.numeric(nexus_ref$lon)
nexus_point$expected_original_lat <- as.numeric(nexus_ref$lat)

ids <- c("498141", "859326", "911866", "464309", "909849")
attributes <- list()
for (spec in list(c("property_profile", "propertyProf_pID"),
                 c("property_characteristics", "propertyChar_pID"),
                 c("situses", "situs_pID"), c("coords", "coord_pID"))) {
  x <- fread(paste0("../landlord-mapper/output/", spec[1], ".csv"), colClasses = "character")
  attributes[[spec[1]]] <- as.data.frame(x[x[[spec[2]]] %in% ids, ])
}
jsonlite::write_json(attributes, file.path(root, "county_account_attributes.json"),
                    pretty = TRUE, dataframe = "rows", na = "null")
cert <- fread("output/residential_property_batch2/caliza_certified_account.csv", colClasses = "character")
stopifnot(nrow(cert) == 1L, cert$PropertyStatusDesc == "Active", cert$DBA == "CALIZA",
          as.numeric(cert$TotalSqFtLivingArea) == 341389)
file.copy("output/residential_property_batch2/caliza_certified_account.csv", root)
raw <- readRDS("data/raw_parcels/williamson/wcad_parcels.rds")
raw <- raw[!is.na(raw$parcelid) & raw$parcelid == "R500219", ]
stopifnot(nrow(raw) == 1L, raw$ownernme1 == "CALIZA PROPERTY LP")
raw_point <- suppressWarnings(st_point_on_surface(st_make_valid(st_transform(raw, 3857)))) |>
  st_transform(4326) |> st_coordinates()
fwrite(st_drop_geometry(raw), file.path(root, "caliza_raw_parcel_attributes.csv"))
supplement <- list(parcel_id = "WILLIAMSON:R500219", source_county = "Williamson",
  situs_address = trimws(cert$SitusAddress), situs_city = cert$City, situs_state = cert$State,
  situs_zip = cert$Zip, propertyProf_imprvStateCd = cert$PropertyTypeDesc,
  improvement_sqft = as.numeric(cert$TotalSqFtLivingArea),
  land_sqft = as.numeric(cert$Acres) * 43560, units_raw = 0,
  lon = raw_point[1, 1], lat = raw_point[1, 2],
  is_residential = TRUE, is_owner_occupied = FALSE, is_corporate_owned = TRUE,
  owner_names = raw$ownernme1, n_owner_rows = "1", parcel_count = 1,
  corporate_parcel_count = 1, corporate_improvement_sqft = as.numeric(cert$TotalSqFtLivingArea),
  county_unit_exclude_from_unit_universe = FALSE, coord_source = "wcad_point_on_surface_outside_grid")
project <- function(id, name, county, units, basis, geometry = NULL, supplement = NULL) {
  list(review_id = paste0("batch2_20261007:", id), project_id = paste0("project:", id),
    name = name, source_county = county, parcel_ids = list(id), units = units,
    basis = basis, evidence_paths = as.list(c(file.path(root, "public_sources.json"),
      footprint_file, city_file, file.path(root, "county_account_attributes.json"))),
    geometry = geometry, supplement = supplement)
}
caliza <- project("WILLIAMSON:R500219", "Caliza", "Williamson", 270L,
  "JLL April 19, 2023 sale release identifies 270 apartments at 12638 Ridgeline. Active WCAD C3 account has 341389 living square feet and DBA CALIZA. Its original point-on-surface lies outside the grid. Explicitly reviewed reference uses the largest in-grid property portion; over 99.9% of the parcel is in full-purpose Austin.",
  caliza_point, supplement)
caliza$omission_reason <- "residential_property_reference_outside_fixed_grid"
caliza$evidence_paths <- c(caliza$evidence_paths, as.list(file.path(root,
  c("caliza_certified_account.csv", "caliza_raw_parcel_attributes.csv"))))
new <- list(caliza,
  project("911866", "Nexus at Goodnight Ranch", "Travis", 294L,
    "Retain the existing county-reported 294 units. Explicitly reviewed reference uses the largest in-grid property portion because the county coordinate falls outside the fixed grid; over 99.9% of the parcel is in full-purpose Austin.", nexus_point),
  project("498141", "Bridge at Canyon Creek", "Travis", 332L,
    "Austin Apartment Association property directory reports 332 units at 9009 FM 620; county DBA and court review corroborate the property. Replace the modeled count once; retain the existing property reference."),
  project("859326", "Ocotillo", "Travis", 308L,
    "Ardent developer project list reports 308 units, completed May 2017; project page identifies 8000 US 290 Highway W. County DBA OCOTILLO and situs corroborate the property. Replace the modeled count once; retain the existing property reference."))
review$projects <- c(review$projects, new)
files <- sort(unique(unlist(lapply(new, `[[`, "evidence_paths"))))
review$evidence <- c(review$evidence, lapply(files, function(p)
  list(path = p, sha256 = digest::digest(file = p, algo = "sha256"))))
jsonlite::write_json(review, config, pretty = TRUE, auto_unbox = TRUE, null = "null", digits = NA)
cat("Prepared four project reviews; two boundary references and one omitted account.\n")
