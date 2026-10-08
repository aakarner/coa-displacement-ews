# One-time, explicit evidence packaging. Production reads the pinned config.
suppressPackageStartupMessages({library(sf); library(dplyr); library(data.table)})
root <- "data/reviewed_unit_properties/batch1_20261007"
config <- "config/residential_unit_property_reviews.json"
if (file.exists(config)) stop("Review config already exists; create a new version instead of overwriting evidence.")
dir.create(root, recursive = TRUE, showWarnings = FALSE)
polys <- readRDS("output/property_geography/county_polygons.rds")$polygons %>%
  filter(source_county == "Travis", polygon_parcel_id %in% c("737155", "774333", "878332", "291453")) %>%
  st_transform(4326)
st_write(polys, file.path(root, "reviewed_footprints.geojson"), quiet = TRUE, delete_dsn = TRUE)
map <- st_read(file.path(root, "domain_site_map.geojson"), quiet = TRUE) %>%
  filter(type == "UnitLabel") %>% st_transform(3083)
stopifnot(nrow(map) == 438L, !anyDuplicated(map$unit_id))
footprint <- function(id, count, hex) {
  p <- st_transform(polys[polys$polygon_parcel_id == id, ], 3083)
  u <- map[lengths(st_within(map, p)) > 0, ]
  stopifnot(nrow(u) == count)
  # The mean can fall in a courtyard/parcel hole. Use the mapped home nearest
  # that mean so the reference is demonstrably inside the residential footprint.
  coords <- st_coordinates(u)
  j <- which.min(rowSums(sweep(coords, 2, colMeans(coords), "-")^2))
  point <- st_geometry(u[j, ]) %>% st_transform(4326)
  xy <- st_coordinates(point)
  list(lon = xy[1, 1], lat = xy[1, 2], polygon_parcel_id = id,
    expected_hex_id = hex, evidence_path = file.path(root, "reviewed_footprints.geojson"),
    method = "mapped_home_nearest_mean_projected_coordinate")
}
south <- footprint("774333", 412L, 3485L)
south$expected_original_lon <- -97.73014171159582
south$expected_original_lat <- 30.391223491348946
north <- footprint("737155", 26L, 3459L)
ids <- c("774341", "774342", "774343", "774344", "774412", "291453")
source_rows <- list()
for (spec in list(c("property_profile", "propertyProf_pID"),
    c("property_characteristics", "propertyChar_pID"), c("coords", "coord_pID"),
    c("situses", "situs_pID"), c("links", "link_pID"))) {
  path <- file.path("../landlord-mapper/output", paste0(spec[1], ".csv"))
  x <- fread(path, colClasses = "character", showProgress = FALSE)
  source_rows[[spec[1]]] <- as.data.frame(x[x[[spec[2]]] %in% ids, ])
}
owners <- fread("../landlord-mapper/output/owners.csv", colClasses = "character", showProgress = FALSE,
  select = c("owner_pID", "owner_name", "owner_ownerPct"))
source_rows$owners <- as.data.frame(owners[owner_pID %in% ids, ])
jsonlite::write_json(source_rows, file.path(root, "county_account_attributes.json"),
  pretty = TRUE, na = "null", dataframe = "rows")
attr <- source_rows$property_profile %>% filter(propertyProf_pID == "774412")
owner <- source_rows$owners %>% filter(owner_pID == "774412")
stopifnot(nrow(attr) == 1L, nrow(owner) == 1L,
  owner$owner_name == "LPF VILLAGES DOMAIN LLC", attr$propertyProf_imprvStateCd == "B1")
supplement <- list(parcel_id = "774412", source_county = "Travis",
  situs_address = "CENTURY OAKS TER P AUSTIN TX 78759", situs_city = "AUSTIN", situs_state = "TX", situs_zip = "78759",
  propertyProf_imprvStateCd = "B1", propertyProf_landStateCd = "B1",
  propertyProf_imprvActualYearBuilt = attr$propertyProf_imprvActualYearBuilt,
  improvement_sqft = as.numeric(attr$propertyProf_imprvTotalArea),
  land_sqft = as.numeric(attr$propertyProf_landSizeSqft), units_raw = 0,
  is_residential = TRUE, is_owner_occupied = FALSE, is_corporate_owned = TRUE,
  has_financialized_owner = TRUE, owner_names = owner$owner_name,
  n_owner_rows = "1", parcel_count = 1, corporate_parcel_count = 1,
  corporate_improvement_sqft = as.numeric(attr$propertyProf_imprvTotalArea),
  county_unit_exclude_from_unit_universe = FALSE, coord_source = "reviewed_operator_unit_map_reference")
project <- function(id, name, members, units, basis, evidence, geometry = NULL, supplement = NULL) {
  list(review_id = paste0("units_20261007_", id), project_id = paste0("project:", id),
    name = name, source_county = "Travis", parcel_ids = as.list(members), units = units,
    basis = basis, evidence_paths = as.list(file.path(root, evidence)), geometry = geometry, supplement = supplement)
}
projects <- list(
  project("878332", "Bell Southpark Springs", "878332", 400L,
    "400 distinct homes on the operator's September 25, 2025 phase map within parcel 878332; replace the project estimate once.",
    c("bell_site_map.geojson", "reviewed_footprints.geojson")),
  project("533185", "Bridge at Monarch Bluffs", c("533185", "975264"), 330L,
    "Austin Energy December 28, 2023 final-inspection fact sheet, page 5: 330 rentable units at 8515 S IH 35; keep the land account at zero.",
    "monarch_austin_energy_2023.pdf"),
  project("513751", "Bridge at Asher", "513751", 452L,
    "HACA April 18, 2019 acquisition record reports 452 apartments; corroborated by its 2019-2020 annual report and the current apartment association listing. The former 449 is a modeled fallback, not a conflicting direct count.",
    c("asher_haca.web.json", "asher_association.html")),
  project("774341", "Villages at the Domain southern footprint", as.character(774341:774344), 412L,
    "Four county residential accounts link to parent 774333. Operator April 21, 2025 map has 412 southern homes; move existing housing to that footprint and replace its combined estimate. City inventory 436 versus operator development total 438 remains a documented two-unit discrepancy.",
    c("domain_site_map.geojson", "county_account_attributes.json", "reviewed_footprints.geojson"), south),
  project("774412", "Villages at the Domain Building P", "774412", 26L,
    "County DBA and owner identify the omitted Building P account. City October 23, 2009 compliance report page 6 identifies Building P as 26 homes; the operator map independently locates 26 northern homes. Reviewed physical association to footprint 737155; the county's original link to unmapped 866401 is preserved, not rewritten.",
    c("domain_site_map.geojson", "domain_compliance.web.json", "county_account_attributes.json", "reviewed_footprints.geojson"), north, supplement)
)
files <- sort(unique(c(unlist(lapply(projects, `[[`, "evidence_paths")), file.path(root, "ben_white_reentry.web.json"))))
evidence <- lapply(files, function(f) list(path = f, sha256 = digest::digest(file = f, algo = "sha256")))
out <- list(schema_version = 1L, review_date = "2026-10-07", analytical_cutoff = "2026-04-01",
  projects = projects, evidence = evidence,
  facilities = list(list(review_id = "facility_20261007_291453", source_county = "Travis",
    polygon_parcel_id = "291453", facility_type = "transitional_residential_facility",
    denominator_status = "independent_housing_unit_count_unverified",
    basis = "Residential use supported; historical 100 beds and approximately 178 rooms are not a verified April 2026 housing-unit count. Retain filing counts and flag denominator uncertainty; do not suppress cells.")))
jsonlite::write_json(out, config, pretty = TRUE, auto_unbox = TRUE, null = "null", digits = NA)
cat("Prepared five reviewed project totals and one unresolved facility review.\n")
