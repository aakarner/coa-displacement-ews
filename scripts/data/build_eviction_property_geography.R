# Build a reusable, address-level residential property crosswalk from local
# county polygons and the exact operational unit reference points.
suppressPackageStartupMessages({library(sf);library(dplyr);library(readr);library(tidyr)})
source("R/pipeline.R")
source("R/eviction_property_geography.R")
source("R/reviewed_unit_properties.R")
unit_reviews <- read_reviewed_unit_properties()
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) %in% c(0L, 2L))
staging <- length(args) == 2L
root <- if (staging) args[1] else "output/property_geography"
unit_path <- if (staging) args[2] else "output/residential_parcels_unit_promoted.rds"
if (staging) stopifnot(startsWith(root, "tmp/"), startsWith(unit_path, "tmp/"))
dir.create(root, recursive = TRUE, showWarnings = FALSE)
grid <- readRDS("output/hex_grid.rds"); crs <- 3083
p <- readRDS(unit_path)
if (!"unit_review_property_group" %in% names(p)) p$unit_review_property_group <- NA_character_
p <- p %>%
  filter(is.finite(lon), is.finite(lat)) %>%
  transmute(parcel_id, source_county, situs_address, project_id = coalesce(unit_review_property_group, unit_model_project_id), lon, lat,
    operational_units = if_else(coalesce(unit_land_use_validation_excluded, FALSE), 0, coalesce(units_calibrated_targeted, 0)))
pp <- st_as_sf(p, coords = c("lon", "lat"), crs = 4326, remove = FALSE)
h <- st_within(st_transform(pp, st_crs(grid)), grid)
stopifnot(max(lengths(h)) <= 1L)
p$unit_hex_id <- grid$hex_id[vapply(h, function(x) if (length(x)) x[1] else NA_integer_, integer(1))]
canonical <- st_drop_geometry(readRDS("output/corporate_ownership_by_hex.rds"))
totals <- p %>% filter(!is.na(unit_hex_id)) %>% group_by(unit_hex_id) %>% summarise(units = sum(operational_units), .groups = "drop")
if (!staging) stopifnot(max(abs(coalesce(totals$units[match(canonical$hex_id, totals$unit_hex_id)], 0) - canonical$residential_units)) < 1e-6)
write_csv(p, file.path(root, "residential_unit_references.csv"))
pp <- st_transform(st_as_sf(p, coords = c("lon", "lat"), crs = 4326, remove = FALSE), crs)
registries <- c(Travis = "output/eviction_addresses_geocoded.csv",
  Williamson = "output/williamson_eviction_addresses_geocoded_with_arcgis.csv")
g <- bind_rows(lapply(names(registries), function(county) read_csv(registries[county],
  col_types = cols(.default = col_skip(), address_for_geocoding = "c", status = "c", score = "d",
    longitude = "d", latitude = "d", st_addr = "c", addr_type = "c"), show_col_types = FALSE) %>%
    mutate(source_county = county))) %>%
  filter(status %in% c("M", "T"), score >= 90, is.finite(longitude), is.finite(latitude)) %>%
  mutate(location_id = paste(source_county, format(longitude, digits = 14), format(latitude, digits = 14), sep = ":"))
locations <- g %>% distinct(location_id, source_county, longitude, latitude)
gp <- st_transform(st_as_sf(locations, coords = c("longitude", "latitude"), crs = 4326), crs)
polygon_sources <- c("data/raw_parcels/travis/Parcel_poly.zip", "data/raw_parcels/williamson/wcad_parcels.rds")
fingerprint <- vapply(c(polygon_sources, "output/hex_grid.rds"), digest::digest, character(1), file = TRUE, algo = "sha256")
cache_path <- file.path(root, "county_polygons.rds")
read_cache <- if (staging && !file.exists(cache_path)) "output/property_geography/county_polygons.rds" else cache_path
cached <- if (file.exists(read_cache)) readRDS(read_cache) else NULL
if (!is.null(cached) && identical(cached$fingerprint, fingerprint)) {
  polys <- cached$polygons
} else {
  message("Loading county parcel polygons and validating local geometry...")
  shape_dir <- tempfile("ews_parcels_"); dir.create(shape_dir)
  unzip(polygon_sources[1], exdir = shape_dir)
  tc <- st_read(file.path(shape_dir, "Parcel_poly.shp"),
    query = "SELECT PROP_ID FROM Parcel_poly", quiet = TRUE) %>%
    transmute(polygon_parcel_id = as.character(PROP_ID), source_county = "Travis") %>% st_transform(crs)
  wc <- readRDS(polygon_sources[2]) %>%
    transmute(polygon_parcel_id = paste0("WILLIAMSON:", trimws(parcelid)), source_county = "Williamson") %>% st_transform(crs)
  polys <- bind_rows(tc, wc)
  bbox <- st_as_sfc(st_bbox(st_transform(grid, crs)))
  polys <- polys[lengths(st_intersects(polys, bbox)) > 0,] %>% st_make_valid() %>% mutate(polygon_id = row_number())
  saveRDS(list(fingerprint = fingerprint, polygons = polys), cache_path)
  unlink(shape_dir, recursive = TRUE)
}
message("Matching ", nrow(locations), " distinct filing coordinates to residential parcels...")
hits <- st_intersects(gp, polys)
used <- sort(unique(unlist(hits)))
unit_rows <- which(p$operational_units > 0 & !is.na(p$unit_hex_id))
inside <- st_intersects(polys[used,], pp[unit_rows,])
accounts <- bind_rows(lapply(seq_along(used), function(i) {
  j <- unit_rows[inside[[i]]]
  if (!length(j)) return(NULL)
  data.frame(polygon_id = polys$polygon_id[used[i]], parcel_id = p$parcel_id[j],
    unit_hex_id = p$unit_hex_id[j], project_id = p$project_id[j], unit_row = j)
}))
geometry_aliases <- read_csv("output/williamson_residential_geometry_links.csv",
  col_types = cols(.default = "c"), show_col_types = FALSE) %>%
  transmute(polygon_parcel_id = paste0("WILLIAMSON:", geometry_source_parcel_id),
    parcel_id = paste0("WILLIAMSON:", certified_quick_ref_id)) %>%
  inner_join(st_drop_geometry(polys) %>% select(polygon_id, polygon_parcel_id), by = "polygon_parcel_id",
    relationship = "many-to-many") %>%
  inner_join(p %>% mutate(unit_row = row_number()) %>%
    filter(operational_units > 0, !is.na(unit_hex_id)) %>% select(parcel_id, unit_hex_id, project_id, unit_row),
    by = "parcel_id", relationship = "many-to-one") %>% select(all_of(names(accounts)))
accounts <- bind_rows(accounts, geometry_aliases) %>% distinct()
# Parent/condominium polygons can represent multiple accounts. They qualify
# only if their occupied accounts represent one project at one unit hex.
anchor <- accounts %>% group_by(polygon_id) %>% summarise(
  account_count = n_distinct(parcel_id), project_count = n_distinct(project_id),
  hex_count = n_distinct(unit_hex_id), property_hex_id = first(unit_hex_id),
  unit_row = first(unit_row), project_id = first(project_id), parcel_id = first(parcel_id), .groups = "drop") %>%
  filter(project_count == 1L, hex_count == 1L, !is.na(project_id)) %>%
  mutate(property_id = if_else(account_count == 1L, paste0("parcel:", parcel_id), project_id))
links <- bind_rows(lapply(seq_along(hits), function(i) data.frame(location_row = i,
  polygon_id = if (length(hits[[i]])) polys$polygon_id[hits[[i]]] else NA_integer_))) %>%
  left_join(st_drop_geometry(polys) %>% select(polygon_id, polygon_parcel_id, polygon_county = source_county), by = "polygon_id") %>%
  left_join(anchor, by = "polygon_id") %>%
  mutate(same_county = polygon_county == locations$source_county[location_row])
verified <- links %>% filter(same_county, !is.na(property_id)) %>% group_by(location_row) %>%
  filter(n_distinct(property_id) == 1L, n_distinct(property_hex_id) == 1L) %>% slice_head(n = 1L) %>% ungroup()
v <- locations %>% mutate(location_row = row_number(), containing_parcels = lengths(hits)) %>%
  left_join(verified %>% select(location_row, polygon_parcel_id, property_id, property_hex_id, unit_row, account_count), by = "location_row") %>%
  mutate(property_link_status = case_when(!is.na(property_id) ~ "verified",
    containing_parcels == 0L ~ "outside_parcel_polygon_review",
    TRUE ~ "unresolved_residential_account_review"), reference_distance_m = NA_real_)
k <- which(!is.na(v$unit_row))
v$reference_distance_m[k] <- as.numeric(st_distance(gp[k,], pp[v$unit_row[k],], by_element = TRUE))
# Preserve facility type and denominator quality, including explicitly adopted
# provisional assumptions. These flags never suppress cases or cells.
facilities <- bind_rows(unit_reviews$facilities)
facility_links <- links %>% filter(same_county) %>%
  inner_join(facilities, by = c("polygon_parcel_id", "polygon_county" = "source_county")) %>%
  transmute(location_row, property_facility_type = facility_type,
    property_denominator_status = denominator_status) %>% distinct()
stopifnot(!anyDuplicated(facility_links$location_row))
v <- left_join(v, facility_links, by = "location_row", relationship = "one-to-one")
out <- g %>% left_join(v %>% select(location_id, polygon_parcel_id, property_id, property_hex_id,
  property_link_status, reference_distance_m, property_facility_type, property_denominator_status), by = "location_id") %>%
  mutate(property_link_status = if_else(!addr_type %in% c("PointAddress", "Subaddress", "APT"),
    "geocode_precision_review", property_link_status), property_geography_rule = eviction_property_geography_version())
stopifnot(!anyDuplicated(out[c("source_county", "address_for_geocoding")]))
address_reviews <- apply_reviewed_property_addresses(out, p)
out <- address_reviews$rows
reviewed <- compile_property_case_reviews(p, g)
attr(out, "case_reviews") <- reviewed$rows
write_csv(reviewed$rows, file.path(root, "eviction_case_property_reviews.csv"))
saveRDS(out, file.path(root, "eviction_address_properties.rds"))
write_csv(out, file.path(root, "eviction_address_properties.csv"))
write_csv(out %>% count(source_county, property_link_status, name = "addresses"), file.path(root, "property_link_summary.csv"))
write_csv(accounts, file.path(root, "polygon_residential_accounts.csv"))
jsonlite::write_json(list(staging = staging, rule = eviction_property_geography_version(), inputs = build_file_manifest(c(
  polygon_sources, registries, unit_path, "output/corporate_ownership_by_hex.rds",
  "output/hex_grid.rds", "output/williamson_residential_geometry_links.csv", "R/eviction_property_geography.R",
  "scripts/data/build_eviction_property_geography.R", "R/eviction_property_reviews.R",
  reviewed$input_paths, address_reviews$input_paths, unit_reviews$input_paths), hash_files = TRUE, require_all = TRUE),
  outputs = build_file_manifest(file.path(root, c("eviction_address_properties.rds",
    "eviction_case_property_reviews.csv")), hash_files = TRUE, require_all = TRUE)),
  file.path(root, "property_geography_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
print(out %>% count(source_county, property_link_status))
