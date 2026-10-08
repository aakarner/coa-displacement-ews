# Freeze this incremental batch-2 regression at its preserved endpoint.
# Later precision/home-inventory integration has separate staging/output checks.
historical_output <- function(path) {
  p <- file.path("output/residential_followup_20261007/before", path)
  if (!file.exists(p)) stop("Missing batch-2 endpoint: ", path)
  p
}
# Independent checks for the four-property repair and unchanged exclusions.
suppressPackageStartupMessages({library(dplyr); library(sf); library(readr)})
source("R/reviewed_unit_properties.R")
before <- "output/residential_property_batch2/before"
old <- readRDS(file.path(before, "output/residential_parcels_unit_promoted.rds"))
p <- readRDS(historical_output("output/residential_parcels_unit_promoted.rds"))
reviews <- read_reviewed_unit_properties(historical_output("config/residential_unit_property_reviews.json"))
validate_reviewed_unit_surface(p, reviews)
ids <- c("WILLIAMSON:R500219", "911866", "498141", "859326")
counts <- c(270, 294, 332, 308)
stopifnot(nrow(p) == nrow(old) + 1L, !anyDuplicated(p$parcel_id),
  setequal(setdiff(p$parcel_id, old$parcel_id), "WILLIAMSON:R500219"),
  identical(as.numeric(p$units_calibrated_targeted[match(ids, p$parcel_id)]), counts))
untouched <- !old$parcel_id %in% ids
j <- match(old$parcel_id[untouched], p$parcel_id)
stopifnot(identical(old$units_calibrated_targeted[untouched], p$units_calibrated_targeted[j]),
  identical(old$lon[untouched], p$lon[j]), identical(old$lat[untouched], p$lat[j]))
# Exactly two reviewed references change; existing in-grid references persist.
existing <- match(old$parcel_id, p$parcel_id)
moved <- old$parcel_id[abs(old$lon - p$lon[existing]) > 1e-8 |
                       abs(old$lat - p$lat[existing]) > 1e-8]
stopifnot(setequal(moved, "911866"))
caliza <- p[p$parcel_id == "WILLIAMSON:R500219", ]
stopifnot(caliza$unit_review_added_account,
  caliza$propertyProf_imprvStateCd == "C3", caliza$units_raw == 0,
  abs(caliza$unit_review_original_lon - (-97.81012138536475)) < 1e-8,
  abs(caliza$unit_review_original_lat - 30.466055950074562) < 1e-8)
refs <- read_csv(historical_output("output/property_geography/residential_unit_references.csv"), show_col_types = FALSE)
stopifnot(identical(as.integer(refs$unit_hex_id[match(ids, refs$parcel_id)]),
                   c(3261L, 6929L, 3321L, 596L)))
grid <- readRDS("output/residential_cluster_rebuild_20261007/before/output/hex_grid.rds")
city <- st_read("data/BOUNDARIES_jurisdictions_20260429.geojson", quiet = TRUE) |>
  filter(city_name == "CITY OF AUSTIN", jurisdiction_type == "FULL") |>
  st_transform(3083) |> st_make_valid() |> st_union()
polys <- st_read("data/reviewed_unit_properties/batch2_20261007/reviewed_footprints.geojson", quiet = TRUE)
for (id in ids[1:2]) {
  a <- p[p$parcel_id == id, ]
  point <- st_as_sf(a, coords = c("lon", "lat"), crs = 4326)
  original <- st_as_sf(a, coords = c("unit_review_original_lon", "unit_review_original_lat"), crs = 4326)
  footprint <- polys[polys$polygon_parcel_id == id, ]
  stopifnot(lengths(st_within(point, footprint)) == 1L,
    lengths(st_within(st_transform(point, 3083), city)) == 1L,
    lengths(st_within(point, grid)) == 1L,
    lengths(st_intersects(original, grid)) == 0L)
}

bundle <- jsonlite::read_json("data/reviewed_eviction_properties/batch2_20261007/cases.json")
review_cases <- bind_rows(lapply(bundle$cases, function(r)
  data.frame(case_number = r$case_number, hex = r$expected_unit_hex_id)))
stopifnot(nrow(review_cases) == 31L,
  all(table(review_cases$hex)[c("3321", "3261", "6929", "596")] == c(18, 9, 3, 1)))
a <- readRDS(historical_output("output/part2/evictions/eviction_case_ledger.rds"))
b <- readRDS(file.path(before, "output/part2/evictions/eviction_case_ledger.rds"))
b <- b[match(a$case_number, b$case_number), ]
stopifnot(identical(a$case_number, b$case_number),
  identical(a$assignment_status, b$assignment_status),
  identical(a$original_assigned_hex_key, b$original_assigned_hex_key),
  identical(is.na(a$assigned_hex_key), is.na(b$assigned_hex_key)))
r <- a[match(review_cases$case_number, a$case_number), ]
stopifnot(identical(r$assigned_hex_key, as.character(review_cases$hex)),
  all(grepl("^batch2_20261007:", r$property_review_id)))
changed <- !is.na(a$assigned_hex_key) & a$assigned_hex_key != b$assigned_hex_key
stopifnot(all(a$case_number[changed] %in% review_cases$case_number |
  a$property_id[changed] %in% paste0("parcel:", ids[1:2])))
annual <- read_csv(historical_output("output/part3/eviction_property_assignment_ledger.csv"),
  col_types = cols(.default = col_guess(), case_number = "c", assigned_hex_key = "c"),
  show_col_types = FALSE)
stopifnot(all(a$assignment_status[!a$case_number %in% annual$case_number] ==
  "excluded_missing_valid_case_identifier"))
paired_shared <- a[a$case_number %in% annual$case_number, ]
shared <- annual[match(paired_shared$case_number, annual$case_number), ]
stopifnot(identical(paired_shared$case_number, shared$case_number),
  identical(paired_shared$assignment_status, shared$assignment_status),
  identical(paired_shared$assigned_hex_key, shared$assigned_hex_key))
queue <- read_csv("output/residential_property_batch2/case_review_queue.csv", show_col_types = FALSE)
mobile <- queue |> filter(hex_id %in% c(3767, 3769))
stopifnot(nrow(mobile) == 24L,
  identical(a$assigned_hex_key[match(mobile$case_number, a$case_number)], as.character(mobile$hex_id)))
m <- st_drop_geometry(readRDS(historical_output("output/part1/measurement/current_measurement.rds")))
remaining <- a |> filter(case_number %in% queue$case_number) |>
  mutate(hex_id = as.integer(assigned_hex_key)) |>
  left_join(select(m, hex_id, residential_units), by = "hex_id") |>
  filter(residential_units < 20)
stopifnot(nrow(remaining) == 24L, setequal(remaining$case_number, mobile$case_number))
snap <- jsonlite::read_json(file.path(before, "snapshot.json"), simplifyVector = TRUE)
for (i in seq_len(nrow(snap))) stopifnot(identical(snap$sha256[i],
  digest::digest(file = file.path(before, snap$path[i]), algo = "sha256")))
cat("Four project totals/references, 31 reviews, original exclusions, annual parity, 24 unresolved mobile-home filings, and immutable baseline pass.\n")
