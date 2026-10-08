# Review/staging only: no writes to promoted units, filing assignments or models.
suppressPackageStartupMessages({library(sf); library(dplyr)})
root <- "tmp/oak_ranch_20261007"
h <- read.csv(file.path(root, "home_inventory_review.csv"), colClasses = c(parcel_id = "character"))
p <- readRDS("output/property_geography/county_polygons.rds")$polygons
p <- st_transform(p[p$source_county == "Travis" & p$polygon_parcel_id %in% c("464309", "909849"), ], 4326)
stopifnot(nrow(p) == 2L)
st_write(p, file.path(root, "park_parent_polygons.geojson"), delete_dsn = TRUE, quiet = TRUE)
pts <- st_as_sf(h[!is.na(h$longitude), ], coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)
ix <- st_intersects(pts, p)
pts$spatial_parent_id <- vapply(ix, function(i) paste(p$polygon_parcel_id[i], collapse = ";"), character(1))
g <- st_transform(readRDS("output/hex_grid.rds"), 4326)
ix <- st_intersects(pts, g)
pts$point_hex_id <- vapply(ix, function(i) if(length(i) == 1L) as.integer(g$hex_id[i]) else NA_integer_, integer(1))
pts$spatial_ready <- as.logical(pts$ready_before_spatial_validation) & lengths(st_intersects(pts, p)) == 1L & !is.na(pts$point_hex_id)
promoted <- readRDS("output/residential_parcels_unit_promoted.rds")
pts$already_promoted <- pts$parcel_id %in% promoted$parcel_id
pts$existing_units <- promoted$units_calibrated_targeted[match(pts$parcel_id, promoted$parcel_id)]
pts$existing_units[!pts$already_promoted] <- 0
write.csv(st_drop_geometry(pts), file.path(root, "home_spatial_review.csv"), row.names = FALSE)
write.csv(st_drop_geometry(pts)[pts$spatial_ready, ], file.path(root, "ready_home_locations.csv"), row.names = FALSE)
st_write(pts, file.path(root, "home_point_candidates.geojson"), delete_dsn = TRUE, quiet = TRUE)
summary <- st_drop_geometry(pts) |> filter(spatial_ready) |>
  group_by(point_hex_id) |> summarise(candidate_home_units = n(),
    newly_recovered_units = sum(!already_promoted), relocated_existing_units = sum(existing_units), .groups="drop")
write.csv(summary, file.path(root, "ready_units_by_hex.csv"), row.names = FALSE)
cat(nrow(h), "source homes;", nrow(pts), "City point candidates;", sum(pts$spatial_ready),
    "unambiguous in-grid home references;", sum(pts$already_promoted), "already promoted.\n")
