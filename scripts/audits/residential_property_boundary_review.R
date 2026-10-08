# Diagnose the Caliza/Nexus reference points without changing production geography.
suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(readr)
})

out <- here::here("output", "residential_property_batch2", "boundary_review")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
grid <- readRDS(here::here("output", "hex_grid.rds")) |> st_transform(3083)
grid_union <- st_union(grid)
city <- st_read(here::here("data", "BOUNDARIES_jurisdictions_20260429.geojson"),
                quiet = TRUE) |>
  filter(city_name == "CITY OF AUSTIN", jurisdiction_type == "FULL") |>
  st_transform(3083) |> st_make_valid() |> st_union()

wcad <- readRDS(here::here("data", "raw_parcels", "williamson", "wcad_parcels.rds"))
caliza <- wcad[wcad$parcelid == "R500219" & !is.na(wcad$parcelid), ]
stopifnot(nrow(caliza) == 1L)
# Replicate the supplement's actual point-on-surface operation and projection.
caliza_point <- caliza |> st_transform(3857) |> st_make_valid() |>
  st_point_on_surface() |> suppressWarnings() |> st_transform(4326)
parcels <- readRDS(here::here("output", "eviction_low_unit_audit",
                            "nearby_county_parcels.rds"))
nexus <- parcels[parcels$parcel_id == "911866", ]
refs <- read_csv(here::here("output", "property_geography",
                           "residential_unit_references.csv"), show_col_types = FALSE)
nexus_ref <- refs |> filter(parcel_id == "911866")
stopifnot(nrow(nexus) == 1L, nrow(nexus_ref) == 1L)
nexus_point <- st_as_sf(nexus_ref, coords = c("lon", "lat"), crs = 4326)

properties <- list(
  Caliza = list(id = "WILLIAMSON:R500219", parcel = caliza, point = caliza_point),
  Nexus = list(id = "911866", parcel = nexus, point = nexus_point)
)
results <- lapply(names(properties), function(name) {
  x <- properties[[name]]
  parcel <- st_geometry(x$parcel) |> st_transform(3083) |> st_make_valid() |> st_union()
  point <- st_geometry(x$point) |> st_transform(3083)
  coords <- st_coordinates(st_transform(point, 4326))[1, ]
  h3 <- h3jsr::point_to_cell(st_as_sf(st_transform(point, 4326)), res = 9)
  center <- h3jsr::cell_to_point(h3, simple = FALSE) |> st_transform(3083)
  hex <- h3jsr::cell_to_polygon(h3, simple = FALSE) |> st_transform(3083)
  overlaps <- suppressWarnings(st_intersection(grid, parcel)) |>
    mutate(overlap_sq_m = as.numeric(st_area(geometry))) |>
    st_drop_geometry() |> select(hex_id, overlap_sq_m) |>
    arrange(desc(overlap_sq_m), hex_id)
  write_csv(overlaps, file.path(out, paste0(tolower(name), "_grid_overlap.csv")))
  parcel_area <- as.numeric(st_area(parcel))
  data.frame(
    property = name, parcel_id = x$id, parcel_area_sq_m = parcel_area,
    grid_area_fraction = sum(overlaps$overlap_sq_m) / parcel_area,
    city_full_area_fraction = as.numeric(sum(st_area(st_intersection(parcel, city)))) / parcel_area,
    reference_lon = coords[[1]], reference_lat = coords[[2]],
    reference_inside_parcel = lengths(st_intersects(point, parcel)) > 0L,
    reference_inside_city = lengths(st_intersects(point, city)) > 0L,
    reference_grid_cells = paste(grid$hex_id[unlist(st_intersects(point, grid))], collapse = ";"),
    distance_to_grid_m = as.numeric(st_distance(point, grid_union)),
    reference_h3 = h3,
    h3_center_inside_current_city = lengths(st_intersects(center, city)) > 0L,
    h3_city_area_fraction = as.numeric(sum(st_area(suppressWarnings(st_intersection(hex, city))))) /
      as.numeric(st_area(hex)),
    largest_existing_grid_overlap = overlaps$hex_id[[1]],
    production_applied = FALSE
  )
})
summary <- bind_rows(results)
stopifnot(all(summary$reference_inside_parcel), all(summary$reference_inside_city))
write_csv(summary, file.path(out, "property_boundary_summary.csv"))
print(summary)
