# Geographic triage of supplied Hays case lists. This does not assign case premises
# or change analytical source coverage. Run after hays_evictions_prepare.py.
suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(readr)
  library(jsonlite)
})

registry_file <- "config/hays_eviction_property_groups.json"
boundary_file <- "data/BOUNDARIES_jurisdictions_20260429.geojson"
parcel_file <- "data/raw_parcels/hays/hays_parcels.gpkg"
state_parcel_file <- paste0("data/raw_parcels/hays/txgio_2025/fgdb/",
  "stratmap25-landparcels_48209_hays_202503.gdb")
state_archive <- "data/raw_parcels/hays/txgio_2025/stratmap25-landparcels_48209_lp.zip"
property_file <- paste0("data/raw_parcels/hays/property_export_unzipped/nested/",
  "2025-PROPERTY-DATA-EXPORT-FILE-PROPERTY-4.28.2026/PropertyDataExport1361499.txt")
owner_file <- paste0("data/raw_parcels/hays/property_export_unzipped/nested/",
  "2025-PROPERTY-DATA-EXPORT-FILE-OWNER-4.28.2026/PropertyDataExport1361500.txt")
read_chars <- function(path) read_csv(path, col_types = cols(.default = col_character()),
                                    na = character(), show_col_types = FALSE)
normalize <- function(x) trimws(gsub(" +", " ", gsub("[^A-Z0-9]", " ", toupper(x))))
registry <- fromJSON(registry_file, simplifyVector = FALSE)
cases <- read_chars("output/hays_eviction_cases_extracted.csv")
review <- read_chars("data/hays_eviction_address_review.csv")
props <- read_chars(property_file)
owners <- read_chars(owner_file)
parcels <- st_read(parcel_file, quiet = TRUE)
parcels$REFNAME <- trimws(parcels$REFNAME)
boundaries <- st_read(boundary_file, quiet = TRUE)
full <- boundaries[trimws(boundaries$city_name) == "CITY OF AUSTIN" &
                     boundaries$jurisdiction_type == "FULL", ]
stopifnot(nrow(full) > 0, !anyDuplicated(cases$case_key), !anyDuplicated(review$case_key))
# Projected meters; entire parcel geometry is tested, not its label or centroid.
full_geometry <- st_union(st_make_valid(st_transform(full, 26914)))
registered_ids <- unique(unlist(lapply(registry$groups, `[[`, "parcel_ids")))
selected <- parcels[parcels$REFNAME %in% registered_ids, ]
selected$geometry_source <- parcel_file
selected <- selected[, c("REFNAME", "geometry_source")]
# The county service omits some IDs present in the official March 2025 state
# archive. Match its numeric Prop_ID to the CAD QuickRefID's R prefix.
state_parcels <- st_read(state_parcel_file, quiet = TRUE)
state_parcels$REFNAME <- paste0("R", trimws(state_parcels$Prop_ID))
state_selected <- state_parcels[state_parcels$REFNAME %in% registered_ids, ]
st_write(st_transform(state_selected[, c("REFNAME", "SITUS_ADDR", "DATE_ACQ")], 4326),
  "output/hays_eviction_txgio_property_geometry.geojson", delete_dsn = TRUE, quiet = TRUE)
fallback <- state_selected[!state_selected$REFNAME %in% selected$REFNAME, ]
fallback$geometry_source <- state_parcel_file
geometry_columns <- function(x) st_sf(REFNAME = x$REFNAME,
  geometry_source = x$geometry_source, geometry = st_geometry(st_transform(x, 26914)))
selected <- rbind(geometry_columns(selected), geometry_columns(fallback))
selected <- st_make_valid(st_transform(selected, 26914))

property_results <- list()
parcel_results <- list()
plot_geometries <- list()
for (g in registry$groups) {
  ids <- unlist(g$parcel_ids)
  p <- selected[selected$REFNAME %in% ids, ]
  missing_ids <- setdiff(ids, p$REFNAME)
  intersects <- NA
  distance <- NA_real_
  if (nrow(p) > 0) {
    geometry <- st_union(st_geometry(p))
    intersects <- any(lengths(st_intersects(geometry, full_geometry)) > 0)
    distance <- as.numeric(st_distance(geometry, full_geometry))
    plot_geometries[[g$id]] <- geometry[[1]]
  }
  polygon_status <- if (length(ids) == 0 || length(missing_ids) > 0) {
    "unresolved_missing_parcel_geometry"
  } else if (intersects) {
    "intersects_austin_full"
  } else if (distance <= 100) {
    "outside_within_100m_review"
  } else "outside_austin_full"
  # Reversible ordering only. Address/identity conflicts cannot be deferred.
  action <- if (polygon_status == "outside_austin_full" && !g$requires_identity_review) {
    "defer_outside_property_group"
  } else "retain_for_lookup"
  property_results[[length(property_results) + 1L]] <- tibble(
    property_group_id = g$id, property_group_name = g$name, court = g$court,
    plaintiff_pattern = g$pattern, property_address = g$address,
    parcel_ids = paste(ids, collapse = "|"), missing_parcel_ids = paste(missing_ids, collapse = "|"),
    source_url = g$source_url, property_evidence = g$evidence,
    requires_identity_review = g$requires_identity_review,
    property_polygon_status = polygon_status, minimum_distance_to_austin_m = round(distance, 1),
    triage_action = action, reviewed_on = registry$reviewed_on)
  for (id in ids) {
    a <- props[props$QuickRefID == id, ]
    o <- owners[owners$QuickRefID == id, ]
    q <- p[p$REFNAME == id, ]
    parcel_results[[length(parcel_results) + 1L]] <- tibble(
      property_group_id = g$id, parcel_id = id,
      situs = paste(unique(a$Situs), collapse = " | "),
      property_label = paste(unique(a$SitusLocation), collapse = " | "),
      legal_description = paste(unique(a$LegalDesc), collapse = " | "),
      owner_names = paste(unique(o$OwnerName), collapse = " | "),
      geometry_features = nrow(q),
      intersects_austin_full = if (nrow(q)) any(lengths(st_intersects(q, full_geometry)) > 0) else NA,
      property_source = property_file, owner_source = owner_file,
      geometry_source = paste(unique(q$geometry_source), collapse = " | "))
  }
}
groups <- bind_rows(property_results)
aliases <- cases |> distinct(court, plaintiff) |> mutate(plaintiff_normalized = normalize(plaintiff))
aliases$property_group_id <- ""
for (i in seq_len(nrow(aliases))) {
  matches <- groups$property_group_id[groups$court == aliases$court[i] &
    vapply(groups$plaintiff_pattern, function(pattern) grepl(pattern, aliases$plaintiff_normalized[i]), logical(1))]
  if (length(matches) > 1) stop("Ambiguous property pattern for ", aliases$plaintiff[i])
  if (length(matches)) aliases$property_group_id[i] <- matches
}
case_screen <- cases |>
  left_join(aliases |> select(court, plaintiff, property_group_id), by = c("court", "plaintiff")) |>
  left_join(groups |> select(-court, -plaintiff_pattern), by = "property_group_id") |>
  mutate(triage_action = coalesce(triage_action, "retain_for_lookup"),
         property_polygon_status = coalesce(property_polygon_status, "unresolved_property_identity")) |>
  left_join(review |> select(case_key, portal_status, portal_url, portal_filing_date,
                            candidate_premises_address, premises_verification), by = "case_key") |>
  mutate(triage_action = if_else(premises_verification == "candidate_multiple_defendant_addresses",
                                "retain_for_lookup", triage_action),
         effective_filing_date = if_else(filing_date != "", filing_date, portal_filing_date),
         current_window_status = case_when(
           effective_filing_date == "" ~ "unknown_filing_date",
           effective_filing_date >= "2024-04-02" & effective_filing_date <= "2026-04-01" ~ "in_current_two_year_window",
           TRUE ~ "outside_current_two_year_window"),
         lookup_priority = case_when(
           triage_action == "defer_outside_property_group" ~ 4L,
           current_window_status == "in_current_two_year_window" ~ 1L,
           current_window_status == "unknown_filing_date" & as.integer(case_number_year) >= 2024 ~ 1L,
           current_window_status == "unknown_filing_date" ~ 2L,
           TRUE ~ 3L))
stopifnot(nrow(case_screen) == nrow(cases), !anyDuplicated(case_screen$case_key),
          !anyNA(case_screen$triage_action))
group_counts <- case_screen |> filter(property_group_id != "") |>
  group_by(property_group_id) |> summarise(cases = n(), plaintiff_variants = n_distinct(plaintiff), .groups = "drop")
groups <- groups |> left_join(group_counts, by = "property_group_id") |> arrange(desc(cases))
stopifnot(!anyNA(groups$cases))
queue <- case_screen |> filter(triage_action == "retain_for_lookup") |>
  arrange(lookup_priority, court, plaintiff, case_number)
deferred <- case_screen |> filter(triage_action == "defer_outside_property_group")
stopifnot(nrow(queue) + nrow(deferred) == nrow(cases),
          !any(deferred$requires_identity_review),
          all(deferred$property_polygon_status == "outside_austin_full"))
write_csv(groups, "output/hays_eviction_property_screen.csv")
write_csv(bind_rows(parcel_results), "output/hays_eviction_property_parcel_evidence.csv")
write_csv(aliases, "output/hays_eviction_property_aliases.csv")
write_csv(case_screen, "output/hays_eviction_case_geography_screen.csv")
write_csv(queue, "output/hays_eviction_austin_lookup_queue.csv")
write_csv(deferred, "output/hays_eviction_deferred_outside_property_groups.csv")
counts <- case_screen |> count(court, triage_action, current_window_status)
write_csv(counts, "output/hays_eviction_geography_screen_counts.csv")
summary <- list(total_cases = nrow(cases), property_groups = nrow(groups),
  cases_in_identified_groups = sum(groups$cases),
  deferred_cases = nrow(deferred), active_lookup_cases = nrow(queue),
  active_unstarted_lookups = sum(queue$portal_status == "not_started"),
  groups_outside = sum(groups$property_polygon_status == "outside_austin_full"),
  groups_intersecting = sum(groups$property_polygon_status == "intersects_austin_full"),
  priority_1_cases = sum(queue$lookup_priority == 1),
  boundary_file = boundary_file, boundary_filter = "CITY OF AUSTIN / FULL",
  outside_near_boundary_review_m = 100,
  note = "Property-level triage is not case-premises verification; all cases and ambiguous groups are retained.",
  source_sha256 = as.list(vapply(c(registry_file, boundary_file, parcel_file, state_archive, property_file, owner_file),
    function(path) digest::digest(path, algo = "sha256", file = TRUE), character(1))))
write_json(summary, "output/hays_eviction_geography_screen_summary.json", auto_unbox = TRUE, pretty = TRUE)
cat(toJSON(summary[1:9], auto_unbox = TRUE, pretty = TRUE), "\n")
print(groups |> select(property_group_name, cases, property_polygon_status,
                       minimum_distance_to_austin_m, triage_action), n = Inf)

# Inspectable geography output preserves parcel polygons for independent GIS QA.
plot_sf <- st_sf(property_group_id = names(plot_geometries),
                 geometry = st_sfc(plot_geometries, crs = 26914)) |>
  left_join(groups |> select(property_group_id, property_group_name, cases, triage_action),
            by = "property_group_id")
st_write(st_transform(plot_sf, 4326), "output/hays_eviction_screened_properties.geojson",
         delete_dsn = TRUE, quiet = TRUE)
