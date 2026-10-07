# Local matching of current-window single party-address candidates to Hays CAD.
# Coordinates locate a parcel, not a verified eviction premises or apartment.
suppressPackageStartupMessages({library(sf); library(dplyr); library(readr); library(stringr); library(jsonlite)})
input <- "output/hays_eviction_current_window_address_review.csv"
property_file <- paste0("data/raw_parcels/hays/property_export_unzipped/nested/",
  "2025-PROPERTY-DATA-EXPORT-FILE-PROPERTY-4.28.2026/PropertyDataExport1361499.txt")
county_file <- "data/raw_parcels/hays/hays_parcels.gpkg"
state_file <- "data/raw_parcels/hays/txgio_2025/fgdb/stratmap25-landparcels_48209_hays_202503.gdb"
boundary_file <- "data/BOUNDARIES_jurisdictions_20260429.geojson"
read_chars <- function(p) read_csv(p, col_types=cols(.default=col_character()), na=character(), show_col_types=FALSE)
normalize_street <- function(x) {
  x <- toupper(x)
  x <- gsub("[.']", "", x)
  x <- sub("(\\b(APARTMENT|APT|UNIT|SUITE|STE|LOT|SPOT)\\b|#).*$", "", x, perl=TRUE)
  x <- gsub("[^A-Z0-9]", " ", x)
  x <- gsub("\\b(FM|RR|CR)([0-9])", "\\1 \\2", x, perl=TRUE)
  replacements <- c(STREET="ST", ROAD="RD", DRIVE="DR", LANE="LN", TRAIL="TRL",
    COURT="CT", CIRCLE="CIR", BOULEVARD="BLVD", AVENUE="AVE", TERRACE="TER",
    PARKWAY="PKWY", PLACE="PL", HIGHWAY="HWY", NORTH="N", SOUTH="S", EAST="E", WEST="W")
  for (word in names(replacements)) x <- gsub(paste0("\\b",word,"\\b"), replacements[[word]], x, perl=TRUE)
  trimws(gsub(" +", " ", x))
}
parse_address <- function(x) {
  m <- str_match(toupper(x), "^(.*),\\s*([^,]+),\\s*TX\\s*,?\\s*(\\d{5})(?:-\\d{4})?$")
  tibble(street_raw=trimws(m[,2]), postal_city=trimws(m[,3]), postal_zip=m[,4],
         street_key=normalize_street(m[,2]))
}
cases <- read_chars(input)
candidate <- cases |> filter(premises_verification == "candidate_party_address")
addresses <- candidate |> distinct(candidate_premises_address)
# Stable IDs prevent new/reordered cases from attaching cached geocodes to a
# different address. Keep the registry in ignored source storage across reruns.
registry_file <- "data/raw_hays_evictions/geocode_cache/address_registry.csv"
prior_file <- "output/hays_eviction_candidate_addresses_geocoded_local.csv"
registry <- if(file.exists(registry_file)) read_chars(registry_file) else if(file.exists(prior_file)) {
  read_chars(prior_file) |> select(candidate_premises_address,address_id)
} else tibble(candidate_premises_address=character(),address_id=character())
stopifnot(!anyDuplicated(registry$candidate_premises_address), !anyDuplicated(registry$address_id))
next_id <- if(nrow(registry)) max(as.integer(sub("HAYS_ADDR_", "", registry$address_id))) else 0L
new_addresses <- addresses |> anti_join(registry,by="candidate_premises_address") |>
  mutate(address_id=sprintf("HAYS_ADDR_%04d", next_id + row_number()))
registry <- bind_rows(registry,new_addresses)
write_csv(registry,registry_file)
addresses <- addresses |> left_join(registry,by="candidate_premises_address")
addresses <- bind_cols(addresses, parse_address(addresses$candidate_premises_address)) |>
  mutate(match_key=paste(street_key,postal_zip,sep="|"))
stopifnot(!anyNA(addresses$street_key), !anyDuplicated(addresses$address_id))
props <- read_chars(property_file) |> filter(grepl("^R[0-9]+$",QuickRefID))
props <- bind_cols(props, parse_address(props$Situs)) |>
  mutate(match_key=paste(street_key,postal_zip,sep="|"))
matches <- addresses |> select(address_id,match_key) |>
  inner_join(props |> select(match_key,QuickRefID,Situs),by="match_key",relationship="many-to-many") |>
  distinct(address_id,QuickRefID,Situs)
county <- st_read(county_file,quiet=TRUE)
county$parcel_id <- trimws(county$REFNAME)
county <- county[county$parcel_id %in% matches$QuickRefID,]
state <- st_read(state_file,quiet=TRUE)
state$parcel_id <- paste0("R",trimws(state$Prop_ID))
state <- state[state$parcel_id %in% setdiff(matches$QuickRefID,county$parcel_id),]
as_geometry <- function(x,source) st_sf(parcel_id=x$parcel_id,geometry_source=rep(source,nrow(x)),
  geometry=st_geometry(st_make_valid(st_transform(x,26914))))
parcels <- rbind(as_geometry(county,county_file),as_geometry(state,state_file))
boundaries <- st_read(boundary_file,quiet=TRUE)
full <- boundaries[trimws(boundaries$city_name)=="CITY OF AUSTIN" & boundaries$jurisdiction_type=="FULL",]
stopifnot(nrow(full)>0)
full <- st_union(st_make_valid(st_transform(full,26914)))
result <- list()
for (i in seq_len(nrow(addresses))) {
  a <- addresses[i,]
  m <- matches[matches$address_id==a$address_id,]
  ids <- unique(m$QuickRefID)
  p <- parcels[parcels$parcel_id %in% ids,]
  missing <- setdiff(ids,p$parcel_id)
  status <- "unmatched_local_reference"
  boundary <- "unresolved"
  distance <- lon <- lat <- NA_real_
  if(length(ids)>0 && length(missing)>0) status <- "matched_address_missing_geometry"
  if(length(ids)>0 && !length(missing)) {
    g <- st_union(st_geometry(p))
    intersects <- lengths(st_intersects(g,full))>0
    inside <- lengths(st_covered_by(g,full))>0
    distance <- as.numeric(st_distance(g,full))
    boundary <- if(inside) "inside_austin_full" else if(intersects) "parcel_crosses_boundary_review" else if(distance<=100) "outside_within_100m_review" else "outside_austin_full"
    status <- "matched_local_parcel_address"
    point <- st_transform(st_point_on_surface(g),4326)
    lon <- st_coordinates(point)[1,1]; lat <- st_coordinates(point)[1,2]
  }
  result[[i]] <- bind_cols(a,tibble(geocode_status=status,candidate_boundary_status=boundary,
    parcel_ids=paste(ids,collapse="|"), matched_addresses=paste(unique(m$Situs),collapse=" | "),
    missing_parcel_ids=paste(missing,collapse="|"),longitude=lon,latitude=lat,
    distance_to_austin_m=round(distance,1),geometry_source=paste(unique(p$geometry_source),collapse=" | "),
    evidence="Normalized street + exact ZIP match to CAD situs; unit ignored for parcel location. Eviction premises remain unverified."))
}
geocodes <- bind_rows(result)
write_csv(geocodes,"output/hays_eviction_candidate_addresses_geocoded_local.csv",na="")
all_cases <- cases |> left_join(geocodes,by="candidate_premises_address") |>
  mutate(geocode_status=coalesce(geocode_status,"not_geocoded_address_review_required"),
         candidate_boundary_status=coalesce(candidate_boundary_status,"unresolved"))
stopifnot(nrow(all_cases)==nrow(cases),!anyDuplicated(all_cases$case_key))
write_csv(all_cases,"output/hays_eviction_current_window_local_geography.csv",na="")
write_csv(geocodes |> filter(geocode_status!="matched_local_parcel_address"),
          "output/hays_eviction_candidate_geocode_followup.csv",na="")
summary <- list(current_window_cases=nrow(cases),single_address_cases=nrow(candidate),
  unique_address_strings=nrow(addresses),local_matches=sum(geocodes$geocode_status=="matched_local_parcel_address"),
  case_boundary_counts=as.list(table(all_cases$candidate_boundary_status)),
  note="Candidate party-address geography only. Source coverage and eviction premises remain unverified.",
  input_sha256=digest::digest(input,algo="sha256",file=TRUE),
  boundary_sha256=digest::digest(boundary_file,algo="sha256",file=TRUE))
write_json(summary,"output/hays_eviction_local_geography_summary.json",pretty=TRUE,auto_unbox=TRUE)
cat(toJSON(summary,pretty=TRUE,auto_unbox=TRUE),"\n")
