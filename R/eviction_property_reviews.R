# Curated, case-specific property links. Private evidence is pinned by a public
# manifest; it is never an address-only alias or a correction to source records.
# Separately reviewed address aliases below are limited to uniquely identified
# properties; ambiguous multi-phase and mixed-project addresses are not aliases.
reviewed_property_address_paths <- function(config_path = "config/eviction_property_address_reviews.json") {
  config <- jsonlite::read_json(config_path, simplifyVector = FALSE)
  unique(c(config_path, unlist(lapply(config$reviews, function(r) vapply(r$evidence, `[[`, character(1), "path")))))
}

apply_reviewed_property_addresses <- function(geography, unit_references,
    config_path = "config/eviction_property_address_reviews.json") {
  config <- jsonlite::read_json(config_path, simplifyVector = FALSE)
  stopifnot(identical(config$schema_version, 1L))
  geography$property_address_review_id <- NA_character_
  normalized <- toupper(gsub("[.,]", "", geography$address_for_geocoding))
  normalized <- trimws(gsub("\\s+", " ", normalized))
  paths <- config_path
  for (r in config$reviews) {
    for (e in r$evidence) verify_property_review_file(e$path, e$sha256)
    paths <- c(paths, vapply(r$evidence, `[[`, character(1), "path"))
    refs <- unit_references[unit_references$project_id %in% r$project_id & unit_references$operational_units > 0, ]
    stopifnot(nrow(refs) >= 1L, r$parcel_id %in% refs$parcel_id, all(refs$source_county == r$source_county),
      all(!is.na(refs$unit_hex_id)), all(refs$unit_hex_id == r$expected_unit_hex_id),
      all(abs(refs$lon-r$reference_longitude)<1e-8), all(abs(refs$lat-r$reference_latitude)<1e-8))
    i <- which(geography$source_county == r$source_county & grepl(r$normalized_address_pattern, normalized, perl=TRUE))
    if (any(!is.na(geography$property_address_review_id[i]))) stop("Overlapping reviewed property aliases.")
    if (!length(i)) next
    points <- sf::st_transform(sf::st_as_sf(geography[i, ], coords=c("longitude","latitude"),crs=4326),3083)
    ref <- sf::st_transform(sf::st_as_sf(refs[1L, , drop=FALSE],coords=c("lon","lat"),crs=4326),3083)
    geography$property_id[i] <- r$project_id
    geography$property_hex_id[i] <- r$expected_unit_hex_id
    geography$property_link_status[i] <- "verified"
    geography$reference_distance_m[i] <- as.numeric(sf::st_distance(points,ref))
    geography$property_address_review_id[i] <- r$review_id
  }
  list(rows=geography,input_paths=unique(paths))
}

verify_property_review_file <- function(path, sha256) {
  if (length(path) != 1L || !file.exists(path) || length(sha256) != 1L ||
      !identical(digest::digest(file = path, algo = "sha256"), sha256))
    stop("Missing or changed property review evidence: ", path, call. = FALSE)
}

read_property_review_batches <- function(config_path = "config/eviction_property_reviews.json") {
  config <- jsonlite::read_json(config_path, simplifyVector = FALSE)
  stopifnot(identical(config$schema_version, 1L), length(config$batches) > 0L)
  batches <- lapply(config$batches, function(x) {
    verify_property_review_file(x$path, x$sha256)
    b <- jsonlite::read_json(x$path, simplifyVector = FALSE)
    stopifnot(identical(b$schema_version, 1L), identical(b$batch_id, x$batch_id),
      length(b$cases) == x$case_count, length(b$evidence) > 0L)
    for (s in b$evidence) verify_property_review_file(s$path, s$sha256)
    b
  })
  list(batches = batches, paths = unique(c(config_path,
    vapply(config$batches, `[[`, character(1), "path"),
    unlist(lapply(batches, function(b) vapply(b$evidence, `[[`, character(1), "path"))))))
}

compile_property_case_reviews <- function(unit_references, geocodes,
    config_path = "config/eviction_property_reviews.json") {
  source <- read_property_review_batches(config_path)
  out <- dplyr::bind_rows(lapply(source$batches, function(b) dplyr::bind_rows(lapply(b$cases, function(x) {
    required <- c("case_number", "source_county", "source_jp_district", "file_date",
      "parcel_id", "project_id", "review_basis", "review_id")
    stopifnot(all(vapply(x[required], function(v) length(v) == 1L && !is.na(v) && nzchar(v), logical(1))),
      !is.na(as.Date(x$file_date)), length(x$addresses) > 0L,
      is.logical(x$apartment_conflict), length(x$apartment_conflict) == 1L)
    p <- unit_references[unit_references$project_id %in% x$project_id &
      unit_references$operational_units > 0, , drop = FALSE]
    members <- unit_references[unit_references$project_id %in% x$project_id, , drop = FALSE]
    if (!nrow(p) || !x$parcel_id %in% members$parcel_id ||
        anyNA(p$unit_hex_id) || length(unique(p$unit_hex_id)) != 1L ||
        !all(p$source_county == x$source_county) || unique(p$unit_hex_id) != x$expected_unit_hex_id)
      stop("Reviewed property no longer has its supported unit reference: ", x$review_id)
    # Distances are diagnostic only, never a nearest-property selection rule.
    ref <- p[order(p$parcel_id), ][1L, ]
    addresses <- dplyr::bind_rows(x$addresses)
    stopifnot(!anyDuplicated(addresses$address_for_geocoding))
    g <- geocodes[geocodes$source_county == x$source_county &
      geocodes$address_for_geocoding %in% addresses$address_for_geocoding, , drop = FALSE]
    g <- g[match(addresses$address_for_geocoding, g$address_for_geocoding), , drop = FALSE]
    if (anyNA(g$address_for_geocoding) || any(!is.finite(g$longitude) | !is.finite(g$latitude)) ||
        any(abs(g$longitude - addresses$longitude) > 1e-8 | abs(g$latitude - addresses$latitude) > 1e-8))
      stop("Reviewed filing geocode changed; re-review required: ", x$review_id)
    points <- sf::st_transform(sf::st_as_sf(g, coords = c("longitude", "latitude"), crs = 4326), 3083)
    point <- sf::st_transform(sf::st_as_sf(ref, coords = c("lon", "lat"), crs = 4326), 3083)
    data.frame(case_number = x$case_number, source_county = x$source_county,
      address_for_geocoding = addresses$address_for_geocoding,
      expected_file_date = x$file_date, expected_source_jp = x$source_jp_district,
      property_id = x$project_id, property_hex_id = as.integer(unique(p$unit_hex_id)),
      property_link_status = "verified", reference_distance_m = as.numeric(sf::st_distance(points, point)),
      property_review_id = x$review_id, property_review_basis = x$review_basis,
      property_apartment_conflict = x$apartment_conflict)
  }))))
  stopifnot(!anyDuplicated(out[c("case_number", "source_county", "address_for_geocoding")]))
  case_ids <- unlist(lapply(source$batches, function(b) vapply(b$cases, `[[`, character(1), "case_number")))
  stopifnot(!anyDuplicated(case_ids))
  list(rows = out, input_paths = source$paths)
}

apply_property_case_reviews <- function(rows, cases, reviews) {
  rows$property_review_id <- NA_character_
  rows$property_review_basis <- NA_character_
  rows$property_apartment_conflict <- FALSE
  if (is.null(reviews) || !nrow(reviews)) return(rows)
  # Original ambiguous/excluded cases are never rescued by this second stage.
  active <- intersect(reviews$case_number, cases$case_number[cases$assignment_status == "assigned_unique_hex"])
  for (id in active) {
    r <- reviews[reviews$case_number == id, , drop = FALSE]
    c <- cases[cases$case_number == id, , drop = FALSE]
    i <- which(rows$case_number == id)
    actual <- rows[i, c("source_county", "address_for_geocoding"), drop = FALSE]
    if (nrow(c) != 1L || is.na(c$file_date) ||
        !all(as.character(c$file_date) == r$expected_file_date) ||
        !all(c$source_county == r$source_county) ||
        !all(c$source_jp_district == r$expected_source_jp) ||
        !setequal(paste(actual$source_county, actual$address_for_geocoding),
          paste(r$source_county, r$address_for_geocoding)))
      stop("Case identity or reliable addresses changed; re-review required: ", id, call. = FALSE)
    stopifnot(length(unique(r$property_id)) == 1L, length(unique(r$property_hex_id)) == 1L,
      !anyDuplicated(r[c("source_county", "address_for_geocoding")]))
    j <- match(paste(actual$source_county, actual$address_for_geocoding),
      paste(r$source_county, r$address_for_geocoding))
    fields <- c("property_id", "property_hex_id", "property_link_status", "reference_distance_m",
      "property_review_id", "property_review_basis", "property_apartment_conflict")
    rows[i, fields] <- r[j, fields]
  }
  rows
}
