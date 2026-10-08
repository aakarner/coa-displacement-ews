# Reviewed project totals take precedence at promotion, without changing raw
# appraisal data or retraining the estimation model on this purposive audit.
read_reviewed_unit_properties <- function(path = "config/residential_unit_property_reviews.json") {
  x <- jsonlite::read_json(path, simplifyVector = FALSE)
  stopifnot(identical(x$schema_version, 1L), length(x$projects) > 0L)
  if (identical(path, "config/residential_unit_property_reviews.json")) {
    for (home_path in c("config/manufactured_home_property_reviews.json",
                        "config/residual_property_reviews.json")) {
      homes <- jsonlite::read_json(home_path, simplifyVector = FALSE)
      stopifnot(identical(homes$schema_version, 1L))
      x$projects <- c(x$projects, homes$projects)
      x$evidence <- c(x$evidence, homes$evidence)
      x$ownership_snapshot_path <- c(x$ownership_snapshot_path, homes$ownership_snapshot_path)
      x$extra_config_paths <- c(x$extra_config_paths, home_path)
    }
  }
  files <- vapply(x$evidence, `[[`, character(1), "path")
  for (e in x$evidence) {
    if (!file.exists(e$path) || !identical(digest::digest(file = e$path,
        algo = "sha256"), e$sha256)) stop("Missing or changed unit-review evidence: ", e$path)
  }
  ids <- vapply(x$projects, `[[`, character(1), "project_id")
  members <- unlist(lapply(x$projects, `[[`, "parcel_ids"))
  stopifnot(!anyDuplicated(ids), !anyDuplicated(members))
  for (r in x$projects) {
    status <- if (is.null(r$count_status)) "reviewed_direct_count" else r$count_status
    stopifnot(status %in% c("reviewed_direct_count", "provisional_assumption", "reviewed_geometry_only"))
    stopifnot(length(r$units) == 1L, is.finite(r$units), r$units > 0,
      (status == "reviewed_geometry_only" || r$units == round(r$units)), length(r$parcel_ids) > 0L,
      nzchar(r$review_id), nzchar(r$basis), length(r$evidence_paths) > 0L,
      all(unlist(r$evidence_paths) %in% files))
    if (!is.null(r$geometry)) stopifnot(
      is.finite(r$geometry$lon), abs(r$geometry$lon) <= 180,
      is.finite(r$geometry$lat), abs(r$geometry$lat) <= 90,
      r$geometry$evidence_path %in% files)
    if (!is.null(r$geometry$city_boundary_evidence_path)) stopifnot(
      r$geometry$city_boundary_evidence_path %in% files,
      is.finite(r$geometry$minimum_city_overlap_fraction),
      r$geometry$minimum_city_overlap_fraction >= 0.999,
      r$geometry$minimum_city_overlap_fraction <= 1)
    if (!is.null(r$supplement)) stopifnot(length(r$parcel_ids) == 1L,
      identical(r$supplement$parcel_id, r$parcel_ids[[1]]),
      identical(r$supplement$source_county, r$source_county))
  }
  if (!is.null(x$ownership_snapshot_path)) {
    owner_rows <- read_reviewed_owner_files(x$ownership_snapshot_path)
    current <- owner_rows[owner_rows$tax_year == "2025", ]
    stopifnot(!anyDuplicated(current$parcel_id))
    x$projects <- lapply(x$projects, function(r) {
      if (!is.null(r$inventory_group) && !is.null(r$supplement)) {
        o <- current[current$parcel_id == r$supplement$parcel_id, ]
        stopifnot(nrow(o) == 1L, o$classification_status == "matched_classified")
        for (field in c("is_owner_occupied", "is_corporate_owned", "has_financialized_owner"))
          stopifnot(identical(r$supplement[[field]], o[[field]] == "TRUE"))
        # Names stay in the private owner snapshot, not the public config.
        r$supplement$owner_names <- o$owner_names
      }
      r
    })
  }
  x$input_paths <- unique(c(path, x$extra_config_paths, files, "R/reviewed_unit_properties.R"))
  x
}

reviewed_unit_supplement_ids <- function(reviews) {
  unlist(lapply(reviews$projects, function(r) if (!is.null(r$supplement)) r$supplement$parcel_id))
}

apply_reviewed_unit_properties <- function(promoted, reviews, grid,
    promotion_version = unique(promoted$unit_model_promotion_version)) {
  stopifnot(!anyDuplicated(promoted$parcel_id))
  # Apply large individual-home inventories on their small subset, then bind
  # once. Repeatedly growing the full county surface is unnecessarily costly.
  grouped <- vapply(reviews$projects, function(r) !is.null(r$inventory_group), logical(1))
  if (any(grouped)) {
    regular <- reviews
    regular$projects <- reviews$projects[!grouped]
    base <- apply_reviewed_unit_properties(promoted, regular, grid, promotion_version)
    inventory <- reviews
    inventory$projects <- lapply(reviews$projects[grouped], function(r) {
      r$inventory_group <- NULL
      r
    })
    ids <- unlist(lapply(inventory$projects, `[[`, "parcel_ids"))
    # Retain any overlap so the normal supplement guard fails on duplicates.
    keep <- base$parcels$parcel_id %in% ids
    homes <- apply_reviewed_unit_properties(base$parcels[keep, ], inventory, grid, promotion_version)
    return(list(parcels = dplyr::bind_rows(base$parcels[!keep, ], homes$parcels),
      audit = dplyr::bind_rows(base$audit, homes$audit)))
  }
  footprints <- new.env(parent = emptyenv())
  promoted$unit_review_property_group <- NA_character_
  promoted$unit_review_location_precision <- NA_character_
  promoted$unit_review_id <- NA_character_
  promoted$unit_review_basis <- NA_character_
  promoted$unit_review_added_account <- FALSE
  promoted$unit_review_count_status <- NA_character_
  promoted$unit_review_original_lon <- promoted$lon
  promoted$unit_review_original_lat <- promoted$lat
  promoted$unit_review_geometry_parcel_id <- NA_character_
  promoted$unit_review_pre_review_units <- promoted$promoted_units
  promoted$unit_review_pre_review_method <- promoted$unit_model_selection_method
  audit <- list()
  for (r in reviews$projects) {
    members <- unlist(r$parcel_ids)
    if (!is.null(r$supplement)) {
      # A supplement is explicitly outside the original estimator universe.
      # Fail if a later upstream import already contains it: review the overlap.
      if (any(members %in% promoted$parcel_id) || r$project_id %in% promoted$unit_model_project_id)
        stop("Reviewed supplement now overlaps the upstream inventory: ", r$review_id)
      row <- promoted[NA_integer_, , drop = FALSE]
      for (name in names(r$supplement)) {
        if (!name %in% names(row)) stop("Unknown supplemental attribute: ", name)
        row[[name]] <- vctrs::vec_cast(r$supplement[[name]], promoted[[name]])
      }
      row$unit_model_project_id <- r$project_id
      row$unit_model_promotion_version <- promotion_version
      row$promotion_baseline_targeted_units <- 0
      row$units_calibrated <- 0
      row$promoted_units <- 0
      row$unit_review_pre_review_units <- 0
      row$unit_review_pre_review_method <- if (is.null(r$omission_reason))
        "omitted_account_without_coordinate" else r$omission_reason
      row$unit_review_added_account <- TRUE
      # Supplements may retain an original source point before its reviewed
      # analytical reference is applied below.
      row$unit_review_original_lon <- row$lon
      row$unit_review_original_lat <- row$lat
      row$unit_model_allocation_method <- "reviewed_single_account_supplement"
      promoted <- dplyr::bind_rows(promoted, row)
    }
    i <- which(promoted$unit_model_project_id == r$project_id)
    if (!setequal(promoted$parcel_id[i], members) ||
        !all(promoted$source_county[i] == r$source_county))
      stop("Reviewed project membership changed: ", r$review_id)
    if (any(promoted$county_unit_exclude_from_unit_universe[i] %in% TRUE))
      stop("Reviewed project includes an excluded appraisal account: ", r$review_id)
    before <- sum(promoted$promoted_units[i])
    weights <- if (!is.null(r$supplement)) 1 else promoted$promoted_units[i] / before
    if (any(!is.finite(weights)) || abs(sum(weights) - 1) > 1e-9)
      stop("Reviewed count lacks a valid existing allocation: ", r$review_id)
    status <- if (is.null(r$count_status)) "reviewed_direct_count" else r$count_status
    if (status == "reviewed_geometry_only") {
      if (!is.null(r$supplement) || abs(before - r$units) > 1e-7)
        stop("Geometry-only review units changed: ", r$review_id)
    } else {
      promoted$promoted_units[i] <- r$units * weights
    method <- if (status == "provisional_assumption") "reviewed_assumed_project_total" else "reviewed_direct_project_total"
    promoted$unit_model_selection_method[i] <- method
    promoted$unit_model_used[i] <- FALSE
    promoted$unit_model_promotion_applied[i] <- TRUE
    promoted$unit_estimation_method_targeted[i] <- paste0("promoted_", method)
    promoted$unit_estimation_confidence_targeted[i] <- if (status == "provisional_assumption") "low" else "high"
    }
    if (!is.null(r$property_group_id)) promoted$unit_review_property_group[i] <- r$property_group_id
    if (!is.null(r$geometry$location_precision)) promoted$unit_review_location_precision[i] <- r$geometry$location_precision
    promoted$unit_review_count_status[i] <- status
    if (status != "reviewed_geometry_only") promoted$unit_estimation_notes_targeted[i] <- r$basis
    promoted$unit_review_id[i] <- r$review_id
    promoted$unit_review_basis[i] <- r$basis
    if (!is.null(r$geometry)) {
      if (!is.null(r$geometry$expected_original_lon)) {
        if (any(abs(promoted$lon[i] - r$geometry$expected_original_lon) > 1e-8) ||
            any(abs(promoted$lat[i] - r$geometry$expected_original_lat) > 1e-8))
          stop("Reviewed original geometry changed: ", r$review_id)
      }
      point <- sf::st_as_sf(data.frame(lon = r$geometry$lon, lat = r$geometry$lat),
        coords = c("lon", "lat"), crs = 4326)
      h <- sf::st_within(sf::st_transform(point, sf::st_crs(grid)), grid)[[1]]
      if (length(h) != 1L || grid$hex_id[h] != r$geometry$expected_hex_id)
        stop("Reviewed geometry no longer has the expected grid cell: ", r$review_id)
      key <- r$geometry$evidence_path
      if (!exists(key, envir = footprints, inherits = FALSE))
        assign(key, sf::st_read(key, quiet = TRUE), envir = footprints)
      footprint <- get(key, envir = footprints, inherits = FALSE)
      footprint <- footprint[footprint$polygon_parcel_id == r$geometry$polygon_parcel_id, ]
      if (nrow(footprint) != 1L || !lengths(sf::st_within(point, sf::st_transform(footprint, 4326))))
        stop("Reviewed geometry falls outside its verified footprint: ", r$review_id)
      if (!is.null(r$geometry$city_boundary_evidence_path)) {
        city <- sf::st_read(r$geometry$city_boundary_evidence_path, quiet = TRUE)
        if (all(c("city_name", "jurisdiction_type") %in% names(city))) {
          city <- dplyr::filter(city, toupper(trimws(city_name)) == "CITY OF AUSTIN",
            toupper(trimws(jurisdiction_type)) == "FULL")
        } else {
          # Earlier reviews pin a geometry-only, already selected City union.
          # Reject partly described jurisdiction tables; do not union all cities.
          stopifnot(nrow(city) == 1L,
            length(setdiff(names(city), attr(city, "sf_column"))) == 0L)
        }
        city <- city |> sf::st_transform(3083) |> sf::st_make_valid() |> sf::st_union()
        parcel <- sf::st_transform(footprint, 3083) |> sf::st_make_valid()
        share <- as.numeric(sum(sf::st_area(suppressWarnings(sf::st_intersection(parcel, city)))) /
          sum(sf::st_area(parcel)))
        if (!is.finite(share) || share < r$geometry$minimum_city_overlap_fraction ||
            !all(lengths(sf::st_within(sf::st_transform(point, 3083), city)) == 1L))
          stop("Reviewed boundary reference no longer meets City scope: ", r$review_id)
      }
      promoted$lon[i] <- r$geometry$lon
      promoted$lat[i] <- r$geometry$lat
      promoted$coord_source[i] <- if (is.null(r$geometry$coord_source))
        "reviewed_operator_unit_map_reference" else r$geometry$coord_source
      promoted$unit_review_geometry_parcel_id[i] <- r$geometry$polygon_parcel_id
    }
    audit[[length(audit) + 1L]] <- data.frame(review_id = r$review_id,
      project_id = r$project_id, property = r$name, previous_units = before,
      reviewed_units = r$units, delta = r$units - before,
      added_account = !is.null(r$supplement), count_status = status, basis = r$basis)
  }
  stopifnot(!anyDuplicated(promoted$parcel_id))
  list(parcels = promoted, audit = dplyr::bind_rows(audit))
}

# Validate the consumed promoted product against the active review configuration.
# The original baseline still has to match exactly, apart from named supplements.
validate_reviewed_unit_surface <- function(parcels, reviews) {
  expected_added <- reviewed_unit_supplement_ids(reviews)
  stopifnot(setequal(parcels$parcel_id[parcels$unit_review_added_account %in% TRUE], expected_added))
  for (r in reviews$projects) {
    p <- parcels[parcels$unit_model_project_id %in% r$project_id, ]
    status <- if (is.null(r$count_status)) "reviewed_direct_count" else r$count_status
    method <- if (status == "provisional_assumption") "reviewed_assumed_project_total" else "reviewed_direct_project_total"
    stopifnot(setequal(p$parcel_id, unlist(r$parcel_ids)),
      all(p$unit_review_id == r$review_id),
      abs(sum(p$units_calibrated_targeted) - r$units) < 1e-7,
      all(p$unit_review_count_status == status),
      (status == "reviewed_geometry_only" || all(p$unit_model_selection_method == method)),
      (status == "reviewed_geometry_only" || all(p$unit_estimation_confidence_targeted == if (status == "provisional_assumption") "low" else "high")))
    if (!is.null(r$geometry)) stopifnot(
      all(abs(p$lon - r$geometry$lon) < 1e-8),
      all(abs(p$lat - r$geometry$lat) < 1e-8))
  }
  invisible(TRUE)
}

# Import independently classified home-level vintages, retaining unknown 2024
# evidence as unknown and preserving any original upstream target rows.
append_reviewed_home_owners <- function(owners, reviews, rule_version) {
  if (is.null(reviews$ownership_snapshot_path)) return(owners)
  homes <- read_reviewed_owner_files(reviews$ownership_snapshot_path) |>
    dplyr::select(-dplyr::any_of(c("property_units", "residential_use_category"))) |>
    dplyr::mutate(tax_year = as.integer(tax_year))
  ids <- unlist(lapply(Filter(function(r) !is.null(r$inventory_group), reviews$projects), `[[`, "parcel_ids"))
  stopifnot(!anyDuplicated(homes[c("parcel_id", "tax_year")]),
    setequal(homes$parcel_id, ids), all(homes$tax_year %in% c(2024L, 2025L)),
    nrow(homes) == length(ids) * 2L,
    all(homes$classification_rule_version == rule_version))
  keys <- c("parcel_id", "tax_year")
  overlap <- dplyr::inner_join(owners, homes, by = keys, suffix = c(".old", ".new"))
  for (field in c("classification_status", "is_owner_occupied", "is_corporate_owned", "has_financialized_owner")) {
    a <- overlap[[paste0(field, ".old")]]
    b <- overlap[[paste0(field, ".new")]]
    if (!identical(a, b)) stop("Existing home ownership disagrees with reviewed source: ", field)
  }
  dplyr::bind_rows(owners, dplyr::anti_join(homes, owners, by = keys))
}

read_reviewed_owner_files <- function(paths) {
  dplyr::bind_rows(lapply(paths, function(path) readr::read_csv(path,
    col_types = readr::cols(.default = readr::col_character()),
    na = c("", "NA"), show_col_types = FALSE)))
}
