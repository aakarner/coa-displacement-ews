#!/usr/bin/env Rscript
# Prepare raw, target-restricted historical ownership evidence; no classification.
# Run from the repository root: Rscript scripts/data/prepare_williamson_txgio.R --year 2025
# This file can also be sourced to call prepare_williamson_txgio().

WILLIAMSON_TXGIO_SOURCE_FIELDS <- c(
  "Prop_ID", "GEO_ID", "OWNER_NAME", "NAME_CARE", "SITUS_ADDR", "SITUS_NUM",
  "SITUS_STRE", "SITUS_ST_1", "SITUS_ST_2", "SITUS_CITY", "SITUS_STAT",
  "SITUS_ZIP", "MAIL_ADDR", "MAIL_LINE1", "MAIL_LINE2", "MAIL_CITY",
  "MAIL_STAT", "MAIL_ZIP", "SOURCE", "DATE_ACQ", "TAX_YEAR", "COUNTY", "FIPS",
  "OBJECTID"
)
WILLIAMSON_TXGIO_EVIDENCE_FIELDS <- setdiff(
  WILLIAMSON_TXGIO_SOURCE_FIELDS, "OBJECTID"
)

txgio_require_packages <- function() {
  needed <- c("sf", "digest", "jsonlite")
  missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Missing package(s): ", paste(missing, collapse = ", "))
}

txgio_sha256 <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

txgio_read_csv <- function(path) {
  # Literal placeholders (including UNAVAILABLE), whitespace, and blank source
  # components remain raw evidence. No current parcel attributes are consulted.
  utils::read.csv(path, colClasses = "character", na.strings = NULL,
                  check.names = FALSE, stringsAsFactors = FALSE,
                  strip.white = FALSE, fileEncoding = "UTF-8")
}

txgio_gdb_members <- function(member_names) {
  if (!length(member_names) || anyNA(member_names) ||
      any(grepl("(^/|^[A-Za-z]:|\\\\|(^|/)\\.\\.?(/|$))", member_names)) ||
      anyDuplicated(member_names)) {
    stop("Unsafe or duplicate ZIP member path")
  }
  members <- member_names[grepl("^fgdb/[^/]+\\.gdb(/|$)", member_names)]
  roots <- unique(sub("(^fgdb/[^/]+\\.gdb).*$", "\\1", members))
  if (length(roots) != 1L || !length(members)) {
    stop("Expected one FileGDB below fgdb/ in the pinned ZIP")
  }
  list(root = roots[[1]], members = members)
}

txgio_normalize_evidence <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  toupper(trimws(gsub("[[:space:]]+", " ", x)))
}

txgio_annotate_evidence <- function(rows) {
  missing <- setdiff(WILLIAMSON_TXGIO_SOURCE_FIELDS, names(rows))
  if (length(missing)) stop("Missing TxGIO evidence fields: ", paste(missing, collapse = ", "))
  rows <- as.data.frame(rows, stringsAsFactors = FALSE)
  ids <- trimws(rows$Prop_ID)
  if (anyNA(ids) || any(!nzchar(ids))) stop("Blank target-matched TxGIO Prop_ID")
  rows$parcel_id <- paste0("WILLIAMSON:", ids)
  rows$source_row_id <- as.character(rows$OBJECTID)
  if (anyNA(rows$source_row_id) || any(!nzchar(rows$source_row_id)) ||
      anyDuplicated(rows$source_row_id)) stop("Invalid or duplicate TxGIO OBJECTID")
  rows <- rows[order(rows$parcel_id, as.numeric(rows$source_row_id), method = "radix"), ]
  normalized <- lapply(rows[WILLIAMSON_TXGIO_EVIDENCE_FIELDS], txgio_normalize_evidence)
  # Length prefixes avoid ambiguous concatenations and preserve field boundaries.
  rows$source_evidence_group <- vapply(seq_len(nrow(rows)), function(i) {
    values <- vapply(normalized, `[`, character(1), i)
    digest::digest(paste0(nchar(values, type = "bytes"), ":", values, collapse = ""),
                   algo = "sha256", serialize = FALSE)
  }, character(1))
  key <- paste(rows$parcel_id, rows$source_evidence_group, sep = "|")
  multiplicities <- table(key)
  groups <- split(rows$source_evidence_group, rows$parcel_id)
  counts <- vapply(groups, function(x) length(unique(x)), integer(1))
  rows$source_evidence_duplicate_count <- as.integer(multiplicities[key])
  rows$source_evidence_is_duplicate <- duplicated(key)
  rows$source_evidence_conflict <- unname(counts[rows$parcel_id]) > 1L
  rows <- rows[c("parcel_id", "source_row_id", "source_evidence_group",
                 "source_evidence_duplicate_count", "source_evidence_is_duplicate",
                 "source_evidence_conflict", WILLIAMSON_TXGIO_SOURCE_FIELDS)]
  rownames(rows) <- NULL
  rows
}

txgio_count_values <- function(values) {
  counts <- table(values, useNA = "ifany")
  stats::setNames(as.list(as.integer(counts)), names(counts))
}

txgio_validate_source_config <- function(config, tax_year) {
  required <- c("path", "sha256", "source_url", "tax_year", "acquisition_date",
                "layer", "county_rows", "field_count")
  if (!is.list(config) || !all(required %in% names(config)) ||
      any(vapply(config[required], function(x) length(x) != 1L || anyNA(x), logical(1)))) {
    stop("Incomplete TxGIO source configuration for tax year ", tax_year)
  }
  if (!identical(as.character(config$tax_year), as.character(tax_year)) ||
      !grepl("^[0-9a-f]{64}$", config$sha256) ||
      !grepl("^[A-Za-z0-9_]+$", config$layer) ||
      !grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", config$acquisition_date) ||
      is.na(as.Date(config$acquisition_date)) ||
      substr(config$acquisition_date, 1L, 4L) != as.character(tax_year) ||
      !nzchar(config$path) || !nzchar(config$source_url) ||
      !is.numeric(config$county_rows) || config$county_rows < 1 ||
      !is.finite(config$county_rows) || config$county_rows %% 1 != 0 ||
      !is.numeric(config$field_count) || config$field_count < length(WILLIAMSON_TXGIO_SOURCE_FIELDS) ||
      !is.finite(config$field_count) || config$field_count %% 1 != 0) {
    stop("Invalid or cross-year TxGIO source configuration for tax year ", tax_year)
  }
  invisible(config)
}

txgio_validate_county <- function(county, config) {
  txgio_validate_source_config(config, config$tax_year)
  expected <- c(TAX_YEAR = as.character(config$tax_year),
                DATE_ACQ = gsub("-", "", config$acquisition_date, fixed = TRUE),
                COUNTY = "WILLIAMSON", FIPS = "48491")
  if (nrow(county) != config$county_rows) stop("Unexpected row count for pinned county source")
  for (field in names(expected)) {
    if (!field %in% names(county) || anyNA(county[[field]]) ||
        any(trimws(county[[field]]) != expected[[field]])) {
      stop("County source fails pinned ", config$tax_year, " validation: ", field)
    }
  }
  invisible(county)
}

prepare_williamson_txgio <- function(
    tax_year = 2025L,
    source_zip = NULL,
    target_csv = "output/part2/ownership/ownership_target_parcels.csv",
    output_csv = NULL,
    manifest_json = NULL,
    source_config = "config/williamson_ownership_sources.json") {
  txgio_require_packages()
  if (length(tax_year) != 1L || is.na(tax_year) || !grepl("^[0-9]{4}$", as.character(tax_year))) {
    stop("Expected a four-digit TxGIO tax year")
  }
  tax_year <- as.integer(tax_year)
  if (!file.exists(source_config)) stop("Missing TxGIO source configuration: ", source_config)
  config_hash <- txgio_sha256(source_config)
  config <- jsonlite::read_json(source_config, simplifyVector = TRUE)[[paste0("gis_", tax_year)]]
  txgio_validate_source_config(config, tax_year)
  if (is.null(source_zip)) source_zip <- config$path
  if (is.null(output_csv)) output_csv <- sprintf("output/part2/ownership/williamson_txgio_%s_evidence.csv", tax_year)
  if (is.null(manifest_json)) manifest_json <- sprintf("output/part2/ownership/williamson_txgio_%s_manifest.json", tax_year)
  if (!file.exists(source_zip)) stop("Missing pinned TxGIO ZIP: ", source_zip)
  if (!file.exists(target_csv)) stop("Missing ownership target CSV: ", target_csv)
  source_hash <- txgio_sha256(source_zip)
  if (!identical(source_hash, config$sha256)) {
    stop("TxGIO ZIP SHA-256 differs from the pinned ", tax_year, " source")
  }
  target_hash <- txgio_sha256(target_csv)
  target <- txgio_read_csv(target_csv)
  required_target <- c("parcel_id", "source_county", "property_units")
  if (!all(required_target %in% names(target))) stop("Incomplete ownership target schema")
  target <- target[target$source_county == "Williamson", , drop = FALSE]
  if (!nrow(target) || anyNA(target$parcel_id) || anyDuplicated(target$parcel_id) ||
      any(!grepl("^WILLIAMSON:[A-Za-z0-9-]+$", target$parcel_id))) {
    stop("Invalid or duplicate Williamson target identifiers")
  }
  units <- suppressWarnings(as.numeric(target$property_units))
  if (anyNA(units) || any(!is.finite(units)) || any(units < 0)) {
    stop("Williamson target property_units must be validated, finite, and nonnegative")
  }
  ids <- sub("^WILLIAMSON:", "", target$parcel_id)
  members <- txgio_gdb_members(utils::unzip(source_zip, list = TRUE)$Name)
  # Every invocation extracts only the pinned GDB into a newly created temporary
  # directory. There is no persistent extracted cache that could become stale.
  scratch <- tempfile(paste0("ews-williamson-txgio-", tax_year, "-"))
  if (!dir.create(scratch)) stop("Could not create temporary extraction directory")
  on.exit(unlink(scratch, recursive = TRUE), add = TRUE)
  utils::unzip(source_zip, files = members$members, exdir = scratch)
  gdb <- file.path(scratch, members$root)
  layers <- sf::st_layers(gdb)
  if (!identical(layers$name, config$layer) ||
      layers$features[[1]] != config$county_rows || layers$fields[[1]] != config$field_count) {
    stop("Unexpected layer/schema in pinned TxGIO FileGDB")
  }
  schema <- sf::st_drop_geometry(sf::st_read(
    gdb, query = paste("SELECT * FROM", config$layer, "LIMIT 1"),
    quiet = TRUE
  ))
  if (!all(WILLIAMSON_TXGIO_SOURCE_FIELDS %in% names(schema))) {
    stop("Pinned TxGIO FileGDB is missing expected source fields")
  }
  # A single geometry-free attribute pass avoids repeated full-layer scans for
  # large IN predicates. Retain only target rows in the durable evidence output.
  county_csv <- file.path(scratch, "county_attributes.csv")
  sf::gdal_utils("vectortranslate", gdb, county_csv, options = c(
    "-f", "CSV", "-nlt", "NONE", "-select",
    paste(WILLIAMSON_TXGIO_SOURCE_FIELDS, collapse = ",")
  ))
  county <- txgio_read_csv(county_csv)
  txgio_validate_county(county, config)
  evidence <- txgio_annotate_evidence(county[trimws(county$Prop_ID) %in% ids, , drop = FALSE])
  if (any(!evidence$parcel_id %in% target$parcel_id)) {
    stop("TxGIO export unexpectedly contains a non-target parcel")
  }
  matched_ids <- unique(evidence$parcel_id)
  matched <- target$parcel_id %in% matched_ids
  repeated <- table(evidence$parcel_id)
  conflict_ids <- unique(evidence$parcel_id[evidence$source_evidence_conflict])
  if (!identical(txgio_sha256(source_config), config_hash) ||
      !identical(txgio_sha256(target_csv), target_hash)) {
    stop("TxGIO source configuration or target changed during preparation; rerun with stable inputs")
  }
  dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(manifest_json), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(evidence, output_csv, row.names = FALSE, na = "", fileEncoding = "UTF-8")
  manifest <- list(
    schema_version = "ews-williamson-txgio-evidence-v2",
    generated_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    inputs = list(
      source_zip = list(path = source_zip, sha256 = source_hash,
                        expected_sha256 = config$sha256,
                        size_bytes = file.info(source_zip)$size),
      source_config = list(path = source_config, sha256 = config_hash,
                           key = paste0("gis_", tax_year)),
      target_csv = list(path = target_csv, sha256 = target_hash,
                        size_bytes = file.info(target_csv)$size)
    ),
    output = list(path = output_csv, sha256 = txgio_sha256(output_csv),
                  size_bytes = file.info(output_csv)$size,
                  rows = nrow(evidence), columns = names(evidence)),
    source = list(
      snapshot_id = paste0("williamson-", tax_year, "-txgio-land-parcels"),
      title = paste("TxGIO StratMap", tax_year, "Land Parcels - Williamson County"),
      tax_year = tax_year, acquisition_date = config$acquisition_date,
      acquisition_date_semantics = "Source-coded DATE_ACQ; consult the retained metadata description and extraction date before interpreting exact timing.",
      metadata_acquisition_description = config$metadata_acquisition_description,
      source_extract_date = config$source_extract_date,
      processing_date = config$processing_date,
      public_object_modified_date = config$public_object_modified_date,
      collection_id = config$collection_id,
      collection_api_url = if (!is.null(config$collection_id)) paste0(
        "https://api.tnris.org/api/v1/collections?collection_id=", config$collection_id) else NULL,
      download_url = config$source_url,
      filegdb_layer = config$layer,
      raw_fields = names(schema),
      semantics = c(
        config$semantics,
        paste0("Source DATE_ACQ is coded ", config$acquisition_date, " with TAX_YEAR ", tax_year,
               "; not a certified roll or proof of ownership on the coded date or January 1."),
        "No homestead/exemption fields are present in the standardized source schema.",
        "Raw owner, care-of, situs, and mailing values are retained, including UNAVAILABLE placeholders; this stage performs no classification.",
        "Owner names may still be truncated in the source; standardized field width does not establish completeness.",
        "Some international mailing addresses occur in MAIL_LINE1 with blank city/state/ZIP components.",
        "Target CSV supplies only parcel identifiers and validated unit weights for coverage, not historical ownership attributes.",
        "Only the configuration-pinned same-year ZIP is read; no other-year fallback or network access is used."
      )
    ),
    extraction = list(strategy = "fresh temporary GDB-only extraction; removed after use",
                      attribute_read = "One geometry-free county attribute pass followed by target-ID restriction in R",
                      source_sha256_verified = TRUE, zip_paths_validated = TRUE,
                      extracted_members = length(members$members)),
    county = list(rows = nrow(county), fields = length(names(schema)),
                  unique_nonblank_ids = length(unique(county$Prop_ID[nzchar(trimws(county$Prop_ID))])),
                  blank_id_rows = sum(!nzchar(trimws(county$Prop_ID))),
                  tax_year_counts = txgio_count_values(county$TAX_YEAR),
                  acquisition_date_counts = txgio_count_values(county$DATE_ACQ),
                  source_counts = txgio_count_values(county$SOURCE)),
    target_coverage = list(
      target_parcels = nrow(target), matched_parcels = sum(matched),
      unmatched_parcels = sum(!matched), parcel_coverage = mean(matched),
      target_units = sum(units), matched_units = sum(units[matched]),
      unmatched_units = sum(units[!matched]),
      unit_coverage = if (sum(units) > 0) sum(units[matched]) / sum(units) else NA_real_,
      unmatched_parcel_ids = unname(target$parcel_id[!matched])
    ),
    repeated_evidence = list(
      exported_source_rows = nrow(evidence), repeated_parcel_ids = sum(repeated > 1L),
      source_rows_beyond_one_per_parcel = nrow(evidence) - length(matched_ids),
      unique_normalized_evidence_rows = sum(!evidence$source_evidence_is_duplicate),
      exact_normalized_duplicate_rows = sum(evidence$source_evidence_is_duplicate),
      conflict_parcel_count = length(conflict_ids), conflict_parcel_ids = conflict_ids,
      normalization = "Case-fold and collapse whitespace only; preserve raw output values and literal placeholders.",
      comparison_fields = WILLIAMSON_TXGIO_EVIDENCE_FIELDS,
      retention = "All source rows retained; source_evidence_group identifies exact normalized duplicates. Conflicting evidence groups are retained and flagged."
    )
  )
  jsonlite::write_json(manifest, manifest_json, pretty = TRUE, auto_unbox = TRUE,
                       na = "null", digits = NA)
  message(sprintf("TxGIO %s: %s source rows, %s/%s target parcels, %.0f/%.0f units; %s exact duplicate rows, %s conflicting parcels.",
                  tax_year, nrow(evidence), sum(matched), nrow(target), sum(units[matched]),
                  sum(units), sum(evidence$source_evidence_is_duplicate), length(conflict_ids)))
  invisible(c(evidence = output_csv, manifest = manifest_json))
}

txgio_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if ("--help" %in% args) {
    cat("Usage: Rscript scripts/data/prepare_williamson_txgio.R [--year YYYY] [--source-config PATH] [--source-zip PATH] [--target-csv PATH] [--output-csv PATH] [--manifest-json PATH]\n")
    return(invisible(NULL))
  }
  options <- c("--year" = "tax_year", "--source-config" = "source_config",
               "--source-zip" = "source_zip", "--target-csv" = "target_csv",
               "--output-csv" = "output_csv", "--manifest-json" = "manifest_json")
  if (length(args) %% 2L != 0L) stop("Every command-line option needs a path")
  parsed <- list()
  for (i in seq_len(length(args) / 2L) * 2L - 1L) {
    if (!args[[i]] %in% names(options)) stop("Unknown option: ", args[[i]])
    name <- unname(options[[args[[i]]]])
    if (name %in% names(parsed)) stop("Duplicate option: ", args[[i]])
    parsed[[name]] <- args[[i + 1L]]
  }
  do.call(prepare_williamson_txgio, parsed)
}

if (sys.nframe() == 0L) txgio_main()
