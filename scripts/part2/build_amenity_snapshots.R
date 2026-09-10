# Reconstruct paired amenity features only; does not build readiness or clusters.
# First run scripts/data/prepare_historical_amenity_sources.py.
# Usage: Rscript scripts/part2/build_amenity_snapshots.R [source-manifest.json]
# EWS_AMENITY_PREPARE_ONLY=true writes candidate tables without geocoding.
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tibble)
})
source("R/amenity_classification.R")
source("R/pipeline.R")

args <- commandArgs(trailingOnly = TRUE)
manifest_path <- if (length(args)) args[[1]] else
  "config/amenity_historical_sources.json"
if (!file.exists(manifest_path)) stop("Missing source manifest: ", manifest_path, call. = FALSE)
source_manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
sources <- source_manifest$sources
if (length(sources) != 2L) stop("Expected one archive and one live source.", call. = FALSE)
read_source <- function(source) {
  path <- source$normalized_path
  if (!file.exists(path)) stop("Missing pinned normalized source: ", path, call. = FALSE)
  if (!identical(digest::digest(file = path, algo = "sha256"), source$normalized_sha256)) {
    stop("Normalized source checksum mismatch: ", path, call. = FALSE)
  }
  amenity_validate_sales(read_csv(path, col_types = cols(.default = col_character()),
                                  show_col_types = FALSE))
}
archive_spec <- sources[[which(vapply(sources, function(x) x$kind == "archive", logical(1)))]]
live_spec <- sources[[which(vapply(sources, function(x) x$kind == "live", logical(1)))]]
archive <- read_source(archive_spec)
live <- read_source(live_spec)
output_root <- "output/part2/amenities"
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
taxonomy_path <- "config/amenity_categories.csv"
taxonomy <- read_csv(taxonomy_path, show_col_types = FALSE,
  col_types = cols(naics = col_character(), category = col_character(), evidence_tier = col_character(),
                   include_in_index = col_logical(), name_filter_required = col_logical(),
                   name_pattern = col_character(), notes = col_character()))
reconciled <- amenity_reconcile_sources(archive, live, archive_spec$source_id, live_spec$source_id)
ledger <- reconciled$ledger
write_csv(ledger, file.path(output_root, "amenity_reconciled_source_ledger.csv"))
write_csv(reconciled$dispositions, file.path(output_root, "amenity_source_id_reconciliation.csv"))
write_csv(reconciled$field_changes, file.path(output_root, "amenity_source_field_changes.csv"))

# Exact opening fingerprints identify possible identifier changes, without
# conflating different businesses at one street address. Keep unit text here.
fingerprints <- ledger %>% mutate(
  opening_fingerprint = paste(stringr::str_squish(stringr::str_to_upper(loc_name)),
    stringr::str_squish(stringr::str_to_upper(paste(address_number, address_text))),
    loc_zip, naics, first_sale_date, sep = "|")) %>%
  group_by(opening_fingerprint) %>% filter(n() > 1L) %>% ungroup()
write_csv(fingerprints, file.path(output_root, "amenity_cross_id_duplicate_review.csv"))

cutoffs <- as.Date(c("2025-04-01", "2026-04-01"))
candidate_paths <- character()
summaries <- list()
candidate_events <- list()
for (i in seq_along(cutoffs)) {
  cutoff <- cutoffs[[i]]
  windows <- amenity_windows(cutoff, 18L)
  date_dir <- file.path(output_root, format(cutoff, "%Y-%m-%d"))
  dir.create(date_dir, recursive = TRUE, showWarnings = FALSE)
  categorized <- amenity_classify_sales(ledger, taxonomy, cutoff, 18L) %>%
    left_join(reconciled$dispositions, by = "event_id") %>%
    mutate(mixed_beverage_address_match = NA, austin_food_address_match = NA,
           corroboration_status = "not_rebuilt_not_used_in_index")
  aliases <- amenity_deduplicate_events(categorized)
  categorized <- aliases$categorized
  write_csv(aliases$alias_audit, file.path(date_dir, "amenity_opening_alias_audit.csv"))
  categorized <- categorized %>% mutate(
    street_source = street, address_key_source = address_key,
    street = amenity_geocode_street(street_source),
    geocode_street_cleaned = street != street_source,
    street_key = amenity_normalize_address(street), address_key = paste(street_key, zip, sep = "|"))
  write_csv(categorized %>% filter(core_index_eligible, geocode_street_cleaned) %>%
    select(event_id, location_name, street_source, street, city, zip),
    file.path(date_dir, "amenity_address_cleanup_qa.csv"))
  events <- categorized %>% filter(core_index_eligible)
  if (anyDuplicated(events$event_id) || any(events$opening_date > cutoff) ||
      any(events$permit_date > cutoff, na.rm = TRUE)) stop("Invalid opening-event eligibility.", call. = FALSE)
  coverage <- tidyr::expand_grid(county = c("Hays", "Travis", "Williamson"),
    category_classified = c("cafe", "full_service_restaurant", "drinking_place"),
    event_window = c("previous", "recent")) %>%
    left_join(count(events, county, category_classified, event_window, name = "opening_events"),
      by = c("county", "category_classified", "event_window")) %>%
    mutate(opening_events = coalesce(opening_events, 0L), analysis_as_of_date = cutoff)
  if (any(coverage$opening_events == 0L)) stop("An amenity county/category/window cell is empty.", call. = FALSE)
  write_csv(coverage, file.path(date_dir, "amenity_category_window_qa.csv"))
  write_csv(categorized, file.path(date_dir, "amenity_source_candidates.csv"))
  write_csv(events %>% count(county, category_classified, event_window, disposition,
                             source_fields_changed, name = "opening_events"),
    file.path(date_dir, "amenity_source_disposition_qa.csv"))
  write_csv(events %>% group_by(county, category_classified, event_window) %>% summarise(
    opening_events = n(), closed_by_cutoff = sum(!active_as_of),
    missing_permit_dates = sum(is.na(permit_date)), first_of_month_pct = 100 * mean(first_of_month_flag),
    january_first_pct = 100 * mean(january_first_flag), .groups = "drop"),
    file.path(date_dir, "amenity_event_date_qa.csv"))
  coverage_contract <- list(
    window_complete = NA, status = "retrospective_reconstructed_completeness_unverified",
    retrospective_usable = TRUE,
    semantics = "Event-date reconstruction, not information published by cutoff",
    source_rule = "Archive-first stable-ID ledger; live-only IDs supplement; same ledger for both cutoffs",
    limitation = "Archived and rolling records cannot establish exhaustive historical coverage",
    corroboration_status = "not_rebuilt_not_used_in_index")
  candidate <- c(windows, list(sales_tax_locations = categorized, taxonomy = taxonomy,
    corroboration_status = "not_rebuilt_not_used_in_index",
    coverage_contract = coverage_contract, source_manifest_path = manifest_path,
    source_manifest_sha256 = digest::digest(file = manifest_path, algo = "sha256"),
    source_snapshot_ids = c(archive_spec$source_id, live_spec$source_id),
    reconstruction_rule_version = "archive_first_stable_id_v1",
    processed_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")))
  candidate_path <- file.path(date_dir, "amenity_source_candidates.rds")
  saveRDS(candidate, candidate_path)
  candidate_paths <- c(candidate_paths, candidate_path)
  candidate_events[[i]] <- events
  summaries[[i]] <- tibble(analysis_as_of_date = cutoff,
    previous_window_start = windows$previous_window_start, recent_window_start = windows$recent_window_start,
    opening_events = nrow(events), unique_addresses = n_distinct(events$address_key),
    previous_openings = sum(events$event_window == "previous"), recent_openings = sum(events$event_window == "recent"),
    archive_only_events = sum(events$disposition == "archive_only_retained"),
    live_only_supplements = sum(events$disposition == "live_only_supplement"),
    shared_ids_with_changed_source_fields = sum(events$source_fields_changed),
    duplicate_id_aliases_removed = sum(categorized$duplicate_opening_alias),
    window_complete = NA, retrospective_usable = TRUE)
}
summary <- bind_rows(summaries)
write_csv(summary, file.path(output_root, "amenity_snapshot_summary.csv"))
write_csv(bind_rows(lapply(seq_along(candidate_events), function(i) {
  candidate_events[[i]] %>% transmute(event_id, analysis_as_of_date = cutoffs[[i]],
    category_classified, event_window, opening_date, address_key, selected_source_id)
})), file.path(output_root, "amenity_paired_event_membership.csv"))

prepare_only <- tolower(Sys.getenv("EWS_AMENITY_PREPARE_ONLY", "false")) %in% c("true", "1", "yes")
if (!prepare_only) {
  geocode_cache <- "data/raw_amenities/historical/amenity_geocodes.csv"
  legacy_cache <- "data/raw_amenities/amenity_census_geocodes.csv"
  if (!file.exists(geocode_cache) && file.exists(legacy_cache)) {
    if (!file.copy(legacy_cache, geocode_cache)) stop("Could not seed historical geocode cache.", call. = FALSE)
  }
  for (i in seq_along(cutoffs)) {
    date_dir <- dirname(candidate_paths[[i]])
    environment <- c(paste0("EWS_ANALYSIS_AS_OF_DATE=", cutoffs[[i]]),
      paste0("EWS_AMENITY_CANDIDATES_FILE=", candidate_paths[[i]]),
      paste0("EWS_AMENITY_OUTPUT_DIR=", date_dir),
      paste0("EWS_AMENITY_GEOCODE_CACHE=", geocode_cache),
      paste0("EWS_AMENITY_SCALING_REFERENCE=", if (i == 1L) "" else
        file.path(dirname(candidate_paths[[1]]), "amenity_scaling.rds")))
    status <- system2(file.path(R.home("bin"), "Rscript"),
      args = "scripts/data/amenities.R", env = environment)
    if (status != 0L) stop("Amenity processing failed for ", cutoffs[[i]], call. = FALSE)
  }
  features <- lapply(candidate_paths, function(path) readRDS(file.path(dirname(path), "amenity_change_features_by_hex.rds")))
  if (!identical(features[[1]]$hex_id, features[[2]]$hex_id)) stop("Paired amenity hex support differs.", call. = FALSE)
  combined <- bind_rows(features)
  write_csv(combined, file.path(output_root, "amenity_features_paired.csv"))
  saveRDS(combined, file.path(output_root, "amenity_features_paired.rds"))
  measures <- c("amenity_change_index", "amenity_recent_weighted_openings", "amenity_previous_weighted_openings",
                 "amenity_weighted_opening_change", "amenity_recent_opening_events", "amenity_previous_opening_events")
  changes <- inner_join(select(features[[1]], hex_id, all_of(measures)),
    select(features[[2]], hex_id, all_of(measures)), by = "hex_id", suffix = c("_2025", "_2026"))
  for (measure in measures) changes[[paste0("delta_", measure)]] <-
    changes[[paste0(measure, "_2026")]] - changes[[paste0(measure, "_2025")]]
  write_csv(changes, file.path(output_root, "amenity_feature_changes_by_hex.csv"))
  qa <- bind_rows(lapply(candidate_paths, function(path)
    read_csv(file.path(dirname(path), "amenity_hex_distribution_qa.csv"), show_col_types = FALSE)))
  write_csv(qa, file.path(output_root, "amenity_paired_spatial_qa.csv"))
}

code_paths <- c("R/amenity_classification.R", "R/amenity_scoring.R", "scripts/data/amenities.R",
  "scripts/part2/build_amenity_snapshots.R", "scripts/data/prepare_historical_amenity_sources.py")
run_manifest <- list(schema_version = 1L,
  status = if (prepare_only) "candidates_prepared" else "paired_features_complete",
  processed_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  git_commit = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  source_manifest = manifest_path, source_manifest_sha256 = digest::digest(file = manifest_path, algo = "sha256"),
  source_ids = c(archive_spec$source_id, live_spec$source_id), cutoffs = as.character(cutoffs),
  reconstruction = "archive_first_stable_id_v1", retrospective = TRUE,
  scoring = "Equal category weights, 800m linear exposure, two 18-month windows; 2025 component bounds frozen for 2026",
  coverage = "Retrospective usable; exhaustive source completeness unverified; corroboration not rebuilt",
  inputs = build_file_manifest(c(taxonomy_path, "output/hex_grid.rds", code_paths), hash_files = TRUE),
  outputs = build_file_manifest(c(candidate_paths, if (!prepare_only) c(
    file.path(output_root, "amenity_features_paired.rds"), file.path(output_root, "amenity_feature_changes_by_hex.csv"),
    file.path(output_root, "2025-04-01", "amenity_scaling.rds"),
    "data/raw_amenities/historical/amenity_geocodes.csv")), hash_files = TRUE))
jsonlite::write_json(run_manifest, file.path(output_root, "amenity_run_manifest.json"),
  auto_unbox = TRUE, pretty = TRUE, na = "null")
print(summary)
cat("Amenity-only outputs:", output_root, "\n")
