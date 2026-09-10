#!/usr/bin/env Rscript
# Import pinned ownership vintages without replacing the current Part 1 surface.
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
  library(tidyr)
})
source("R/ownership_snapshots.R")
spec_path <- "config/ownership_snapshot_spec.json"
spec <- jsonlite::read_json(spec_path, simplifyVector = TRUE)
out <- "output/part2/ownership"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
upstream <- normalizePath(spec$upstream_repository, mustWork = TRUE)
classifier <- file.path(upstream, spec$classifier_path)
travis_path <- file.path(upstream, "output/historical_ownership/travis_owner_snapshots_2024_2025.csv")
travis_manifest_path <- file.path(upstream, "output/historical_ownership/travis_owner_snapshot_manifest.json")
ownership_assert_hash(classifier, spec$classifier_sha256)
ownership_assert_hash(travis_path, spec$travis_snapshot_sha256)
committed_blob <- system2("git", c("-C", shQuote(upstream), "rev-parse",
  paste0(spec$upstream_commit, ":", spec$classifier_path)), stdout = TRUE)
working_blob <- system2("git", c("-C", shQuote(upstream), "hash-object", shQuote(classifier)), stdout = TRUE)
stopifnot(length(committed_blob) == 1L, identical(committed_blob, working_blob))
travis_manifest <- jsonlite::read_json(travis_manifest_path, simplifyVector = FALSE)
stopifnot(identical(travis_manifest$pipeline$script$sha256, spec$classifier_sha256),
  identical(travis_manifest$pipeline$classification_rule_version, spec$classification_rule_version))
ownership_assert_hash("data/residential_parcels_for_hex.csv", travis_manifest$sources$ews_surface$sha256)

surface_path <- "output/residential_parcels_unit_promoted.rds"
grid_path <- "output/hex_grid.rds"
surface <- readRDS(surface_path) |> as_tibble() |>
  mutate(parcel_id = as.character(parcel_id),
    residential_units = .data[[spec$unit_field]],
    land_use_excluded = coalesce(unit_land_use_validation_excluded, FALSE),
    across(c(is_owner_occupied, has_financialized_owner, is_corporate_owned),
      .names = "baseline_{.col}"))
stopifnot(!anyDuplicated(surface$parcel_id), !anyNA(surface$residential_units),
  all(is.finite(surface$residential_units)), all(surface$residential_units >= 0),
  all(surface$is_residential), all(surface$parcel_count == 1))
target_path <- file.path(out, "ownership_target_parcels.csv")
surface |>
  select(parcel_id, source_county, is_residential, starts_with("situs"),
    any_of(c("residential_use_category", "improvement_state_code", "land_state_code")),
    residential_units, starts_with("baseline_")) |>
  mutate(property_units = residential_units) |>
  write_csv(target_path, na = "")
if ("--prepare-only" %in% commandArgs(trailingOnly = TRUE)) {
  message("Prepared ownership target: ", target_path)
  quit(status = 0)
}

source("scripts/data/prepare_williamson_txgio.R")
txgio_paths <- unlist(lapply(spec$tax_years, function(year) {
  message("Preparing target-restricted evidence from the pinned ", year, " TxGIO source...")
  prepare_williamson_txgio(tax_year = year, target_csv = target_path)
}), use.names = FALSE)
message("Classifying Hays and Williamson through the pinned upstream classifier...")
python <- Sys.getenv("EWS_PYTHON", unset = "python3")
status <- system2(python, c("-B", "scripts/data/build_other_county_ownership.py",
  "--target", shQuote(target_path), "--classifier", shQuote(classifier),
  "--output-dir", shQuote(out)))
if (status != 0) stop("Other-county ownership adapter failed.")
other_path <- file.path(out, "other_county_owner_snapshots_2024_2025.csv")
other_manifest_path <- file.path(out, "other_county_sources_manifest.json")
certified_only_paths <- file.path(out,
  paste0("williamson_", spec$tax_years, "_certified_only_snapshots.csv"))
read_owners <- function(path) read_csv(path, col_types = cols(.default = col_character()),
  na = c("", "NA"), show_col_types = FALSE) |>
  select(-any_of(c("property_units", "residential_use_category")))
owners <- bind_rows(read_owners(travis_path), read_owners(other_path)) |>
  mutate(tax_year = as.integer(tax_year))
owners <- ownership_validate_rows(owners, surface, spec$tax_years, spec$classification_rule_version)
owners$source_reconciliation_conflict <- ownership_bool(owners$source_reconciliation_conflict)

# Sensitivities share the same target IDs, geometry, and promoted unit weights.
# Replace both Williamson years for the certified-only variant; the additional
# prior-year variant isolates the effect of supplementing Williamson 2024.
certified_only <- bind_rows(lapply(certified_only_paths, read_owners)) |>
  mutate(tax_year = as.integer(tax_year))
certified_only <- ownership_validate_rows(certified_only,
  filter(surface, source_county == "Williamson"), spec$tax_years, spec$classification_rule_version)
if ("source_reconciliation_conflict" %in% names(certified_only)) {
  certified_only$source_reconciliation_conflict <- ownership_bool(certified_only$source_reconciliation_conflict)
}
certified_only_owners <- bind_rows(
  filter(owners, source_county != "Williamson"), certified_only)
certified_only_owners <- ownership_validate_rows(certified_only_owners, surface,
  spec$tax_years, spec$classification_rule_version)
prior_2024_certified_only_owners <- bind_rows(
  filter(owners, !(source_county == "Williamson" & tax_year == 2024L)),
  filter(certified_only, tax_year == 2024L))
prior_2024_certified_only_owners <- ownership_validate_rows(prior_2024_certified_only_owners,
  surface, spec$tax_years, spec$classification_rule_version)
source_agreement_owners <- owners
conflicted <- source_agreement_owners$source_county == "Williamson" &
  source_agreement_owners$tax_year %in% spec$tax_years &
  source_agreement_owners$source_reconciliation_conflict %in% TRUE
source_agreement_owners[conflicted,
  c("is_owner_occupied", "has_financialized_owner", "is_corporate_owned")] <- NA
source_agreement_owners$classification_status[conflicted] <- "matched_ambiguous"
source_agreement_owners <- ownership_validate_rows(source_agreement_owners, surface,
  spec$tax_years, spec$classification_rule_version)

message("Assigning fixed promoted parcel coordinates to the existing H3 grid...")
grid <- readRDS(grid_path)
stopifnot(!anyDuplicated(grid$hex_id), all(grid$area_km2 > 0))
points <- surface |>
  filter(!land_use_excluded) |>
  select(parcel_id, lon, lat) |>
  st_as_sf(coords = c("lon", "lat"), crs = 4326) |>
  st_transform(st_crs(grid))
hits <- st_within(points, grid)
stopifnot(all(lengths(hits) <= 1L))
assignment <- tibble(parcel_id = points$parcel_id,
  hex_id = grid$hex_id[vapply(hits, function(h) if (length(h)) h else NA_integer_, integer(1))])
support <- surface |>
  select(parcel_id, source_county, residential_units, land_use_excluded,
    starts_with("baseline_")) |>
  left_join(assignment, by = "parcel_id") |>
  mutate(support_status = case_when(land_use_excluded ~ "land_use_excluded",
    is.na(hex_id) ~ "outside_hex_grid", TRUE ~ "inside_hex_grid"))
panel <- owners |> left_join(support, by = c("parcel_id", "source_county"))
stopifnot(nrow(panel) == 2L * nrow(surface))
analytical <- filter(panel, !land_use_excluded)
hex <- ownership_hex_outputs(analytical, grid, spec$minimum_common_coverage, spec$minimum_hex_units)
panel <- panel |> left_join(hex$pair_support, by = "parcel_id")
county_qa <- bind_rows(
  ownership_summarise(analytical, c("source_county", "tax_year")) |> mutate(scope = "all_eligible_parcels"),
  ownership_summarise(filter(analytical, !is.na(hex_id)), c("source_county", "tax_year")) |> mutate(scope = "inside_hex_grid")
)
support_qa <- support |> group_by(source_county, support_status) |>
  summarise(parcels = n(), units = sum(residential_units), .groups = "drop")
status_qa <- panel |> group_by(source_county, tax_year, support_status, classification_status) |>
  summarise(parcels = n(), units = sum(residential_units), .groups = "drop")
parity <- analytical |> filter(tax_year == max(spec$tax_years)) |>
  select(parcel_id, source_county, hex_id, residential_units, starts_with("baseline_"),
    is_owner_occupied, has_financialized_owner, is_corporate_owned) |>
  pivot_longer(c(is_owner_occupied, has_financialized_owner, is_corporate_owned),
    names_to = "indicator", values_to = "snapshot_value") |>
  mutate(baseline_value = case_when(indicator == "is_owner_occupied" ~ baseline_is_owner_occupied,
    indicator == "has_financialized_owner" ~ baseline_has_financialized_owner,
    TRUE ~ baseline_is_corporate_owned),
    change = case_when(is.na(snapshot_value) ~ "snapshot_unknown",
      is.na(baseline_value) ~ "baseline_unknown", snapshot_value == baseline_value ~ "same",
      snapshot_value ~ "false_to_true", TRUE ~ "true_to_false"))
parity_qa <- parity |> group_by(source_county, indicator, change) |>
  summarise(parcels = n(), units = sum(residential_units), .groups = "drop")
common_panel <- panel |> filter(common_owner_support %in% TRUE, !land_use_excluded, !is.na(hex_id))
common_county <- ownership_summarise(common_panel, c("source_county", "tax_year"))

variant_owners <- list(main = owners, certified_only = certified_only_owners,
  source_agreement = source_agreement_owners,
  prior_2024_certified_only = prior_2024_certified_only_owners)
variant_results <- lapply(names(variant_owners), function(variant_name) {
  variant_panel <- variant_owners[[variant_name]] |>
    left_join(support, by = c("parcel_id", "source_county")) |>
    filter(!land_use_excluded)
  variant_hex <- if (variant_name == "main") hex else ownership_hex_outputs(
    variant_panel, grid, spec$minimum_common_coverage, spec$minimum_hex_units)
  variant_common <- variant_panel |>
    inner_join(filter(variant_hex$pair_support, common_owner_support), by = "parcel_id") |>
    filter(!is.na(hex_id))
  variant_county <- bind_rows(
    ownership_summarise(variant_panel, c("source_county", "tax_year")) |>
      mutate(scope = "all_eligible_parcels"),
    ownership_summarise(filter(variant_panel, !is.na(hex_id)), c("source_county", "tax_year")) |>
      mutate(scope = "inside_hex_grid"),
    ownership_summarise(variant_common, c("source_county", "tax_year")) |>
      mutate(scope = "common_support_inside_hex_grid")
  ) |> mutate(source_variant = variant_name, .before = 1)
  latest_common <- filter(variant_hex$common, tax_year == max(spec$tax_years))
  latest_full <- filter(variant_hex$full, tax_year == max(spec$tax_years))
  main_ready <- hex$change$comparison_ready[match(variant_hex$change$hex_id, hex$change$hex_id)]
  suppressed <- variant_panel |>
    filter(source_county == "Williamson", tax_year %in% spec$tax_years,
      source_reconciliation_conflict %in% TRUE)
  suppressed_unique <- distinct(suppressed, parcel_id, residential_units)
  variant_summary <- tibble(
    source_variant = variant_name,
    total_hexes = nrow(grid), residential_hexes = sum(latest_full$has_residential_support),
    comparison_ready_hexes = sum(variant_hex$change$comparison_ready),
    readiness_changed_from_main_hexes = sum(variant_hex$change$comparison_ready != main_ready),
    hexes_with_common_support = sum(latest_common$common_parcels > 0),
    residential_parcels_inside_hex_grid = sum(latest_full$residential_parcels),
    residential_units_inside_hex_grid = sum(latest_full$residential_units),
    common_parcels_inside_hex_grid = sum(latest_common$common_parcels),
    common_units_inside_hex_grid = sum(latest_common$common_units),
    common_parcel_coverage = ownership_safe_share(sum(latest_common$common_parcels),
      sum(latest_full$residential_parcels)),
    common_unit_coverage = ownership_safe_share(sum(latest_common$common_units),
      sum(latest_full$residential_units)),
    comparison_ready_common_parcels = sum(latest_common$common_parcels[latest_common$comparison_ready]),
    comparison_ready_common_units = sum(latest_common$common_units[latest_common$comparison_ready]),
    source_conflict_parcel_years_suppressed = if (variant_name == "source_agreement") nrow(suppressed) else 0L,
    source_conflict_unit_years_suppressed = if (variant_name == "source_agreement") sum(suppressed$residential_units) else 0,
    source_conflict_unique_parcels_suppressed = if (variant_name == "source_agreement") nrow(suppressed_unique) else 0L,
    source_conflict_unique_units_suppressed = if (variant_name == "source_agreement") sum(suppressed_unique$residential_units) else 0
  )
  list(hex = variant_hex, county = variant_county, summary = variant_summary)
})
names(variant_results) <- names(variant_owners)
source_variant_county_qa <- bind_rows(lapply(variant_results, `[[`, "county"))
source_variant_summary <- bind_rows(lapply(variant_results, `[[`, "summary"))

transitions <- common_panel |>
  select(parcel_id, source_county, hex_id, residential_units, tax_year,
    is_corporate_owned, has_financialized_owner, owner_ids, owner_names,
    source_reported_owner_names, source_snapshot_id, homestead_positive, address_match_positive) |>
  pivot_wider(names_from = tax_year, values_from = c(is_corporate_owned, has_financialized_owner,
    owner_ids, owner_names, source_reported_owner_names, source_snapshot_id,
    homestead_positive, address_match_positive))
for (field in c("is_corporate_owned", "owner_ids", "owner_names", "homestead_positive", "address_match_positive")) {
  transitions[[paste0(field, "_changed")]] <-
    transitions[[paste0(field, "_", min(spec$tax_years))]] != transitions[[paste0(field, "_", max(spec$tax_years))]]
}
both_certified <- transitions$source_county == "Williamson"
for (year in spec$tax_years) {
  both_certified <- both_certified & grepl(paste0("wcad-", year, "-certification-printed-report"),
    coalesce(transitions[[paste0("source_snapshot_id_", year)]], ""), fixed = TRUE)
}
comparison_names <- lapply(spec$tax_years, function(year) {
  reported <- transitions[[paste0("source_reported_owner_names_", year)]]
  ifelse(both_certified, toupper(trimws(gsub("[[:space:]]+", " ", reported))),
    transitions[[paste0("owner_names_", year)]])
})
transitions$comparison_name_changed <- comparison_names[[1]] != comparison_names[[2]]
transitions$comparison_name_basis <- ifelse(both_certified,
  "certified_reported_owner_names", "classified_owner_names")
transition_qa <- transitions |>
  group_by(source_county, .data[[paste0("is_corporate_owned_", min(spec$tax_years))]],
    .data[[paste0("is_corporate_owned_", max(spec$tax_years))]]) |>
  summarise(parcels = n(), units = sum(residential_units), .groups = "drop")
transition_evidence_qa <- transitions |> filter(is_corporate_owned_changed) |>
  group_by(source_county, comparison_name_changed, comparison_name_basis, owner_ids_changed,
    homestead_positive_changed, address_match_positive_changed) |>
  summarise(parcels = n(), units = sum(residential_units), .groups = "drop")

# Verify geography and weights match the current canonical input surface.
current <- readRDS("output/residential_parcels_for_hex_sf.rds") |> st_drop_geometry() |>
  transmute(parcel_id = as.character(parcel_id), current_units = property_units)
eligible <- filter(support, !land_use_excluded)
stopifnot(setequal(current$parcel_id, eligible$parcel_id))
unit_check <- left_join(eligible, current, by = "parcel_id")
stopifnot(all(abs(unit_check$residential_units - unit_check$current_units) < 1e-8))
canonical <- readRDS("output/corporate_ownership_by_hex.rds") |> st_drop_geometry() |>
  select(hex_id, canonical_parcels = residential_parcels, canonical_units = residential_units)
canonical_check <- hex$full |> filter(tax_year == max(spec$tax_years)) |>
  left_join(canonical, by = "hex_id")
stopifnot(!anyNA(canonical_check$canonical_units),
  all(abs(canonical_check$residential_units - canonical_check$canonical_units) < 1e-7),
  all(canonical_check$residential_parcels == canonical_check$canonical_parcels))

saveRDS(panel, file.path(out, "parcel_ownership_snapshots.rds"))
saveRDS(hex$full, file.path(out, "ownership_features_by_hex_year.rds"))
saveRDS(hex$common, file.path(out, "ownership_common_support_by_hex_year.rds"))
artifacts <- list(ownership_county_qa = county_qa, ownership_support_qa = support_qa,
  ownership_classification_status_qa = status_qa, ownership_2025_parity_qa = parity_qa,
  ownership_common_support_county_qa = common_county,
  ownership_common_support_transitions = transition_qa,
  ownership_transition_evidence_qa = transition_evidence_qa,
  ownership_transition_review = transitions |> filter(is_corporate_owned_changed) |>
    arrange(desc(residential_units)),
  ownership_large_hex_changes = hex$change |>
    filter(comparison_ready, abs(delta_pct_corporate_units) >= 10) |>
    arrange(desc(abs(delta_pct_corporate_units))),
  ownership_features_by_hex_year = hex$full,
  ownership_common_support_by_hex_year = hex$common, ownership_hex_change = hex$change,
  ownership_certified_only_hex_change = variant_results$certified_only$hex$change,
  ownership_source_agreement_hex_change = variant_results$source_agreement$hex$change,
  ownership_2024_certified_only_hex_change = variant_results$prior_2024_certified_only$hex$change,
  ownership_source_variant_county_qa = source_variant_county_qa,
  ownership_source_variant_summary = source_variant_summary,
  ownership_2025_parity_differences = filter(parity, change != "same"))
for (name in names(artifacts)) write_csv(artifacts[[name]], file.path(out, paste0(name, ".csv")), na = "NA")
write_csv(panel |> filter(is.na(is_corporate_owned) | is.na(has_financialized_owner)),
  file.path(out, "ownership_unknown_review.csv"), na = "NA")
input_paths <- c(spec_path, classifier, travis_path, travis_manifest_path, surface_path,
  grid_path, "output/residential_parcels_for_hex_sf.rds", "output/corporate_ownership_by_hex.rds", "R/ownership_snapshots.R",
  "scripts/part2/build_ownership_snapshots.R", "scripts/data/build_other_county_ownership.py",
  "scripts/data/prepare_williamson_txgio.R", "scripts/data/williamson_ownership_reconciliation.py",
  "config/williamson_ownership_sources.json", unname(txgio_paths), certified_only_paths,
  other_path, other_manifest_path)
input_hashes <- lapply(input_paths, function(path) list(path = path, sha256 = ownership_sha256(path)))
manifest <- list(schema_version = 2L, generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  specification = spec, pinned_upstream_classifier_blob = committed_blob,
  williamson_source_tax_years = as.integer(spec$tax_years),
  promoted_unit_version = unique(surface$unit_model_promotion_version),
  inputs = input_hashes, other_counties = jsonlite::read_json(other_manifest_path),
  semantics = list(common_support = "Same parcels with known corporate and financialized flags in both years; fixed coordinates and validated units.",
    missing = "Unknown flags retained as NA; strict full-support features NA where required ownership is unknown; bounds include unknown units.",
    coverage = "Common-cohort coverage measures data completeness, not statistical representativeness.",
    cutoff = spec$semantics, source_comparability = "2024 Travis property-year owner; 2025 special-export owner. County sources retain their own appraisal-vintage semantics.",
    source_variants = list(main = "Reconcile each Williamson year separately against its own pinned GIS and certified sources; annual GIS names, mailing/situs evidence, and certified homesteads never cross years.",
      certified_only = "Replace both Williamson 2024 and 2025 rows with independently revalidated certified-only rows; retain every other county-year, fixed coordinates, and validated units.",
      source_agreement = "Retain primary rows and set all three ownership flags unknown for source-reconciliation conflicts in either Williamson year; mark affected records matched_ambiguous and recompute common support.",
      prior_2024_certified_only = "Replace only Williamson 2024 with certified-only evidence, retaining reconciled Williamson 2025 and every other county-year; isolates the 2024 GIS supplement effect.",
      suppression_counts = "Conflict parcel-years and unit-years count flagged eligible annual records and their fixed unit weights across both years; unique parcel/unit counts count each affected parcel once.",
      outputs = "Explicit sensitivity outputs only; the primary ownership outputs retain the main reconciliation."),
    transition_name_comparison = "When both Williamson vintages contain certified evidence, compare original certified reported names; retain full owner_names_changed separately so annual GIS name extensions do not become reported ownership transitions.",
    scope = "Ownership-only Part 2 input; no cluster refit/reassignment or ML training; no change to canonical Part 1 outputs."),
  checks = list(parcel_year_rows = nrow(panel), unit_surface_exactly_matches_current = TRUE,
    unique_parcel_years = TRUE, unique_hex_assignment = TRUE,
    canonical_hex_counts_and_units_match = TRUE,
    total_hexes = nrow(grid), residential_hexes = sum(hex$full$tax_year == max(spec$tax_years) & hex$full$has_residential_support),
    comparison_ready_hexes = sum(hex$change$comparison_ready),
    hexes_with_common_support = sum(hex$common$tax_year == max(spec$tax_years) & hex$common$common_parcels > 0),
    county_coverage = county_qa, source_variant_summary = source_variant_summary))
output_paths <- file.path(out, c("parcel_ownership_snapshots.rds",
  "ownership_features_by_hex_year.rds", "ownership_common_support_by_hex_year.rds",
  paste0(names(artifacts), ".csv"), "ownership_unknown_review.csv"))
manifest$outputs <- lapply(output_paths, function(path) list(path = path,
  bytes = file.info(path)$size, sha256 = ownership_sha256(path)))
jsonlite::write_json(manifest, file.path(out, "ownership_snapshot_manifest.json"), pretty = TRUE, auto_unbox = TRUE, na = "null", digits = 12)
message("Ownership import complete: ", nrow(panel), " parcel-years; ", sum(hex$change$comparison_ready), " screened comparison-ready hexes.")
print(county_qa |> filter(scope == "inside_hex_grid") |>
  select(source_county, tax_year, residential_parcels, unit_match_rate, corporate_unknown_units))
