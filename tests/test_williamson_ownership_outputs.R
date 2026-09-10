#!/usr/bin/env Rscript
# Output-level regressions for a completed ownership-only Part 2 stage.
# Run after rebuilding: Rscript tests/test_williamson_ownership_outputs.R
# Reads existing artifacts only; never rebuilds, downloads, or edits inputs.
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
})
source("R/ownership_snapshots.R")

assert <- function(ok, message) {
  if (!isTRUE(ok)) stop(message, call. = FALSE)
}
same_values <- function(x, y, tolerance = 1e-8) {
  isTRUE(all.equal(x, y, check.attributes = FALSE, tolerance = tolerance))
}
read_output <- function(name, character_only = FALSE) {
  read_csv(file.path(out, name), show_col_types = FALSE, na = c("", "NA"),
    col_types = if (character_only) cols(.default = col_character()) else NULL)
}
normalized_names <- function(x) toupper(trimws(gsub("[[:space:]]+", " ", x)))

out <- "output/part2/ownership"
spec <- jsonlite::read_json("config/ownership_snapshot_spec.json", simplifyVector = TRUE)
annual_output_names <- function(year) c(
  paste0("williamson_txgio_", year, "_evidence.csv"),
  paste0("williamson_txgio_", year, "_manifest.json"),
  paste0("williamson_", year, "_certified_only_snapshots.csv"),
  paste0("williamson_", year, "_source_reconciliation.csv"),
  paste0("williamson_", year, "_source_reconciliation_qa.csv"),
  paste0("williamson_", year, "_source_conflict_review.csv")
)
expected_outputs <- c(
  "ownership_target_parcels.csv", "parcel_ownership_snapshots.rds",
  "ownership_features_by_hex_year.rds", "ownership_common_support_by_hex_year.rds",
  "ownership_hex_change.csv", "ownership_county_qa.csv",
  "ownership_transition_review.csv", "ownership_transition_evidence_qa.csv",
  "ownership_certified_only_hex_change.csv", "ownership_source_agreement_hex_change.csv",
  "ownership_2024_certified_only_hex_change.csv",
  "ownership_source_variant_county_qa.csv", "ownership_source_variant_summary.csv",
  "ownership_snapshot_manifest.json", "other_county_owner_snapshots_2024_2025.csv",
  "other_county_sources_manifest.json", unlist(lapply(spec$tax_years, annual_output_names), use.names = FALSE)
)
missing <- expected_outputs[!file.exists(file.path(out, expected_outputs))]
assert(!length(missing), paste("Complete the ownership stage before this test; missing:",
  paste(missing, collapse = ", ")))
manifest <- jsonlite::read_json(file.path(out, "ownership_snapshot_manifest.json"))
assert(!is.null(manifest$semantics$source_variants),
  "Ownership manifest predates the source-reconciliation stage; rebuild it first.")
assert(setequal(as.integer(unlist(manifest$williamson_source_tax_years)), spec$tax_years),
  "Ownership manifest does not declare both annual Williamson source vintages.")
assert(!is.null(manifest$semantics$source_variants$prior_2024_certified_only) &&
  !is.null(manifest$semantics$source_variants$suppression_counts),
  "Ownership manifest lacks annual sensitivity or parcel-year suppression semantics.")

# Check every declared input/output hash, including nested county/GIS sources.
# Cache by absolute path so repeated provenance entries hash large files once.
hashes <- new.env(parent = emptyenv())
checked_paths <- character()
verify_hashes <- function(node) {
  if (!is.list(node)) return(invisible(NULL))
  if (is.character(node$path) && length(node$path) == 1L && !is.null(node$sha256)) {
    assert(file.exists(node$path), paste("Manifest file is missing:", node$path))
    path <- normalizePath(node$path, mustWork = TRUE)
    if (!exists(path, envir = hashes, inherits = FALSE)) {
      assign(path, ownership_sha256(path), envir = hashes)
    }
    assert(identical(get(path, envir = hashes, inherits = FALSE), node$sha256),
      paste("Manifest checksum mismatch:", node$path))
    checked_paths <<- union(checked_paths, path)
  }
  for (item in node) if (is.list(item)) verify_hashes(item)
  invisible(NULL)
}
verify_hashes(manifest)
verify_hashes(jsonlite::read_json(file.path(out, "other_county_sources_manifest.json")))
gis_manifests <- setNames(lapply(spec$tax_years, function(year) {
  result <- jsonlite::read_json(file.path(out, paste0("williamson_txgio_", year, "_manifest.json")))
  verify_hashes(result)
  result
}), as.character(spec$tax_years))
declared_inputs <- vapply(manifest$inputs, `[[`, character(1), "path")
assert(all(c("scripts/data/prepare_williamson_txgio.R",
  "scripts/data/williamson_ownership_reconciliation.py", "config/williamson_ownership_sources.json",
  file.path(out, unlist(lapply(spec$tax_years, function(year) annual_output_names(year)[1:3]),
    use.names = FALSE))) %in% declared_inputs),
  "Main manifest lacks required source-reconciliation input provenance.")
declared_outputs <- basename(vapply(manifest$outputs, `[[`, character(1), "path"))
assert(all(c("ownership_certified_only_hex_change.csv", "ownership_source_agreement_hex_change.csv",
  "ownership_2024_certified_only_hex_change.csv",
  "ownership_source_variant_county_qa.csv", "ownership_source_variant_summary.csv") %in% declared_outputs),
  "Main manifest lacks the source-sensitivity outputs.")

surface <- readRDS("output/residential_parcels_unit_promoted.rds") |>
  mutate(parcel_id = as.character(parcel_id))
panel <- readRDS(file.path(out, "parcel_ownership_snapshots.rds"))
assert(nrow(panel) == length(spec$tax_years) * nrow(surface),
  "Ownership rows no longer enumerate the fixed surface once per vintage.")
assert(!anyDuplicated(panel[c("parcel_id", "tax_year")]), "Duplicate parcel-year rows.")
panel <- ownership_validate_rows(panel, surface, spec$tax_years, spec$classification_rule_version)
for (year in spec$tax_years) {
  annual <- panel |> filter(tax_year == year)
  order <- match(surface$parcel_id, annual$parcel_id)
  assert(same_values(annual$residential_units[order], surface[[spec$unit_field]]),
    paste("Promoted unit weights changed in", year))
}
first <- panel |> filter(tax_year == min(spec$tax_years)) |> arrange(parcel_id)
last <- panel |> filter(tax_year == max(spec$tax_years)) |> arrange(parcel_id)
assert(identical(first$hex_id, last$hex_id), "Parcel hex assignment changed across vintages.")
assert(identical(first$land_use_excluded, last$land_use_excluded),
  "Land-use eligibility changed across vintages.")

# Fixed coordinates must agree with the current canonical point surface.
current <- readRDS("output/residential_parcels_for_hex_sf.rds")
eligible <- surface |> filter(!coalesce(unit_land_use_validation_excluded, FALSE))
assert(setequal(as.character(current$parcel_id), eligible$parcel_id),
  "Canonical and ownership eligibility differ.")
current_order <- match(eligible$parcel_id, as.character(current$parcel_id))
coordinates <- st_coordinates(st_transform(current, 4326))[current_order, , drop = FALSE]
assert(same_values(unname(coordinates[, "X"]), eligible$lon, 1e-10) &&
  same_values(unname(coordinates[, "Y"]), eligible$lat, 1e-10),
  "Promoted parcel coordinates differ from the canonical point surface.")
assert(same_values(current$property_units[current_order], eligible[[spec$unit_field]]),
  "Canonical point units differ from the ownership weights.")
canonical <- readRDS("output/corporate_ownership_by_hex.rds") |> st_drop_geometry()
full <- readRDS(file.path(out, "ownership_features_by_hex_year.rds"))
for (year in spec$tax_years) {
  annual <- full |> filter(tax_year == year)
  order <- match(canonical$hex_id, annual$hex_id)
  assert(!anyNA(order) && same_values(annual$residential_parcels[order], canonical$residential_parcels) &&
    same_values(annual$residential_units[order], canonical$residential_units),
    paste("Canonical hex support is not conserved in", year))
}

target_ids <- surface$parcel_id[surface$source_county == "Williamson"]
certified_by_year <- list()
for (year in spec$tax_years) {
  w <- panel |> filter(source_county == "Williamson", tax_year == year) |> arrange(parcel_id)
  gis <- read_output(paste0("williamson_txgio_", year, "_evidence.csv"), TRUE)
  gis_manifest <- gis_manifests[[as.character(year)]]
  assert(nrow(w) == length(target_ids) && !anyDuplicated(w$parcel_id) &&
    setequal(w$parcel_id, target_ids), paste("Williamson", year, "does not enumerate each target ID once."))
  assert(all(gis$parcel_id %in% target_ids) &&
    length(unique(gis$parcel_id)) == gis_manifest$target_coverage$matched_parcels &&
    setequal(setdiff(target_ids, gis$parcel_id), unlist(gis_manifest$target_coverage$unmatched_parcel_ids)),
    paste("GIS evidence disagrees with matched/unmatched target diagnostics in", year))
  assert(all(as.integer(gis$TAX_YEAR) == year) &&
    all(gsub("-", "", gis$DATE_ACQ, fixed = TRUE) ==
      gsub("-", "", gis_manifest$source$acquisition_date, fixed = TRUE)),
    paste("GIS evidence does not match its own verified vintage in", year))
  assert(nrow(gis) == gis_manifest$repeated_evidence$exported_source_rows &&
    sum(ownership_bool(gis$source_evidence_is_duplicate)) ==
      gis_manifest$repeated_evidence$exact_normalized_duplicate_rows,
    paste("GIS duplicate diagnostics disagree with raw evidence in", year))
  assert(!any(grepl("^Shape|geometry", names(gis), ignore.case = TRUE)),
    "Prepared ownership evidence unexpectedly contains geometry.")
  diagnostic <- read_output(paste0("williamson_", year, "_source_reconciliation.csv"), TRUE) |>
    arrange(parcel_id)
  assert(identical(w$parcel_id, diagnostic$parcel_id),
    paste("Reconciliation diagnostics do not enumerate Williamson targets in", year))
  for (field in c("reconciliation_status", "reconciliation_version")) {
    assert(identical(w[[paste0("source_", field)]], diagnostic[[field]]),
      paste("Source diagnostic lost from parcel snapshots:", year, field))
  }
  for (field in c("reconciliation_conflict", "name_extended", "name_completeness_confirmed",
                  "mailing_extended", "situs_extended")) {
    assert(identical(ownership_bool(w[[paste0("source_", field)]]), ownership_bool(diagnostic[[field]])),
      paste("Boolean source diagnostic lost from parcel snapshots:", year, field))
  }
  certified <- read_output(paste0("williamson_", year, "_certified_only_snapshots.csv"), TRUE) |>
    mutate(tax_year = as.integer(tax_year)) |> arrange(parcel_id)
  certified <- ownership_validate_rows(certified,
    filter(surface, source_county == "Williamson"), year, spec$classification_rule_version)
  certified_by_year[[as.character(year)]] <- certified
  has_certified <- ownership_bool(diagnostic$certified_present)
  assert(identical(w$source_reported_owner_names[has_certified],
    certified$source_reported_owner_names[has_certified]),
    paste("Shared Williamson parcels lost their original certified names in", year))
  conflict_review <- read_output(paste0("williamson_", year, "_source_conflict_review.csv"), TRUE)
  assert(setequal(conflict_review$parcel_id,
    w$parcel_id[ownership_bool(w$source_reconciliation_conflict) %in% TRUE]),
    paste("Conflict review does not enumerate exactly the flagged parcels in", year))
  for (other_year in setdiff(spec$tax_years, year)) {
    assert(!any(grepl(paste0("txgio-williamson-", other_year, "-"),
      coalesce(w$source_snapshot_id, ""), fixed = TRUE)),
      paste("Another GIS vintage leaked into Williamson", year))
  }
}

# The 2024 supplement may change only Williamson 2024 source classifications.
backup_path <- file.path(out, "pre_williamson_2024_integration",
  "other_county_owner_snapshots_2024_2025.csv")
if (file.exists(backup_path)) {
  previous <- read_csv(backup_path, col_types = cols(.default = col_character()),
    na = c("", "NA"), show_col_types = FALSE) |>
    mutate(tax_year = as.integer(tax_year)) |>
    filter(!(source_county == "Williamson" & tax_year == 2024L)) |>
    arrange(source_county, tax_year, parcel_id)
  current_other <- read_output("other_county_owner_snapshots_2024_2025.csv", TRUE) |>
    mutate(tax_year = as.integer(tax_year)) |>
    filter(!(source_county == "Williamson" & tax_year == 2024L)) |>
    arrange(source_county, tax_year, parcel_id)
  assert(identical(previous[c("source_county", "tax_year", "parcel_id")],
    current_other[c("source_county", "tax_year", "parcel_id")]),
    "Adding 2024 GIS changed another county-year's parcel support.")
  for (field in c("is_owner_occupied", "has_financialized_owner", "is_corporate_owned")) {
    assert(identical(ownership_bool(previous[[field]]), ownership_bool(current_other[[field]])),
      paste("Adding 2024 GIS changed Williamson 2025 or a Hays classification:", field))
  }
}
travis_reference <- read_csv(file.path(spec$upstream_repository,
  "output/historical_ownership/travis_owner_snapshots_2024_2025.csv"),
  col_types = cols(.default = col_character()), na = c("", "NA"), show_col_types = FALSE) |>
  mutate(tax_year = as.integer(tax_year)) |> arrange(tax_year, parcel_id)
travis_current <- panel |> filter(source_county == "Travis") |> arrange(tax_year, parcel_id)
assert(identical(travis_reference[c("parcel_id", "tax_year")],
  travis_current[c("parcel_id", "tax_year")]), "Williamson integration changed Travis parcel support.")
for (field in c("is_owner_occupied", "has_financialized_owner", "is_corporate_owned")) {
  assert(identical(ownership_bool(travis_reference[[field]]), ownership_bool(travis_current[[field]])),
    paste("Williamson integration changed a pinned Travis ownership flag:", field))
}

transitions <- read_output("ownership_transition_review.csv", TRUE)
assert(all(c("owner_names_changed", "comparison_name_changed", "comparison_name_basis") %in% names(transitions)),
  "Transition review lacks separate full-name and comparable-name diagnostics.")
both_certified <- transitions$source_county == "Williamson"
for (year in spec$tax_years) {
  both_certified <- both_certified & grepl(paste0("wcad-", year, "-certification-printed-report"),
    coalesce(transitions[[paste0("source_snapshot_id_", year)]], ""), fixed = TRUE)
}
earlier_name <- transitions[[paste0("owner_names_", min(spec$tax_years))]]
later_name <- transitions[[paste0("owner_names_", max(spec$tax_years))]]
expected_change <- earlier_name != later_name
expected_change[both_certified] <- normalized_names(transitions[[paste0(
  "source_reported_owner_names_", min(spec$tax_years))]][both_certified]) !=
  normalized_names(transitions[[paste0("source_reported_owner_names_", max(spec$tax_years))]][both_certified])
assert(identical(ownership_bool(transitions$owner_names_changed), earlier_name != later_name) &&
  identical(ownership_bool(transitions$comparison_name_changed), expected_change),
  "Transition review confuses full-name changes with comparable certified-name changes.")
assert(identical(transitions$comparison_name_basis, ifelse(both_certified,
  "certified_reported_owner_names", "classified_owner_names")), "Wrong transition name-comparison basis.")
extension_only <- both_certified & ownership_bool(transitions$owner_names_changed) %in% TRUE &
  expected_change %in% FALSE
if (any(extension_only)) assert(all(ownership_bool(transitions$comparison_name_changed[extension_only]) %in% FALSE),
  "A mere source-name extension was reported as a comparable ownership-name change.")
evidence_qa <- read_output("ownership_transition_evidence_qa.csv", TRUE)
assert(all(c("comparison_name_changed", "comparison_name_basis") %in% names(evidence_qa)) &&
  !"owner_names_changed" %in% names(evidence_qa), "Transition evidence QA does not group comparable names.")

variants <- read_output("ownership_source_variant_summary.csv")
variant_labels <- c("main", "certified_only", "source_agreement", "prior_2024_certified_only")
assert(setequal(variants$source_variant, variant_labels) &&
  !anyDuplicated(variants$source_variant), "Missing or duplicate source variants.")
for (field in c("total_hexes", "residential_hexes", "residential_parcels_inside_hex_grid",
                "residential_units_inside_hex_grid")) {
  assert(same_values(variants[[field]], rep(variants[[field]][1], nrow(variants))),
    paste("Source variants changed fixed full support:", field))
}
main_variant <- variants |> filter(source_variant == "main")
agreement <- variants |> filter(source_variant == "source_agreement")
assert(agreement$common_parcels_inside_hex_grid <= main_variant$common_parcels_inside_hex_grid &&
  agreement$common_units_inside_hex_grid <= main_variant$common_units_inside_hex_grid + 1e-7,
  "Source-agreement masking increased common support.")
conflicts <- panel |> filter(source_county == "Williamson", tax_year %in% spec$tax_years,
  !land_use_excluded,
  ownership_bool(source_reconciliation_conflict) %in% TRUE)
unique_conflicts <- distinct(conflicts, parcel_id, residential_units)
assert(all(c("source_conflict_parcel_years_suppressed", "source_conflict_unit_years_suppressed",
  "source_conflict_unique_parcels_suppressed", "source_conflict_unique_units_suppressed") %in% names(variants)),
  "Sensitivity suppression lacks separate annual-record and unique-parcel counts.")
assert(agreement$source_conflict_parcel_years_suppressed == nrow(conflicts) &&
  same_values(agreement$source_conflict_unit_years_suppressed, sum(conflicts$residential_units)) &&
  agreement$source_conflict_unique_parcels_suppressed == nrow(unique_conflicts) &&
  same_values(agreement$source_conflict_unique_units_suppressed, sum(unique_conflicts$residential_units)),
  "Sensitivity suppression does not match flagged eligible parcel-years and unique parcels.")
variant_county <- read_output("ownership_source_variant_county_qa.csv")
fixed <- variant_county |> filter(scope != "common_support_inside_hex_grid")
reference <- fixed |> filter(source_variant == "main") |>
  select(source_county, tax_year, scope, residential_parcels, residential_units)
for (variant in setdiff(variant_labels, "main")) {
  compared <- fixed |> filter(source_variant == variant) |>
    left_join(reference, by = c("source_county", "tax_year", "scope"), suffix = c("", "_main"))
  assert(same_values(compared$residential_parcels, compared$residential_parcels_main) &&
    same_values(compared$residential_units, compared$residential_units_main),
    paste("County full-support totals changed for", variant))
}
unaffected <- variant_county |> filter(source_county != "Williamson") |>
  arrange(source_county, tax_year, scope)
for (variant in setdiff(variant_labels, "main")) {
  assert(same_values(unaffected |> filter(source_variant == "main") |> select(-source_variant),
    unaffected |> filter(source_variant == variant) |> select(-source_variant)),
    paste("Williamson sensitivity changed another county's QA:", variant))
}

# Annual full-support QA must reflect exactly the requested source replacement
# or masking; common-support QA can change in either year with the pair cohort.
flags <- c("is_owner_occupied", "has_financialized_owner", "is_corporate_owned")
for (year in spec$tax_years) {
  annual_support <- panel |>
    filter(source_county == "Williamson", tax_year == year) |>
    select(parcel_id, source_county, tax_year, residential_units, land_use_excluded, hex_id)
  certified_panel <- certified_by_year[[as.character(year)]] |>
    select(parcel_id, source_county, tax_year, all_of(flags), classification_status) |>
    left_join(annual_support, by = c("parcel_id", "source_county", "tax_year")) |>
    filter(!land_use_excluded)
  agreement_panel <- panel |>
    filter(source_county == "Williamson", tax_year == year, !land_use_excluded) |>
    mutate(.conflicted = ownership_bool(source_reconciliation_conflict) %in% TRUE,
      across(all_of(flags), ~ if_else(.conflicted, NA, .x)),
      classification_status = if_else(.conflicted, "matched_ambiguous", classification_status))
  for (scope_name in c("all_eligible_parcels", "inside_hex_grid")) {
    reference_certified <- ownership_summarise(
      if (scope_name == "inside_hex_grid") filter(certified_panel, !is.na(hex_id)) else certified_panel,
      c("source_county", "tax_year"))
    observed_certified <- variant_county |>
      filter(source_variant == "certified_only", source_county == "Williamson",
        tax_year == year, scope == scope_name) |>
      select(all_of(names(reference_certified)))
    assert(same_values(observed_certified, reference_certified),
      paste("Certified-only sensitivity did not replace the intended annual evidence:", year, scope_name))
    reference_agreement <- ownership_summarise(
      if (scope_name == "inside_hex_grid") filter(agreement_panel, !is.na(hex_id)) else agreement_panel,
      c("source_county", "tax_year"))
    observed_agreement <- variant_county |>
      filter(source_variant == "source_agreement", source_county == "Williamson",
        tax_year == year, scope == scope_name) |>
      select(all_of(names(reference_agreement)))
    assert(same_values(observed_agreement, reference_agreement),
      paste("Source-agreement sensitivity did not mask every annual conflict:", year, scope_name))
    prior_reference <- if (year == 2024L) reference_certified else variant_county |>
      filter(source_variant == "main", source_county == "Williamson", tax_year == year, scope == scope_name) |>
      select(all_of(names(reference_certified)))
    observed_prior <- variant_county |>
      filter(source_variant == "prior_2024_certified_only", source_county == "Williamson",
        tax_year == year, scope == scope_name) |>
      select(all_of(names(reference_certified)))
    assert(same_values(observed_prior, prior_reference),
      paste("Prior-year sensitivity changed the wrong annual evidence:", year, scope_name))
  }
}
main_change <- read_output("ownership_hex_change.csv") |> arrange(hex_id)
agreement_change <- read_output("ownership_source_agreement_hex_change.csv") |> arrange(hex_id)
certified_change <- read_output("ownership_certified_only_hex_change.csv") |> arrange(hex_id)
prior_change <- read_output("ownership_2024_certified_only_hex_change.csv") |> arrange(hex_id)
assert(identical(main_change$hex_id, agreement_change$hex_id) &&
  identical(main_change$hex_id, certified_change$hex_id) &&
  identical(main_change$hex_id, prior_change$hex_id), "Source variants changed the hex universe.")
previous_change_path <- file.path(out, "pre_williamson_2024_integration", "ownership_hex_change.csv")
if (file.exists(previous_change_path)) {
  previous_change <- read_csv(previous_change_path, show_col_types = FALSE, na = c("", "NA")) |>
    arrange(hex_id)
  assert(same_values(prior_change, previous_change),
    "The 2024 certified-only sensitivity does not reproduce the prior integration baseline.")
}
for (year in spec$tax_years) {
  for (measure in c("common_parcels", "common_units")) {
    field <- paste0(measure, "_", year)
    assert(all(agreement_change[[field]] <= main_change[[field]] + 1e-7),
      paste("Source-agreement support increased within a hex:", field))
  }
}
message("Williamson ownership output regressions passed: ", nrow(panel),
  " parcel-years, ", length(target_ids), " Williamson parcels in each source year, ", length(checked_paths),
  " verified file hashes; ", sum(extension_only), " comparable-name extension cases checked.")
