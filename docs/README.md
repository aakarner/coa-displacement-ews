# Documentation

This directory separates current guidance from dated analytical audits.
The root [analytical changelog](../CHANGELOG.md) provides the running record of
major decisions and corrections; superseded generated model runs are not
archived automatically.

## Getting Started

- [`quickstart.md`](quickstart.md): installation, configuration, pipeline
  execution, and diagnostics.
- [`../data/README.md`](../data/README.md): source inventory, coverage limits,
  caches, and generated data artifacts.

## Current Methods

These documents describe the intended analytical architecture and should change
with the corresponding code and configuration:

- [`methods/analytical-workflow.md`](methods/analytical-workflow.md): the
  three-part pipeline and dependency structure.
- [`methods/cluster-methodology.md`](methods/cluster-methodology.md): baseline
  cluster selection and fixed future-vintage assignment.
- [`methods/current-measurement.md`](methods/current-measurement.md): corrected
  single-cutoff Part 1 measurement, shared recipes and current-only eligibility.
- [`methods/unit-count-modeling.md`](methods/unit-count-modeling.md): parcel
  unit-source hierarchy, model validation, integration, and promotion.
- [`methods/historical-ownership.md`](methods/historical-ownership.md): pinned
  2024/2025 ownership inputs, fixed parcel/unit support, and missingness-aware
  common-support comparisons for Part 2.
- [`methods/historical-amenities.md`](methods/historical-amenities.md): paired
  April 2025/2026 retrospective amenity sources, event reconciliation,
  geocoding, fixed scoring and reproduction commands.
- [`methods/historical-acs.md`](methods/historical-acs.md): paired ACS rent and
  vulnerability, common dollars, fixed allocation, uncertainty handling and
  frozen component scoring.
- [`methods/historical-events.md`](methods/historical-events.md): paired 311
  and demolition windows, fixed city footprint, historical coverage screens,
  missingness-aware counts and frozen indices.
- [`methods/historical-feature-matrix.md`](methods/historical-feature-matrix.md):
  coverage-aware eviction reconstruction, ownership index assembly and the
  common seven-feature sample; no cluster fitting or ML.
- [`methods/historical-cluster-comparison.md`](methods/historical-cluster-comparison.md):
  the separate retrospective 2025 baseline, fixed-2026 assignments, aligned
  later refit and conditional random/spatial robustness checks.

## Analytical Decisions

Files under [`decisions/`](decisions/) explain why consequential current choices
were made, what alternatives remain plausible, and what evidence should trigger
reconsideration. They are documentation, not pipeline inputs. The
[`decision index`](decisions/README.md) distinguishes accepted, provisional,
and superseded choices.

## Audit Snapshots

Files under `audits/` record results for a stated data and method vintage. They
support methodological decisions but are not automatically updated when the
pipeline changes:

- [`audits/parcel-acs-unit-audit-2026-07.md`](audits/parcel-acs-unit-audit-2026-07.md):
  parcel/ACS housing-unit reconciliation and populated zero-unit review.
- [`audits/part1-cluster-selection-2026-08.md`](audits/part1-cluster-selection-2026-08.md):
  superseded August selection evidence, not validation of the corrected fit.
- [`audits/part1-harmonized-measurement-2026-09.md`](audits/part1-harmonized-measurement-2026-09.md):
  current complete-component Part 1 refit, current-only eligibility, profile
  interpretation, coverage, validation and retirement of obsolete outputs.
- [`audits/amenity-historical-coverage-2026-09.md`](audits/amenity-historical-coverage-2026-09.md):
  archived-source coverage and reconstruction options for a one-year-back
  amenity feature.
- [`audits/amenity-historical-processing-2026-09.md`](audits/amenity-historical-processing-2026-09.md):
  implemented paired amenity features, geocoding coverage and validation results.
- [`audits/part2-historical-readiness-2026-09.md`](audits/part2-historical-readiness-2026-09.md):
  seven-feature readiness table for the April 2025/2026 retrospective comparison,
  exact source windows, completed corrected analysis and coverage requirements.
- [`audits/acs-historical-processing-2026-09.md`](audits/acs-historical-processing-2026-09.md):
  completed paired ACS inputs, coverage, uncertainty corrections and validation.
- [`audits/part2-rent-tract-fallback-2026-09.md`](audits/part2-rent-tract-fallback-2026-09.md):
  test findings now integrated into the ACS pair: reliable full-series tract
  fallback with geographic level and the three-term recipe fixed across dates.
- [`audits/events-historical-processing-2026-09.md`](audits/events-historical-processing-2026-09.md):
  completed 311/demolition pairs, coverage and location exclusions, source-count
  reconciliation and validation.
- [`audits/part2-paired-matrix-processing-2026-09.md`](audits/part2-paired-matrix-processing-2026-09.md):
  corrected eviction/ownership indices and the verified 2,351-cell common
  seven-feature sample, with complete fixed recipes and explicit exclusions.
- [`audits/part2-cluster-comparison-2026-09.md`](audits/part2-cluster-comparison-2026-09.md):
  corrected historical results, fixed-definition movement, aligned later refit,
  qualitative concern transitions, robustness and interpretation cautions.
- [`audits/historical-ownership-import-2026-09.md`](audits/historical-ownership-import-2026-09.md):
  initial ownership import and validation, before the Williamson 2025 source was found.
- [`audits/williamson-2025-ownership-source-search-2026-09.md`](audits/williamson-2025-ownership-source-search-2026-09.md):
  verified public certified-roll and GIS sources and why the earlier search missed them.
- [`audits/williamson-2025-ownership-integration-2026-09.md`](audits/williamson-2025-ownership-integration-2026-09.md):
  intermediate 2025 certified/GIS reconciliation, coverage, and source-sensitivity results.
- [`audits/williamson-2024-ownership-integration-2026-09.md`](audits/williamson-2024-ownership-integration-2026-09.md):
  subsequent 2024 GIS supplementation of the existing certified roll, updated
  two-year coverage, and source-sensitive comparisons.

Obsolete prototype summaries and informal version histories are retained in Git
history rather than the active documentation tree. Formal future releases
should use Git tags and GitHub release notes.

## Work Handoffs

- [`handoffs/landlord-mapper-historical-ownership-poc.md`](handoffs/landlord-mapper-historical-ownership-poc.md):
  speed-first instructions for producing comparable 2024 and 2025 Travis
  ownership snapshots in the sibling `landlord-mapper` repository.
