# Documentation

This directory separates current guidance from dated analytical audits.

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
- [`methods/unit-count-modeling.md`](methods/unit-count-modeling.md): parcel
  unit-source hierarchy, model validation, integration, and promotion.

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
  random and spatially blocked evidence supporting the seven-cluster baseline.

Obsolete prototype summaries and informal version histories are retained in Git
history rather than the active documentation tree. Formal future releases
should use Git tags and GitHub release notes.
