# Williamson 2025 ownership integration: September 7, 2026

**Subsequent update:** this audit preserves the intermediate run that
supplemented only 2025. The [2024 integration audit](williamson-2024-ownership-integration-2026-09.md)
records the later same-year supplementation of the existing 2024 certified
roll. References below to 2024 clipping as the principal bottleneck describe
the earlier evidence state, not a missing 2024 file.

## Bottom line

The public 2025 certified roll and July 2025 TxGIO/WCAD parcel archive have
been integrated into the isolated Part 2 ownership workflow. The stage ran
end to end, retaining 474,610 parcel-years and the existing 7,027-cell grid.
The pinned landlord-mapper classifier, current Part 1 outputs, and fixed
parcel coordinates/unit weights are unchanged. No clusters were refit or
reassigned, no ML work was resumed, and issue #5's full upstream migration
remains outside this integration.

The missing-2025-file problem is closed. The remaining important limitation
is **incomplete 2024 ownership evidence**, especially clipped printed names.
Do not use longer 2025 names to fill that historical gap.

## Sources and reconciliation

The [source-search audit](williamson-2025-ownership-source-search-2026-09.md)
records the public download links, exact archive hashes, and why the earlier
search missed the certified report. Those archives are now pinned by
[`config/williamson_ownership_sources.json`](../../config/williamson_ownership_sources.json).

The report's internal date is July 11, 2025; its online posting is July 16,
2026. The GIS acquisition metadata is July 1, 2025, with July 28 processing.
This is an appraisal-vintage reconstruction, not an exact same-day ownership
inventory or proof of public availability at a historical cutoff.

The source adapter keeps certified ownership primary. Longer names require
same-parcel identity, exact/clipped-prefix name agreement, and matching
mailing street plus city/state/ZIP. GIS mailing and situs are used together
only when both corroborate the certified evidence. Independent name agreement
can resolve a name even if situs is unconfirmed, but the certified address
pair is retained and the disagreement is flagged. Owner/mailing disagreement
keeps certified evidence. GIS-only parcels receive no inferred homestead or
owner ID. Missing/suppressed evidence remains unknown.

Across the full fixed Williamson target of 13,626 parcels:

- The certified report matches 13,179; GIS supplies evidence for all 13,626.
- GIS has 13,738 matching features. The 112 repeated features have identical
  evidence and collapse before classification/weighting: no duplicated units.
- 3,940 printed owner names are safely extended. Including exact names at the
  printed column boundary, 4,652 potentially clipped names have their
  completeness confirmed. Names at GIS's observed 80-character boundary remain
  potentially clipped; longer fields are not proof of a complete co-owner list.
- 12,663 parcels permit corroborated name/address reconciliation; 447 are
  GIS-only. Twenty GIS-placeholder cases retain their certified evidence.
- 496 parcels are flagged: 27 owner disagreements, 73 unconfirmed mailing
  comparisons, and 396 unconfirmed situs comparisons. These are differences
  in evidence, not verified ownership transfers.

One 276.70-unit parcel accounts for most remaining unknown corporate units.
Its mailing comparison appears to involve an expanded care-of line and the
same PO box, rather than a different destination, but the conservative rule
does not confirm it. It remains unknown and in the review output. All other
flagged parcels have at most one validated unit. Improving that case is a
targeted review opportunity, not a reason to relax matching globally.

## Coverage in the actual hex analysis

These figures describe eligible fixed parcel points assigned to the project's
H3 grid, not countywide totals or an exact City of Austin polygon clip. The
Williamson denominator is **13,014 parcels and 18,917.23 validated units**.
Source matching and usable classification are distinct.

| Williamson evidence | Source-matched parcels | Units with known corporate status | Units with known financialized/entity status |
| --- | ---: | ---: | ---: |
| 2024 certified report, unchanged | 12,911 (99.21%) | 87.8% | 69.0% |
| 2025 certified report only | 12,593 (96.76%) | 85.6% | 69.1% |
| 2025 certified + reconciled GIS | 13,014 (100%) | 98.1% | 99.2% |

The reconciled 2025 result leaves 88 mapped parcels / 362.70 units with
unknown corporate status, and 147 mapped parcels with unknown financialized
status. All three flags are known for 12,808 mapped parcels.

The full fixed target, including outside-grid Williamson parcels, has 13,531
known corporate classifications and 13,471 known financialized classifications
out of 13,626. Neither source matches nor these flag counts imply a complete
co-owner roster.

## What the two-year comparison can support

The common cohort requires known corporate and financialized flags in **both**
years. Williamson has **8,435 common parcels / 12,707.07 units**, covering
64.8% of its mapped parcels and **67.2% of its mapped units**. Better 2025
coverage cannot repair the 2024 evidence limitations under this rule.

Across all counties, 3,063 cells pass the provisional screen of at least 20
common units and at least 95% common parcel and unit coverage. That is eight
more than the pre-integration run's 3,055, with no previously passing cell
lost. There are 4,082 cells with some common support, versus 3,879 before.
Of the 228 cells containing Williamson residential parcels, nine pass; this
includes mixed-county boundary cells and is not a Williamson-only grid count.

Three variants hold the full parcel/unit denominator fixed:

| Variant | Williamson common parcels | Williamson common units | All-county screened cells |
| --- | ---: | ---: | ---: |
| Main reconciled sources | 8,435 | 12,707.07 | 3,063 |
| 2025 certified report only | 7,980 | 12,094.53 | 3,063 |
| Exclude flagged source disagreements | 8,096 | 12,368.07 | 3,063 |

The same cells pass in all three variants, and their corporate-unit-share
changes are identical. The GIS supplement improves coverage outside that
screened sample; it does not expand the screened sample beyond what the
certified report alone allows. `source_agreement` is the output label for the
conflict-exclusion variant, **not** a requirement that both sources affirmatively
corroborate every parcel. GIS-only rows and valid certified rows paired with
unavailable GIS evidence remain eligible if not otherwise conflicted.

On Williamson's observed common cohort, corporate units increase by four
(32 one-unit parcels become corporate; 29 parcels representing 28 units cease
to be classified corporate). The corporate-unit share moves from 35.057% to
35.089%. These are cohort/classification changes, not a countywide trend or
confirmed acquisitions. The cohort excludes substantial 2024 uncertainty.

For parcels in both certified rolls, transition review compares original
printed names across years. The separate full-name change field can reflect
2025 name extension; it is not counted as independent evidence of a transfer.
Equal clipped names also cannot rule out a hidden owner change.

## Verification and reproducibility

```sh
Rscript scripts/part2/build_ownership_snapshots.R
python3 -B -m unittest discover -s tests -p 'test_*ownership*.py'
Rscript tests/test_ownership_snapshots.R
Rscript tests/test_williamson_txgio.R
Rscript tests/test_williamson_ownership_outputs.R
```

The Python suite contains 42 tests. Synthetic tests cover year isolation,
name/address reconciliation, missing homesteads, placeholders, duplicate
features, original certified names, unknown-value serialization, and stale
source/target/evidence hash rejection. R checks cover aggregation, source
preparation, and the completed outputs/provenance. The targets graph parses
and tracks the new sources, scripts, and outputs.

All variants retain 233,334 mapped residential parcels / 502,256.94 units.
Parcel-year keys are unique; weights and hex totals agree with canonical
current inputs. An independent read-only review reproduced all 13,626
reconciliation outputs without an invariant failure. Source ZIPs, extracted
evidence, and owner-level review outputs remain Git-ignored.

Key outputs under `output/part2/ownership/`:

- `ownership_county_qa.csv`, `ownership_hex_change.csv`, and
  `ownership_snapshot_manifest.json`: primary results and provenance.
- `williamson_2025_source_reconciliation_qa.csv` and
  `williamson_2025_source_conflict_review.csv`: source resolution and remaining
  review cases; parcel-level evidence is in `williamson_2025_source_reconciliation.csv`.
- `ownership_source_variant_county_qa.csv`,
  `ownership_source_variant_summary.csv`, and the certified-only/source-agreement
  hex-change files: explicit sensitivity comparisons.
- `ownership_transition_review.csv`: classification-change evidence with the
  original-name comparison basis preserved.

The earlier run's manifest and summary/change tables are retained in the
ignored `pre_williamson_2025_integration/` subdirectory. The
[method document](../methods/historical-ownership.md) is the current contract;
the [initial import audit](historical-ownership-import-2026-09.md) preserves the
earlier results.

## Recommended next step

Investigate a comparable **2024 Williamson GIS/owner archive** to address
printed-field clipping using evidence from the same year. If sufficient
evidence cannot be recovered, choose explicitly between a narrower
coverage-screened proof of concept and a missingness/bounds-based comparison.
In parallel with that source decision, the next distinct Part 2 work is the
remaining historical smoke-signal assembly (including amenities), followed by
cross-vintage transformations and a final comparable cluster sample. This
ownership integration alone does not authorize or complete those steps.
