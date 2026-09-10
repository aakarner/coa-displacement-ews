"""Synthetic source-contract tests for county ownership adapters."""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("county_ownership", ROOT / "scripts/data/build_other_county_ownership.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
LM = adapter.load_classifier(ROOT.parent / "landlord-mapper/historical_ownership.py")
LM.ews_rule_version = LM.classifier_rule_version()


def report_line(owner="", description="", exemption="", values=""):
    return owner.ljust(31) + description.ljust(35) + exemption.ljust(16) + values


def report_block(name="SAMPLE LLC", address="200 OTHER RD", exemption=""):
    return [
        report_line("PID: R123 (O44)", "SUBDIVISION", exemption),
        report_line("R-123", "LOT 1"),
        report_line("LOCAL ID:", "BLOCK 2"),
        report_line(name),
        report_line(address, "SITUS: 100 MAIN ST"),
        report_line("DALLAS TX 75201", "AUSTIN, TX 78727"),
        report_line("", "LAND SPTB: A1, IMP SPTB: B1"),
        report_line("AGENT:", "ENTS: CAU,GWI"),
        report_line("DO NOT READ AGENT LLC"),
    ]


def target(residential="TRUE"):
    return {"parcel_id": "WILLIAMSON:R123", "source_county": "Williamson",
            "is_residential": residential, "property_units": "10",
            "situs_address": "2099 CURRENT ST", "is_corporate_owned": "FALSE"}


def gis_raw(parcel="R123", name="SAMPLE LLC", **changes):
    row = {
        "Prop_ID": parcel, "OWNER_NAME": name, "MAIL_LINE1": "200 OTHER RD",
        "MAIL_ADDR": "200 OTHER RD DALLAS TX 75201", "MAIL_CITY": "DALLAS",
        "MAIL_STAT": "TX", "MAIL_ZIP": "75201",
        "SITUS_ADDR": "100 MAIN ST, AUSTIN, TX 78727", "SITUS_STAT": "TX",
        "SOURCE": "WILLIAMSON APPRAISAL DISTRICT", "TAX_YEAR": "2025",
        "DATE_ACQ": "20250701",
    }
    row.update(changes)
    return row


def prepared_gis_fixture(root):
    """Create independent, tiny source/evidence/target files with matching hashes."""
    args = SimpleNamespace(root=root, output_dir=root / "prepared", target=root / "target.csv")
    args.output_dir.mkdir()
    archive = root / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("metadata.txt", "Synthetic July 2025 ownership source")
    adapter.write_table(args.target, [target()])
    evidence = args.output_dir / "williamson_txgio_2025_evidence.csv"
    adapter.write_table(evidence, [gis_raw(OBJECTID=1), gis_raw(OBJECTID=2)])
    config = {"path": "source.zip", "sha256": adapter.digest(archive)}
    manifest = {
        "inputs": {
            "source_zip": {"sha256": config["sha256"]},
            "target_csv": {"sha256": adapter.digest(args.target)},
        },
        "output": {"sha256": adapter.digest(evidence)},
    }
    manifest_path = args.output_dir / "williamson_txgio_2025_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return args, config, archive, evidence, manifest_path


class HistoricalCountyTests(unittest.TestCase):
    def test_report_columns_and_agent_exclusion(self):
        row = adapter.report_block_to_standard(report_block(), 2024)
        self.assertEqual(row["owner_name"], "SAMPLE LLC")
        self.assertEqual(row["owner_addr_line1"], "200 OTHER RD")
        self.assertEqual(row["owner_addr_city"], "DALLAS")
        self.assertEqual(row["situs_street"], "100 MAIN ST")
        self.assertEqual(row["improvement_state_code"], "B1")
        result = adapter.snapshot(target(), 2024, [row], LM)
        self.assertTrue(result["is_corporate_owned"])
        self.assertEqual(result["classification_status"], "matched_classified")

    def test_zero_amount_homestead_is_positive(self):
        row = adapter.report_block_to_standard(report_block(exemption="HS $         0"), 2024)
        self.assertEqual(row["homestead_flag"], "TRUE")
        result = adapter.snapshot(target(), 2024, [row], LM)
        self.assertTrue(result["is_owner_occupied"])
        self.assertFalse(result["is_corporate_owned"])

    def test_truncated_negative_name_is_unknown(self):
        row = adapter.report_block_to_standard(report_block(name="NATURAL PERSON AND SECOND OWNER"), 2024)
        self.assertTrue(row["source_name_may_be_truncated"])
        result = adapter.snapshot(target(), 2024, [row], LM)
        self.assertIsNone(result["has_financialized_owner"])
        self.assertIsNone(result["is_corporate_owned"])
        self.assertIn("printed name", result["classification_note"])

    def test_positive_marker_survives_truncation(self):
        row = adapter.report_block_to_standard(report_block(name="LLC WITH A VERY LONG OWNER NAME"), 2024)
        result = adapter.snapshot(target(), 2024, [row], LM)
        self.assertTrue(result["has_financialized_owner"])
        self.assertTrue(result["is_corporate_owned"])

    def test_no_historical_situs_is_unknown_without_current_fallback(self):
        row = adapter.report_block_to_standard(report_block(), 2024)
        row["situs_street"] = ""
        result = adapter.snapshot(target(), 2024, [row], LM)
        self.assertIsNone(result["is_owner_occupied"])
        self.assertIsNone(result["is_corporate_owned"])

    def test_truncated_address_mismatch_is_unknown(self):
        row = adapter.report_block_to_standard(report_block(address="9876 AN EXCEPTIONALLY LONG ROAD"), 2024)
        result = adapter.snapshot(target(), 2024, [row], LM)
        self.assertIsNone(result["is_owner_occupied"])
        self.assertIsNone(result["is_corporate_owned"])

    def test_missing_parcel_is_not_noncorporate(self):
        result = adapter.snapshot(target("FALSE"), 2024, [], LM)
        self.assertEqual(result["classification_status"], "source_parcel_not_found")
        self.assertIsNone(result["is_corporate_owned"])

    def test_multiple_owners_and_conflicts_use_shared_rules(self):
        first = adapter.report_block_to_standard(report_block(), 2024)
        second = dict(first, owner_id="O45", owner_name="JANE SMITH", homestead_flag="TRUE")
        result = adapter.snapshot(target(), 2024, [first, second], LM)
        self.assertTrue(result["has_financialized_owner"])
        self.assertTrue(result["is_owner_occupied"])
        self.assertFalse(result["is_corporate_owned"])
        second["owner_id"] = first["owner_id"]
        result = adapter.snapshot(target(), 2024, [first, second], LM)
        self.assertEqual(result["classification_status"], "matched_ambiguous")
        self.assertIsNone(result["is_corporate_owned"])

    def test_hays_uses_annual_owner_situs_and_homestead(self):
        owner = {"QuickRefID": "R12", "OwnerID": "O44", "OwnerName": "SAMPLE LLC",
                 "Address1": "200 OTHER RD", "City": "DALLAS", "State": "TX",
                 "Zip": "75201", "ExemptionList": "HS,OV65", "ConfidentialOwner": "False"}
        prop = {"SitusStreetNumber": "100", "SitusStreetName": "MAIN",
                "SitusStreetSuffix": "ST", "SitusCity": "AUSTIN", "SitusState": "TX", "SitusZip": "78727"}
        row = adapter.hays_standard(owner, prop, 2024, LM)
        self.assertEqual(row["tax_year"], "2024")
        self.assertEqual(row["homestead_flag"], "TRUE")
        self.assertEqual(row["situs_street"], "MAIN")
        self.assertTrue(LM.aggregate_owner_rows([row])["is_owner_occupied"])

    def test_hays_rejects_wrong_source_vintage_before_reading_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hays_2024.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("2025 Certified Data Export/2025-CERTIFIED-OWNER.zip", b"")
            with self.assertRaisesRegex(ValueError, "Unexpected source year"):
                adapter.parse_hays(path, 2024, {}, LM)


class ReconciledCountyIntegrationTests(unittest.TestCase):
    def test_2024_reconciliation_is_year_local_and_keeps_missing_gis_certified(self):
        full_name = "THE NORTH AUSTIN HOUSING COMPANY LLC"
        cert = adapter.report_block_to_standard(report_block(name=full_name[:30]), 2024)
        gis, _ = adapter.reconciliation.deduplicate_gis_rows(
            [gis_raw(name=full_name, TAX_YEAR="2024", DATE_ACQ="20240601")],
            LM, year=2024, acquisition_date="2024-06-01")
        primary, original, review = adapter.reconcile_snapshots(
            {"WILLIAMSON:R123": target()}, {"WILLIAMSON:R123": [cert]}, gis, LM, year=2024)
        self.assertEqual(primary[0]["tax_year"], 2024)
        self.assertTrue(primary[0]["has_financialized_owner"])
        self.assertIsNone(original[0]["has_financialized_owner"])
        self.assertEqual(review[0]["tax_year"], 2024)
        no_gis, _, diagnostics = adapter.reconcile_snapshots(
            {"WILLIAMSON:R123": target()}, {"WILLIAMSON:R123": [cert]}, {}, LM, year=2024)
        self.assertEqual(no_gis[0]["source_snapshot_id"], cert["source_snapshot_id"])
        self.assertIsNone(no_gis[0]["has_financialized_owner"])
        self.assertEqual(diagnostics[0]["reconciliation_status"], "certified_only")
        with self.assertRaises(ValueError):
            adapter.reconcile_snapshots({"WILLIAMSON:R123": target()},
                {"WILLIAMSON:R123": [cert]}, gis, LM, year=2025)

    def test_verified_gis_rejects_wrong_year_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, _, _, _ = prepared_gis_fixture(Path(tmp))
            config["tax_year"] = 2024
            with self.assertRaisesRegex(ValueError, "configuration disagrees"):
                adapter.verified_gis(args, config, LM, year=2025)

    def test_verified_gis_rejects_inconsistent_declared_coverage(self):
        for section, field in (("output", "rows"), ("target_coverage", "matched_parcels")):
            with self.subTest(section=section), tempfile.TemporaryDirectory() as tmp:
                args, config, _, _, manifest_path = prepared_gis_fixture(Path(tmp))
                manifest = json.loads(manifest_path.read_text())
                manifest.setdefault(section, {})[field] = 999
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "disagrees with preparation manifest"):
                    adapter.verified_gis(args, config, LM)

    def test_reconciled_outputs_keep_fixed_parcels_units_and_original_certified_name(self):
        full_name = "THE NORTH AUSTIN HOUSING COMPANY LLC"
        primary = adapter.report_block_to_standard(report_block(name=full_name[:30]), 2025)
        selected = {
            "WILLIAMSON:R123": target(),
            "WILLIAMSON:R124": dict(target(), parcel_id="WILLIAMSON:R124", property_units="3.5"),
            "WILLIAMSON:R125": dict(target(), parcel_id="WILLIAMSON:R125", property_units="6"),
        }
        gis, _ = adapter.reconciliation.deduplicate_gis_rows(
            [gis_raw(name=full_name, OBJECTID=1), gis_raw(name=full_name, OBJECTID=2),
             gis_raw(parcel="R124"), gis_raw(parcel="R999")], LM
        )
        snapshots, certified_only, review = adapter.reconcile_snapshots(
            selected, {"WILLIAMSON:R123": [primary]}, gis, LM
        )
        for table in (snapshots, certified_only, review):
            self.assertEqual(len(table), len(selected))
            self.assertEqual({row["parcel_id"] for row in table}, set(selected))
        self.assertEqual(sum(row["residential_units"] for row in review), 19.5)
        self.assertEqual(snapshots[0]["n_owner_rows"], 1)
        self.assertEqual(snapshots[0]["owner_names"], full_name)
        self.assertEqual(snapshots[0]["source_reported_owner_names"], full_name[:30])
        self.assertEqual(review[0]["certified_owner_names"], full_name[:30])
        self.assertEqual(certified_only[0]["source_reported_owner_names"], full_name[:30])
        self.assertIsNone(certified_only[0]["has_financialized_owner"])
        self.assertTrue(snapshots[0]["has_financialized_owner"])
        self.assertIsNone(snapshots[2]["is_corporate_owned"])

    def test_snapshot_and_diagnostic_csv_preserve_true_false_and_unknown(self):
        primary = adapter.report_block_to_standard(
            report_block(name="THE NORTH AUSTIN HOUSING COMPA"), 2025
        )
        selected = {
            "WILLIAMSON:R123": target(),
            "WILLIAMSON:R124": dict(target(), parcel_id="WILLIAMSON:R124"),
        }
        gis, _ = adapter.reconciliation.deduplicate_gis_rows(
            [gis_raw(name="THE NORTH AUSTIN HOUSING COMPANY LLC"),
             gis_raw(parcel="R124", name="UNAVAILABLE", MAIL_LINE1="UNAVAILABLE")], LM
        )
        snapshots, _, review = adapter.reconcile_snapshots(
            selected, {"WILLIAMSON:R123": [primary]}, gis, LM
        )
        self.assertTrue(snapshots[0]["source_name_completeness_confirmed"])
        self.assertFalse(snapshots[0]["source_reconciliation_conflict"])
        self.assertIsNone(snapshots[1]["is_corporate_owned"])
        with tempfile.TemporaryDirectory() as tmp:
            for name, rows in (("snapshots", snapshots), ("review", review)):
                path = Path(tmp) / (name + ".csv")
                adapter.write_table(path, rows)
                with path.open(newline="", encoding="utf-8") as handle:
                    restored = list(csv.DictReader(handle))
                for original, serialized in zip(rows, restored):
                    for field, value in original.items():
                        if isinstance(value, bool) or value is None:
                            with self.subTest(table=name, field=field, value=value):
                                self.assertEqual(serialized[field], LM.bool_csv(value))
                                self.assertIs(LM.parse_optional_bool(serialized[field]), value)

    def test_verified_gis_accepts_consistent_fixture_and_deduplicates_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, _, _, manifest_path = prepared_gis_fixture(Path(tmp))
            owners, provenance = adapter.verified_gis(args, config, LM)
            self.assertEqual(set(owners), {"WILLIAMSON:R123"})
            self.assertEqual(len(owners["WILLIAMSON:R123"]), 1)
            self.assertEqual(provenance["deduplication"]["duplicate_rows_removed"], 1)
            self.assertEqual(provenance["preparation_manifest_sha256"], adapter.digest(manifest_path))

    def test_verified_gis_rejects_preparation_for_a_stale_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, _, _, _ = prepared_gis_fixture(Path(tmp))
            adapter.write_table(args.target, [dict(target(), property_units="11")])
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                adapter.verified_gis(args, config, LM)

    def test_verified_gis_rejects_modified_prepared_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, _, evidence, _ = prepared_gis_fixture(Path(tmp))
            adapter.write_table(evidence, [gis_raw(name="CHANGED OWNER LLC")])
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                adapter.verified_gis(args, config, LM)

    def test_verified_gis_rejects_modified_source_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, archive, _, _ = prepared_gis_fixture(Path(tmp))
            with zipfile.ZipFile(archive, "a") as zipped:
                zipped.writestr("changed.txt", "Changed source after evidence preparation")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                adapter.verified_gis(args, config, LM)

    def test_verified_gis_rejects_manifest_hashes_from_another_preparation(self):
        for section, entry in (("inputs", "source_zip"), ("inputs", "target_csv"), ("output", None)):
            with self.subTest(section=section, entry=entry), tempfile.TemporaryDirectory() as tmp:
                args, config, _, _, manifest_path = prepared_gis_fixture(Path(tmp))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                record = manifest[section][entry] if entry else manifest[section]
                record["sha256"] = "0" * 64
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    adapter.verified_gis(args, config, LM)

    def test_input_paths_use_pinned_annual_sources_and_exclude_current_owner_extract(self):
        paths = set(adapter.input_paths(ROOT))
        config_path = ROOT / "config/williamson_ownership_sources.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertIn(config_path, paths)
        self.assertIn(ROOT / config["certified_2025"]["path"], paths)
        self.assertIn(ROOT / config["gis_2025"]["path"], paths)
        self.assertIn(ROOT / config["certified_2024"]["path"], paths)
        self.assertIn(ROOT / config["gis_2024"]["path"], paths)
        self.assertNotIn(
            ROOT / "data/raw_parcels/appraisal_history/williamson/2025/williamson_2025.zip", paths
        )
        self.assertFalse(any(path.name == "wcad_owners.csv" or "2026" in path.parts for path in paths))
        for county, year in (("hays", 2024), ("hays", 2025), ("williamson", 2024)):
            self.assertIn(
                ROOT / "data/raw_parcels/appraisal_history" / county / str(year) / f"{county}_{year}.zip",
                paths,
            )


if __name__ == "__main__":
    unittest.main()
