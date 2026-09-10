"""Synthetic checks for certified-primary Williamson 2025 ownership evidence."""

import importlib.util
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/data" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adapter = load_module("williamson_test_adapter", "build_other_county_ownership.py")
reconciliation = load_module(
    "williamson_test_reconciliation", "williamson_ownership_reconciliation.py"
)
LM = adapter.load_classifier(ROOT.parent / "landlord-mapper/historical_ownership.py")
LM.ews_rule_version = LM.classifier_rule_version()
PARCEL_ID = "WILLIAMSON:R123"


def report_line(owner="", description="", exemption=""):
    return owner.ljust(31) + description.ljust(35) + exemption.ljust(16)


def certified(name="JANE SMITH", mail="200 OTHER RD", homestead=False, year=2025):
    return adapter.report_block_to_standard(
        [
            report_line("PID: R123 (O44)", "SUBDIVISION", "HS $ 0" if homestead else ""),
            report_line("R-123", "LOT 1"),
            report_line("LOCAL ID:", "BLOCK 2"),
            report_line(name),
            report_line(mail, "SITUS: 100 MAIN ST"),
            report_line("DALLAS TX 75201", "AUSTIN TX 78727"),
            report_line("", "LAND SPTB: A1, IMP SPTB: B1"),
            report_line("AGENT:"),
        ],
        year,
    )


def raw_gis(**changes):
    row = {
        "Prop_ID": "R123",
        "OWNER_NAME": "JANE SMITH",
        "NAME_CARE": "",
        "MAIL_ADDR": "200 OTHER RD DALLAS TX 75201",
        "MAIL_LINE1": "200 OTHER RD",
        "MAIL_LINE2": "",
        "MAIL_CITY": "DALLAS",
        "MAIL_STAT": "TX",
        "MAIL_ZIP": "75201",
        "SITUS_ADDR": "100 MAIN ST AUSTIN TX 78727",
        "SITUS_NUM": "100",
        "SITUS_STRE": "MAIN",
        "SITUS_ST_1": "ST",
        "SITUS_ST_2": "",
        "SITUS_CITY": "AUSTIN",
        "SITUS_STAT": "TX",
        "SITUS_ZIP": "78727",
        "SOURCE": "WILLIAMSON APPRAISAL DISTRICT",
        "TAX_YEAR": "2025",
        "DATE_ACQ": "20250701",
    }
    row.update(changes)
    return row


def gis(**changes):
    return reconciliation.standardize_gis(raw_gis(**changes), LM)


def classify(rows, year=2025):
    target = {
        "parcel_id": PARCEL_ID,
        "source_county": "Williamson",
        "is_residential": "TRUE",
        "property_units": "10",
        # Current target attributes must not substitute for source evidence.
        "owner_name": "CURRENT OWNER LLC",
        "homestead_flag": "TRUE",
        "situs_address": "2099 CURRENT ST AUSTIN TX 78727",
    }
    return adapter.snapshot(target, year, rows, LM)


class WilliamsonOwnershipReconciliationTests(unittest.TestCase):
    def test_clipped_name_extends_only_after_owner_and_mail_agreement(self):
        full_name = "THE NORTH AUSTIN HOUSING COMPANY LLC"
        primary = certified(name=full_name[:30], homestead=True)
        before = deepcopy(primary)
        selected, diagnostic = reconciliation.reconcile_2025(
            [primary], [gis(OWNER_NAME=full_name)], LM
        )
        self.assertEqual(primary, before, "Reconciliation must not mutate cached certified evidence")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["owner_name"], full_name)
        self.assertEqual(selected[0]["owner_id"], primary["owner_id"])
        self.assertEqual(selected[0]["homestead_flag"], "TRUE")
        self.assertEqual(diagnostic["name_relation"], "certified_prefix")
        self.assertEqual(diagnostic["mailing_relation"], "exact")
        self.assertTrue(diagnostic["name_extended"])
        self.assertFalse(diagnostic["reconciliation_conflict"])
        result = classify(selected)
        self.assertTrue(result["has_financialized_owner"])
        self.assertTrue(result["is_owner_occupied"])
        self.assertFalse(result["is_corporate_owned"])

    def test_short_unclipped_prefix_does_not_identify_same_owner(self):
        primary = certified(name="JANE SMITH")
        selected, diagnostic = reconciliation.reconcile_2025(
            [primary], [gis(OWNER_NAME="JANE SMITH HOLDINGS LLC")], LM
        )
        self.assertEqual(selected[0]["owner_name"], "JANE SMITH")
        self.assertFalse(diagnostic["name_extended"])
        self.assertTrue(diagnostic["reconciliation_conflict"])
        self.assertFalse(classify(selected)["has_financialized_owner"])

    def test_exact_gis_name_confirms_printed_name_ends_at_column_boundary(self):
        name = "ALEXANDRA CHRISTOPHERSON SMITH"
        primary = certified(name=name)
        self.assertTrue(primary["source_name_may_be_truncated"])
        self.assertIsNone(classify([primary])["has_financialized_owner"])
        selected, diagnostic = reconciliation.reconcile_2025([primary], [gis(OWNER_NAME=name)], LM)
        self.assertEqual(diagnostic["name_relation"], "exact")
        self.assertFalse(diagnostic["name_extended"])
        self.assertTrue(diagnostic["name_completeness_confirmed"])
        self.assertFalse(selected[0]["source_name_may_be_truncated"])
        self.assertEqual(selected[0]["source_reported_owner_name"], primary["source_reported_owner_name"])
        self.assertIn(reconciliation.GIS_SNAPSHOT_ID, selected[0]["source_snapshot_id"])
        self.assertFalse(classify(selected)["has_financialized_owner"])

    def test_matching_name_prefix_with_changed_mail_keeps_primary_unknown(self):
        full_name = "THE NORTH AUSTIN HOUSING COMPANY LLC"
        primary = certified(name=full_name[:30])
        supplement = gis(
            OWNER_NAME=full_name,
            MAIL_LINE1="300 DIFFERENT RD",
            MAIL_ADDR="300 DIFFERENT RD DALLAS TX 75201",
        )
        selected, diagnostic = reconciliation.reconcile_2025([primary], [supplement], LM)
        self.assertTrue(diagnostic["reconciliation_conflict"])
        self.assertFalse(diagnostic["name_extended"])
        self.assertEqual(selected[0]["owner_name"], primary["owner_name"])
        self.assertEqual(selected[0]["owner_addr_line1"], "200 OTHER RD")
        self.assertIsNone(classify(selected)["has_financialized_owner"])
        self.assertIsNone(classify(selected)["is_corporate_owned"])

    def test_new_gis_owner_does_not_receive_certified_homestead(self):
        primary = certified(name="JANE SMITH", homestead=True)
        supplement = gis(OWNER_NAME="NEW OWNER LLC")
        selected, diagnostic = reconciliation.reconcile_2025([primary], [supplement], LM)
        self.assertTrue(diagnostic["reconciliation_conflict"])
        self.assertEqual(selected[0]["owner_name"], "JANE SMITH")
        self.assertEqual(supplement["homestead_flag"], "")
        result = classify(selected)
        self.assertTrue(result["homestead_positive"])
        self.assertTrue(result["is_owner_occupied"])
        self.assertFalse(result["has_financialized_owner"])
        self.assertFalse(result["is_corporate_owned"])

    def test_missing_gis_mailing_locality_cannot_verify_name_extension(self):
        full_name = "THE NORTH AUSTIN HOUSING COMPANY LLC"
        primary = certified(name=full_name[:30])
        supplement = gis(OWNER_NAME=full_name, MAIL_CITY="", MAIL_STAT="", MAIL_ZIP="")
        selected, diagnostic = reconciliation.reconcile_2025([primary], [supplement], LM)
        self.assertFalse(diagnostic["name_extended"])
        self.assertEqual(selected[0]["owner_name"], primary["owner_name"])
        self.assertIsNone(classify(selected)["has_financialized_owner"])

    def test_conflicting_situs_preserves_certified_source_fields(self):
        primary = certified(name="SAMPLE LLC")
        supplement = gis(
            OWNER_NAME="SAMPLE LLC",
            SITUS_ADDR="900 DIFFERENT ST AUSTIN TX 78727",
            SITUS_NUM="900",
            SITUS_STRE="DIFFERENT",
        )
        selected, diagnostic = reconciliation.reconcile_2025([primary], [supplement], LM)
        self.assertTrue(diagnostic["reconciliation_conflict"])
        self.assertEqual(selected[0]["situs_street"], primary["situs_street"])
        self.assertFalse(diagnostic["situs_extended"])
        self.assertFalse(diagnostic["mailing_extended"])

    def test_name_corroboration_keeps_situs_conflict_and_certified_address_pair(self):
        for full_name in (
            "THE NORTH AUSTIN HOUSING COMPANY LLC",
            "ALEXANDRA CHRISTOPHERSON SMITH",
        ):
            with self.subTest(name=full_name):
                primary = certified(name=full_name[:30])
                supplement = gis(
                    OWNER_NAME=full_name,
                    SITUS_ADDR="900 DIFFERENT ST AUSTIN TX 78727",
                    SITUS_NUM="900",
                    SITUS_STRE="DIFFERENT",
                )
                selected, diagnostic = reconciliation.reconcile_2025([primary], [supplement], LM)
                self.assertTrue(diagnostic["reconciliation_conflict"])
                self.assertEqual(diagnostic["situs_relation"], "disagree")
                self.assertTrue(diagnostic["name_completeness_confirmed"])
                self.assertEqual(selected[0]["owner_name"], full_name)
                self.assertEqual(selected[0]["situs_street"], primary["situs_street"])
                self.assertEqual(selected[0]["owner_addr_line1"], primary["owner_addr_line1"])
                self.assertEqual(selected[0]["source_reported_owner_name"], primary["source_reported_owner_name"])
                self.assertIn(reconciliation.GIS_SNAPSHOT_ID, selected[0]["source_snapshot_id"])
                self.assertFalse(diagnostic["situs_extended"])
                self.assertFalse(diagnostic["mailing_extended"])

    def test_verified_clipped_mailing_and_situs_extend_as_same_source_pair(self):
        mailing = "9876 AN EXTRAORDINARILY LONGNAMED ROAD NORTH"
        situs = "100 AN EXTRAORDINARILY LONGNAMED STREET NORTH"
        primary = certified(name="SAMPLE LLC")
        primary.update(
            owner_addr_line1=mailing[:30],
            situs_street=situs[:28],
            source_address_may_be_truncated=True,
            source_situs_may_be_truncated=True,
        )
        supplement = gis(
            OWNER_NAME="SAMPLE LLC",
            MAIL_LINE1=mailing,
            MAIL_ADDR=mailing + " DALLAS TX 75201",
            SITUS_ADDR=situs + " AUSTIN TX 78727",
            SITUS_NUM="100",
            SITUS_STRE="AN EXTRAORDINARILY LONGNAMED",
            SITUS_ST_1="STREET",
            SITUS_ST_2="NORTH",
        )
        selected, diagnostic = reconciliation.reconcile_2025([primary], [supplement], LM)
        self.assertFalse(diagnostic["reconciliation_conflict"])
        self.assertTrue(diagnostic["mailing_extended"])
        self.assertTrue(diagnostic["situs_extended"])
        self.assertEqual(selected[0]["owner_addr_line1"], mailing)
        self.assertFalse(selected[0]["source_address_may_be_truncated"])
        self.assertFalse(selected[0]["source_situs_may_be_truncated"])
        self.assertTrue(classify(selected)["is_corporate_owned"])

    def test_gis_only_nonresident_uses_own_address_without_homestead(self):
        selected, diagnostic = reconciliation.reconcile_2025([], [gis(OWNER_NAME="SAMPLE LLC")], LM)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["owner_id"], "")
        self.assertEqual(selected[0]["homestead_flag"], "")
        self.assertFalse(diagnostic["reconciliation_conflict"])
        result = classify(selected)
        self.assertFalse(result["homestead_evidence_available"])
        self.assertFalse(result["homestead_positive"])
        self.assertFalse(result["is_owner_occupied"])
        self.assertTrue(result["is_corporate_owned"])

    def test_gis_only_matching_address_is_occupied_without_homestead(self):
        supplement = gis(
            OWNER_NAME="SAMPLE LLC",
            MAIL_LINE1="100 MAIN ST",
            MAIL_CITY="AUSTIN",
            MAIL_ZIP="78727",
            MAIL_ADDR="100 MAIN ST AUSTIN TX 78727",
        )
        selected, _ = reconciliation.reconcile_2025([], [supplement], LM)
        result = classify(selected)
        self.assertFalse(result["homestead_evidence_available"])
        self.assertTrue(result["address_match_positive"])
        self.assertTrue(result["is_owner_occupied"])
        self.assertFalse(result["is_corporate_owned"])

    def test_gis_full_situs_supplies_its_own_missing_components(self):
        for address in ("100 MAIN ST, AUSTIN, TX 78727", "100 MAIN ST, AUSTIN TX 78727"):
            with self.subTest(situs=address):
                supplement = gis(
                    OWNER_NAME="SAMPLE LLC",
                    SITUS_ADDR=address,
                    SITUS_NUM="",
                    SITUS_STRE="",
                    SITUS_ST_1="",
                    SITUS_ST_2="",
                    SITUS_CITY="",
                    SITUS_ZIP="",
                    MAIL_LINE1="100 MAIN ST",
                    MAIL_CITY="AUSTIN",
                    MAIL_ZIP="78727",
                    MAIL_ADDR="100 MAIN ST AUSTIN TX 78727",
                )
                selected, _ = reconciliation.reconcile_2025([], [supplement], LM)
                result = classify(selected)
                self.assertTrue(result["address_match_positive"])
                self.assertTrue(result["is_owner_occupied"])
                self.assertFalse(result["homestead_evidence_available"])

    def test_component_backed_situs_has_one_locality_with_or_without_full_text(self):
        for address in ("100 MAIN ST", "100 MAIN ST AUSTIN TX 78727"):
            with self.subTest(situs=address):
                supplement = gis(SITUS_ADDR=address)
                self.assertEqual(LM.source_situs_address(supplement), "100 MAIN ST AUSTIN TX 78727")

    def test_incomplete_gis_situs_does_not_borrow_mail_or_current_locality(self):
        supplement = gis(
            OWNER_NAME="SAMPLE LLC",
            SITUS_ADDR="100 MAIN ST",
            SITUS_NUM="",
            SITUS_STRE="",
            SITUS_ST_1="",
            SITUS_ST_2="",
            SITUS_CITY="",
            SITUS_ZIP="",
            MAIL_LINE1="100 MAIN ST",
            MAIL_CITY="AUSTIN",
            MAIL_ZIP="78727",
            MAIL_ADDR="100 MAIN ST AUSTIN TX 78727",
        )
        selected, _ = reconciliation.reconcile_2025([], [supplement], LM)
        result = classify(selected)
        self.assertFalse(result["homestead_evidence_available"])
        self.assertIsNone(result["is_owner_occupied"])
        self.assertIsNone(result["is_corporate_owned"])

    def test_gis_placeholders_do_not_create_negative_evidence(self):
        supplement = gis(OWNER_NAME="UNAVAILABLE", MAIL_ADDR="UNAVAILABLE", MAIL_LINE1="UNAVAILABLE")
        selected, _ = reconciliation.reconcile_2025([], [supplement], LM)
        result = classify(selected)
        self.assertFalse(result["owner_name_available"])
        self.assertFalse(result["owner_address_available"])
        for field in ("is_owner_occupied", "has_financialized_owner", "is_corporate_owned"):
            self.assertIsNone(result[field])

    def test_gis_placeholder_cannot_replace_valid_primary_or_create_conflict(self):
        primary = certified(name="SAMPLE LLC")
        supplement = gis(OWNER_NAME="UNAVAILABLE", MAIL_ADDR="UNAVAILABLE", MAIL_LINE1="UNAVAILABLE")
        selected, diagnostic = reconciliation.reconcile_2025([primary], [supplement], LM)
        self.assertFalse(diagnostic["reconciliation_conflict"])
        self.assertEqual(selected[0]["owner_name"], "SAMPLE LLC")
        self.assertTrue(classify(selected)["is_corporate_owned"])

    def test_eighty_character_name_withholds_negative_but_retains_positive_marker(self):
        for name, financialized, corporate in (("A" * 80, None, None), ("A" * 76 + " LLC", True, True)):
            with self.subTest(name=name):
                supplement = gis(OWNER_NAME=name)
                self.assertTrue(supplement["source_name_may_be_truncated"])
                selected, _ = reconciliation.reconcile_2025([], [supplement], LM)
                result = classify(selected)
                self.assertIs(result["has_financialized_owner"], financialized)
                self.assertIs(result["is_corporate_owned"], corporate)

    def test_geometry_repetitions_produce_one_owner_record_per_target(self):
        first = raw_gis(OWNER_NAME="SAMPLE LLC", OBJECTID=1, Shape_Area=100)
        second = dict(first, OBJECTID=2, Shape_Area=200)
        owners, _ = reconciliation.deduplicate_gis_rows([first, second], LM)
        self.assertEqual(set(owners), {PARCEL_ID})
        self.assertEqual(len(owners[PARCEL_ID]), 1)
        selected, diagnostic = reconciliation.reconcile_2025([], owners[PARCEL_ID], LM)
        result = classify(selected)
        self.assertEqual(result["n_owner_rows"], 1)
        self.assertEqual(result["parcel_id"], PARCEL_ID)
        self.assertFalse(diagnostic["reconciliation_conflict"])

    def test_conflicting_gis_features_without_primary_are_ambiguous_not_coowners(self):
        owners, _ = reconciliation.deduplicate_gis_rows(
            [raw_gis(OWNER_NAME="JANE SMITH"), raw_gis(OWNER_NAME="SAMPLE LLC")], LM
        )
        self.assertEqual(len(owners[PARCEL_ID]), 2)
        selected, diagnostic = reconciliation.reconcile_2025([], owners[PARCEL_ID], LM)
        self.assertTrue(diagnostic["reconciliation_conflict"])
        self.assertEqual(len(selected), 1)
        result = classify(selected)
        self.assertEqual(result["classification_status"], "matched_ambiguous")
        for field in ("is_owner_occupied", "has_financialized_owner", "is_corporate_owned"):
            self.assertIsNone(result[field])

    def test_conflicting_gis_features_keep_certified_primary(self):
        primary = certified(name="JANE SMITH", homestead=True)
        owners, _ = reconciliation.deduplicate_gis_rows(
            [raw_gis(OWNER_NAME="JANE SMITH"), raw_gis(OWNER_NAME="SAMPLE LLC")], LM
        )
        selected, diagnostic = reconciliation.reconcile_2025([primary], owners[PARCEL_ID], LM)
        self.assertTrue(diagnostic["reconciliation_conflict"])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["owner_name"], "JANE SMITH")
        self.assertTrue(classify(selected)["homestead_positive"])

    def test_2025_supplement_is_rejected_for_2024_certified_rows(self):
        primary = certified()
        primary["tax_year"] = "2024"
        primary["source_snapshot_id"] = "wcad-2024-certification-printed-report"
        before = deepcopy(primary)
        with self.assertRaises(ValueError):
            reconciliation.reconcile_2025([primary], [gis(OWNER_NAME="NEW OWNER LLC")], LM)
        self.assertEqual(primary, before)

    def test_reconciliation_rejects_supplement_from_another_year(self):
        supplement = gis()
        supplement["tax_year"] = "2024"
        with self.assertRaises(ValueError):
            reconciliation.reconcile_2025([certified()], [supplement], LM)

    def test_gis_vintage_is_verified_before_standardizing(self):
        for changes in ({"TAX_YEAR": "2024"}, {"DATE_ACQ": "20260701"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                gis(**changes)

    def test_care_of_text_is_not_reinterpreted_as_owner_entity_name(self):
        selected, _ = reconciliation.reconcile_2025(
            [], [gis(OWNER_NAME="JANE SMITH", NAME_CARE="C/O MANAGEMENT LLC")], LM
        )
        self.assertEqual(selected[0]["owner_name"], "JANE SMITH")
        self.assertFalse(classify(selected)["has_financialized_owner"])


class AnnualWilliamsonReconciliationTests(unittest.TestCase):
    def test_2024_merge_uses_explicit_synthetic_vintage_and_preserves_certified_name(self):
        name = "THE NORTH AUSTIN HOUSING COMPANY LLC"
        primary = certified(name=name[:30], homestead=True, year=2024)
        supplement = reconciliation.standardize_gis(
            raw_gis(OWNER_NAME=name, TAX_YEAR="2024", DATE_ACQ="20241115"),
            LM, year=2024, acquisition_date="2024-11-15",
        )
        selected, diagnostic = reconciliation.reconcile_year([primary], [supplement], LM, year=2024)
        self.assertEqual(selected[0]["tax_year"], "2024")
        self.assertEqual(selected[0]["owner_name"], name)
        self.assertEqual(selected[0]["source_reported_owner_name"], name[:30])
        self.assertEqual(selected[0]["homestead_flag"], "TRUE")
        self.assertEqual(
            selected[0]["source_snapshot_id"],
            "wcad-2024-certification-printed-report; txgio-williamson-2024-11-wcad-parcels",
        )
        self.assertTrue(diagnostic["name_extended"])
        self.assertFalse(diagnostic["reconciliation_conflict"])
        result = classify(selected, year=2024)
        self.assertEqual(result["tax_year"], 2024)
        self.assertTrue(result["has_financialized_owner"])
        self.assertTrue(result["is_owner_occupied"])
        self.assertFalse(result["is_corporate_owned"])

    def test_2024_has_no_implicit_acquisition_date(self):
        raw = raw_gis(TAX_YEAR="2024", DATE_ACQ="20241115")
        with self.assertRaisesRegex(ValueError, "explicit verified acquisition date"):
            reconciliation.standardize_gis(raw, LM, year=2024)
        with self.assertRaisesRegex(ValueError, "explicit verified acquisition date"):
            reconciliation.deduplicate_gis_rows([], LM, year=2024)

    def test_annual_standardization_rejects_unverified_or_invalid_vintage(self):
        raw = raw_gis(TAX_YEAR="2024", DATE_ACQ="20241115")
        for year, acquisition_date in (
            (2024, "20241116"), (2024, "20240230"),
            (2024, "2025-11-15"), (2025, "2025-11-15"),
        ):
            with self.subTest(year=year, date=acquisition_date), self.assertRaises(ValueError):
                reconciliation.standardize_gis(raw, LM, year=year, acquisition_date=acquisition_date)

    def test_generic_reconciliation_rejects_both_directions_of_cross_year_mixing(self):
        supplement_2024 = reconciliation.standardize_gis(
            raw_gis(TAX_YEAR="2024", DATE_ACQ="20241115"),
            LM, year=2024, acquisition_date="20241115",
        )
        combinations = (
            (2024, certified(year=2024), gis()),
            (2024, certified(year=2025), supplement_2024),
            (2025, certified(year=2024), gis()),
            (2025, certified(year=2025), supplement_2024),
        )
        for year, primary, supplement in combinations:
            before = deepcopy((primary, supplement))
            with self.subTest(year=year, primary_year=primary["tax_year"], gis_year=supplement["tax_year"]):
                with self.assertRaisesRegex(ValueError, "cannot supplement another tax year"):
                    reconciliation.reconcile_year([primary], [supplement], LM, year=year)
                self.assertEqual((primary, supplement), before)

    def test_separate_annual_runs_keep_their_own_owner_names_and_classifications(self):
        results = {}
        for year, name, date in ((2024, "JANE SMITH", "20241115"), (2025, "SAMPLE LLC", "20250701")):
            primary = certified(name=name, year=year)
            supplement = reconciliation.standardize_gis(
                raw_gis(OWNER_NAME=name, TAX_YEAR=str(year), DATE_ACQ=date),
                LM, year=year, acquisition_date=date,
            )
            selected, _ = reconciliation.reconcile_year([primary], [supplement], LM, year=year)
            results[year] = classify(selected, year=year)
        self.assertEqual(results[2024]["owner_names"], "JANE SMITH")
        self.assertFalse(results[2024]["has_financialized_owner"])
        self.assertFalse(results[2024]["is_corporate_owned"])
        self.assertNotIn("2025", results[2024]["source_snapshot_id"])
        self.assertEqual(results[2025]["owner_names"], "SAMPLE LLC")
        self.assertTrue(results[2025]["has_financialized_owner"])
        self.assertTrue(results[2025]["is_corporate_owned"])
        self.assertNotIn("2024", results[2025]["source_snapshot_id"])

    def test_2024_deduplication_validates_every_rows_vintage(self):
        raw = raw_gis(TAX_YEAR="2024", DATE_ACQ="20241115", OBJECTID=1)
        owners, qa = reconciliation.deduplicate_gis_rows(
            [raw, dict(raw, OBJECTID=2)], LM, year=2024, acquisition_date="2024-11-15"
        )
        self.assertEqual(len(owners[PARCEL_ID]), 1)
        self.assertEqual(qa["duplicate_rows_removed"], 1)
        self.assertEqual(owners[PARCEL_ID][0]["tax_year"], "2024")
        with self.assertRaises(ValueError):
            reconciliation.deduplicate_gis_rows(
                [raw, raw_gis(Prop_ID="R999")], LM, year=2024, acquisition_date="20241115"
            )

    def test_2025_compatibility_defaults_equal_explicit_annual_api(self):
        name = "THE NORTH AUSTIN HOUSING COMPANY LLC"
        primary = certified(name=name[:30])
        default_row = gis(OWNER_NAME=name)
        explicit_row = reconciliation.standardize_gis(
            raw_gis(OWNER_NAME=name), LM, year=2025, acquisition_date="2025-07-01"
        )
        self.assertEqual(default_row, explicit_row)
        self.assertEqual(default_row["source_snapshot_id"], reconciliation.GIS_SNAPSHOT_ID)
        self.assertEqual(
            reconciliation.reconcile_2025([primary], [default_row], LM),
            reconciliation.reconcile_year([primary], [explicit_row], LM, year=2025),
        )


if __name__ == "__main__":
    unittest.main()
