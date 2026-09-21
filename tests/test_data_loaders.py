"""Loaders, cleaning rules and the vendor-to-part mapping (mini fixture)."""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import data_loaders as DL  # noqa: E402
from mini_fixture import scms_rows, willems_rows  # noqa: E402


# ---------------------------------------------------------------- Willems

def test_chain_products_are_manufacturing_stages_with_no_manufacturing_after():
    ch = DL.chain_structure(*willems_rows())
    assert ch["products"] == ["M1", "M2", "M3"]
    assert ch["subs"] == ["S1", "S2", "S3"]
    assert len(ch["parts"]) == 12


def test_demand_rolls_up_from_distribution_stages():
    ch = DL.chain_structure(*willems_rows())
    assert ch["demand_avg"] == {"M1": 10.0, "M2": 6.0, "M3": 3.0}
    assert ch["demand_sd"]["M1"] == pytest.approx(4.0)


def test_demand_is_split_when_two_products_feed_one_stage():
    rows, arcs = willems_rows()
    arcs = arcs + [("M2", "D1")]
    ch = DL.chain_structure(rows, arcs)
    assert ch["demand_stages_split"] == 1
    assert ch["demand_avg"]["M1"] == pytest.approx(5.0)
    assert ch["demand_avg"]["M2"] == pytest.approx(11.0)


def test_lead_distribution_is_read_and_normalised():
    rows, _ = willems_rows()
    p01 = next(r for r in rows if r["Stage Name"] == "P01")
    vals, probs = DL.lead_distribution(p01)
    assert vals == (3, 6) and sum(probs) == pytest.approx(1.0)
    p02 = next(r for r in rows if r["Stage Name"] == "P02")
    assert DL.lead_distribution(p02) == ((4.0,), (1.0,))


def test_bom_keeps_only_arcs_into_manufacturing():
    ch = DL.chain_structure(*willems_rows())
    assert ("M1", "D1") not in ch["bom"]
    assert ("P01", "S1") in ch["bom"] and ("S3", "S1") in ch["bom"]


# ---------------------------------------------------------------- SCMS

def test_parse_date_handles_both_formats_and_junk():
    assert DL.parse_date("2-Jun-06") == dt.date(2006, 6, 2)
    assert DL.parse_date("8/27/2014") == dt.date(2014, 8, 27)
    assert DL.parse_date("Date Not Captured") is None
    assert DL.parse_date("") is None


def test_cleaning_drops_warehouse_missing_dates_and_zero_leads():
    lines, log = DL.clean_scms(scms_rows())
    assert log["rows"] == 8 * 40 + 6
    assert log["direct_drop"] == 8 * 40 + 5        # the RDC row is gone
    assert log["dropped_missing_date"] == 1
    assert log["dropped_lead_not_positive"] == 1
    assert all(x["vendor"] != DL.IN_HOUSE_VENDOR for x in lines)
    assert all(x["lead_days"] > 0 for x in lines)


def test_vendor_table_applies_the_minimum_and_measures_on_time():
    lines, _ = DL.clean_scms(scms_rows())
    vt = DL.vendor_table(lines, min_lines=10)
    assert "Tiny Vendor" not in vt and len(vt) == 8
    for v in vt.values():
        assert 0 <= v["on_time_vs_scheduled"] <= 1
        assert v["exact_on_scheduled_date"] <= v["on_time_vs_scheduled"] + 1e-12
    # Vendors A and B are the only shippers of their molecules
    assert vt["Vendor A"]["sole_source_share"] == 1.0
    assert vt["Vendor C"]["sole_source_share"] == 0.0


# ---------------------------------------------------------------- mapping

def test_mapping_is_monotone_in_lead_time():
    parts = {f"p{i}": float(i) for i in range(40)}
    vendors = {"slow": 200.0, "fast": 10.0, "mid": 90.0, "mid2": 100.0}
    m = DL.map_vendors_to_parts(parts, vendors)
    rank = {v: i for i, v in enumerate(sorted(vendors, key=vendors.get))}
    ordered = sorted(parts, key=parts.get)
    ranks = [rank[m[p]] for p in ordered]
    assert ranks == sorted(ranks), "a longer-lead part got a faster vendor"
    assert set(m.values()) == set(vendors), "every vendor should get parts"


def test_mapping_bands_are_balanced_and_deterministic():
    parts = {f"p{i:03d}": float(i % 7) for i in range(209)}
    vendors = {f"v{i:02d}": float(i) for i in range(27)}
    a = DL.map_vendors_to_parts(parts, vendors)
    b = DL.map_vendors_to_parts(dict(reversed(list(parts.items()))), vendors)
    assert a == b
    counts = {}
    for v in a.values():
        counts[v] = counts.get(v, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1


def test_mapping_with_no_vendors_is_empty():
    assert DL.map_vendors_to_parts({"p": 1.0}, {}) == {}
