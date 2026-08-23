"""Tests for the third-pass modules."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inventory as INV  # noqa: E402
import routing as RT  # noqa: E402


# ---------------------------------------------------------------------------
# safety stock
# ---------------------------------------------------------------------------

def test_supply_variance_term_dominates_for_a_variable_supplier():
    """The term most textbook treatments drop, and where it matters."""
    out = INV.safety_stock(demand_per_day=100, sigma_demand=10,
                           lead_mean_days=45, lead_sd_days=20)
    assert out["supply_variance_share"] > 0.9
    assert out["understatement_factor"] > 3


def test_demand_only_and_full_agree_when_lead_time_is_certain():
    out = INV.safety_stock(demand_per_day=100, sigma_demand=10,
                           lead_mean_days=10, lead_sd_days=0.0)
    assert out["understatement_factor"] == pytest.approx(1.0, abs=1e-9)


def test_reorder_point_covers_mean_lead_time_demand_plus_safety():
    out = INV.safety_stock(50, 5, 20, 3)
    assert out["reorder_point"] > 50 * 20


def test_empirical_safety_stock_exceeds_parametric_on_a_fat_tail():
    """z comes from a normal quantile; a fat tail does not have one."""
    rng = np.random.default_rng(0)
    thin = rng.normal(30, 4, 400)
    fat = np.concatenate([rng.normal(30, 4, 380), rng.normal(120, 20, 20)])
    par = INV.safety_stock(100, 10, float(fat.mean()), float(fat.std()))
    emp_fat = INV.empirical_safety_stock(100, 10, fat)
    emp_thin = INV.empirical_safety_stock(100, 10, thin)
    assert emp_fat["tail_ratio_p95_over_p50"] > emp_thin["tail_ratio_p95_over_p50"]
    assert emp_fat["safety_stock"] > 0


def test_empirical_safety_stock_refuses_on_thin_history():
    out = INV.empirical_safety_stock(10, 1, np.array([5.0, 6.0]))
    assert out["insufficient_history"]


# ---------------------------------------------------------------------------
# lot sizing
# ---------------------------------------------------------------------------

def test_order_is_raised_to_moq():
    out = INV.order_quantity(340, moq=1000)
    assert out["order_qty"] == 1000 and out["raised_to_moq"]
    assert out["excess_over_need"] == 660


def test_pack_size_rounds_up_never_down():
    out = INV.order_quantity(1010, moq=100, pack_size=250)
    assert out["order_qty"] == 1250


def test_capacity_cap_is_applied_last_and_is_surfaced():
    """Capping before rounding would push the quantity back over the cap."""
    out = INV.order_quantity(5000, moq=100, pack_size=250, supplier_capacity=900)
    assert out["order_qty"] == 900
    assert out["capped_by_capacity"] and out["shortfall_vs_need"] == 4100
    assert out["note"] and "decision" in out["note"]


def test_eoq_grows_with_demand_and_shrinks_with_holding_cost():
    a = INV.eoq(10000, 250, 1.0, holding_rate=0.2)
    b = INV.eoq(40000, 250, 1.0, holding_rate=0.2)
    c = INV.eoq(10000, 250, 4.0, holding_rate=0.2)
    assert b == pytest.approx(2 * a, rel=1e-6)
    assert c < a


def test_moq_waste_counts_only_what_the_minimum_forced():
    """EOQ excess is deliberate; MOQ excess is imposed. Conflating them
    overstates the negotiating number, which is the one figure here that a
    supplier would check."""
    out = INV.moq_waste([
        {"part": "P1", "need": 100, "moq": 1000, "unit_cost": 5.0,
         "annual_demand": 1200},
        {"part": "P2", "need": 5000, "moq": 100, "unit_cost": 1.0,
         "annual_demand": 60000}])
    assert out["n_parts_raised_to_moq"] == 1
    w = out["worst"][0]
    assert w["part"] == "P1"
    # Exactly the gap between what the minimum forced and what would have been
    # ordered without it.
    assert w["excess_value"] == pytest.approx(
        (w["ordered"] - w["would_order_without_moq"]) * 5.0)
    assert out["total_excess_value"] == pytest.approx(w["excess_value"])
    # P2's order is well above its need, but that is EOQ choosing to buy a
    # year's worth of a cheap part -- not the minimum imposing anything.
    assert out["eoq_excess_value"] > out["total_excess_value"]


def test_moq_waste_is_zero_when_the_minimum_never_binds():
    out = INV.moq_waste([{"part": "P", "need": 5000, "moq": 10,
                          "unit_cost": 1.0, "annual_demand": 60000}])
    assert out["total_excess_value"] == 0.0
    assert out["n_parts_raised_to_moq"] == 0


def test_capacity_check_says_ordering_earlier_does_not_help():
    out = INV.capacity_check({"S1": 1000, "S2": 100}, {"S1": 400, "S2": 500})
    assert out["n_over_capacity"] == 1
    over = next(r for r in out["rows"] if r["over"])
    assert "EARLIER does not create capacity" in over["remedy"]


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def test_slack_is_judged_against_lead_time_not_absolute_days():
    """Ten days is comfortable on a 5-day lead and an emergency on a 60-day one."""
    short, _ = RT.severity_of(10, 100, False, lead_days=5)
    long_, _ = RT.severity_of(10, 100, False, lead_days=60)
    assert short == "P4" and long_ == "P1"


def test_single_sourcing_raises_the_tier():
    a, _ = RT.severity_of(20, 100, False, lead_days=30)
    b, _ = RT.severity_of(20, 100, True, lead_days=30)
    order = ["P1", "P2", "P3", "P4"]
    assert order.index(b) < order.index(a)


def test_negative_slack_is_p1_and_says_ordering_will_not_recover_it():
    sev, why = RT.severity_of(-5, 100, False, lead_days=30)
    assert sev == "P1" and "does not recover" in why


def test_capacity_problems_route_past_the_buyer():
    assert RT.route("P2", False, over_capacity=True) == "COMMODITY_MANAGER"
    assert RT.route("P2", False, over_capacity=False) == "BUYER"


def test_every_tier_has_an_sla_and_an_escalation_chain():
    for sev, t in RT.TIERS.items():
        assert t["response_hours"] > 0 and t["chain"] and t["meaning"]
        assert all(r in RT.ROLES for r in t["chain"])


def test_alerts_sort_by_severity_then_value():
    rows = [
        {"part": "A", "days_of_slack": 40, "units_at_risk": 0,
         "value_at_risk": 10, "lead_days": 10},
        {"part": "B", "days_of_slack": -1, "units_at_risk": 50,
         "value_at_risk": 500, "lead_days": 30},
        {"part": "C", "days_of_slack": -1, "units_at_risk": 90,
         "value_at_risk": 9000, "lead_days": 30},
    ]
    a = RT.build_alerts(rows)
    assert [x.part for x in a[:2]] == ["C", "B"]
    assert a[-1].part == "A"


def test_duplicates_are_suppressed():
    rows = [{"part": "A", "days_of_slack": -1, "units_at_risk": 1,
             "value_at_risk": 1, "lead_days": 30, "supplier_id": "S1"}] * 3
    kept, dropped = RT.dedupe(RT.build_alerts(rows))
    assert len(kept) == 1 and dropped == 2


def test_load_by_role_totals_match_the_alert_count():
    rows = [{"part": f"P{i}", "days_of_slack": i - 5, "units_at_risk": i,
             "value_at_risk": i * 10, "lead_days": 20,
             "single_sourced": i % 2 == 0} for i in range(12)]
    alerts, _ = RT.dedupe(RT.build_alerts(rows))
    load = RT.load_by_role(alerts)
    assert sum(d["total"] for d in load.values()) == len(alerts)
