"""Tests: BOM explosion, buildability, allocation, supplier analytics.

Run on the mini fixture (same raw formats as the real data), so they pass in CI
without data/raw. Real-data checks live in test_real_data.py.
"""
from __future__ import annotations

import copy
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import buildability as B  # noqa: E402
import supplier_analytics as SA  # noqa: E402
import supply  # noqa: E402
from mini_fixture import build_mini  # noqa: E402


@pytest.fixture(scope="module")
def base():
    return build_mini()


# ---------------------------------------------------------------- BOM

def test_bom_direction_parent_is_the_consumer(base):
    """Willems arcs run component -> consumer. Reading them the other way
    round makes every product's BOM empty and the plan 100% buildable."""
    assert {b.child for b in base.children("M1")} >= {"S1", "P07", "P12"}
    assert not base.children("P01")


def test_explosion_is_linear_in_quantity(base):
    one = supply.explode(base, "M1", 1.0)
    ten = supply.explode(base, "M1", 10.0)
    for part, q in one.items():
        assert ten[part] == pytest.approx(10 * q)


def test_explosion_is_multi_level(base):
    need = supply.explode(base, "M1", 1.0)
    assert "S3" in need and "P03" in need      # P03 -> S3 -> S1 -> M1
    levels = {base.components[p].level for p in need}
    assert levels == {1, 2}


def test_shared_components_accumulate_across_branches(base):
    """P01 feeds S1 (inside M1, M2) and M3 directly."""
    req = supply.unit_requirements(base)
    assert all("P01" in req[p] for p in ("M1", "M2", "M3"))
    assert len(base.parents_of("P01")) == 2


def test_bom_has_no_cycles(base):
    def walk(node, stack):
        assert node not in stack, f"cycle through {node}"
        for b in base.children(node):
            walk(b.child, stack | {node})

    for prod in base.products:
        walk(prod, set())


def test_quantity_per_is_one_everywhere(base):
    assert {b.qty_per for b in base.bom} == {1.0}


# ---------------------------------------------------------------- buildability

def test_buildable_never_exceeds_demand(base):
    r = B.buildable(base)
    for p in base.products:
        assert (r["built"][p] <= base.demand[p] + 1e-9).all()


def test_buildable_is_non_negative_and_integral(base):
    r = B.buildable(base)
    for p in base.products:
        v = r["built"][p]
        assert (v >= 0).all()
        assert np.allclose(v, np.floor(v))


def test_gating_events_name_a_part(base):
    r = B.buildable(base)
    short = [g for g in r["gating"] if g["short_by"] > 0]
    assert short, "the plan should be constrained somewhere"
    assert any(g["gating_part"] == "P01" for g in short)


def test_unlimited_inventory_builds_the_whole_plan(base):
    """Control: if nothing is short, buildability equals demand exactly."""
    rich = copy.deepcopy(base)
    for c in rich.components.values():
        if c.level == 2:
            c.on_hand = 1e12
    rich.meta.pop("_matrix", None)
    r = B.buildable(rich)
    for p in rich.products:
        assert np.allclose(r["built"][p], rich.demand[p])


def test_monte_carlo_percentiles_are_ordered(base):
    mc = B.monte_carlo(base, n_sims=30)
    for p in base.products:
        f = mc["fan"][p]
        assert all(a <= b <= c for a, b, c in zip(f["p10"], f["p50"], f["p90"]))


def test_even_allocation_protects_the_worst_product(base):
    """Proportional allocation must protect the worst-off product at least as
    well as strict priority; that is the only thing it is for."""
    pols = {a["policy"]: a for a in B.compare_allocation_policies(base, n_sims=20)}
    even_worst = pols["even"]["worst_product_fill_pct"]
    priority_worst = min(pols["margin"]["worst_product_fill_pct"],
                         pols["contract"]["worst_product_fill_pct"])
    assert even_worst >= priority_worst - 1e-9


def test_shortage_drivers_use_the_part_p95(base):
    mc = B.monte_carlo(base, n_sims=25)
    drivers = B.shortage_drivers(base, mc, top_n=5)
    assert drivers
    for d in drivers:
        c = base.components[d["part"]]
        assert d["lead_p95_days"] == pytest.approx(c.lead_p95)
        assert d["lead_p95_days"] >= d["lead_mean_days"] - 1e-9


def test_discrete_quantile():
    assert supply.discrete_quantile((5, 10), (0.65, 0.35), 0.95) == 10
    assert supply.discrete_quantile((5, 10), (0.96, 0.04), 0.95) == 5
    assert supply.discrete_quantile((7.0,), (1.0,), 0.95) == 7


def test_fixed_lead_parts_get_no_monte_carlo_spread(base):
    c = base.components["P02"]           # no distribution in the fixture
    assert not c.has_distribution and c.lead_sd == 0


# ---------------------------------------------------------------- suppliers

def test_otif_against_the_scheduled_date_is_never_worse(base):
    """The original promise is only ever earlier, so the gap is >= 0."""
    rows = SA.otif_gap(base)
    assert rows
    assert all(r["gaming_gap_pts"] >= -1e-9 for r in rows)
    assert max(r["gaming_gap_pts"] for r in rows) > 10


def test_only_simulated_reschedulers_have_a_gap(base):
    for r in SA.otif_gap(base):
        if not r["habitual_rescheduler_simulated"]:
            assert r["gaming_gap_pts"] == pytest.approx(0.0)


def test_deterioration_flags_are_consistent(base):
    flags = SA.detect_deterioration(base)
    sc = SA.score_detection(base, flags)
    assert set(sc["true_positives"]) <= set(sc["planted"])
    assert set(sc["true_positives"]) <= set(sc["flagged"])
    assert 0.0 <= sc["precision"] <= 1.0 and 0.0 <= sc["recall"] <= 1.0


def test_planted_disruptions_live_only_in_the_sandbox(base):
    """The real history must never be touched by the planted disruptions."""
    import data_loaders as DL
    from mini_fixture import scms_rows
    lines, _ = DL.clean_scms(scms_rows())
    real_leads = sorted(x["lead_days"] for x in lines if x["vendor"] != "Tiny Vendor")
    ours = sorted(d.received_day - d.order_day for d in base.history)
    assert ours == pytest.approx(real_leads)
    assert base.planted_disruptions
    assert all(d.source == "sandbox" for d in base.sandbox)


def test_sandbox_resamples_each_vendors_own_leads(base):
    real = {}
    for d in base.history:
        real.setdefault(d.supplier_id, set()).add(round(d.received_day - d.order_day, 6))
    planted = {d["supplier_id"] for d in base.planted_disruptions}
    for d in base.sandbox:
        if d.supplier_id in planted:
            continue
        assert round(d.received_day - d.order_day, 6) in real[d.supplier_id]


def test_risk_contributions_sum_to_the_score(base):
    for r in SA.risk_score(base):
        assert sum(r["contributions"].values()) == pytest.approx(r["risk_score"])


def test_risk_weights_sum_to_one():
    assert sum(SA.RISK_WEIGHTS.values()) == pytest.approx(1.0)


def test_runout_day():
    # 100 on hand, 10/day, nothing arriving: runs out on day 10, 810 short by day 91
    day, short = SA.runout_day(100, 10, [], 91)
    assert day == pytest.approx(10) and short == pytest.approx(810)
    # an order of 1000 on day 5 carries it past the horizon
    day, short = SA.runout_day(100, 10, [(5, 1000)], 91)
    assert day > 91 and short == 0
