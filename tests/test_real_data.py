"""Smoke tests on the REAL data. Skipped when data/raw is missing (as in CI).

Run `python download_data.py` first to enable them.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import data_loaders as DL  # noqa: E402

pytestmark = pytest.mark.skipif(not DL.raw_data_present(),
                                reason="data/raw missing; run download_data.py")


@pytest.fixture(scope="module")
def base():
    import supply
    return supply.build()


def test_chain_24_shape(base):
    """Willems chain 24 (power-driven hand tools) as published."""
    assert len(base.products) == 17
    assert sum(c.level == 1 for c in base.components.values()) == 31
    assert sum(c.level == 2 for c in base.components.values()) == 209
    assert len(base.bom) == 1168
    assert base.meta["parts_with_lead_distribution"] == 114
    assert base.meta["demand_stages_split"] == 0


def test_scms_cleaning_counts(base):
    log = base.meta["scms_cleaning"]
    assert log["rows"] == 10324
    assert log["direct_drop"] == 4920
    assert log["kept"] == 4235
    assert len(base.suppliers) == 27
    assert all(s.n_lines >= DL.MIN_VENDOR_LINES for s in base.suppliers.values())
    assert "SCMS from RDC" not in {s.name for s in base.suppliers.values()}


def test_every_vendor_gets_parts_in_its_band(base):
    parts = [c for c in base.components.values() if c.level == 2]
    per = {}
    for c in parts:
        per.setdefault(c.supplier_id, []).append(c.lead_mean)
    assert set(per) == set(base.suppliers)
    ids = sorted(base.suppliers, key=lambda s: base.suppliers[s].lead_p50_days)
    for a, b in zip(ids, ids[1:]):
        assert max(per[a]) <= min(per[b]) + 1e-9


def test_most_real_deliveries_hit_the_scheduled_date_exactly(base):
    import supplier_analytics as SA
    ev = SA.scheduled_date_evidence(base)
    assert ev["exact_on_scheduled_pct"] > 80


def test_real_plan_is_short_but_nothing_is_past_due(base):
    import buildability as B
    mc = B.monte_carlo(base, n_sims=20)
    fill = (sum(sum(f["p50"]) for f in mc["fan"].values())
            / sum(sum(f["demand"]) for f in mc["fan"].values()))
    assert 0.2 < fill < 0.8
    drivers = B.shortage_drivers(base, mc)
    assert drivers and not any(d["already_too_late"] for d in drivers)


def test_build_is_deterministic_across_processes():
    """Python salts string hashing per process; any rng call driven by set
    iteration would make the supply base differ between runs."""
    prog = (f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
            "import supply\n"
            "b = supply.build()\n"
            "print(sum(c.on_hand for c in b.components.values()), len(b.pos), "
            "sum(d.received_day for d in b.sandbox), b.planted_disruptions)\n")
    outs = {subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, check=True).stdout.strip()
            for _ in range(2)}
    assert len(outs) == 1, outs
