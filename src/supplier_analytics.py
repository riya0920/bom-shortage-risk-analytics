"""Supplier performance beyond OTD%, and the deterioration detector.

Two things here that a supplier scorecard usually gets wrong.

1. OTIF AND ITS DEFINITIONAL TRAP.
   On-time-in-full, measured against WHICH promise? The ORIGINAL promise made when
   the order was placed, or the LATEST reschedule agreed last Tuesday?

   Against the latest reschedule, a supplier who is chronically late but always
   calls ahead scores ~100%. That is the number suppliers prefer, and it is the
   default in a lot of ERP configurations, because the promise field gets
   overwritten in place. Against the original promise, the same supplier scores
   badly -- and correctly, because the plant re-planned around every one of those
   reschedules.

   Both are computed here. The one that goes on the scorecard is the ORIGINAL, and
   the gap between the two is reported as its own metric: it is a direct measure of
   how much a supplier is managing their score rather than their delivery.

2. DISTRIBUTIONS, NOT AVERAGES.
   A supplier whose mean lead time is unchanged but whose P95 has doubled is a
   supplier who will stop your line, and every mean-based monitor says they are
   fine. `detect_deterioration` compares recent to baseline on the mean AND the
   P95 AND the variance, and flags on any of them.
"""
from __future__ import annotations

import numpy as np

from supply import SupplyBase


def realised_lead_times(base: SupplyBase) -> dict[str, list[tuple[float, float]]]:
    """supplier -> [(order_day, realised_lead_days)] for received POs."""
    out: dict[str, list[tuple[float, float]]] = {}
    for po in base.pos:
        if po.received_day is None:
            continue
        s = base.suppliers[po.supplier_id]
        order_day = po.original_promise_day - s.lead_mean_days
        out.setdefault(po.supplier_id, []).append(
            (order_day, po.received_day - order_day))
    for k in out:
        out[k].sort(key=lambda t: t[0])
    return out


def otif(base: SupplyBase, against: str = "original") -> dict[str, dict]:
    """On-time-in-full per supplier, against the chosen promise definition."""
    agg: dict[str, dict] = {}
    for po in base.pos:
        if po.received_day is None:
            continue
        promise = po.original_promise_day if against == "original" else po.promise_day
        a = agg.setdefault(po.supplier_id, {"n": 0, "on_time": 0, "late_days": []})
        a["n"] += 1
        late = po.received_day - promise
        a["on_time"] += int(late <= 0)
        a["late_days"].append(late)
    for sid, a in agg.items():
        a["otif_pct"] = 100.0 * a["on_time"] / max(1, a["n"])
        a["mean_late_days"] = float(np.mean(a["late_days"]))
        a["p95_late_days"] = float(np.percentile(a["late_days"], 95))
        del a["late_days"]
    return agg


def otif_gap(base: SupplyBase) -> list[dict]:
    """How much better a supplier looks when scored against their reschedules."""
    orig = otif(base, "original")
    latest = otif(base, "latest")
    rows = []
    for sid in sorted(orig):
        o, l = orig[sid], latest.get(sid, orig[sid])
        rows.append({
            "supplier_id": sid, "n_receipts": o["n"],
            "otif_vs_original_pct": o["otif_pct"],
            "otif_vs_latest_promise_pct": l["otif_pct"],
            "gaming_gap_pts": l["otif_pct"] - o["otif_pct"],
        })
    return sorted(rows, key=lambda r: -r["gaming_gap_pts"])


def detect_deterioration(base: SupplyBase, baseline_days: int = 365,
                         recent_days: int = 120, min_n: int = 5) -> list[dict]:
    """Flag suppliers whose lead-time DISTRIBUTION has moved.

    Three independent triggers, because they catch different things:
      mean shift    -- the obvious one
      P95 blowout   -- the mean can be flat while the tail explodes; this is the
                       one that stops lines and the one averages cannot see
      variance up   -- unpredictability itself is a cost, because safety stock
                       scales with it
    """
    rl = realised_lead_times(base)
    rows = []
    for sid, series in rl.items():
        recent = [v for d, v in series if d >= -recent_days]
        base_pts = [v for d, v in series if -baseline_days <= d < -recent_days]
        if len(recent) < min_n or len(base_pts) < min_n:
            continue
        b_mean, r_mean = float(np.mean(base_pts)), float(np.mean(recent))
        b_p95, r_p95 = float(np.percentile(base_pts, 95)), float(np.percentile(recent, 95))
        b_sd, r_sd = float(np.std(base_pts, ddof=1)), float(np.std(recent, ddof=1))
        trig = []
        if r_mean > b_mean * 1.25:
            trig.append("mean")
        if r_p95 > b_p95 * 1.30:
            trig.append("p95")
        if r_sd > b_sd * 1.50:
            trig.append("variance")
        rows.append({
            "supplier_id": sid, "n_recent": len(recent), "n_baseline": len(base_pts),
            "mean_baseline": b_mean, "mean_recent": r_mean,
            "mean_ratio": r_mean / max(1e-9, b_mean),
            "p95_baseline": b_p95, "p95_recent": r_p95,
            "p95_ratio": r_p95 / max(1e-9, b_p95),
            "sd_ratio": r_sd / max(1e-9, b_sd),
            "triggers": trig, "flagged": bool(trig),
        })
    return sorted(rows, key=lambda r: -r["p95_ratio"])


def score_detection(base: SupplyBase, flags: list[dict]) -> dict:
    """Precision and recall against the planted disruptions."""
    planted = {d["supplier_id"] for d in base.planted_disruptions}
    flagged = {r["supplier_id"] for r in flags if r["flagged"]}
    tp = planted & flagged
    fp = flagged - planted
    fn = planted - flagged
    by_kind: dict[str, dict] = {}
    for d in base.planted_disruptions:
        k = by_kind.setdefault(d["kind"], {"planted": 0, "caught": 0, "by_p95_only": 0})
        k["planted"] += 1
        if d["supplier_id"] in flagged:
            k["caught"] += 1
            r = next(x for x in flags if x["supplier_id"] == d["supplier_id"])
            if r["triggers"] == ["p95"]:
                k["by_p95_only"] += 1
    return {
        "planted": sorted(planted), "flagged": sorted(flagged),
        "true_positives": sorted(tp), "false_positives": sorted(fp),
        "missed": sorted(fn),
        "precision": len(tp) / max(1, len(flagged)),
        "recall": len(tp) / max(1, len(planted)),
        "by_kind": by_kind,
    }


def risk_score(base: SupplyBase) -> list[dict]:
    """Composite supplier risk with EXPLICIT weights.

    Weights are stated here and printed in the report. A composite score with
    hidden weights is an opinion wearing a number's clothes, and the first thing a
    sourcing manager asks is 'why is this supplier red' -- which is unanswerable
    unless the contributions decompose.
    """
    weights = {"single_source": 0.30, "lead_variability": 0.25,
               "quality": 0.25, "financial": 0.20}
    rl = realised_lead_times(base)
    single_by_supplier: dict[str, int] = {}
    for c in base.components.values():
        if c.supplier_id and c.single_sourced:
            single_by_supplier[c.supplier_id] = single_by_supplier.get(c.supplier_id, 0) + 1

    rows = []
    for sid, s in base.suppliers.items():
        series = [v for _, v in rl.get(sid, [])]
        cv = float(np.std(series) / np.mean(series)) if len(series) > 3 else 0.0
        parts = {
            "single_source": min(1.0, single_by_supplier.get(sid, 0) / 5.0),
            "lead_variability": min(1.0, cv / 0.6),
            "quality": min(1.0, s.quality_reject_rate / 0.05),
            "financial": 1.0 - s.financial_score,
        }
        total = sum(weights[k] * v for k, v in parts.items())
        rows.append({
            "supplier_id": sid, "risk_score": total,
            "contributions": {k: weights[k] * v for k, v in parts.items()},
            "single_sourced_parts": single_by_supplier.get(sid, 0),
            "lead_cv": cv, "quality_reject_rate": s.quality_reject_rate,
            "financial_score": s.financial_score,
        })
    return sorted(rows, key=lambda r: -r["risk_score"])


RISK_WEIGHTS = {"single_source": 0.30, "lead_variability": 0.25,
                "quality": 0.25, "financial": 0.20}
