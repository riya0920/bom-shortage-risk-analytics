"""Supplier performance beyond on-time %, and the deterioration detector.

1. OTIF AND ITS DEFINITIONAL TRAP.
   On time against WHICH promise: the original one, or the latest reschedule?
   Against the latest reschedule, a supplier who is always late but always
   calls ahead scores ~100%. The real SCMS file shows the symptom: most
   Direct-Drop lines are delivered EXACTLY on the scheduled date, which is what
   a promise field that gets updated looks like. SCMS keeps only one date, so
   the original promise is SIMULATED here (see supply.py) to show the size of
   the trap. The scorecard uses the original promise and reports the gap.

2. DISTRIBUTIONS, NOT AVERAGES.
   `detect_deterioration` compares a recent window to a baseline on the mean,
   the P95 and the variance, and flags on any of them. It is scored on the
   sandbox (real vendor baselines + planted disruptions) and also run on the
   real SCMS history, where there is no ground truth.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from supply import BASELINE_DAYS, RECENT_DAYS, SupplyBase


def _deliveries(base: SupplyBase, source: str):
    return base.sandbox if source == "sandbox" else base.history


def realised_lead_times(base: SupplyBase, source: str = "real"
                        ) -> dict[str, list[tuple[float, float]]]:
    """supplier -> [(order_day, realised_lead_days)], sorted by order day."""
    out: dict[str, list[tuple[float, float]]] = {}
    for d in _deliveries(base, source):
        out.setdefault(d.supplier_id, []).append(
            (d.order_day, d.received_day - d.order_day))
    for k in out:
        out[k].sort(key=lambda t: t[0])
    return out


def otif(base: SupplyBase, against: str = "original", source: str = "real"
         ) -> dict[str, dict]:
    """On-time per supplier against the chosen promise.

    "latest" = the SCMS scheduled date (real). "original" = the simulated
    original promise. SCMS has no in-full field we can trust per line, so this
    is on-time only; the name OTIF is kept because that is what the scorecard
    is called.
    """
    agg: dict[str, dict] = {}
    for d in _deliveries(base, source):
        promise = d.original_promise_day if against == "original" else d.promise_day
        a = agg.setdefault(d.supplier_id, {"n": 0, "on_time": 0, "late_days": []})
        a["n"] += 1
        late = d.received_day - promise
        a["on_time"] += int(late <= 0)
        a["late_days"].append(late)
    for sid, a in agg.items():
        a["otif_pct"] = 100.0 * a["on_time"] / max(1, a["n"])
        a["mean_late_days"] = float(np.mean(a["late_days"]))
        a["p95_late_days"] = float(np.percentile(a["late_days"], 95))
        del a["late_days"]
    return agg


def otif_gap(base: SupplyBase) -> list[dict]:
    """How much better a supplier looks when scored against reschedules."""
    orig = otif(base, "original")
    latest = otif(base, "latest")
    rows = []
    for sid in sorted(orig):
        o, l = orig[sid], latest.get(sid, orig[sid])
        s = base.suppliers[sid]
        rows.append({
            "supplier_id": sid, "name": s.name, "n_receipts": o["n"],
            "otif_vs_original_pct": o["otif_pct"],
            "otif_vs_latest_promise_pct": l["otif_pct"],
            "gaming_gap_pts": l["otif_pct"] - o["otif_pct"],
            "exact_on_scheduled_date_pct": 100 * s.exact_on_scheduled_date,
            "habitual_rescheduler_simulated": s.habitual_rescheduler,
        })
    return sorted(rows, key=lambda r: (-r["gaming_gap_pts"], r["supplier_id"]))


def scheduled_date_evidence(base: SupplyBase) -> dict:
    """REAL: how often SCMS deliveries land exactly on the scheduled date."""
    late = np.array([d.received_day - d.promise_day for d in base.history])
    per = sorted(((s.exact_on_scheduled_date, sid) for sid, s in base.suppliers.items()),
                 reverse=True)
    return {
        "n_lines": int(len(late)),
        "exact_on_scheduled_pct": float(100 * np.mean(late == 0)),
        "on_or_before_scheduled_pct": float(100 * np.mean(late <= 0)),
        "late_pct": float(100 * np.mean(late > 0)),
        "late_by_more_than_7d_pct": float(100 * np.mean(late > 7)),
        "vendors_exact_over_90pct": sum(1 for v, _ in per if v > 0.9),
        "n_vendors": len(per),
    }


def _window_stats(series, lo_recent, hi_recent, lo_base, min_n):
    recent = [v for d, v in series if lo_recent <= d < hi_recent]
    base_pts = [v for d, v in series if lo_base <= d < lo_recent]
    if len(recent) < min_n or len(base_pts) < min_n:
        return None
    return recent, base_pts


def _triggers(recent, base_pts, mean_t=1.25, p95_t=1.30, sd_t=1.50):
    b_mean, r_mean = float(np.mean(base_pts)), float(np.mean(recent))
    b_p95, r_p95 = float(np.percentile(base_pts, 95)), float(np.percentile(recent, 95))
    b_sd, r_sd = float(np.std(base_pts, ddof=1)), float(np.std(recent, ddof=1))
    trig = []
    if r_mean > b_mean * mean_t:
        trig.append("mean")
    if r_p95 > b_p95 * p95_t:
        trig.append("p95")
    if r_sd > b_sd * sd_t:
        trig.append("variance")
    return {
        "n_recent": len(recent), "n_baseline": len(base_pts),
        "mean_baseline": b_mean, "mean_recent": r_mean,
        "mean_ratio": r_mean / max(1e-9, b_mean),
        "p95_baseline": b_p95, "p95_recent": r_p95,
        "p95_ratio": r_p95 / max(1e-9, b_p95),
        "sd_ratio": r_sd / max(1e-9, b_sd),
        "triggers": trig, "flagged": bool(trig),
    }


def detect_deterioration(base: SupplyBase, baseline_days: int = BASELINE_DAYS,
                         recent_days: int = RECENT_DAYS, min_n: int = 5,
                         source: str = "sandbox", thresholds=(1.25, 1.30, 1.50)
                         ) -> list[dict]:
    """Flag suppliers whose lead-time DISTRIBUTION has moved.

    Windows are by order date, relative to day 0: recent = the last year,
    baseline = the two years before it (the same windows as the yearly scan of
    the real history). Scored with hindsight: every order in the window is
    assumed delivered (true for the SCMS history, which only has delivered
    lines). `thresholds` = (mean, P95, sd) ratios that fire each trigger.
    """
    rows = []
    for sid, series in realised_lead_times(base, source).items():
        w = _window_stats(series, -recent_days, 1e9, -baseline_days, min_n)
        if w is None:
            continue
        rows.append({"supplier_id": sid, **_triggers(*w, *thresholds)})
    return sorted(rows, key=lambda r: -r["p95_ratio"])


def scan_real_history(base: SupplyBase, min_n: int = 5) -> dict:
    """Run the detector over the REAL SCMS history, one calendar year at a time.

    For each vendor and year Y: recent = orders placed in Y, baseline = the two
    years before. There is no ground truth, so this reports what gets flagged,
    not whether it was right.
    """
    day0 = dt.date.fromisoformat(base.meta["day0"])
    rl = realised_lead_times(base, "real")
    rows, evaluated = [], 0
    for year in range(2008, day0.year + 1):
        lo = (dt.date(year, 1, 1) - day0).days
        hi = (dt.date(year + 1, 1, 1) - day0).days
        blo = (dt.date(year - 2, 1, 1) - day0).days
        for sid, series in rl.items():
            recent = [v for d, v in series if lo <= d < hi]
            base_pts = [v for d, v in series if blo <= d < lo]
            if len(recent) < min_n or len(base_pts) < min_n:
                continue
            evaluated += 1
            t = _triggers(recent, base_pts)
            if t["flagged"]:
                rows.append({"supplier_id": sid, "name": base.suppliers[sid].name,
                             "year": year, **t})
    rows.sort(key=lambda r: -r["p95_ratio"])
    by_trigger = {k: sum(1 for r in rows if k in r["triggers"])
                  for k in ("mean", "p95", "variance")}
    return {"vendor_years_evaluated": evaluated, "vendor_years_flagged": len(rows),
            "flag_rate": len(rows) / max(evaluated, 1),
            "by_trigger": by_trigger,
            "vendors_flagged_at_least_once": len({r["supplier_id"] for r in rows}),
            "top": rows[:12]}


def score_detection(base: SupplyBase, flags: list[dict]) -> dict:
    """Precision and recall against the planted (simulated) disruptions."""
    planted = {d["supplier_id"] for d in base.planted_disruptions}
    evaluated = {r["supplier_id"] for r in flags}
    flagged = {r["supplier_id"] for r in flags if r["flagged"]}
    tp = planted & flagged
    by_kind: dict[str, dict] = {}
    for d in base.planted_disruptions:
        k = by_kind.setdefault(d["kind"], {"planted": 0, "caught": 0,
                                           "by_p95_only": 0, "not_evaluable": 0})
        k["planted"] += 1
        if d["supplier_id"] not in evaluated:
            k["not_evaluable"] += 1
        if d["supplier_id"] in flagged:
            k["caught"] += 1
            r = next(x for x in flags if x["supplier_id"] == d["supplier_id"])
            if r["triggers"] == ["p95"]:
                k["by_p95_only"] += 1
    return {
        "planted": sorted(planted), "flagged": sorted(flagged),
        "evaluated": len(evaluated),
        "true_positives": sorted(tp), "false_positives": sorted(flagged - planted),
        "missed": sorted(planted - flagged),
        "missed_not_evaluable": sorted(planted - evaluated),
        "precision": len(tp) / max(1, len(flagged)),
        "recall": len(tp) / max(1, len(planted)),
        "by_kind": by_kind,
    }


RISK_WEIGHTS = {"single_source": 0.30, "lead_variability": 0.30,
                "late_vs_scheduled": 0.20, "recent_drift": 0.20}


def risk_score(base: SupplyBase) -> list[dict]:
    """Composite supplier risk with EXPLICIT weights, all inputs real SCMS.

      single_source      share of the vendor's lines for products no other
                         SCMS vendor shipped
      lead_variability   coefficient of variation of real lead times
      late_vs_scheduled  share of lines delivered after the scheduled date
      recent_drift       last 12 months' mean lead / the vendor's earlier mean

    Weights are a stated judgement, not fitted: there is no outcome to fit to.
    """
    rl = realised_lead_times(base, "real")
    parts_by = {}
    for c in base.components.values():
        if c.supplier_id:
            parts_by[c.supplier_id] = parts_by.get(c.supplier_id, 0) + 1
    rows = []
    for sid, s in base.suppliers.items():
        series = rl.get(sid, [])
        leads = [v for _, v in series]
        cv = float(np.std(leads) / np.mean(leads)) if len(leads) > 3 else 0.0
        last = max((d for d, _ in series), default=0.0)
        recent = [v for d, v in series if d >= last - 365]
        older = [v for d, v in series if d < last - 365]
        drift = (float(np.mean(recent)) / float(np.mean(older))
                 if len(recent) >= 3 and len(older) >= 3 else 1.0)
        parts = {
            "single_source": s.sole_source_share,
            "lead_variability": min(1.0, cv / 1.0),
            "late_vs_scheduled": min(1.0, s.late_rate / 0.15),
            "recent_drift": min(1.0, max(drift - 1.0, 0.0) / 0.5),
        }
        total = sum(RISK_WEIGHTS[k] * v for k, v in parts.items())
        rows.append({
            "supplier_id": sid, "name": s.name, "risk_score": total,
            "contributions": {k: RISK_WEIGHTS[k] * v for k, v in parts.items()},
            "parts_mapped": parts_by.get(sid, 0),
            "sole_source_share": s.sole_source_share,
            "lead_cv": cv, "late_rate": s.late_rate, "recent_drift": drift,
            "n_lines": s.n_lines,
        })
    return sorted(rows, key=lambda r: (-r["risk_score"], r["supplier_id"]))


def runout_day(on_hand: float, daily: float, arrivals: list[tuple[float, float]],
               horizon_days: float) -> tuple[float, float]:
    """(day the part runs out, units short over the horizon).

    Stock falls at `daily` units a day; open orders add their quantity on
    their promise day. If it never runs out inside the horizon, the run-out
    day is extrapolated past it from what is left.
    """
    if daily <= 0:
        return float("inf"), 0.0
    s, cur, out = on_hand, 0.0, None
    for day, qty in sorted(arrivals) + [(horizon_days, 0.0)]:
        day = min(max(day, 0.0), horizon_days)
        if out is None and s - daily * (day - cur) < 0:
            out = cur + s / daily
        s -= daily * (day - cur)
        cur = day
        s += qty
    if out is None:
        out = horizon_days + s / daily
    return float(out), float(max(-s, 0.0))


def part_risk_rows(base: SupplyBase, stock_scale: float = 1.0) -> list[dict]:
    """One row per purchased part: slack against its own lead time, and value.

    slack = projected run-out day (simulated stock + simulated open orders,
    drawn down by the real plan) - the part's real mean lead time. Negative
    slack means the part runs out before a new order could arrive.
    Units at risk = the shortfall over the 13-week horizon; value = that x the
    real unit cost. `stock_scale` multiplies on-hand stock, for sensitivity.

    The simulated version measured slack from on-hand stock only. With real
    lead times of days rather than months, that said every part was fine while
    the Monte Carlo said most of the plan could not be built, so slack now
    uses the projected run-out day, which sees the whole position.
    """
    from supply import WEEKS, part_daily_demand
    risk = {r["supplier_id"]: r for r in risk_score(base)}
    d_day = part_daily_demand(base)
    arrivals: dict[str, list] = {}
    for po in base.pos:
        if po.received_day is None:
            arrivals.setdefault(po.part, []).append((po.promise_day, po.qty))
    rows = []
    for part, c in sorted(base.components.items()):
        if c.level != 2 or c.supplier_id is None:
            continue
        d = d_day.get(part, 0.0)
        if d <= 0:
            continue
        ro, short = runout_day(c.on_hand * stock_scale, d, arrivals.get(part, []),
                               WEEKS * 7.0)
        slack = ro - c.lead_mean
        rows.append({"part": part, "supplier_id": c.supplier_id,
                     "days_of_slack": slack, "lead_days": c.lead_mean,
                     "runout_day": ro,
                     "units_at_risk": short,
                     "value_at_risk": short * c.unit_cost,
                     "single_sourced": c.single_sourced,
                     "risk_score": risk.get(c.supplier_id, {}).get("risk_score", 0.0)})
    return rows
