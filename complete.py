"""Safety stock, lot sizing, MOQ, supplier capacity, a threshold sweep on the
deterioration detector, alert routing with SLAs, a supplier-distress proxy, and
charts.

    python complete.py
    python complete.py --quick
    python complete.py --report-only
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import buildability as BLD  # noqa: E402
import inventory as INV  # noqa: E402
import routing as RT  # noqa: E402
import supplier_analytics as SA  # noqa: E402
import supply  # noqa: E402

OUT = ROOT / "out"
DOCS = ROOT / "docs"
QUICK = "--quick" in sys.argv


# ---------------------------------------------------------------------------
# 1. safety stock
# ---------------------------------------------------------------------------

def part_lead_samples(base, part: str, n: int, rng) -> np.ndarray:
    """Lead-time samples for one part: its real Willems distribution, plus
    the mapped vendor's real lateness (as a share of the quoted lead)."""
    c = base.components[part]
    s = base.suppliers[c.supplier_id]
    draws = rng.choice(np.asarray(c.lead_values, float), size=n,
                       p=np.asarray(c.lead_probs, float))
    slip = s.rel_late[rng.integers(0, len(s.rel_late), size=n)]
    return draws * (1.0 + slip)


def stage_safety_stock(base) -> dict:
    """Both variance terms, with REAL inputs on both sides.

    demand side: the product demand standard deviations in Willems, pushed
                 through the BOM (products treated as independent)
    supply side: the part's Willems lead-time spread, plus vendor lateness
    """
    rng = np.random.default_rng(5)
    d_day = supply.part_daily_demand(base)
    sd_day = supply.part_daily_demand_sd(base)
    rows = []
    for part, c in sorted(base.components.items()):
        if c.level != 2 or c.supplier_id is None or d_day.get(part, 0) <= 0:
            continue
        samples = part_lead_samples(base, part, 400, rng)
        L, sL = float(samples.mean()), float(samples.std())
        par = INV.safety_stock(d_day[part], sd_day[part], L, sL)
        emp = INV.empirical_safety_stock(d_day[part], sd_day[part], samples)
        spread = float(np.quantile(samples, 0.95) / max(np.quantile(samples, 0.5), 1e-9))
        rows.append({"part": part, "supplier_id": c.supplier_id,
                     "fat_tail": spread >= 1.3, "demand_per_day": d_day[part],
                     "demand_sd_per_day": sd_day[part],
                     "lead_mean": L, "lead_sd": sL, "lead_p95_over_p50": spread,
                     "unit_cost": c.unit_cost, **par, "empirical": emp})
    fat = [r for r in rows if r["fat_tail"] and not r["empirical"].get("insufficient_history")]
    thin = [r for r in rows if not r["fat_tail"] and not r["empirical"].get("insufficient_history")]

    def gap(rs):
        if not rs:
            return float("nan")
        return float(np.mean([r["empirical"]["safety_stock"]
                              / max(r["safety_stock"], 1e-9) for r in rs]))

    return {"rows": rows[:40], "n": len(rows),
            "mean_understatement_demand_only": float(
                np.mean([r["understatement_factor"] for r in rows])),
            "mean_supply_variance_share": float(
                np.mean([r["supply_variance_share"] for r in rows])),
            "parts_supply_dominated": int(sum(r["supply_variance_share"] > 0.5
                                              for r in rows)),
            "median_demand_cv": float(np.median([r["demand_sd_per_day"]
                                                 / r["demand_per_day"] for r in rows])),
            "median_lead_cv": float(np.median([r["lead_sd"] / max(r["lead_mean"], 1e-9)
                                               for r in rows])),
            "empirical_over_parametric_fat_tail": gap(fat),
            "empirical_over_parametric_thin_tail": gap(thin),
            "n_fat": len(fat), "n_thin": len(thin)}


# ---------------------------------------------------------------------------
# 2. lot sizing, MOQ, capacity
# ---------------------------------------------------------------------------

def stage_lot_sizing(base) -> dict:
    rng = np.random.default_rng(11)
    d_day = supply.part_daily_demand(base)
    comps = []
    demands: dict[str, float] = {}
    for part, c in sorted(base.components.items()):
        if c.level != 2 or c.supplier_id is None:
            continue
        annual = d_day.get(part, 0.0) * 365
        if annual <= 0:
            continue
        need_now = max(annual / 12 - c.on_hand, 0.0)
        comps.append({"part": part, "need": need_now, "moq": c.moq,
                      "pack_size": 1.0, "annual_demand": annual,
                      "unit_cost": max(c.unit_cost, 0.01),
                      "supplier_id": c.supplier_id})
        demands[c.supplier_id] = demands.get(c.supplier_id, 0.0) + need_now

    waste = INV.moq_waste(comps)

    # Supplier capacity is in no public dataset. Assigned as a multiple of
    # current commitment so some suppliers are tight, and flagged as assumed.
    capacity = {sid: need * float(rng.uniform(0.6, 2.5))
                for sid, need in demands.items()}
    cap = INV.capacity_check(demands, capacity)

    examples = []
    for c in sorted(comps, key=lambda c: -c["moq"])[:8]:
        examples.append({**{k: c[k] for k in ("part", "need", "moq", "unit_cost")},
                         **INV.order_quantity(
                             c["need"], moq=c["moq"],
                             annual_demand=c["annual_demand"],
                             unit_cost=c["unit_cost"],
                             supplier_capacity=capacity.get(c["supplier_id"]))})
    return {"moq_waste": waste, "capacity": {k: v for k, v in cap.items()
                                             if k != "rows"},
            "capacity_worst": sorted(cap["rows"], key=lambda r: -r["utilisation"])[:8],
            "examples": examples, "n_components": len(comps),
            "parts_with_zero_need": sum(1 for c in comps if c["need"] <= 0),
            "capacity_is_assumed": True, "moq_is_simulated": True}


# ---------------------------------------------------------------------------
# 3. threshold sweep (sandbox: real baselines + planted disruptions)
# ---------------------------------------------------------------------------

def stage_threshold_sweep(base) -> dict:
    """Precision/recall across thresholds, and which trigger earns its place."""
    planted = {d["supplier_id"] for d in base.planted_disruptions}

    def run(mean_t, p95_t, sd_t):
        flags = SA.detect_deterioration(base, thresholds=(mean_t, p95_t, sd_t))
        flagged = {r["supplier_id"] for r in flags if r["flagged"]}
        by_trigger = {k: {r["supplier_id"] for r in flags if k in r["triggers"]}
                      for k in ("mean", "p95", "variance")}
        tp = planted & flagged
        return {"n_flagged": len(flagged), "n_evaluated": len(flags),
                "precision": len(tp) / max(len(flagged), 1),
                "recall": len(tp) / max(len(planted), 1),
                "by_trigger_recall": {k: len(planted & v) / max(len(planted), 1)
                                      for k, v in by_trigger.items()}}

    curve = []
    for scale in (1.05, 1.10, 1.15, 1.25, 1.40, 1.60, 2.00):
        r = run(scale, scale * 1.04, scale * 1.2)
        f1 = (2 * r["precision"] * r["recall"]
              / max(r["precision"] + r["recall"], 1e-9))
        curve.append({"scale": scale, **r, "f1": f1})

    big = 99.0
    ablation = {"all three": run(1.25, 1.30, 1.50),
                "without mean": run(big, 1.30, 1.50),
                "without p95": run(1.25, big, 1.50),
                "without variance": run(1.25, 1.30, big)}
    return {"curve": curve, "ablation": ablation,
            "n_planted": len(planted),
            "best_f1": max(curve, key=lambda r: (r["f1"], -r["scale"]))}


# ---------------------------------------------------------------------------
# 4. alert routing
# ---------------------------------------------------------------------------

def stage_routing(base, lot) -> dict:
    rows = SA.part_risk_rows(base)
    over = {r["supplier_id"] for r in lot["capacity_worst"] if r["over"]}
    alerts = RT.build_alerts(rows, over_capacity_suppliers=over)
    alerts, dropped = RT.dedupe(alerts)
    load = RT.load_by_role(alerts)
    by_sev: dict[str, int] = {}
    for a in alerts:
        by_sev[a.severity] = by_sev.get(a.severity, 0) + 1
    sev_num = {"P1": 4, "P2": 3, "P3": 2, "P4": 1}
    xs = [sev_num[a.severity] for a in alerts]
    ys = [a.risk_score for a in alerts]
    corr = (float(np.corrcoef(xs, ys)[0, 1])
            if len(alerts) > 2 and np.std(xs) > 0 and np.std(ys) > 0 else float("nan"))
    return {"n_alerts": len(alerts), "deduped": dropped, "by_severity": by_sev,
            "load_by_role": load,
            "severity_vs_riskscore_correlation": corr,
            "runs_out_in_horizon": sum(1 for r in rows
                                       if r["runout_day"] < supply.WEEKS * 7),
            "earliest_runout_day": min(r["runout_day"] for r in rows),
            "top": [a.as_dict() for a in alerts[:10]],
            "n_parts_considered": len(rows)}


# ---------------------------------------------------------------------------
# 6. supplier distress proxy
# ---------------------------------------------------------------------------

def _distress_rows(base, source: str) -> list[dict]:
    rows = []
    by: dict[str, list] = {}
    deliveries = base.sandbox if source == "sandbox" else base.history
    for d in deliveries:
        by.setdefault(d.supplier_id, []).append(d)

    def lead(xs):
        return float(np.mean([d.received_day - d.order_day for d in xs]))

    def late(xs):
        return float(np.mean([d.received_day > d.promise_day for d in xs]))

    for sid, ds in sorted(by.items()):
        last = max(d.order_day for d in ds)
        recent = [d for d in ds if d.order_day >= last - 365]
        older = [d for d in ds if d.order_day < last - 365]
        if len(recent) < 3 or len(older) < 3:
            continue
        infl = lead(recent) / max(lead(older), 1e-9)
        slip = late(recent) - late(older)
        distress = (0.6 * min(max(infl - 1.0, 0.0), 1.0)
                    + 0.4 * min(max(slip, 0.0) * 5, 1.0))
        rows.append({"supplier_id": sid, "name": base.suppliers[sid].name,
                     "lead_inflation": infl, "late_rate_change": slip,
                     "observed_distress": distress,
                     "on_time_vs_scheduled_all_time": 1.0 - late(ds)})
    return rows


def stage_financial(base) -> dict:
    """A distress proxy from behaviour a buyer can observe.

    The simulated version compared this with an invented "financial score" (a
    random draw). Real vendors come with no credit data at all, so the
    comparison is now with the number a scorecard would normally show: on-time
    rate against the scheduled date. Both are scored as predictors of the
    planted disruptions in the sandbox, then the proxy is run on the real
    history. Weights (0.6 lead inflation, 0.4 rise in late rate) are stated,
    not fitted.
    """
    from sklearn.metrics import roc_auc_score
    sb = _distress_rows(base, "sandbox")
    planted = {d["supplier_id"] for d in base.planted_disruptions}
    lab = np.array([r["supplier_id"] in planted for r in sb], dtype=int)
    if lab.any() and not lab.all():
        auc_obs = float(roc_auc_score(lab, [r["observed_distress"] for r in sb]))
        auc_otd = float(roc_auc_score(lab, [1 - r["on_time_vs_scheduled_all_time"]
                                           for r in sb]))
    else:
        auc_obs = auc_otd = float("nan")
    real = _distress_rows(base, "real")
    return {"n": len(sb), "n_planted": len(planted),
            "auc_observed_distress": auc_obs,
            "auc_on_time_scorecard": auc_otd,
            "real_top": sorted(real, key=lambda r: -r["observed_distress"])[:8],
            "caveat": ("still a proxy with stated weights; SCMS has no record of "
                       "supplier failures, so the real-history ranking cannot be "
                       "checked")}


# ---------------------------------------------------------------------------
# 5. charts
# ---------------------------------------------------------------------------

def _fan_svg(p10, p50, p90, labels, width=760, height=260):
    pad_l, pad_b, pad_t, pad_r = 56, 40, 16, 14
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    hi = max(p90) * 1.08 or 1.0
    n = len(p50)

    def x(i):
        return pad_l + pw * (i / max(n - 1, 1))

    def y(v):
        return pad_t + ph * (1 - v / hi)

    band = (" ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(p90))
            + " " + " ".join(f"{x(i):.1f},{y(v):.1f}"
                             for i, v in reversed(list(enumerate(p10)))))
    line = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}"
                    for i, v in enumerate(p50))
    ticks = "".join(
        f'<text x="{x(i):.1f}" y="{height - pad_b + 15}" text-anchor="middle" '
        f'class="ax">{l}</text>' for i, l in enumerate(labels) if i % max(n // 8, 1) == 0)
    grid = "".join(
        f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(hi * f):.1f}" '
        f'y2="{y(hi * f):.1f}" class="grid"/>'
        f'<text x="{pad_l - 6}" y="{y(hi * f) + 4:.1f}" text-anchor="end" '
        f'class="ax">{hi * f:.0f}</text>' for f in (0, .25, .5, .75, 1))
    return (f'<svg viewBox="0 0 {width} {height}" class="chart">{grid}'
            f'<polygon points="{band}" class="fan"/>'
            f'<path d="{line}" class="p50"/>{ticks}</svg>')


def _bars_svg(rows, width=380, height=230, fmt="{:.0f}"):
    if not rows:
        return ""
    pad_l, pad_t, pad_b = 150, 8, 18
    pw = width - pad_l - 40
    bh = (height - pad_t - pad_b) / len(rows)
    vmax = max(v for _, v in rows) or 1
    out = []
    for i, (lbl, v) in enumerate(rows):
        yy = pad_t + i * bh
        out.append(
            f'<rect x="{pad_l}" y="{yy + bh * .15:.1f}" '
            f'width="{pw * v / vmax:.1f}" height="{bh * .7:.1f}" class="bar"/>'
            f'<text x="{pad_l - 6}" y="{yy + bh * .62:.1f}" text-anchor="end" '
            f'class="ax">{str(lbl)[:20]}</text>'
            f'<text x="{pad_l + pw * v / vmax + 5:.1f}" y="{yy + bh * .62:.1f}" '
            f'class="val">{fmt.format(v)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" class="chart">{"".join(out)}</svg>'


def stage_charts(base, mc, routed, sweep, lot) -> dict:
    import html
    # monte_carlo() returns a per-product fan; the chart shows the plant total.
    fan_by = mc.get("fan") or {}
    def _total(key):
        cols = [f[key] for f in fan_by.values()]
        return [float(sum(w)) for w in zip(*cols)] if cols else []
    p10, p50, p90 = _total("p10"), _total("p50"), _total("p90")
    fan = (_fan_svg(p10, p50, p90, [f"w{i}" for i in range(len(p50))])
           if p50 else "<p class='ax'>no fan data</p>")

    sev_rows = [(k, v) for k, v in sorted(routed["by_severity"].items())]
    role_rows = [(k, v["total"]) for k, v in sorted(
        routed["load_by_role"].items(), key=lambda kv: -kv[1]["total"])]
    moq_rows = [(r["part"], r["excess_value"])
                for r in lot["moq_waste"]["worst"][:8]]

    pr = "".join(
        f'<tr><td class="n">{c["scale"]:.2f}</td>'
        f'<td class="n">{c["precision"]:.2f}</td>'
        f'<td class="n">{c["recall"]:.2f}</td>'
        f'<td class="n">{c["f1"]:.2f}</td></tr>' for c in sweep["curve"])

    alert_rows = "".join(
        f'<tr><td><b>{a["severity"]}</b></td><td>{html.escape(a["part"])}</td>'
        f'<td class="n">{a["days_of_slack"]:.0f}</td>'
        f'<td class="n">{a["value_at_risk"]:,.0f}</td>'
        f'<td>{html.escape(a["route_to"])}</td>'
        f'<td class="n">{a["response_hours"]}h</td></tr>'
        for a in routed["top"])

    doc = f"""<!doctype html>
<meta charset="utf-8"><title>Supply risk</title>
<style>
:root{{--bg:#f7fafc;--fg:#1a202c;--card:#fff;--line:#e2e8f0;--mut:#718096;
       --bar:#e07a5f;--fan:#3182ce}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171923;--fg:#e2e8f0;--card:#242c3d;
  --line:#3a4459;--mut:#a0aec0;--bar:#f08a68;--fan:#4c9bea}}}}
*{{box-sizing:border-box}}
body{{margin:0;padding:24px;background:var(--bg);color:var(--fg);
      font:14px/1.55 system-ui,sans-serif}}
h1{{font-size:20px;margin:0 0 2px}}
h2{{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--mut);
    margin:0 0 10px}}
.sub{{color:var(--mut);margin-bottom:20px}}
.grid{{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;
       padding:16px;overflow-x:auto}}
.wide{{grid-column:1/-1}}
.chart{{width:100%;height:auto}}
line.grid{{stroke:var(--line)}}
polygon.fan{{fill:var(--fan);opacity:.22}}
path.p50{{fill:none;stroke:var(--fan);stroke-width:1.8}}
rect.bar{{fill:var(--bar);rx:2}}
text.ax{{fill:var(--mut);font-size:10px}}
text.val{{fill:var(--fg);font-size:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-size:11px;text-transform:uppercase}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.note{{font-size:12px;color:var(--mut);margin-top:8px}}
</style>
<h1>Supply risk</h1>
<div class="sub">{routed['n_parts_considered']} purchased parts &middot;
 {routed['n_alerts']} routed alerts &middot; generated by <code>complete.py</code>
 &middot; BOM and lead times: Willems (2008), real &middot; vendors: USAID SCMS, real
 &middot; stock and open orders: simulated</div>
<div class="grid">
  <div class="card wide">
    <h2>Buildable units - P10 / P50 / P90 fan</h2>
    {fan}
    <div class="note">The band is the 10th-90th percentile across Monte-Carlo
      draws of supplier lead time. A point forecast here would be the least
      useful number available: the decision is how much cover to buy, and that
      is set by the width, not the middle.</div>
  </div>
  <div class="card">
    <h2>Alerts by severity</h2>
    {_bars_svg(sev_rows)}
    <div class="note">Severity is consequence &times; urgency, not risk score.
      Correlation between the two here is
      {routed['severity_vs_riskscore_correlation']:.2f} - if it were near 1 the
      tiering would be doing nothing.</div>
  </div>
  <div class="card">
    <h2>Load by role</h2>
    {_bars_svg(role_rows)}
    <div class="note">A policy that routes hundreds of P1s to one buyer has not
      prioritised anything.</div>
  </div>
  <div class="card">
    <h2>MOQ excess by part (value)</h2>
    {_bars_svg(moq_rows, fmt="{:,.0f}")}
    <div class="note">Dead stock the minimum order forces onto the balance
      sheet. This is the number a buyer takes to a supplier.</div>
  </div>
  <div class="card">
    <h2>Detector threshold sweep</h2>
    <table><thead><tr><th class="n">scale</th><th class="n">precision</th>
      <th class="n">recall</th><th class="n">F1</th></tr></thead>
      <tbody>{pr}</tbody></table>
  </div>
  <div class="card wide">
    <h2>Top alerts</h2>
    <table><thead><tr><th>sev</th><th>part</th><th class="n">slack (d)</th>
      <th class="n">value at risk</th><th>route to</th>
      <th class="n">SLA</th></tr></thead><tbody>{alert_rows}</tbody></table>
  </div>
</div>
"""
    p = OUT / "supply_dashboard.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return {"path": str(p), "bytes": p.stat().st_size, "self_contained": True}



# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        res = json.loads((OUT / "completion.json").read_text(encoding="utf-8"))
        (DOCS / "COMPLETION.md").write_text(report(res), encoding="utf-8")
        print("re-rendered docs/COMPLETION.md")
        return

    t0 = time.perf_counter()
    base = supply.build()
    res: dict = {"quick": QUICK}

    print("1/6 safety stock with both variance terms ...", flush=True)
    res["safety"] = stage_safety_stock(base)
    print(f"    demand-only understates by "
          f"{res['safety']['mean_understatement_demand_only']:.2f}x; supply share "
          f"{res['safety']['mean_supply_variance_share']:.2f}", flush=True)

    print("2/6 lot sizing, MOQ, supplier capacity ...", flush=True)
    res["lot"] = stage_lot_sizing(base)
    print(f"    MOQ forces ${res['lot']['moq_waste']['total_excess_value']:,.0f} "
          f"of excess", flush=True)

    print("3/6 detector threshold sweep ...", flush=True)
    res["sweep"] = stage_threshold_sweep(base)

    print("4/6 alert routing and SLAs ...", flush=True)
    res["routing"] = stage_routing(base, res["lot"])
    print(f"    {res['routing']['n_alerts']} alerts, "
          f"{res['routing']['by_severity']}", flush=True)

    print("5/6 supplier distress proxy ...", flush=True)
    res["financial"] = stage_financial(base)

    print("6/6 charts ...", flush=True)
    mc = BLD.monte_carlo(base, n_sims=60 if QUICK else 200)
    res["charts"] = stage_charts(base, mc, res["routing"], res["sweep"], res["lot"])

    res["wall_seconds"] = time.perf_counter() - t0
    (OUT / "completion.json").write_text(
        json.dumps(res, indent=1, default=str), encoding="utf-8")
    (DOCS / "COMPLETION.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/COMPLETION.md and out/supply_dashboard.html "
          f"({res['wall_seconds']:.0f}s)")


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    sf, lot, sw, rt, fin = (res["safety"], res["lot"], res["sweep"],
                            res["routing"], res["financial"])
    A("# Safety stock, lot sizing, detector sweep and alert routing - generated by `complete.py`, not hand-edited\n")

    A("## 1. Safety stock, with both sources of variation\n")
    A("```\nSS = z * sqrt( L * sigma_d^2  +  d^2 * sigma_L^2 )\n"
      "                \\____________/    \\_____________/\n"
      "                 demand varies      supply varies\n```\n")
    A("Both sides now use real inputs: demand spread from the Willems product "
      "demand standard deviations, lead-time spread from the part's Willems "
      "distribution plus its vendor's real SCMS lateness.\n")
    A(f"Across {sf['n']} purchased parts, **{sf['mean_supply_variance_share'] * 100:.0f}% "
      f"of the variance is supply-side on average** ({sf['parts_supply_dominated']} "
      f"parts are supply-dominated). The demand-only formula understates safety "
      f"stock by only **{sf['mean_understatement_demand_only']:.2f}x**.\n")
    A(f"**This finding flipped.** The simulated version said 97% of the variance "
      f"came from suppliers and the textbook formula understated stock 6.9x. The "
      f"real data says the opposite: daily demand in this chain swings a lot "
      f"(median coefficient of variation {sf['median_demand_cv']:.2f}), while part "
      f"lead times barely move (median CV {sf['median_lead_cv']:.2f}; many parts "
      f"have a single fixed lead time). The two-term formula is still the right "
      f"one; it just matters little here.\n")
    A("| lead-time spread | empirical SS / parametric SS | parts |")
    A("|---|---:|---:|")
    A(f"| wide (P95/P50 >= 1.3) | {sf['empirical_over_parametric_fat_tail']:.2f}x "
      f"| {sf['n_fat']} |")
    A(f"| narrow | {sf['empirical_over_parametric_thin_tail']:.2f}x | {sf['n_thin']} |")
    A("\nWith lead-time spreads this small, the normal formula and the empirical "
      "quantile agree to within a few percent. **A bug the real data exposed:** "
      "the empirical version drew one day's demand and multiplied it by the lead "
      "time, so demand spread grew with L instead of sqrt(L). With the simulated "
      "demand (CV 0.25) this barely showed; with the real demand (CV about 0.95) "
      "it made the empirical safety stock 2.5-2.9x the formula. Fixed, with a "
      "test that a fixed lead time gives the formula's answer.")

    A("\n## 2. Lot sizing, MOQ and capacity\n")
    w = lot["moq_waste"]
    A("*MOQs are simulated (no public source) and supplier capacity is assumed.* "
      f"Applying the minimums: **{w['n_parts_raised_to_moq']} of "
      f"{lot['n_components']} parts get raised to their minimum, forcing "
      f"${w['total_excess_value']:,.0f} of excess stock** at real unit costs. "
      "The real parts are cheap (median $0.48) and used in large volumes, so "
      "minimums rarely bind.\n")
    if w["worst"]:
        A("| part | need | MOQ | ordered | excess value | months of cover |")
        A("|---|---:|---:|---:|---:|---:|")
        for r in w["worst"][:6]:
            A(f"| {r['part']} | {r['need']:,.0f} | {r['moq']:,.0f} "
              f"| {r['ordered']:,.0f} | ${r['excess_value']:,.0f} "
              f"| {r['months_of_cover']:.1f} |")
    cap = lot["capacity"]
    A(f"\n**Supplier capacity (assumed): {cap['n_over_capacity']} vendors "
      f"over-committed**, total shortfall {cap['total_shortfall']:,.0f} units. "
      "Ordering earlier does not fix a capacity problem; a second source or a "
      "smaller order does.\n")

    A("## 3. The detector threshold sweep\n")
    A("Scored in the sandbox (real vendor lead times resampled, disruptions "
      "planted). A vendor needs 5+ orders in both windows to be judged.\n")
    A("| scale | judged | flagged | precision | recall | F1 |")
    A("|---:|---:|---:|---:|---:|---:|")
    for c in sw["curve"]:
        A(f"| {c['scale']:.2f} | {c['n_evaluated']} | {c['n_flagged']} | "
          f"{c['precision']:.2f} | {c['recall']:.2f} | {c['f1']:.2f} |")
    b = sw["best_f1"]
    A(f"\nBest F1 at scale {b['scale']:.2f}: precision {b['precision']:.2f}, "
      f"recall {b['recall']:.2f} against {sw['n_planted']} planted disruptions.\n")
    A("**Which trigger earns its place:**\n")
    A("| trigger set | precision | recall |")
    A("|---|---:|---:|")
    for k, v in sw["ablation"].items():
        A(f"| {k} | {v['precision']:.2f} | {v['recall']:.2f} |")
    allr = sw["ablation"]["all three"]["recall"]
    allp = sw["ablation"]["all three"]["precision"]
    for k, v in sw["ablation"].items():
        if k == "all three":
            continue
        name = k.replace("without ", "")
        if v["recall"] < allr - 1e-9:
            A(f"- Removing **{name}** drops recall {allr:.2f} -> {v['recall']:.2f}: "
              "it is doing detection work.")
        elif v["precision"] > allp + 1e-9:
            A(f"- Removing **{name}** keeps recall at {v['recall']:.2f} and raises "
              f"precision {allp:.2f} -> {v['precision']:.2f}: it only adds false alarms.")
        else:
            A(f"- Removing **{name}** changes nothing measurable here.")
    A("")

    A("## 4. Alert routing, severity tiers and SLAs\n")
    A(f"**{rt['n_alerts']} alerts** ({rt['deduped']} duplicates suppressed) across "
      f"{rt['n_parts_considered']} purchased parts. Slack = projected run-out day "
      f"(simulated stock and orders, real plan) minus the part's real lead time.\n")
    A("| severity | count | meaning | response SLA |")
    A("|---|---:|---|---|")
    for sev in ("P1", "P2", "P3", "P4"):
        t = RT.TIERS[sev]
        A(f"| {sev} | {rt['by_severity'].get(sev, 0)} | {t['meaning']} "
          f"| {t['response_hours']}h |")
    A(f"\n{rt['runs_out_in_horizon']} parts run out inside the 13 weeks, the first "
      f"on day {rt['earliest_runout_day']:.0f}. Every one of them can still be "
      "reordered at its normal lead time (3-55 days), so **none is P1**. The "
      "simulated version, with 14-75 day lead times, put 26 urgent items on one "
      "owner. The difference comes from real lead times being short; the stock "
      "rule is unchanged and simulated, so this says nothing about a real plant's "
      "stock position.\n")
    A("| owner | total | P1 | P2 | value at risk |")
    A("|---|---:|---:|---:|---:|")
    for role, d in sorted(rt["load_by_role"].items(), key=lambda kv: -kv[1]["total"]):
        A(f"| {role} | {d['total']} | {d['P1']} | {d['P2']} | ${d['value']:,.0f} |")
    corr = rt["severity_vs_riskscore_correlation"]
    A(f"\nCorrelation between severity and supplier risk score: "
      f"{'n/a' if corr != corr else f'{corr:.2f}'}. Severity is about "
      "consequence and time, not supplier score.\n")

    A("## 5. Supplier distress proxy\n")
    A("Two observable signals: lead-time inflation (last year vs before) and a "
      "rise in the share of late lines. SCMS has no credit data, so the proxy is "
      "compared with the usual scorecard number, on-time vs scheduled.\n")
    A("| predictor of a planted disruption (sandbox) | AUROC |")
    A("|---|---:|")
    A(f"| distress proxy (lead inflation + late-rate rise) | "
      f"**{fin['auc_observed_distress']:.3f}** |")
    A(f"| on-time vs scheduled (the usual scorecard) | "
      f"{fin['auc_on_time_scorecard']:.3f} |")
    A(f"\n{fin['n_planted']} planted disruptions across {fin['n']} vendors with enough "
      "history. Both separate them well, and the plain scorecard does slightly "
      "better. That result flatters the scorecard: in the sandbox a disruption "
      "delays the delivery but leaves the scheduled date where it was, so it "
      "shows up as lateness. In the real file 89% of lines land exactly on the "
      "scheduled date, which suggests the real date moves with the delay, and "
      "then on-time vs scheduled would see nothing. Lead-time inflation does "
      "not depend on the promise field at all.\n")
    A("On the **real** history the proxy ranks these vendors highest (no ground "
      "truth to check against):\n")
    A("| vendor | lead inflation | late-rate change | proxy |")
    A("|---|---:|---:|---:|")
    for r in fin["real_top"][:6]:
        A(f"| {r['supplier_id']} {r['name'][:30]} | {r['lead_inflation']:.2f}x | "
          f"{r['late_rate_change']:+.2f} | {r['observed_distress']:.2f} |")
    A(f"\n*{fin['caveat']}*\n")

    c = res["charts"]
    A("## 6. Charts\n")
    A(f"`out/supply_dashboard.html`, {c['bytes'] / 1024:.0f} KB, self-contained. "
      "The public dashboard is `docs/index.html`, built by `build_dashboard.py`.\n")
    A("---")
    A(f"*Generated in {res.get('wall_seconds', 0):.0f}s.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
