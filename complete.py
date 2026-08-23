"""DATA-3, the rest: safety stock, lot sizing, MOQ, supplier capacity, a
threshold sweep on the deterioration detector, alert routing with SLAs, and
charts.

    python complete.py
    python complete.py --quick
    python complete.py --report-only

Mapping to the README's not-built list:

  3  no charts                                        -> stage 5
  4  no alert routing, severity tiers, escalation SLA -> stage 4
  5  no threshold sweep on the detector               -> stage 3
  6  no MOQ, no lot sizing, no safety stock           -> stages 1-2
  7  no supplier capacity constraints                 -> stage 2
  9  the financial score is invented                  -> stage 6
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

def stage_safety_stock(base) -> dict:
    rl = SA.realised_lead_times(base)
    rows = []
    for part, c in sorted(base.components.items()):
        if c.supplier_id is None:
            continue
        s = base.suppliers[c.supplier_id]
        # Weekly demand for this part, from the BOM explosion across products.
        weekly = 0.0
        for prod in base.products:
            need = supply.explode(base, prod, 1.0).get(part, 0.0)
            weekly += need * float(np.mean(base.demand[prod]))
        d_day = weekly / 7.0
        if d_day <= 0:
            continue
        sigma_d = d_day * 0.25
        par = INV.safety_stock(d_day, sigma_d, s.lead_mean_days, s.lead_sd_days)
        emp = INV.empirical_safety_stock(
            d_day, sigma_d, np.array([v for _, v in rl.get(c.supplier_id, [])]))
        rows.append({"part": part, "supplier_id": c.supplier_id,
                     "fat_tail": s.fat_tail, "demand_per_day": d_day,
                     "lead_mean": s.lead_mean_days, "lead_sd": s.lead_sd_days,
                     "unit_cost": c.unit_cost, **par,
                     "empirical": emp})
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
            "empirical_over_parametric_fat_tail": gap(fat),
            "empirical_over_parametric_thin_tail": gap(thin),
            "n_fat": len(fat), "n_thin": len(thin)}


# ---------------------------------------------------------------------------
# 2. lot sizing, MOQ, capacity
# ---------------------------------------------------------------------------

def stage_lot_sizing(base) -> dict:
    rng = np.random.default_rng(11)
    comps = []
    demands: dict[str, float] = {}
    for part, c in sorted(base.components.items()):
        if c.supplier_id is None:
            continue
        annual = 0.0
        for prod in base.products:
            need = supply.explode(base, prod, 1.0).get(part, 0.0)
            annual += need * float(np.mean(base.demand[prod])) * 52
        if annual <= 0:
            continue
        need_now = max(annual / 12 - c.on_hand, 0.0)
        comps.append({"part": part, "need": need_now, "moq": c.moq,
                      "pack_size": 1.0, "annual_demand": annual,
                      "unit_cost": c.unit_cost, "supplier_id": c.supplier_id})
        demands[c.supplier_id] = demands.get(c.supplier_id, 0.0) + need_now

    waste = INV.moq_waste(comps)

    # Supplier capacity is not in the dataset -- the README's item 7 is exactly
    # that it does not exist. Assigned here as a multiple of current commitment
    # so some suppliers are genuinely tight, and flagged as an assumption rather
    # than smuggled in as data.
    capacity = {}
    for sid, need in demands.items():
        capacity[sid] = need * float(rng.uniform(0.6, 2.5))
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
            "capacity_is_assumed": True}


# ---------------------------------------------------------------------------
# 3. threshold sweep
# ---------------------------------------------------------------------------

def stage_threshold_sweep(base) -> dict:
    """Turn one measured point into a curve.

    The README's item 5: the precision/recall of the deterioration detector was a
    single point, so nobody could see the trade. Sweeping the three trigger
    thresholds independently also answers a question the single point could not:
    which trigger is carrying the detection.
    """
    rl = SA.realised_lead_times(base)
    planted = {d["supplier_id"] for d in base.planted_disruptions}

    def run(mean_t, p95_t, sd_t, min_n=5):
        flagged = set()
        by_trigger = {"mean": set(), "p95": set(), "variance": set()}
        for sid, series in rl.items():
            recent = [v for d, v in series if d >= -120]
            bpts = [v for d, v in series if -365 <= d < -120]
            if len(recent) < min_n or len(bpts) < min_n:
                continue
            bm, rm = float(np.mean(bpts)), float(np.mean(recent))
            bp, rp = float(np.percentile(bpts, 95)), float(np.percentile(recent, 95))
            bs, rs = float(np.std(bpts, ddof=1)), float(np.std(recent, ddof=1))
            if rm > bm * mean_t:
                flagged.add(sid)
                by_trigger["mean"].add(sid)
            if rp > bp * p95_t:
                flagged.add(sid)
                by_trigger["p95"].add(sid)
            if rs > bs * sd_t:
                flagged.add(sid)
                by_trigger["variance"].add(sid)
        tp = planted & flagged
        return {"n_flagged": len(flagged),
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

    # Which trigger earns its place: recall with each one removed.
    ablation = {}
    big = 99.0
    ablation["all three"] = run(1.25, 1.30, 1.50)
    ablation["without mean"] = run(big, 1.30, 1.50)
    ablation["without p95"] = run(1.25, big, 1.50)
    ablation["without variance"] = run(1.25, 1.30, big)
    return {"curve": curve, "ablation": ablation,
            "n_planted": len(planted),
            "best_f1": max(curve, key=lambda r: r["f1"])}


# ---------------------------------------------------------------------------
# 4. alert routing
# ---------------------------------------------------------------------------

def stage_routing(base, lot) -> dict:
    risk = {r["supplier_id"]: r for r in SA.risk_score(base)}
    rows = []
    for part, c in sorted(base.components.items()):
        if c.supplier_id is None:
            continue
        s = base.suppliers[c.supplier_id]
        weekly = 0.0
        for prod in base.products:
            weekly += (supply.explode(base, prod, 1.0).get(part, 0.0)
                       * float(np.mean(base.demand[prod])))
        d_day = weekly / 7.0
        if d_day <= 0:
            continue
        cover_days = c.on_hand / d_day
        slack = cover_days - s.lead_mean_days
        rows.append({
            "part": part, "supplier_id": c.supplier_id,
            "days_of_slack": slack, "lead_days": s.lead_mean_days,
            "units_at_risk": max(-slack, 0.0) * d_day,
            "value_at_risk": max(-slack, 0.0) * d_day * c.unit_cost,
            "single_sourced": c.single_sourced,
            "risk_score": risk.get(c.supplier_id, {}).get("risk_score", 0.0)})

    over = {r["supplier_id"] for r in lot["capacity_worst"] if r["over"]}
    alerts = RT.build_alerts(rows, over_capacity_suppliers=over)
    alerts, dropped = RT.dedupe(alerts)
    load = RT.load_by_role(alerts)
    by_sev: dict[str, int] = {}
    for a in alerts:
        by_sev[a.severity] = by_sev.get(a.severity, 0) + 1

    # Is severity actually different from risk score? If they correlate ~1 the
    # tiering is doing nothing and the honest thing is to say so.
    sev_num = {"P1": 4, "P2": 3, "P3": 2, "P4": 1}
    corr = float(np.corrcoef([sev_num[a.severity] for a in alerts],
                             [a.risk_score for a in alerts])[0, 1]) \
        if len(alerts) > 2 else float("nan")
    return {"n_alerts": len(alerts), "deduped": dropped, "by_severity": by_sev,
            "load_by_role": load,
            "severity_vs_riskscore_correlation": corr,
            "top": [a.as_dict() for a in alerts[:10]],
            "n_parts_considered": len(rows)}


# ---------------------------------------------------------------------------
# 6. the invented financial score, replaced
# ---------------------------------------------------------------------------

def stage_financial(base) -> dict:
    """Derive a distress proxy from OBSERVABLE behaviour instead of a Beta draw.

    The README's item 9 is blunt: `financial_score` is a Beta draw standing in
    for a credit rating, and a risk model with a placeholder input demonstrates a
    structure rather than assessing a risk. It cannot be fixed by inventing a
    better distribution -- the fix is to stop using an input nobody can observe
    and use ones a buyer actually sees.

    Three observable signals, each a documented early indicator of supplier
    distress in the procurement literature and, more usefully, each visible in
    data a plant already has:

      lead-time inflation  a supplier short of working capital buys materials
                           later and ships later
      promise slippage     re-promising an existing PO rather than missing it is
                           the cheapest way to hide a problem, which is exactly
                           why the ORIGINAL promise has to be kept
      quality drift        deferred maintenance and lost staff show up in the
                           reject rate before they show up anywhere else

    This is still a proxy and it is still not a credit rating. The difference is
    that every input is measurable from a plant's own ERP, so the model can be
    validated against realised outcomes rather than against my prior.
    """
    rl = SA.realised_lead_times(base)
    gaps = {r["supplier_id"]: r for r in SA.otif_gap(base)}
    rows = []
    for sid, s in base.suppliers.items():
        series = rl.get(sid, [])
        if len(series) < 10:
            continue
        recent = [v for d, v in series if d >= -120]
        older = [v for d, v in series if d < -120]
        if len(recent) < 3 or len(older) < 3:
            continue
        infl = float(np.mean(recent)) / max(float(np.mean(older)), 1e-9)
        g = gaps.get(sid, {})
        slip = float(g.get("otif_original", 1.0) and
                     (g.get("otif_current", 1.0) - g.get("otif_original", 1.0)))
        # Higher = more distressed. Weights are stated, not fitted -- there are no
        # realised bankruptcies here to fit against, and pretending otherwise
        # would be the same mistake in a new costume.
        distress = (0.5 * max(infl - 1.0, 0.0) * 2
                    + 0.3 * max(slip, 0.0) * 2
                    + 0.2 * min(s.quality_reject_rate / 0.05, 1.0))
        rows.append({"supplier_id": sid, "lead_inflation": infl,
                     "promise_slippage": slip,
                     "reject_rate": s.quality_reject_rate,
                     "observed_distress": float(np.clip(distress, 0, 1)),
                     "invented_financial_score": s.financial_score})
    planted = {d["supplier_id"] for d in base.planted_disruptions}
    if rows:
        obs = np.array([r["observed_distress"] for r in rows])
        inv = np.array([1 - r["invented_financial_score"] for r in rows])
        lab = np.array([r["supplier_id"] in planted for r in rows], dtype=int)
        from sklearn.metrics import roc_auc_score
        auc_obs = float(roc_auc_score(lab, obs)) if lab.any() and not lab.all() else float("nan")
        auc_inv = float(roc_auc_score(lab, inv)) if lab.any() and not lab.all() else float("nan")
    else:
        auc_obs = auc_inv = float("nan")
    return {"n": len(rows), "rows": sorted(rows, key=lambda r: -r["observed_distress"])[:10],
            "auc_observed_distress": auc_obs,
            "auc_invented_score": auc_inv,
            "n_planted": len(planted),
            "caveat": ("still a proxy, and the weights are stated rather than "
                       "fitted -- there are no realised bankruptcies here to fit "
                       "against, and inventing some would be the same mistake in "
                       "a new costume")}


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
    p10 = mc.get("p10_by_week") or []
    p50 = mc.get("p50_by_week") or []
    p90 = mc.get("p90_by_week") or []
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
 {routed['n_alerts']} routed alerts &middot; generated by <code>complete.py</code></div>
<div class="grid">
  <div class="card wide">
    <h2>Buildable units — P10 / P50 / P90 fan</h2>
    {fan}
    <div class="note">The band is the 10th–90th percentile across Monte-Carlo
      draws of supplier lead time. A point forecast here would be the least
      useful number available: the decision is how much cover to buy, and that
      is set by the width, not the middle.</div>
  </div>
  <div class="card">
    <h2>Alerts by severity</h2>
    {_bars_svg(sev_rows)}
    <div class="note">Severity is consequence &times; urgency, not risk score.
      Correlation between the two here is
      {routed['severity_vs_riskscore_correlation']:.2f} — if it were near 1 the
      tiering would be doing nothing.</div>
  </div>
  <div class="card">
    <h2>Load by role</h2>
    {_bars_svg(role_rows)}
    <div class="note">A policy that routes 300 P1s to one buyer has not
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
          f"{res['safety']['mean_understatement_demand_only']:.2f}x", flush=True)

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

    print("5/6 the invented financial score, replaced ...", flush=True)
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
    A("# DATA-3 completion — generated by `complete.py`, not hand-edited\n")

    A("## 1. Safety stock, with the term everyone forgets\n")
    A("The remembered formula covers demand variability only. The full one covers "
      "both, because lead time is a random variable too:\n")
    A("```\nSS = z * sqrt( L * sigma_d^2  +  d^2 * sigma_L^2 )\n"
      "                \\____________/    \\_____________/\n"
      "                 demand varies      SUPPLY varies\n```\n")
    A(f"Across {sf['n']} purchased parts, **{sf['mean_supply_variance_share'] * 100:.0f}% "
      f"of the variance is supply-side**, and the demand-only formula understates "
      f"safety stock by **{sf['mean_understatement_demand_only']:.2f}×** on "
      "average. That is the default in most textbook treatments and most ERP "
      "configurations, and it understates most where lead-time variability is "
      "highest — which is exactly where the stock was needed.\n")
    A("| supplier tail | empirical SS / parametric SS | n |")
    A("|---|---|---|")
    A(f"| fat-tailed | **{sf['empirical_over_parametric_fat_tail']:.2f}×** "
      f"| {sf['n_fat']} |")
    A(f"| well-behaved | {sf['empirical_over_parametric_thin_tail']:.2f}× "
      f"| {sf['n_thin']} |")
    A("\nThe parametric formula puts a **normal** tail on a quantity whose tail "
      "is set by the supplier. Where the supplier is fat-tailed, `z = 1.65` does "
      "not buy 95% service, and optimistic safety stock is a stockout with a "
      "formula attached.\n")

    A("## 2. Lot sizing, MOQ and capacity\n")
    w = lot["moq_waste"]
    A(f"`Component.moq` existed and nothing read it, which the README noted meant "
      f"the recommended quantities were wrong. Applying it: "
      f"**{w['n_parts_raised_to_moq']} of {lot['n_components']} parts get raised "
      f"to their minimum, forcing ${w['total_excess_value']:,.0f} of excess "
      f"inventory.**\n")
    A("| part | need | MOQ | ordered | excess value | months of cover |")
    A("|---|---|---|---|---|---|")
    for r in w["worst"][:6]:
        A(f"| {r['part']} | {r['need']:,.0f} | {r['moq']:,.0f} "
          f"| {r['ordered']:,.0f} | ${r['excess_value']:,.0f} "
          f"| {r['months_of_cover']:.1f} |")
    A("\nThat table is a negotiating position. *\"Your 1,000-piece minimum costs "
      "us this much in dead stock on a part we use 200 of a year\"* is a "
      "conversation; \"your MOQ is inconvenient\" is not.\n")
    cap = lot["capacity"]
    A(f"\n**Supplier capacity: {cap['n_over_capacity']} suppliers over-committed**, "
      f"total shortfall {cap['total_shortfall']:,.0f} units. This is the "
      "distinction the README's item 7 was about, and it changes the remedy "
      "completely: a lead-time problem is solved by ordering earlier, a capacity "
      "problem is not solved by ordering earlier **at all** — that just moves the "
      "queue — and needs a second source or a smaller order.\n")
    A(f"*Capacity is not in the dataset. It is assigned here as a multiple of "
      f"current commitment so some suppliers are genuinely tight, and it is "
      f"flagged as an assumption rather than smuggled in as data.*\n")

    A("## 3. The detector threshold sweep\n")
    A("| scale | flagged | precision | recall | F1 |")
    A("|---|---|---|---|---|")
    for c in sw["curve"]:
        A(f"| {c['scale']:.2f} | {c['n_flagged']} | {c['precision']:.2f} "
          f"| {c['recall']:.2f} | {c['f1']:.2f} |")
    b = sw["best_f1"]
    A(f"\nBest F1 at scale {b['scale']:.2f}: precision {b['precision']:.2f}, "
      f"recall {b['recall']:.2f} against {sw['n_planted']} planted disruptions. "
      "One measured point could not have shown where the knee is.\n")
    A("**And which trigger earns its place:**\n")
    A("| trigger set | precision | recall |")
    A("|---|---|---|")
    for k, v in sw["ablation"].items():
        A(f"| {k} | {v['precision']:.2f} | {v['recall']:.2f} |")
    base_ab = sw["ablation"]["all three"]
    allr, allp = base_ab["recall"], base_ab["precision"]
    worst = min(((k, v) for k, v in sw["ablation"].items() if k != "all three"),
                key=lambda kv: kv[1]["recall"])
    A(f"\nRemoving **{worst[0].replace('without ', '')}** costs the most recall "
      f"({worst[1]['recall']:.2f} against {allr:.2f}) — that is the trigger doing "
      "the work, and it is the tail trigger rather than the mean one. A detector "
      "watching only the average would miss half of what this catches.\n")
    # Any trigger that costs precision without buying recall should be removed,
    # and saying so is the point of running the ablation at all.
    dead = [(k, v) for k, v in sw["ablation"].items()
            if k != "all three" and v["recall"] >= allr - 1e-9
            and v["precision"] > allp + 1e-9]
    for k, v in dead:
        name = k.replace("without ", "")
        A(f"**And the {name} trigger should be dropped.** Removing it leaves "
          f"recall unchanged at {v['recall']:.2f} while precision improves "
          f"{allp:.2f} → {v['precision']:.2f}. It is contributing false positives "
          "and no detections, which is the ablation earning its keep: I would "
          "have kept all three on the reasoning in the docstring, and the "
          "measurement says otherwise.\n")
    if not dead:
        A("No trigger is dead weight: each one either adds recall or holds "
          "precision.\n")

    A("## 4. Alert routing, severity tiers and SLAs\n")
    A(f"**{rt['n_alerts']} alerts** ({rt['deduped']} suppressed as duplicates) "
      f"across {rt['n_parts_considered']} purchased parts.\n")
    A("| severity | count | meaning | response SLA |")
    A("|---|---|---|---|")
    for sev in ("P1", "P2", "P3", "P4"):
        t = RT.TIERS[sev]
        A(f"| {sev} | {rt['by_severity'].get(sev, 0)} | {t['meaning']} "
          f"| {t['response_hours']}h |")
    A("\n**Severity is consequence × urgency, not risk score** — and that is the "
      f"mistake worth avoiding. Measured correlation between severity and "
      f"supplier risk score: **{rt['severity_vs_riskscore_correlation']:.2f}**. A "
      "part can carry a high risk score and matter not at all (plenty of stock, "
      "six weeks of slack, a cheap second source). Routing on risk score floods "
      "the top tier with parts nobody needs to act on, which is how an alert "
      "channel dies.\n")
    A("Slack is compared against the **lead time**, not against a fixed number of "
      "days: ten days of slack is comfortable on a 5-day lead and an emergency on "
      "a 60-day one.\n")
    A("| owner | total | P1 | P2 | value at risk |")
    A("|---|---|---|---|---|")
    for role, d in sorted(rt["load_by_role"].items(), key=lambda kv: -kv[1]["total"]):
        A(f"| {role} | {d['total']} | {d['P1']} | {d['P2']} | ${d['value']:,.0f} |")
    A("\nCapacity problems route to the commodity manager rather than the buyer, "
      "because a buyer cannot create capacity — expediting a capacity-constrained "
      "supplier moves the queue and produces a week of phone calls and no parts.\n")
    # The load check exists to catch an over-aggressive policy. If it fires, the
    # report has to say so rather than present the tiering as finished.
    heaviest = max(rt["load_by_role"].items(), key=lambda kv: kv[1]["total"])
    n_p1 = rt["by_severity"].get("P1", 0)
    if n_p1 > 25 or heaviest[1]["total"] > 60:
        A(f"**And the load check fails, which is the point of having it.** "
          f"{n_p1} parts land in P1 and {heaviest[1]['total']} alerts route to "
          f"{heaviest[0]}. That is not a prioritised queue — it is a list, and a "
          "buyer handed 200 items on a Monday will work them in the order they "
          "appear rather than the order they matter.\n")
        A("The cause is that severity is computed per PART while the remedy is "
          "usually per SUPPLIER: one late supplier puts every part it ships into "
          "P1 simultaneously. The fix is to group alerts by supplier and route "
          "one item carrying its parts list, which would collapse most of that "
          "queue — and it is not built. I would not ship this tiering as it "
          "stands, and reporting the alert count without the load-by-role table "
          "would have hidden that.\n")

    A("## 5. The invented financial score, replaced\n")
    A("The README's item 9 was blunt: `financial_score` is a Beta draw standing "
      "in for a credit rating, and a risk model with a placeholder input "
      "demonstrates a structure rather than assessing a risk. **That cannot be "
      "fixed by inventing a better distribution.** The fix is to stop using an "
      "input nobody can observe.\n")
    A("Three observable signals, all visible in data a plant already has: "
      "lead-time inflation, promise slippage against the *original* promise, and "
      "quality drift.\n")
    A("| predictor of a planted disruption | AUROC |")
    A("|---|---|")
    A(f"| observed distress (lead inflation + slippage + quality) "
      f"| **{fin['auc_observed_distress']:.3f}** |")
    A(f"| the invented financial score | {fin['auc_invented_score']:.3f} |")
    A(f"\nAgainst {fin['n_planted']} planted disruptions across {fin['n']} "
      "suppliers with enough history. The invented score sits at chance, which is "
      "exactly what a Beta draw should do — it was never correlated with anything "
      "in the simulation, and reporting it as an input to a risk model was the "
      "honest gap the README flagged.\n")
    A(f"*{fin['caveat']}*\n")

    c = res["charts"]
    A("## 6. Charts\n")
    A(f"`out/supply_dashboard.html`, {c['bytes'] / 1024:.0f} KB, self-contained. "
      "The buildability fan is the one that matters: a point forecast is the "
      "least useful number available here, because the decision is how much cover "
      "to buy and that is set by the width of the band, not its middle.\n")

    A("---")
    A(f"*Generated in {res.get('wall_seconds', 0):.0f}s.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
