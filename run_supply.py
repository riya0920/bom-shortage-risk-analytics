"""End-to-end: BOM -> buildability -> shortage drivers -> order-by dates,
plus supplier analytics on the real SCMS history.

    python run_supply.py
    python run_supply.py --quick
    python run_supply.py --report-only
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import buildability as B  # noqa: E402
import supplier_analytics as SA  # noqa: E402
import supply  # noqa: E402

OUT = ROOT / "out"


def base_summary(base) -> dict:
    parts = [c for c in base.components.values() if c.level == 2]
    req = supply.unit_requirements(base)
    n_prod_using = {p.part: sum(1 for pr in base.products if p.part in req[pr])
                    for p in parts}
    leads = np.array([c.lead_mean for c in parts])
    return {
        "source": {
            "bom": "Willems (2008) chain 24, power-driven hand tools (real)",
            "suppliers": "USAID SCMS delivery history, Direct Drop vendors (real)",
            "vendor_to_part_mapping": "lead-time band matching (assumption)",
            "stock_and_open_orders": "simulated",
            "original_promise_dates": "simulated",
            "planted_disruptions": "simulated, in a sandbox built from real vendor lead times",
        },
        "products": base.products,
        "n_products": len(base.products),
        "purchased_parts": len(parts),
        "sub_assemblies": sum(1 for c in base.components.values() if c.level == 1),
        "bom_lines": len(base.bom),
        "suppliers": len(base.suppliers),
        "single_sourced_parts": sum(1 for c in parts if c.single_sourced),
        "parts_shared_by_2plus_products": sum(1 for v in n_prod_using.values() if v > 1),
        "parts_in_every_product": sum(1 for v in n_prod_using.values()
                                      if v == len(base.products)),
        "parts_with_lead_distribution": sum(1 for c in parts if c.has_distribution),
        "part_lead_days": {"min": float(leads.min()), "median": float(np.median(leads)),
                           "max": float(leads.max())},
        "weekly_demand_total": float(sum(base.demand[p][0] for p in base.products)),
        "open_pos": len(base.pos),
        "real_delivery_lines": len(base.history),
        "sandbox_delivery_lines": len(base.sandbox),
        "horizon_weeks": supply.WEEKS,
        "planted_disruptions": base.planted_disruptions,
        "scms_cleaning": base.meta.get("scms_cleaning"),
        "day0": base.meta.get("day0"),
        "suppliers_table": [
            {"supplier_id": s.supplier_id, "name": s.name, "n_lines": s.n_lines,
             "lead_p50": s.lead_p50_days, "lead_p95": s.lead_p95_days,
             "on_time_vs_scheduled": s.on_time_vs_scheduled,
             "sole_source_share": s.sole_source_share,
             "lines_per_year": s.lines_per_year,
             "parts_mapped": sum(1 for c in parts if c.supplier_id == s.supplier_id),
             "part_lead_band": [
                 min((c.lead_mean for c in parts if c.supplier_id == s.supplier_id), default=0),
                 max((c.lead_mean for c in parts if c.supplier_id == s.supplier_id), default=0)]}
            for s in base.suppliers.values()],
    }


def main() -> None:
    quick = "--quick" in sys.argv
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        prev = json.loads((OUT / "results.json").read_text())
        (ROOT / "docs").mkdir(exist_ok=True)
        (ROOT / "docs" / "RESULTS.md").write_text(report(prev), encoding="utf-8")
        print("re-rendered docs/RESULTS.md")
        return

    t0 = time.perf_counter()
    print("1/5 loading real data and building the supply base ...", flush=True)
    base = supply.build()
    res: dict = {"base": base_summary(base)}
    b = res["base"]
    print(f"    {b['n_products']} products, {b['sub_assemblies']} sub-assemblies, "
          f"{b['purchased_parts']} purchased parts, {b['bom_lines']} BOM lines, "
          f"{b['suppliers']} suppliers", flush=True)

    print("2/5 supplier analytics ...", flush=True)
    flags = SA.detect_deterioration(base)
    res["deterioration"] = {"flags": [f for f in flags if f["flagged"]],
                            "score": SA.score_detection(base, flags)}
    short = SA.detect_deterioration(base, recent_days=120, baseline_days=365)
    res["deterioration"]["window_120d"] = {
        "evaluable": len(short), "score": SA.score_detection(base, short)}
    res["real_scan"] = SA.scan_real_history(base)
    res["otif_gap"] = SA.otif_gap(base)
    res["scheduled_evidence"] = SA.scheduled_date_evidence(base)
    res["risk"] = {"weights": SA.RISK_WEIGHTS, "top": SA.risk_score(base)[:10]}
    sc = res["deterioration"]["score"]
    print(f"    deterioration (sandbox): precision {sc['precision']:.2f}, recall "
          f"{sc['recall']:.2f}; real history: {res['real_scan']['vendor_years_flagged']}"
          f" of {res['real_scan']['vendor_years_evaluated']} vendor-years flagged",
          flush=True)

    print("3/5 deterministic buildability + shortage attribution ...", flush=True)
    det = B.buildable(base)
    res["deterministic"] = {
        "built": {p: v.tolist() for p, v in det["built"].items()},
        "demand": {p: base.demand[p].tolist() for p in base.products},
        "gating_events": len(det["gating"]),
    }

    print("4/5 Monte Carlo buildability ...", flush=True)
    mc = B.monte_carlo(base, n_sims=80 if quick else 300)
    res["monte_carlo"] = {k: v for k, v in mc.items() if not k.startswith("_")}

    print("5/5 shortage drivers with order-by dates ...", flush=True)
    res["drivers"] = B.shortage_drivers(base, mc)
    res["allocation"] = B.compare_allocation_policies(base, n_sims=40 if quick else 120)

    sims = mc["_sims"]
    cal = {}
    for p in base.products:
        s = sims[p]
        p50 = np.percentile(s, 50, axis=0)
        p10 = np.percentile(s, 10, axis=0)
        p90 = np.percentile(s, 90, axis=0)
        cal[p] = {
            "pct_below_p50": float(100 * np.mean(s < p50[None, :])),
            "pct_inside_p10_p90": float(100 * np.mean((s >= p10[None, :]) & (s <= p90[None, :]))),
        }
    res["calibration"] = cal
    res["wall_seconds"] = time.perf_counter() - t0

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/RESULTS.md and out/results.json ({res['wall_seconds']:.0f}s)")


def _totals(res):
    fan = res["monte_carlo"]["fan"]
    dem = sum(sum(f["demand"]) for f in fan.values())
    return {k: sum(sum(f[k]) for f in fan.values()) for k in ("p10", "p50", "p90")}, dem


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    b = res["base"]
    A("# Core results - generated by `run_supply.py`, not hand-edited\n")
    A("## 0. The data, and what is real\n")
    A("| layer | source |")
    A("|---|---|")
    for k, v in b["source"].items():
        A(f"| {k.replace('_', ' ')} | {v} |")
    A(f"\n**BOM (real):** {b['n_products']} finished products, {b['sub_assemblies']} "
      f"in-house sub-assemblies, **{b['purchased_parts']} purchased parts**, "
      f"{b['bom_lines']} BOM lines. {b['parts_shared_by_2plus_products']} parts feed "
      f"two or more products and {b['parts_in_every_product']} feed all "
      f"{b['n_products']}. {b['parts_with_lead_distribution']} parts have a discrete "
      "lead-time distribution in the data; the rest have one fixed lead time and are "
      f"kept fixed. Part lead times run {b['part_lead_days']['min']:.0f}-"
      f"{b['part_lead_days']['max']:.0f} days (median "
      f"{b['part_lead_days']['median']:.1f}). Quantity per parent is 1 on every line "
      "because the data has no quantities.\n")
    cl = b["scms_cleaning"]
    A(f"**Suppliers (real):** SCMS has {cl['rows']:,} lines. {cl['direct_drop']:,} are "
      f"Direct Drop from a vendor (the rest ship from SCMS's own warehouse and have "
      f"no supplier lead time). Dropped: {cl['dropped_missing_date']} with a missing "
      f"date, {cl['dropped_lead_not_positive']} with a lead time of zero or less. "
      f"Kept {cl['kept']:,} lines from {cl['vendors']} vendors; the "
      f"**{b['suppliers']} vendors with at least 10 lines** are the supplier base "
      f"({b['real_delivery_lines']:,} lines, day 0 = {b['day0']}).\n")
    A("**Vendor-to-part mapping (assumption):** the datasets are unrelated, so "
      "each part is given the vendor at the same position in the lead-time "
      "ranking (fast parts to fast vendors). Every vendor gets 7-8 parts.\n")
    A("| vendor | real SCMS name | lines | median lead (d) | P95 lead (d) | "
      "on time vs scheduled | parts mapped | part lead band (d) |")
    A("|---|---|---:|---:|---:|---:|---:|---|")
    for s in b["suppliers_table"]:
        A(f"| {s['supplier_id']} | {s['name'][:40]} | {s['n_lines']} | "
          f"{s['lead_p50']:.0f} | {s['lead_p95']:.0f} | "
          f"{100 * s['on_time_vs_scheduled']:.0f}% | {s['parts_mapped']} | "
          f"{s['part_lead_band'][0]:.1f}-{s['part_lead_band'][1]:.1f} |")
    A("\n**Simulated:** on-hand stock, open orders and MOQs (no public source), "
      "the original promise dates, and the planted disruptions. Why 13 weeks: it "
      "covers every part lead time in this chain plus a planning cycle.")

    tot, dem = _totals(res)
    A("\n## 1. Buildability\n")
    A(f"Median case: **{100 * tot['p50'] / dem:.0f}%** of the {dem:,.0f} planned "
      f"units are buildable (P10 {100 * tot['p10'] / dem:.0f}%, P90 "
      f"{100 * tot['p90'] / dem:.0f}%). Deterministic plan: "
      f"{res['deterministic']['gating_events']} (product, week) pairs are short.\n")
    A("| product | 13-week demand | P10 | P50 | P90 | fill at P50 | first short week (P50) |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for p in b["products"]:
        f = res["monte_carlo"]["fan"][p]
        d = sum(f["demand"])
        first = next((i + 1 for i, (x, y) in enumerate(zip(f["p50"], f["demand"]))
                      if x < y), None)
        A(f"| {p} | {d:,.0f} | {sum(f['p10']):,.0f} | {sum(f['p50']):,.0f} | "
          f"{sum(f['p90']):,.0f} | {100 * sum(f['p50']) / d:.0f}% | "
          f"{('W' + str(first)) if first else '-'} |")
    cal = res["calibration"]
    A("\nSelf-consistency (percentiles checked against the simulations they came "
      "from, so this checks arithmetic, not the model): mean % inside P10-P90 = "
      f"{np.mean([c['pct_inside_p10_p90'] for c in cal.values()]):.0f}%. The real "
      "calibration test is in EXTENSIONS.md.")

    A("\n## 2. Shortage drivers, with order-by dates\n")
    A("| part | vendor | single source | times gating | products using it | first short week | lead mean | lead P95 | order by (day) | too late? |")
    A("|---|---|---|---:|---:|---|---:|---:|---:|---|")
    for d in res["drivers"]:
        A(f"| {d['part']} | {d['supplier']} | {'yes' if d['single_sourced'] else 'no'} | "
          f"{d['times_gating_in_sim']} | {d['n_products_using']} | "
          f"{('W' + str(d['first_impact_week'])) if d['first_impact_week'] else '-'} | "
          f"{d['lead_mean_days']:.1f} d | {d['lead_p95_days']:.0f} d | "
          f"{d['order_by_day']:+.0f} | {'**yes**' if d['already_too_late'] else 'no'} |")
    late = sum(1 for d in res["drivers"] if d["already_too_late"])
    A(f"\n{late} of {len(res['drivers'])} top drivers are past their order-by date. "
      "The order-by date is the first short week minus the part's P95 lead time "
      "from its real distribution. With real lead times of days to weeks (not "
      "months), most drivers still have runway: the problem is the quantity "
      "on order, not the timing.")

    A("\n## 3. Supplier deterioration\n")
    sc = res["deterioration"]["score"]
    w120 = res["deterioration"]["window_120d"]
    A("Scored in the sandbox: order histories resampled from each real vendor's "
      "own lead times, with 6 disruptions planted (simulated). Recent window = last "
      "365 days, baseline = the 730 days before.\n")
    A(f"**Precision {sc['precision']:.2f}, recall {sc['recall']:.2f}.** "
      f"{sc['evaluated']} of {b['suppliers']} vendors have enough orders (5+ in each "
      f"window) to be judged at all. With the 120-day window the simulated version "
      f"used, only **{w120['evaluable']}** could be judged and recall was "
      f"{w120['score']['recall']:.2f}: real vendors order too rarely for it.\n")
    A("| disruption type | planted | caught | caught by P95 alone | too few orders to judge |")
    A("|---|---:|---:|---:|---:|")
    for kind, k in sc["by_kind"].items():
        A(f"| {kind} | {k['planted']} | {k['caught']} | {k['by_p95_only']} | "
          f"{k['not_evaluable']} |")
    A("\n| vendor | mean ratio | P95 ratio | sd ratio | triggers | planted? |")
    A("|---|---:|---:|---:|---|---|")
    planted = set(sc["planted"])
    for f in res["deterioration"]["flags"][:12]:
        A(f"| {f['supplier_id']} | {f['mean_ratio']:.2f}x | {f['p95_ratio']:.2f}x | "
          f"{f['sd_ratio']:.2f}x | {', '.join(f['triggers'])} | "
          f"{'yes' if f['supplier_id'] in planted else 'no'} |")
    A("\nThe false alarms come from real vendors whose lead times are very spread "
      "out (air and ocean shipments, many products): a year of their orders can "
      "move the P95 by 30% with nothing wrong.\n")
    rs = res["real_scan"]
    A("### On the real SCMS history (no ground truth)\n")
    A(f"Each vendor-year compared with the two years before it: "
      f"**{rs['vendor_years_flagged']} of {rs['vendor_years_evaluated']} vendor-years "
      f"flagged ({100 * rs['flag_rate']:.0f}%)**, across "
      f"{rs['vendors_flagged_at_least_once']} vendors. Trigger counts: "
      + ", ".join(f"{k} {v}" for k, v in rs["by_trigger"].items()) + ".\n")
    A("| vendor | year | mean ratio | P95 ratio | sd ratio | triggers |")
    A("|---|---:|---:|---:|---:|---|")
    for r in rs["top"][:10]:
        A(f"| {r['supplier_id']} {r['name'][:28]} | {r['year']} | {r['mean_ratio']:.2f}x | "
          f"{r['p95_ratio']:.2f}x | {r['sd_ratio']:.2f}x | {', '.join(r['triggers'])} |")
    A("\nWe cannot say which of these were real problems: SCMS has no record of "
      "supplier incidents. What it does show is that real lead times move a lot "
      "from year to year, so thresholds that look calm on steady data would page "
      "someone for about a third of vendor-years.")

    A("\n## 4. On-time scores, and which promise they use\n")
    ev = res["scheduled_evidence"]
    A(f"**Real:** of {ev['n_lines']:,} SCMS vendor lines, "
      f"**{ev['exact_on_scheduled_pct']:.0f}% arrive exactly on the scheduled "
      f"date** and {ev['on_or_before_scheduled_pct']:.0f}% on or before it. "
      f"{ev['vendors_exact_over_90pct']} of {ev['n_vendors']} vendors hit the exact "
      "date more than 90% of the time. Real deliveries do not land on the day "
      "that precisely; a scheduled-date field that is updated as plans change "
      "does. The file cannot prove this, and it keeps no original date.\n")
    A("**Simulated:** so the original promise is simulated for a third of vendors "
      "(half their lines promised 10-50% earlier than the SCMS date).\n")
    A("| vendor | receipts | on time vs original (sim) | on time vs SCMS scheduled (real) | gap | exact-date hits (real) |")
    A("|---|---:|---:|---:|---:|---:|")
    for r in res["otif_gap"][:10]:
        A(f"| {r['supplier_id']} {r['name'][:28]} | {r['n_receipts']} | "
          f"{r['otif_vs_original_pct']:.0f}% | {r['otif_vs_latest_promise_pct']:.0f}% | "
          f"**{r['gaming_gap_pts']:+.0f}** | {r['exact_on_scheduled_date_pct']:.0f}% |")

    A("\n## 5. Composite supplier risk, weights visible\n")
    w = res["risk"]["weights"]
    A("All inputs are real SCMS measures. Weights: "
      + ", ".join(f"`{k}` {v:.2f}" for k, v in w.items()) + ".\n")
    A("| vendor | risk | sole-source share | lead CV | late vs scheduled | recent drift | largest part |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for r in res["risk"]["top"]:
        top = max(r["contributions"], key=r["contributions"].get)
        A(f"| {r['supplier_id']} {r['name'][:26]} | {r['risk_score']:.3f} | "
          f"{100 * r['sole_source_share']:.0f}% | {r['lead_cv']:.2f} | "
          f"{100 * r['late_rate']:.1f}% | {r['recent_drift']:.2f}x | {top} |")

    A("\n## 6. Who gets a short shared part\n")
    A("| policy | total P50 units | worst product fill | median product fill | products under 50% |")
    A("|---|---:|---:|---:|---:|")
    for a in res["allocation"]:
        A(f"| {a['policy']} | {a['total_p50']:,.0f} | {a['worst_product_fill_pct']:.0f}% | "
          f"{a['median_product_fill_pct']:.0f}% | {a['products_below_50pct']} |")
    A("\nThe choice is a business policy, so all three are shown side by side.")
    A("\n---\n*Real: BOM, costs, demand, part lead-time distributions (Willems 2008); "
      "vendor lead times, on-time rates, order history (SCMS). Assumed: quantity "
      "per = 1, vendor-to-part mapping. Simulated: stock, open orders, MOQs, "
      "original promises, planted disruptions.*")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
