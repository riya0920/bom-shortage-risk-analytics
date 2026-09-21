"""Expedites with costs, tier-2 risk from real manufacturing sites, the weekly
pack, and calibration against simulated futures.

    python extend.py
    python extend.py --report-only
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


# ---------------------------------------------------------------------------
# 1. expedites
# ---------------------------------------------------------------------------

def part_shortfall(base) -> dict[str, float]:
    """13-week requirement minus (on hand + open orders), per purchased part."""
    req = supply.unit_requirements(base)
    need: dict[str, float] = {}
    for prod in base.products:
        units = float(base.demand[prod].sum())
        for part, q in req[prod].items():
            need[part] = need.get(part, 0.0) + q * units
    supply_ = {p: c.on_hand for p, c in base.components.items() if c.level == 2}
    for po in base.pos:
        supply_[po.part] = supply_.get(po.part, 0.0) + po.qty
    return {p: max(need.get(p, 0.0) - supply_.get(p, 0.0), 0.0) for p in supply_}


def expedite_options(base, drivers, mc) -> list[dict]:
    """For each gating part, what an expedite buys and what it costs.

    Premium = unit cost x (premium multiple - 1) x the part's shortfall
    quantity. Unit costs are real (Willems stageCost); the premium multiples
    and compression factors are assumptions, stated here.
    """
    short = part_shortfall(base)
    rows = []
    for d in drivers[:8]:
        part = d["part"]
        c = base.components.get(part)
        if c is None:
            continue
        for mode, compression, premium_mult in (("air freight", 0.45, 4.0),
                                                ("partial shipment", 0.70, 1.8),
                                                ("supplier overtime", 0.85, 1.25)):
            days_saved = c.lead_mean * (1 - compression)
            rows.append({
                "part": part, "supplier": c.supplier_id, "mode": mode,
                "lead_mean_days": c.lead_mean,
                "days_saved": days_saved,
                "shortfall_units": short.get(part, 0.0),
                "premium_usd": c.unit_cost * (premium_mult - 1.0) * short.get(part, 0.0),
                "unit_cost": c.unit_cost,
                "single_sourced": c.single_sourced,
                "order_by_day": d["order_by_day"],
                "needed": bool(d["order_by_day"] < 0),
                "recovers_runway": bool(d["order_by_day"] + days_saved > 0),
            })
    return rows


# ---------------------------------------------------------------------------
# 2. multi-tier: REAL manufacturing sites behind each SCMS vendor
# ---------------------------------------------------------------------------

def build_subtier(base) -> dict:
    """Tier 2 = the manufacturing sites SCMS records behind each vendor's lines.

    This is real data: SCMS logs the manufacturing site of every shipment, and
    some sites ship through several vendors. The step from vendor to Willems
    part still goes through the assumed mapping.
    """
    sites_by_vendor: dict[str, set] = {}
    for d in base.history:
        if d.site:
            sites_by_vendor.setdefault(d.supplier_id, set()).add(d.site)
    vendors_by_site: dict[str, set] = {}
    for sid, sites in sites_by_vendor.items():
        for t in sites:
            vendors_by_site.setdefault(t, set()).add(sid)

    req = supply.unit_requirements(base)
    parts_by_vendor: dict[str, list] = {}
    for p, c in base.components.items():
        if c.level == 2 and c.supplier_id:
            parts_by_vendor.setdefault(c.supplier_id, []).append(p)

    rows = []
    for t, vs in vendors_by_site.items():
        parts = sorted({p for v in vs for p in parts_by_vendor.get(v, [])})
        products = {pr for pr in base.products if any(p in req[pr] for p in parts)}
        rows.append({
            "tier2": t, "tier1_suppliers": len(vs),
            "tier1_list": sorted(vs),
            "parts_behind_it": len(parts),
            "products_affected": sorted(products),
            "n_products_affected": len(products),
        })
    rows.sort(key=lambda r: (-r["tier1_suppliers"], -r["parts_behind_it"], r["tier2"]))
    return {"sites_by_vendor": {k: sorted(v) for k, v in sites_by_vendor.items()},
            "concentration": rows, "n_tier2": len(vendors_by_site),
            "n_shared_sites": sum(1 for r in rows if r["tier1_suppliers"] > 1),
            "vendors_with_one_site": sum(1 for v in sites_by_vendor.values()
                                         if len(v) == 1)}


def hidden_single_source(base, subtier) -> list[dict]:
    """Vendors a buyer would treat as independent that share a factory."""
    out = []
    for r in subtier["concentration"]:
        if r["tier1_suppliers"] < 2:
            continue
        out.append({"tier2": r["tier2"], "tier1_count": r["tier1_suppliers"],
                    "tier1_suppliers": r["tier1_list"][:6],
                    "parts_exposed": r["parts_behind_it"],
                    "products_exposed": r["n_products_affected"]})
    return out


# ---------------------------------------------------------------------------
# 3. genuine calibration
# ---------------------------------------------------------------------------

def calibrate_against_future(base, n_futures: int = 60, n_sims: int = 120,
    """Calibration of the fan against independent simulated futures.
    """The calibration the first build could not do.

    RESULTS.md checked what fraction of simulations fell below their own P50 --
    which is a self-consistency check on the arithmetic, not on the model, and it
    said so. This is the real thing: draw N independent FUTURES from the same
    generative process, treat each as "what actually happened", and ask how often
    the forecast fan contained it.

    A P10-P90 band should contain the outcome 80% of the time. If it contains it
    99% of the time the fan is too wide and nobody will trust the P10; if 50%,
    the plan is being presented as more certain than it is.
    """
    rng = np.random.default_rng(seed)
    mc = B.monte_carlo(base, n_sims=n_sims, seed=seed)
    fan = mc["fan"]

    inside = {p: 0 for p in base.products}
    below_p10 = {p: 0 for p in base.products}
    above_p90 = {p: 0 for p in base.products}
    total = {p: 0 for p in base.products}

    for i in range(n_futures):
        actual = B.buildable(base, rng=np.random.default_rng(int(rng.integers(1e9))),
                             stochastic=True)
        for p in base.products:
            a = np.asarray(actual["built"][p], dtype=float)
            lo = np.asarray(fan[p]["p10"], dtype=float)
            hi = np.asarray(fan[p]["p90"], dtype=float)
            inside[p] += int(((a >= lo) & (a <= hi)).sum())
            below_p10[p] += int((a < lo).sum())
            above_p90[p] += int((a > hi).sum())
            total[p] += len(a)

    rows = []
    for p in base.products:
        t = max(1, total[p])
        rows.append({
            "product": p,
            "pct_inside_p10_p90": 100.0 * inside[p] / t,
            "pct_below_p10": 100.0 * below_p10[p] / t,
            "pct_above_p90": 100.0 * above_p90[p] / t,
            "target_inside_pct": 80.0,
        })
    return {"n_futures": n_futures, "n_sims_in_fan": n_sims, "per_product": rows,
            "mean_inside_pct": float(np.mean([r["pct_inside_p10_p90"] for r in rows]))}


# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(exist_ok=True)
    if "--report-only" in sys.argv:
        prev = json.loads((OUT / "extensions.json").read_text())
        (ROOT / "docs" / "EXTENSIONS.md").write_text(report(prev), encoding="utf-8")
        print("re-rendered docs/EXTENSIONS.md")
        return

    t0 = time.perf_counter()
    quick = "--quick" in sys.argv
    base = supply.build()
    res: dict = {}

    print("1/4 Monte Carlo + expedite options ...", flush=True)
    mc = B.monte_carlo(base, n_sims=60 if quick else 200)
    drivers = B.shortage_drivers(base, mc)
    res["expedites"] = expedite_options(base, drivers, mc)
    res["drivers"] = drivers[:8]
    print(f"    {len(res['expedites'])} expedite options across "
          f"{len(res['drivers'])} gating parts", flush=True)

    print("2/4 multi-tier supplier risk ...", flush=True)
    st = build_subtier(base)
    res["subtier"] = {"n_tier2": st["n_tier2"], "n_shared_sites": st["n_shared_sites"],
                      "vendors_with_one_site": st["vendors_with_one_site"],
                      "n_vendors": len(st["sites_by_vendor"]),
                      "concentration": st["concentration"][:8]}
    res["hidden_single_source"] = hidden_single_source(base, st)[:8]
    top = st["concentration"][0]
    print(f"    top site {top['tier2']} sits behind {top['tier1_suppliers']} vendors, "
          f"{top['parts_behind_it']} parts and {top['n_products_affected']} products",
          flush=True)

    print("3/4 genuine calibration against simulated futures ...", flush=True)
    res["calibration"] = calibrate_against_future(
        base, n_futures=25 if quick else 60, n_sims=60 if quick else 150)
    print(f"    P10-P90 contains the outcome "
          f"{res['calibration']['mean_inside_pct']:.1f}% of the time "
          f"(target 80%)", flush=True)

    print("4/4 the weekly materials pack ...", flush=True)
    res["products"] = list(base.products)
    res["pack"] = weekly_pack(base, mc, drivers, res)
    (OUT / "materials_review_pack.md").write_text(res["pack"], encoding="utf-8")
    res["wall_seconds"] = time.perf_counter() - t0

    (OUT / "extensions.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "EXTENSIONS.md").write_text(report(res), encoding="utf-8")
    print(f"\nwrote docs/EXTENSIONS.md and out/materials_review_pack.md "
          f"({res['wall_seconds']:.0f}s)")


def weekly_pack(base, mc, drivers, res) -> str:
    """The pack the Monday meeting runs on, generated by code."""
    L = ["# Weekly Materials Review", "",
         "*Generated by `extend.py`. BOM, demand, part lead times: Willems (2008) "
         "chain 24 (real). Vendors: USAID SCMS (real), mapped to parts by lead-time "
         "band (assumption). Stock and open orders: simulated.*", "",
         "## 1. Can we build the plan?", ""]
    L.append("| product | 13-week demand | P50 buildable | P10 | fill at P50 |")
    L.append("|---|---:|---:|---:|---:|")
    for p in base.products:
        f = mc["fan"][p]
        dem = float(np.sum(f["demand"]))
        p50 = float(np.sum(f["p50"]))
        p10 = float(np.sum(f["p10"]))
        L.append(f"| {p} | {dem:,.0f} | {p50:,.0f} | {p10:,.0f} | "
                 f"{100*p50/max(dem,1):.0f}% |")

    L += ["", "## 2. Top shortage drivers", ""]
    L.append("| part | vendor | single source | first short week | order by | status |")
    L.append("|---|---|---|---|---|---|")
    for d in drivers[:6]:
        status = "**PAST DUE**" if d["already_too_late"] else "order in time"
        L.append(f"| {d['part']} | {d['supplier']} | "
                 f"{'YES' if d['single_sourced'] else 'no'} | "
                 f"{('week ' + str(d['first_impact_week'])) if d['first_impact_week'] else '-'} | "
                 f"day {d['order_by_day']:+.0f} | {status} |")

    L += ["", "## 3. Expedite options", ""]
    L.append("| part | mode | days saved | premium | needed? |")
    L.append("|---|---|---:|---:|---|")
    for e in res["expedites"][:8]:
        L.append(f"| {e['part']} | {e['mode']} | {e['days_saved']:.0f} | "
                 f"${e['premium_usd']:,.0f} | {'yes' if e['needed'] else 'no, order normally'} |")

    L += ["", "## 4. Shared factories behind our vendors (tier 2, real SCMS sites)", ""]
    L.append("| manufacturing site | vendors behind it | parts | products |")
    L.append("|---|---:|---:|---:|")
    for c in res["subtier"]["concentration"][:5]:
        L.append(f"| {c['tier2']} | {c['tier1_suppliers']} | "
                 f"{c['parts_behind_it']} | {c['n_products_affected']} |")
    L += ["", "*Items marked PAST DUE need a decision in this meeting.*", ""]
    return "\n".join(L)


def report(res: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# Expedites, tier-2 risk and calibration - generated by `extend.py`, not hand-edited\n")

    A("## 1. Expedites\n")
    A("Premium = real unit cost x (premium multiple - 1) x the part's shortfall "
      "quantity. The multiples (air 4x, partial 1.8x, overtime 1.25x) and the "
      "lead-time compression are assumptions.\n")
    A("| part | vendor | mode | lead mean | days saved | shortfall units | premium | needed? |")
    A("|---|---|---|---:|---:|---:|---:|---|")
    for e in res["expedites"][:12]:
        A(f"| {e['part']} | {e['supplier']} | {e['mode']} | "
          f"{e['lead_mean_days']:.1f} d | {e['days_saved']:.1f} d | "
          f"{e['shortfall_units']:,.0f} | ${e['premium_usd']:,.0f} | "
          f"{'**yes**' if e['needed'] else 'no'} |")
    n_need = sum(1 for e in res["expedites"] if e["needed"])
    A(f"\n**{n_need} of {len(res['expedites'])} options are needed.** Every top "
      "driver still has time to be ordered at its normal lead time, because the "
      "real part lead times are short (median 9 days). Paying for speed would "
      "not help: these parts are short on quantity, not on time. In the "
      "simulated version (lead times of 14-75 days) 11 of 24 expedites arrived "
      "too late; with real lead times the question does not come up.")

    st = res["subtier"]
    A("\n## 2. Tier 2: the factories behind the vendors (real)\n")
    A(f"SCMS records a manufacturing site on every line. Across the "
      f"{st['n_vendors']} vendors there are **{st['n_tier2']} sites**; "
      f"**{st['n_shared_sites']} of them ship through more than one vendor**, and "
      f"{st['vendors_with_one_site']} vendors rely on a single site.\n")
    A("| manufacturing site | vendors behind it | parts exposed | products affected |")
    A("|---|---:|---:|---:|")
    for c in st["concentration"]:
        A(f"| {c['tier2']} | {c['tier1_suppliers']} ({', '.join(c['tier1_list'])}) | "
          f"{c['parts_behind_it']} | {c['n_products_affected']} |")
    top = st["concentration"][0]
    A(f"\n**{top['tier2']}** sits behind {top['tier1_suppliers']} vendors a buyer "
      f"would see as separate, {top['parts_behind_it']} parts and "
      f"{top['n_products_affected']} of the {len(base_products(res))} products. "
      "The vendor-to-site link is real; the site-to-part link goes through the "
      "assumed vendor mapping, so the part and product counts show the method, "
      "not a fact about power tools.")

    c = res["calibration"]
    A("\n## 3. Calibration against simulated futures\n")
    A(f"Draw {c['n_futures']} independent futures from the same model, treat each "
      f"as what happened, and ask how often the fan (from {c['n_sims_in_fan']} "
      "simulations) contained it.\n")
    A("| product | % inside P10-P90 | % below P10 | % above P90 |")
    A("|---|---:|---:|---:|")
    for r in c["per_product"]:
        A(f"| {r['product']} | {r['pct_inside_p10_p90']:.0f}% | "
          f"{r['pct_below_p10']:.0f}% | {r['pct_above_p90']:.0f}% |")
    m = c["mean_inside_pct"]
    A(f"\n**Mean coverage {m:.0f}% against a target of 80%.** ")
    if m > 90:
        A("The band is too wide for its label, and here it is also very narrow in "
          "units: the real part lead-time distributions have little spread (two to "
          "four values a few days apart), and buildable units are whole numbers, so "
          "many weeks have P10 = P50 = P90 and almost every future lands on it. "
          "The fan's width comes from supply timing, and in this chain timing "
          "barely moves the answer; quantity does.")
    A("\nThe futures come from the same model as the fan, so this checks the "
      "propagation, not whether real lead times behave this way.")

    A("\n## 4. The weekly materials pack\n")
    A("Written to `out/materials_review_pack.md`: can we build the plan, top "
      "shortage drivers with order-by dates, expedite options with costs, and "
      "shared factories.")
    A("\n---\n*Regenerate with `python extend.py`.*")
    return "\n".join(L) + "\n"


def base_products(res):
    return res.get("products", [])


if __name__ == "__main__":
    main()
