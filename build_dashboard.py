"""Build the public dashboard (docs/index.html) from the pipeline's own JSON.

Run after run_supply.py, extend.py, complete.py and run_pass4.py. Every number on
the page is read from out/*.json - nothing is typed in by hand.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DOCS = os.path.join(HERE, "docs")


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


def build_data():
    r, e, c, p = (load("results.json"), load("extensions.json"),
                  load("completion.json"), load("pass4.json"))
    fan = r["monte_carlo"]["fan"]
    weeks = len(next(iter(fan.values()))["p50"])
    total = {k: [sum(fan[prd][k][w] for prd in fan) for w in range(weeks)]
             for k in ("p10", "p50", "p90", "demand")}
    demand_total = sum(total["demand"])
    p50_total = sum(total["p50"])

    drivers = [dict(part=d["part"], supplier=d["supplier"],
                    vendor=d.get("supplier_name", ""),
                    single=d["single_sourced"], week=d["first_impact_week"],
                    p95=round(d["lead_p95_days"]), mean=round(d["lead_mean_days"], 1),
                    order_by=round(d["order_by_day"]), late=d["already_too_late"],
                    products=d.get("n_products_using"))
               for d in r["drivers"]]

    ic = p["incidents"]
    lc = ic["load_check"]
    b = r["base"]
    ev = r["scheduled_evidence"]
    kpis = dict(
        fill_p50=p50_total / demand_total,
        units_p50=p50_total, units_demand=demand_total,
        late_drivers=sum(d["late"] for d in drivers), n_drivers=len(drivers),
        value_at_risk=lc["value_before"],
        runs_out=ic["runs_out_in_horizon"],
        n_parts=b["purchased_parts"],
        exact_date_pct=ev["exact_on_scheduled_pct"],
        n_lines=ev["n_lines"],
    )
    fin = c["financial"]
    top_site = e["subtier"]["concentration"][0]
    return dict(
        base={k: b[k] for k in ("n_products", "purchased_parts", "sub_assemblies",
                                "bom_lines", "suppliers", "single_sourced_parts",
                                "parts_shared_by_2plus_products",
                                "parts_in_every_product",
                                "parts_with_lead_distribution", "horizon_weeks",
                                "real_delivery_lines", "day0")}
        | {"planted_disruptions": len(b["planted_disruptions"]),
           "lead_min": b["part_lead_days"]["min"],
           "lead_median": b["part_lead_days"]["median"],
           "lead_max": b["part_lead_days"]["max"],
           "scms_rows": b["scms_cleaning"]["rows"],
           "scms_kept": b["scms_cleaning"]["kept"],
           "n_sims": r["monte_carlo"]["n_sims"]},
        kpis=kpis,
        fan=dict(total=total, **fan),
        drivers=drivers,
        otif=[dict(s=o["supplier_id"], name=o["name"], orig=o["otif_vs_original_pct"],
                   latest=o["otif_vs_latest_promise_pct"], n=o["n_receipts"],
                   exact=o["exact_on_scheduled_date_pct"])
              for o in r["otif_gap"][:9]],
        evidence=ev,
        sweep=[dict(scale=s["scale"], precision=s["precision"],
                    recall=s["recall"], flagged=s["n_flagged"],
                    judged=s["n_evaluated"])
               for s in c["sweep"]["curve"]],
        best=c["sweep"]["best_f1"],
        ablation={k: dict(precision=v["precision"], recall=v["recall"])
                  for k, v in c["sweep"]["ablation"].items()},
        window120=r["deterioration"]["window_120d"]["evaluable"],
        judged=r["deterioration"]["score"]["evaluated"],
        real_scan={k: r["real_scan"][k] for k in
                   ("vendor_years_evaluated", "vendor_years_flagged",
                    "vendors_flagged_at_least_once")},
        load=dict(sens=ic["sensitivity"], limit=lc["p1_limit"],
                  owner_limit=lc["per_owner_limit"],
                  alerts=lc["alerts"], incidents=lc["incidents"],
                  stress=ic["stress"]["stock_scale"]),
        allocation=[dict(policy=a["policy"], total=a["total_p50"],
                         worst=a["worst_product_fill_pct"],
                         median=a["median_product_fill_pct"],
                         under50=a["products_below_50pct"])
                    for a in r["allocation"]],
        tier2=dict(site=top_site["tier2"], vendors=top_site["tier1_suppliers"],
                   shared=e["subtier"]["n_shared_sites"], n=e["subtier"]["n_tier2"]),
        safety=dict(understate=c["safety"]["mean_understatement_demand_only"],
                    supply_share=c["safety"]["mean_supply_variance_share"]),
        expedite=dict(n=len(e["expedites"]),
                      needed=sum(x["needed"] for x in e["expedites"])),
        distress=dict(proxy=fin["auc_observed_distress"],
                      scorecard=fin["auc_on_time_scorecard"]),
        calibration=dict(inside=e["calibration"]["mean_inside_pct"], target=80),
    )


def main():
    data = build_data()
    with open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8") as f:
        page = f.read()
    page = page.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    print(f"wrote docs/index.html ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
