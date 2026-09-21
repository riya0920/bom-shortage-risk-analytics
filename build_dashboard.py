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
                    single=d["single_sourced"], week=d["first_impact_week"],
                    p95=round(d["lead_p95_days"]), mean=round(d["lead_mean_days"]),
                    order_by=round(d["order_by_day"]), late=d["already_too_late"])
               for d in r["drivers"]]

    lc = p["incidents"]["load_check"]
    kpis = dict(
        fill_p50=p50_total / demand_total,
        units_p50=p50_total, units_demand=demand_total,
        late_drivers=sum(d["late"] for d in drivers), n_drivers=len(drivers),
        value_at_risk=lc["value_before"],
        zero_slack=p["incidents"]["already_short"],
        n_parts=r["base"]["purchased_parts"],
        otif_gamers=sum(o["gaming_gap_pts"] >= 50 for o in r["otif_gap"]),
    )
    return dict(
        base=r["base"] | {"planted_disruptions": len(r["base"]["planted_disruptions"])},
        kpis=kpis,
        fan=dict(total=total, **fan),
        drivers=drivers,
        otif=[dict(s=o["supplier_id"], orig=o["otif_vs_original_pct"],
                   latest=o["otif_vs_latest_promise_pct"], n=o["n_receipts"])
              for o in r["otif_gap"]],
        sweep=[dict(scale=s["scale"], precision=s["precision"],
                    recall=s["recall"], flagged=s["n_flagged"])
               for s in c["sweep"]["curve"]],
        best=c["sweep"]["best_f1"],
        ablation={k: dict(precision=v["precision"], recall=v["recall"])
                  for k, v in c["sweep"]["ablation"].items()},
        load=dict(
            before={k: v["P1"] for k, v in p["incidents"]["load_before"].items()},
            after={k: v["P1"] for k, v in p["incidents"]["load_after"].items()},
            limit=lc["p1_limit"], alerts=lc["alerts"], incidents=lc["incidents"],
            top10_share=p["incidents"]["value_triage"]["top_n_share_of_p1_value"],
            n80=p["incidents"]["value_triage"]["incidents_for_80pct_of_p1_value"]),
        allocation=[dict(policy=a["policy"], total=a["total_p50"],
                         worst=a["worst_product_fill_pct"],
                         per={k: v["fill_rate_pct"] for k, v in a["per_product"].items()})
                    for a in r["allocation"]],
        tier2=[dict(t=t["tier2"], parts=t["parts_behind_it"],
                    tier1=t["tier1_suppliers"], products=t["n_products_affected"])
               for t in e["subtier"]["concentration"]],
        safety=dict(understate=c["safety"]["mean_understatement_demand_only"],
                    supply_share=c["safety"]["mean_supply_variance_share"],
                    fat=c["safety"]["empirical_over_parametric_fat_tail"],
                    thin=c["safety"]["empirical_over_parametric_thin_tail"]),
        expedite=dict(n=len(e["expedites"]),
                      no_recover=sum(not x["recovers_runway"] for x in e["expedites"])),
        distress=dict(observed=c["financial"]["auc_observed_distress"],
                      invented=c["financial"]["auc_invented_score"]),
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
