"""BOM propagation to end-item buildability, and the shortage attribution.

The centrepiece. Given on-hand inventory, open purchase orders, and lead-time
DISTRIBUTIONS, compute how many units of each product can be built per week over a
13-week horizon -- as a distribution, not a point estimate.

Why a distribution: supply plans are probabilistic and presenting them as a single
number is how a materials meeting ends with a commitment nobody can keep. P50 = 120
buildable and P10 = 84 are different conversations, and the second one is the
useful one.

Why 13 weeks: it covers most purchased-part lead times plus a planning cycle, so
an order placed today still lands inside the window. Beyond it, FORECAST
uncertainty starts to dominate SUPPLY uncertainty and the analysis is answering a
sales question wearing a materials hat.
"""
from __future__ import annotations

import numpy as np

from supply import WEEKS, SupplyBase, unit_requirements


def _receipt_schedule(base: SupplyBase, rng: np.random.Generator,
                      stochastic: bool) -> dict[str, np.ndarray]:
    """Quantity of each purchased part arriving in each week of the horizon.

    Open POs land at their promise date under the deterministic plan. Under Monte
    Carlo they land at promise + a lead-time deviation drawn from that supplier's
    distribution, and are reduced by the supplier's quality rejection rate --
    because a receipt that fails inspection is not supply.
    """
    arrivals: dict[str, np.ndarray] = {}
    for po in base.pos:
        if po.received_day is not None:
            continue  # already received; already in on_hand
        s = base.suppliers[po.supplier_id]
        day = po.promise_day
        qty = po.qty
        if stochastic:
            dev = rng.normal(0, s.lead_sd_days)
            if s.fat_tail and rng.random() < 0.12:
                dev += abs(rng.normal(0, s.lead_mean_days * 0.9))
            day += dev
            qty *= (1.0 - min(0.5, max(0.0, rng.normal(s.quality_reject_rate,
                                                       s.quality_reject_rate * 0.5))))
        else:
            qty *= (1.0 - s.quality_reject_rate)
        wk = int(np.floor(day / 7.0))
        if wk < 0:
            wk = 0
        if wk >= WEEKS:
            continue
        a = arrivals.setdefault(po.part, np.zeros(WEEKS))
        a[wk] += max(0.0, qty)
    return arrivals


def buildable(base: SupplyBase, rng: np.random.Generator | None = None,
              stochastic: bool = False,
              allocation: str = "even") -> dict:
    """Weekly buildable units per product, plus what gated each week.

    The allocation policy matters because components are SHARED. When a common
    part is short, somebody has to decide which product gets it, and the decision
    is a business policy rather than an arithmetic fact:

      even     -- split the shortfall proportionally to demand
      margin   -- highest-margin product first
      contract -- contractual/priority order first

    The three produce materially different plans and different angry people, which
    is why `compare_allocation_policies()` runs all of them instead of picking.
    """
    rng = rng or np.random.default_rng(0)
    req = unit_requirements(base)
    arrivals = _receipt_schedule(base, rng, stochastic)

    on_hand = {p: c.on_hand for p, c in base.components.items() if c.level == 2}
    built = {p: np.zeros(WEEKS) for p in base.products}
    gating: list[dict] = []

    order = _priority_order(base, allocation)

    for wk in range(WEEKS):
        for part, arr in arrivals.items():
            on_hand[part] = on_hand.get(part, 0.0) + arr[wk]

        want = {p: float(base.demand[p][wk]) for p in base.products}

        # Which part binds each product, if it had the inventory to itself? Kept
        # for shortage attribution -- the "who gated me" answer.
        binding_of = {}
        for p in base.products:
            lim, binding = want[p], None
            for part, per in req[p].items():
                if per <= 0 or part not in on_hand:
                    continue
                can = on_hand[part] / per
                if can < lim:
                    lim, binding = can, part
            binding_of[p] = binding

        if allocation == "even":
            # PROPORTIONAL split of every contended part, then rebuild.
            #
            # The first version consumed inventory in list order and called it
            # "even", which meant the first product in the list took everything it
            # wanted and the last one starved -- a worst-product fill rate of 7.7%
            # that was an artifact of dict ordering rather than of any policy. An
            # allocation policy that depends on the order of a Python list is not a
            # policy.
            share: dict[str, dict[str, float]] = {p: {} for p in base.products}
            for part in on_hand:
                total_need = sum(req[p].get(part, 0.0) * want[p] for p in base.products)
                if total_need <= 0:
                    continue
                avail = on_hand[part]
                if avail >= total_need:
                    for p in base.products:
                        share[p][part] = req[p].get(part, 0.0) * want[p]
                else:
                    for p in base.products:
                        need_p = req[p].get(part, 0.0) * want[p]
                        share[p][part] = avail * (need_p / total_need)
            for p in base.products:
                n = want[p]
                for part, per in req[p].items():
                    if per > 0 and part in on_hand:
                        n = min(n, np.floor(share[p].get(part, 0.0) / per))
                n = max(0.0, n)
                for part, per in req[p].items():
                    if part in on_hand:
                        on_hand[part] -= n * per
                built[p][wk] = n
        else:
            # Strict priority: the highest-priority product takes what it needs,
            # the next takes what is left, and so on.
            for p in order:
                n = want[p]
                for part, per in req[p].items():
                    if per > 0 and part in on_hand:
                        n = min(n, np.floor(on_hand[part] / per))
                n = max(0.0, n)
                for part, per in req[p].items():
                    if part in on_hand:
                        on_hand[part] -= n * per
                built[p][wk] = n

        for p in base.products:
            if built[p][wk] < want[p] - 1e-9:
                gating.append({
                    "week": wk + 1, "product": p, "wanted": want[p],
                    "built": float(built[p][wk]),
                    "short_by": want[p] - float(built[p][wk]),
                    "gating_part": binding_of[p],
                })
    return {"built": built, "gating": gating, "allocation": allocation}


def _priority_order(base: SupplyBase, allocation: str) -> list[str]:
    if allocation == "margin":
        # Stand-in for margin: total BOM cost as a proxy for product value.
        req = unit_requirements(base)
        val = {p: sum(base.components[c].unit_cost * q
                      for c, q in req[p].items() if c in base.components)
               for p in base.products}
        return sorted(base.products, key=lambda p: -val[p])
    if allocation == "contract":
        return sorted(base.products)
    return list(base.products)


def monte_carlo(base: SupplyBase, n_sims: int = 300, seed: int = 7,
                allocation: str = "even") -> dict:
    """Buildability as a DISTRIBUTION over lead-time and quality uncertainty."""
    rng = np.random.default_rng(seed)
    sims = {p: np.zeros((n_sims, WEEKS)) for p in base.products}
    gate_counts: dict[str, int] = {}
    for i in range(n_sims):
        r = buildable(base, rng, stochastic=True, allocation=allocation)
        for p in base.products:
            sims[p][i] = r["built"][p]
        for g in r["gating"]:
            if g["gating_part"]:
                gate_counts[g["gating_part"]] = gate_counts.get(g["gating_part"], 0) + 1

    fan = {}
    for p in base.products:
        fan[p] = {
            "p10": np.percentile(sims[p], 10, axis=0).tolist(),
            "p50": np.percentile(sims[p], 50, axis=0).tolist(),
            "p90": np.percentile(sims[p], 90, axis=0).tolist(),
            "demand": base.demand[p].tolist(),
        }
    top = sorted(gate_counts.items(), key=lambda kv: -kv[1])[:15]
    return {
        "fan": fan, "n_sims": n_sims, "allocation": allocation,
        "top_gating_parts": [{"part": k, "times_gating": v} for k, v in top],
        "_sims": sims,
    }


def compare_allocation_policies(base: SupplyBase, n_sims: int = 120) -> list[dict]:
    out = []
    for pol in ("even", "margin", "contract"):
        mc = monte_carlo(base, n_sims=n_sims, allocation=pol)
        rows = {}
        for p in base.products:
            s = np.array(mc["fan"][p]["p50"])
            d = np.array(mc["fan"][p]["demand"])
            rows[p] = {
                "p50_total": float(s.sum()),
                "demand_total": float(d.sum()),
                "fill_rate_pct": float(100 * s.sum() / max(1e-9, d.sum())),
            }
        out.append({
            "policy": pol,
            "per_product": rows,
            "total_p50": float(sum(r["p50_total"] for r in rows.values())),
            "worst_product_fill_pct": float(min(r["fill_rate_pct"] for r in rows.values())),
        })
    return out


def shortage_drivers(base: SupplyBase, mc: dict, top_n: int = 10) -> list[dict]:
    """The Monday materials meeting list: which parts gate the plan, and by when.

    Each driver carries an ORDER-BY DATE, which is what makes it actionable. An
    alert that says "you will be short in week 7" is a fact; an alert that says
    "order by 3 March or the line stops 21 April" is a decision.
    """
    drivers = []
    for row in mc["top_gating_parts"][:top_n]:
        part = row["part"]
        c = base.components.get(part)
        if c is None or c.supplier_id is None:
            continue
        s = base.suppliers[c.supplier_id]
        # First week the part gates anything, from the deterministic pass.
        det = buildable(base)
        first_wk = min((g["week"] for g in det["gating"] if g["gating_part"] == part),
                       default=None)
        impact_day = (first_wk * 7) if first_wk else WEEKS * 7
        # P95 lead time: ordering to the mean means arriving late half the time,
        # which for a gating part is not a plan.
        p95_lead = s.lead_mean_days + 1.645 * s.lead_sd_days
        order_by_day = impact_day - p95_lead
        drivers.append({
            "part": part, "supplier": s.supplier_id,
            "single_sourced": c.single_sourced,
            "times_gating_in_sim": row["times_gating"],
            "first_impact_week": first_wk,
            "lead_mean_days": s.lead_mean_days,
            "lead_p95_days": p95_lead,
            "order_by_day": order_by_day,
            "runway_days": order_by_day,
            "already_too_late": bool(order_by_day < 0),
            "on_hand": c.on_hand,
            "unit_cost": c.unit_cost,
        })
    return drivers
