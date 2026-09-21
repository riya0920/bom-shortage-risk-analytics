"""BOM propagation to end-item buildability, and the shortage attribution.

Given on-hand stock, open orders and lead-time DISTRIBUTIONS, compute how many
units of each product can be built per week over a 13-week horizon, as a
distribution rather than one number.

Lead-time uncertainty in the Monte Carlo has two real parts:
  * the part's own Willems lead-time distribution (drawn, minus its mean, so a
    part with a fixed lead time in the data gets no spread), and
  * the mapped vendor's real SCMS lateness against its scheduled date, as a
    fraction of the quoted lead, applied to the part's lead time.
Stock and open orders are simulated (see supply.py).

Why 13 weeks: it covers every purchased-part lead time in this chain (the
longest is 55 days) plus a planning cycle. Beyond it, forecast uncertainty
starts to dominate supply uncertainty.
"""
from __future__ import annotations

import numpy as np

from supply import WEEKS, SupplyBase, unit_requirements


def _matrix(base: SupplyBase):
    """Products x purchased-parts requirement matrix, cached on the base."""
    cache = base.meta.get("_matrix")
    if cache is not None:
        return cache
    parts = sorted(p for p, c in base.components.items() if c.level == 2)
    col = {p: j for j, p in enumerate(parts)}
    req = unit_requirements(base)
    R = np.zeros((len(base.products), len(parts)))
    for i, prod in enumerate(base.products):
        for part, q in req[prod].items():
            j = col.get(part)
            if j is not None:
                R[i, j] = q
    cache = (parts, col, R)
    base.meta["_matrix"] = cache
    return cache


def _receipt_schedule(base: SupplyBase, rng: np.random.Generator,
                      stochastic: bool) -> np.ndarray:
    """Quantity of each purchased part arriving in each week (parts x weeks)."""
    parts, col, _ = _matrix(base)
    arr = np.zeros((len(parts), WEEKS))
    for po in base.pos:
        if po.received_day is not None:
            continue
        c = base.components[po.part]
        day = po.promise_day
        if stochastic:
            if c.has_distribution:
                day += float(rng.choice(c.lead_values, p=c.lead_probs)) - c.lead_mean
            s = base.suppliers.get(po.supplier_id)
            if s is not None and len(s.rel_late):
                day += float(s.rel_late[rng.integers(len(s.rel_late))]) * c.lead_mean
        wk = max(0, int(np.floor(day / 7.0)))
        if wk >= WEEKS:
            continue
        arr[col[po.part], wk] += max(0.0, po.qty)
    return arr


def buildable(base: SupplyBase, rng: np.random.Generator | None = None,
              stochastic: bool = False, allocation: str = "even") -> dict:
    """Weekly buildable units per product, plus what gated each week.

    Allocation matters because parts are SHARED (143 of the 209 purchased
    parts in this chain feed more than one product). When a shared part is
    short, somebody decides who gets it:

      even     -- split the shortfall in proportion to demand
      margin   -- highest-value product first (BOM cost as the stand-in)
      contract -- a fixed priority order (alphabetical, as a stand-in)
    """
    rng = rng or np.random.default_rng(0)
    parts, _, R = _matrix(base)
    arrivals = _receipt_schedule(base, rng, stochastic)
    on_hand = np.array([base.components[p].on_hand for p in parts], float)
    P = len(base.products)
    built = {p: np.zeros(WEEKS) for p in base.products}
    gating: list[dict] = []
    order = [base.products.index(p) for p in _priority_order(base, allocation)]
    uses = R > 0

    for wk in range(WEEKS):
        on_hand = on_hand + arrivals[:, wk]
        want = np.array([float(base.demand[p][wk]) for p in base.products])

        # Which part binds each product if it had the stock to itself.
        with np.errstate(divide="ignore", invalid="ignore"):
            can = np.where(uses, on_hand[None, :] / np.where(uses, R, 1.0), np.inf)
        j_min = np.argmin(can, axis=1)
        binding = [parts[j_min[i]] if can[i, j_min[i]] < want[i] else None
                   for i in range(P)]

        n = np.zeros(P)
        if allocation == "even":
            need = R * want[:, None]                 # P x N
            total = need.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(total > 0, np.minimum(1.0, on_hand / total), 1.0)
                share = need * frac[None, :]
                lim = np.where(uses, np.floor(share / np.where(uses, R, 1.0) + 1e-9),
                               np.inf)
            n = np.maximum(0.0, np.minimum(want, lim.min(axis=1)))
            on_hand = on_hand - n @ R
        else:
            for i in order:
                with np.errstate(divide="ignore", invalid="ignore"):
                    lim = np.where(uses[i], np.floor(on_hand / np.where(uses[i], R[i], 1.0)
                                                     + 1e-9), np.inf)
                n[i] = max(0.0, min(want[i], float(lim.min())))
                on_hand = on_hand - n[i] * R[i]
        on_hand = np.maximum(on_hand, 0.0)

        for i, p in enumerate(base.products):
            built[p][wk] = n[i]
            if n[i] < want[i] - 1e-9:
                gating.append({
                    "week": wk + 1, "product": p, "wanted": float(want[i]),
                    "built": float(n[i]), "short_by": float(want[i] - n[i]),
                    "gating_part": binding[i],
                })
    return {"built": built, "gating": gating, "allocation": allocation}


def _priority_order(base: SupplyBase, allocation: str) -> list[str]:
    if allocation == "margin":
        # Stand-in for margin: total purchased-part cost per unit.
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
    """Buildability as a DISTRIBUTION over lead-time uncertainty."""
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
    top = sorted(gate_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
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
        fills = [r["fill_rate_pct"] for r in rows.values()]
        out.append({
            "policy": pol,
            "per_product": rows,
            "total_p50": float(sum(r["p50_total"] for r in rows.values())),
            "worst_product_fill_pct": float(min(fills)),
            "median_product_fill_pct": float(np.median(fills)),
            "products_below_50pct": int(sum(f < 50 for f in fills)),
        })
    return out


def shortage_drivers(base: SupplyBase, mc: dict, top_n: int = 10) -> list[dict]:
    """The Monday materials list: which parts gate the plan, and by when.

    The order-by date uses the part's P95 lead time from its real Willems
    distribution (for a part with a fixed lead time, P95 = that time).
    """
    det = buildable(base)
    drivers = []
    for row in mc["top_gating_parts"][:top_n]:
        part = row["part"]
        c = base.components.get(part)
        if c is None or c.supplier_id is None:
            continue
        first_wk = min((g["week"] for g in det["gating"] if g["gating_part"] == part),
                       default=None)
        impact_day = ((first_wk - 1) * 7) if first_wk else WEEKS * 7
        order_by_day = impact_day - c.lead_p95
        drivers.append({
            "part": part, "supplier": c.supplier_id,
            "supplier_name": base.suppliers[c.supplier_id].name,
            "single_sourced": c.single_sourced,
            "times_gating_in_sim": row["times_gating"],
            "first_impact_week": first_wk,
            "lead_mean_days": c.lead_mean,
            "lead_p95_days": c.lead_p95,
            "lead_has_distribution": c.has_distribution,
            "order_by_day": order_by_day,
            "runway_days": order_by_day,
            "already_too_late": bool(order_by_day < 0),
            "on_hand": c.on_hand,
            "unit_cost": c.unit_cost,
            "n_products_using": sum(1 for p in base.products
                                    if part in unit_requirements(base)[p]),
        })
    return drivers
