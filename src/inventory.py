"""Safety stock, lot sizing, MOQ and supplier capacity -- the constraints that
make a recommended order quantity a real number rather than a gap calculation.

WHY THIS MATTERS FOR THE PROJECT'S OWN HONESTY. The README's item 6 says
`Component.moq` exists and nothing reads it, "which means the recommended order
quantities would be wrong". That is exact: a system that recommends ordering 340
pieces of a part with a 1,000-piece minimum has not produced a purchase order, it
has produced a number a buyer has to correct by hand -- and a buyer who has to
correct every number stops reading them.

===========================================================================
SAFETY STOCK, and the term everyone forgets
===========================================================================

The formula people remember covers demand variability only:

    SS = z * sigma_demand * sqrt(L)

The full one covers BOTH sources, because lead time is a random variable too:

    SS = z * sqrt( L * sigma_d^2  +  d^2 * sigma_L^2 )
                   \\____________/    \\_____________/
                    demand varies      SUPPLY varies

For the suppliers in this project the second term dominates and it is not close:
a fat-tailed supplier with a 45-day mean and a 20-day standard deviation
contributes far more to required stock than demand noise does. Dropping that
term -- which is the default in most textbook treatments and most ERP
configurations -- understates safety stock by a factor that grows with lead-time
variability, which is precisely the case where you needed it.

AND THE NORMAL APPROXIMATION IS WRONG FOR FAT TAILS. `z` comes from a normal
quantile. A supplier with occasional 4x-mean lead times has a right tail the
normal does not have, and z=2.33 buys nowhere near 99% service. Both are computed
here -- the parametric number and an empirical quantile of simulated lead-time
demand -- because the gap between them is the point.

===========================================================================
LOT SIZING
===========================================================================

EOQ balances ordering cost against holding cost:

    Q* = sqrt( 2 * D * S / H )

It assumes constant demand, instant replenishment, no MOQ and no capacity limit,
all of which are false here. It is still the right starting point, because it
gives a defensible quantity to then CONSTRAIN, and the constraining is where the
real answer lives:

    order = max(EOQ_or_gap, MOQ) rounded up to a pack multiple, capped by what
            the supplier can actually ship in the window

The order of operations matters: capping before rounding produces a quantity the
supplier cannot ship, and rounding before applying MOQ produces one below the
minimum.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# safety stock
# ---------------------------------------------------------------------------

def safety_stock(demand_per_day: float, sigma_demand: float,
                 lead_mean_days: float, lead_sd_days: float,
                 service_level: float = 0.95) -> dict:
    """Both variance terms, and the demand-only version alongside it."""
    z = float(stats.norm.ppf(service_level))
    demand_term = lead_mean_days * sigma_demand ** 2
    supply_term = (demand_per_day ** 2) * (lead_sd_days ** 2)
    ss_full = z * math.sqrt(max(demand_term + supply_term, 0.0))
    ss_demand_only = z * sigma_demand * math.sqrt(max(lead_mean_days, 0.0))
    return {
        "z": z, "service_level": service_level,
        "safety_stock": ss_full,
        "safety_stock_demand_only": ss_demand_only,
        "understatement_factor": ss_full / max(ss_demand_only, 1e-9),
        "demand_variance_share": demand_term / max(demand_term + supply_term, 1e-9),
        "supply_variance_share": supply_term / max(demand_term + supply_term, 1e-9),
        "reorder_point": demand_per_day * lead_mean_days + ss_full,
    }


def empirical_safety_stock(demand_per_day: float, sigma_demand: float,
                           lead_samples: np.ndarray, service_level: float = 0.95,
                           n_sims: int = 20000, seed: int = 0) -> dict:
    """Safety stock from the simulated distribution of lead-time demand.

    The parametric formula puts a normal tail on a quantity whose tail is set by
    the supplier's lead time. When that lead time is fat-tailed the normal
    quantile is optimistic, and optimistic safety stock is a stockout with a
    formula attached.
    """
    rng = np.random.default_rng(seed)
    ls = np.asarray(lead_samples, dtype=float)
    ls = ls[np.isfinite(ls) & (ls > 0)]
    if len(ls) < 5:
        return {"insufficient_history": True, "n_samples": int(len(ls))}
    draws = rng.choice(ls, size=n_sims, replace=True)
    dld = rng.normal(demand_per_day, sigma_demand, n_sims) * draws
    q = float(np.quantile(dld, service_level))
    mean_dld = float(np.mean(dld))
    return {"n_samples": int(len(ls)), "insufficient_history": False,
            "lead_time_demand_quantile": q, "mean_lead_time_demand": mean_dld,
            "safety_stock": max(q - mean_dld, 0.0),
            "lead_p50": float(np.quantile(ls, 0.5)),
            "lead_p95": float(np.quantile(ls, 0.95)),
            "tail_ratio_p95_over_p50": float(np.quantile(ls, 0.95)
                                             / max(np.quantile(ls, 0.5), 1e-9))}


# ---------------------------------------------------------------------------
# lot sizing
# ---------------------------------------------------------------------------

def eoq(annual_demand: float, order_cost: float, unit_cost: float,
        holding_rate: float = 0.22) -> float:
    h = max(unit_cost * holding_rate, 1e-9)
    return math.sqrt(max(2.0 * annual_demand * order_cost / h, 0.0))


def order_quantity(gross_need: float, *, moq: float, pack_size: float = 1.0,
                   annual_demand: float | None = None,
                   order_cost: float = 250.0, unit_cost: float = 1.0,
                   supplier_capacity: float | None = None) -> dict:
    """Turn a need into a quantity a buyer can actually place.

    Order of operations, and each step changes the answer:

      1. start from the larger of the gap and EOQ -- ordering the bare gap on a
         cheap part means ordering it again next week
      2. raise to MOQ
      3. round UP to a pack multiple
      4. cap at supplier capacity

    Capping last is deliberate. Cap before rounding and the rounding pushes the
    quantity back over the cap; apply MOQ after rounding and the quantity can
    land below the minimum. The capped case is also the one that must be
    SURFACED rather than silently truncated -- it means the need cannot be met by
    this supplier and somebody has to decide what to do about that.
    """
    q = float(gross_need)
    econ = None
    if annual_demand:
        econ = eoq(annual_demand, order_cost, unit_cost)
        q = max(q, econ)
    raised = q < moq
    q = max(q, float(moq))
    if pack_size > 1:
        q = math.ceil(q / pack_size) * pack_size
    capped = False
    if supplier_capacity is not None and q > supplier_capacity:
        q = float(supplier_capacity)
        capped = True
    return {
        "order_qty": q, "gross_need": float(gross_need), "moq": float(moq),
        "eoq": econ, "raised_to_moq": raised, "capped_by_capacity": capped,
        "excess_over_need": max(q - gross_need, 0.0),
        "shortfall_vs_need": max(gross_need - q, 0.0),
        "note": ("capacity-capped: this supplier cannot cover the need and the "
                 "gap needs a decision, not a rounded number"
                 if capped else None),
    }


def moq_waste(components: list[dict]) -> dict:
    """How much inventory MOQ forces onto the balance sheet.

    Worth computing because it is the argument a buyer takes to a supplier. "Your
    1,000-piece minimum costs us X in dead stock on a part we use 200 of a year"
    is a negotiation; "your MOQ is inconvenient" is not.
    """
    # MOQ-FORCED excess only. Ordering an economic quantity above the immediate
    # need is deliberate -- that is what EOQ is for -- so counting it as waste
    # would overstate the MOQ case and make the negotiating number indefensible
    # the first time a supplier checked it. The two are separated and both are
    # returned.
    moq_forced = 0.0
    eoq_excess = 0.0
    rows = []
    for c in components:
        cost = c.get("unit_cost", 1.0)
        r = order_quantity(c["need"], moq=c["moq"],
                           pack_size=c.get("pack_size", 1.0),
                           annual_demand=c.get("annual_demand"),
                           unit_cost=cost, supplier_capacity=c.get("capacity"))
        # What would have been ordered with NO minimum: the same calculation
        # without the MOQ step.
        no_moq = order_quantity(c["need"], moq=1.0,
                                pack_size=c.get("pack_size", 1.0),
                                annual_demand=c.get("annual_demand"),
                                unit_cost=cost,
                                supplier_capacity=c.get("capacity"))
        forced = max(r["order_qty"] - no_moq["order_qty"], 0.0)
        moq_forced += forced * cost
        eoq_excess += max(no_moq["order_qty"] - c["need"], 0.0) * cost
        if forced > 0:
            rows.append({"part": c["part"], "need": c["need"], "moq": c["moq"],
                         "ordered": r["order_qty"],
                         "would_order_without_moq": no_moq["order_qty"],
                         "excess_value": forced * cost,
                         "months_of_cover": (forced
                                             / max(c.get("annual_demand", 1) / 12, 1e-9))})
    rows.sort(key=lambda r: -r["excess_value"])
    return {"total_excess_value": moq_forced,
            "eoq_excess_value": eoq_excess,
            "n_parts_raised_to_moq": len(rows), "worst": rows[:10]}


# ---------------------------------------------------------------------------
# supplier capacity
# ---------------------------------------------------------------------------

def capacity_check(demands: dict[str, float], capacity: dict[str, float]) -> dict:
    """Which suppliers are over-committed, and by how much.

    The README's item 7: "a supplier can ship unlimited quantity; in reality the
    constraint is often the supplier's LINE, not their lead time." That
    distinction changes the remedy completely. A lead-time problem is solved by
    ordering earlier. A capacity problem is not solved by ordering earlier at
    all -- ordering earlier just moves the queue -- and needs a second source, a
    smaller order, or a different supplier.
    """
    rows = []
    for sup, need in sorted(demands.items(), key=lambda kv: -kv[1]):
        cap = capacity.get(sup, float("inf"))
        util = need / cap if cap else float("inf")
        rows.append({"supplier_id": sup, "committed": need, "capacity": cap,
                     "utilisation": util, "over": need > cap,
                     "shortfall": max(need - cap, 0.0),
                     "remedy": ("second source or reduce the order -- ordering "
                                "EARLIER does not create capacity"
                                if need > cap else "within capacity")})
    over = [r for r in rows if r["over"]]
    return {"rows": rows, "n_over_capacity": len(over),
            "total_shortfall": sum(r["shortfall"] for r in over)}
