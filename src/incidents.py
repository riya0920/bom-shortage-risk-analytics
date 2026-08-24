"""Grouping alerts into incidents, because the remedy is per supplier.

The README's own load check failed and named the cause:

    severity is computed per *part* while the remedy is usually per *supplier* --
    one late supplier puts every part it ships into P1 at once -- and the fix
    (group by supplier, route one item with its parts list) is not built.

WHAT AN INCIDENT IS. Not a summary of alerts. It is one DECISION with one owner.
A supplier that has slipped puts every part it ships at risk simultaneously, and
the response to all of them is the same phone call -- so they are one item on a
buyer's list, not thirty. The parts do not disappear; they become the incident's
contents, which is where they were always useful.

WHAT THIS DOES NOT FIX. Grouping reduces the *count* an owner sees. It does not
reduce the *work*, and a report that leads with "221 alerts became 34" while the
same parts are still at risk has substituted a metric for the problem. So
`load_check` reports both, and the value at risk -- which grouping cannot change
-- is carried through unchanged so the comparison is visibly like for like.

WHY ONE INCIDENT PER (SUPPLIER, SEVERITY) AND NOT PER SUPPLIER. A supplier with
two P1 parts and nine P4 parts is two different conversations on two different
clocks: the P1 needs an answer in four hours and the P4s belong in the weekly
review. Collapsing them into one item forces the whole group onto the P1 SLA,
which floods the urgent queue with parts that did not need to be there -- the
original problem, arrived at from the other direction.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from routing import TIERS, route

SEV_ORDER = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}


@dataclass
class Incident:
    supplier_id: str | None
    severity: str
    parts: list = field(default_factory=list)
    n_parts: int = 0
    value_at_risk: float = 0.0
    units_at_risk: float = 0.0
    worst_slack_days: float = 0.0
    n_single_sourced: int = 0
    any_single_sourced: bool = False
    over_capacity: bool = False
    max_risk_score: float = 0.0
    route_to: str = "BUYER"
    response_hours: int = 168
    escalate_after_hours: int | None = None
    escalation_chain: list = field(default_factory=list)
    reason: str = ""
    key: str = ""

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("supplier_id", "severity", "n_parts", "value_at_risk",
              "units_at_risk", "worst_slack_days", "n_single_sourced",
              "any_single_sourced", "over_capacity", "max_risk_score",
              "route_to", "response_hours", "escalate_after_hours",
              "escalation_chain", "reason", "key")}
        d["parts"] = list(self.parts)
        return d


def group_by_supplier(alerts: list, *, over_capacity_suppliers=None) -> list:
    """One incident per (supplier, severity), parts listed inside.

    Alerts with no supplier stay individual: a part whose risk cannot be
    attributed to a supplier has no shared remedy, and inventing a group for it
    would put unrelated problems behind one owner's single decision.
    """
    over = set(over_capacity_suppliers or ())
    buckets: dict = {}
    for a in alerts:
        sid = a.supplier_id
        gk = (sid, a.severity) if sid else (f"__part__{a.part}", a.severity)
        buckets.setdefault(gk, []).append(a)

    out = []
    for (sid, sev), group in buckets.items():
        standalone = isinstance(sid, str) and sid.startswith("__part__")
        supplier = None if standalone else sid
        parts = sorted({a.part for a in group})
        oc = supplier in over
        worst = min(a.days_of_slack for a in group)
        n_single = sum(1 for a in group if a.single_sourced)
        any_single = n_single > 0
        if standalone:
            reason = group[0].reason
        else:
            reason = (f"{len(parts)} part(s) from {supplier} at {sev}: worst "
                      f"slack {worst:.0f} days")
            if n_single:
                reason += f", {n_single} single-sourced"
            if oc:
                reason += ", supplier over capacity"
        t = TIERS[sev]
        key = hashlib.sha1(
            f"{supplier or group[0].part}|{sev}|{'|'.join(parts)}".encode()
        ).hexdigest()[:12]
        out.append(Incident(
            supplier_id=supplier, severity=sev, parts=parts, n_parts=len(parts),
            value_at_risk=sum(a.value_at_risk for a in group),
            units_at_risk=sum(a.units_at_risk for a in group),
            worst_slack_days=worst, n_single_sourced=n_single,
            any_single_sourced=any_single, over_capacity=oc,
            max_risk_score=max(a.risk_score for a in group),
            route_to=route(sev, any_single, oc),
            response_hours=t["response_hours"],
            escalate_after_hours=t["escalate_after_hours"],
            escalation_chain=t["chain"], reason=reason, key=key))

    out.sort(key=lambda i: (SEV_ORDER[i.severity], -i.value_at_risk))
    return out


def load_by_role(incidents: list) -> dict:
    per: dict = {}
    for i in incidents:
        d = per.setdefault(i.route_to, {"total": 0, "P1": 0, "P2": 0, "P3": 0,
                                        "P4": 0, "value": 0.0, "parts": 0})
        d["total"] += 1
        d[i.severity] += 1
        d["value"] += i.value_at_risk
        d["parts"] += i.n_parts
    return per


def load_check(alerts: list, incidents: list, *, per_owner_limit: int = 15,
               p1_limit: int = 10) -> dict:
    """Does the queue fit in a working day? Both views, side by side.

    The limits are stated because they are judgements, not findings: fifteen open
    items is roughly a day's worth of chasing for one buyer, and ten P1s at a
    four-hour SLA is already more than one person can honour. A policy that
    exceeds them has not prioritised, whatever it calls its top tier.
    """
    from routing import load_by_role as alert_load
    before, after = alert_load(alerts), load_by_role(incidents)

    def worst(d, sev=None):
        if not d:
            return ("-", 0)
        k = max(d, key=lambda r: d[r][sev] if sev else d[r]["total"])
        return (k, d[k][sev] if sev else d[k]["total"])

    b_role, b_n = worst(before)
    a_role, a_n = worst(after)
    b_p1_role, b_p1 = worst(before, "P1")
    a_p1_role, a_p1 = worst(after, "P1")
    return {
        "alerts": len(alerts), "incidents": len(incidents),
        "reduction_pct": 100 * (1 - len(incidents) / max(len(alerts), 1)),
        "parts_covered_before": len({a.part for a in alerts}),
        "parts_covered_after": len({p for i in incidents for p in i.parts}),
        "value_before": sum(a.value_at_risk for a in alerts),
        "value_after": sum(i.value_at_risk for i in incidents),
        "p1_alerts": sum(1 for a in alerts if a.severity == "P1"),
        "p1_incidents": sum(1 for i in incidents if i.severity == "P1"),
        "busiest_before": {"role": b_role, "items": b_n},
        "busiest_after": {"role": a_role, "items": a_n},
        "worst_p1_before": {"role": b_p1_role, "items": b_p1},
        "worst_p1_after": {"role": a_p1_role, "items": a_p1},
        "per_owner_limit": per_owner_limit, "p1_limit": p1_limit,
        "passes_before": b_n <= per_owner_limit and b_p1 <= p1_limit,
        "passes_after": a_n <= per_owner_limit and a_p1 <= p1_limit,
        "load_before": before, "load_after": after,
        "caveat": ("grouping reduces the COUNT an owner sees and not the work; "
                   "the parts and the value at risk are carried through "
                   "unchanged so the comparison is like for like"),
    }


def concentration(alerts: list, top: int = 5) -> list:
    """Which suppliers drive the alert count -- the evidence for grouping at all.

    If alerts were spread evenly across suppliers, grouping would buy nothing and
    the right fix would be a different severity policy. This says which it is.
    """
    per: dict = {}
    for a in alerts:
        if not a.supplier_id:
            continue
        d = per.setdefault(a.supplier_id, {"alerts": 0, "P1": 0, "value": 0.0,
                                           "parts": set()})
        d["alerts"] += 1
        d["value"] += a.value_at_risk
        d["parts"].add(a.part)
        if a.severity == "P1":
            d["P1"] += 1
    rows = [{"supplier_id": k, "alerts": v["alerts"], "P1": v["P1"],
             "n_parts": len(v["parts"]), "value_at_risk": v["value"]}
            for k, v in per.items()]
    rows.sort(key=lambda r: -r["alerts"])
    total = sum(r["alerts"] for r in rows) or 1
    for r in rows:
        r["share_of_alerts"] = r["alerts"] / total
    head = rows[:top]
    return {"top": head, "n_suppliers": len(rows),
            "top_share": sum(r["alerts"] for r in head) / total}


# ---------------------------------------------------------------------------
# calibrating the policy to the capacity that has to absorb it
# ---------------------------------------------------------------------------

def calibrate(rows: list, *, p1_limit: int = 10, per_owner_limit: int = 15,
              over_capacity_suppliers=None,
              ratios=(0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01)) -> dict:
    """Find the P1 threshold whose queue an owner can actually work.

    This exists because building the fix the README named -- group by supplier --
    and MEASURING it showed the fix was not sufficient, and the reason was that
    the stated diagnosis was only half right. Grouping helps by exactly as much
    as alerts are concentrated on a few suppliers, and in this dataset they are
    not: the top five suppliers account for a fifth of the alerts across nearly
    fifty of them. What actually floods the queue is the severity policy putting
    two fifths of the catalogue into its top tier.

    So the remaining lever is the threshold. Sweeping it is not tuning a number
    until the report looks good -- the constraint is external (how many items one
    person can chase in a day) and the sweep says what has to be true of the
    policy for that constraint to hold, plus what it costs: every part that drops
    out of P1 is a part somebody decided not to treat as urgent, and the value
    that moves with them is the size of that bet.
    """
    from routing import build_alerts, dedupe
    import routing as _rt

    out = []
    original = _rt.severity_of
    total_value = None
    try:
        for ratio in ratios:
            def scoped(days, units, single, lead, _r=ratio):
                res = original(days, units, single, lead)
                if res[0] != "P1":
                    return res
                # Keep the original tiering, but demand a tighter slack ratio
                # before the top tier fires. Everything it demotes lands in P2,
                # which has an SLA rather than no SLA.
                if days > 0 and (days / max(lead, 1e-9)) >= _r:
                    return "P2", res[1] + f" (below the P1 ratio of {_r:.2f})"
                return res

            _rt.severity_of = scoped
            alerts, _ = dedupe(build_alerts(
                rows, over_capacity_suppliers=over_capacity_suppliers))
            inc = group_by_supplier(
                alerts, over_capacity_suppliers=over_capacity_suppliers)
            lc = load_check(alerts, inc, per_owner_limit=per_owner_limit,
                            p1_limit=p1_limit)
            p1_value = sum(i.value_at_risk for i in inc if i.severity == "P1")
            if total_value is None:
                total_value = lc["value_after"]
            out.append({
                "p1_ratio": ratio, "alerts": lc["alerts"],
                "incidents": lc["incidents"],
                "p1_alerts": lc["p1_alerts"], "p1_incidents": lc["p1_incidents"],
                "worst_p1_owner": lc["worst_p1_after"]["role"],
                "worst_p1_items": lc["worst_p1_after"]["items"],
                "busiest_items": lc["busiest_after"]["items"],
                "p1_value_at_risk": p1_value,
                "p1_value_share": p1_value / max(total_value, 1e-9),
                "p1_passes": lc["worst_p1_after"]["items"] <= p1_limit,
                "fully_passes": lc["passes_after"]})
    finally:
        _rt.severity_of = original

    ok = [r for r in out if r["p1_passes"]]
    return {"sweep": out, "p1_limit": p1_limit,
            "per_owner_limit": per_owner_limit,
            "first_passing_p1": ok[0] if ok else None,
            "any_fully_passes": any(r["fully_passes"] for r in out),
            "caveat": ("the P1 threshold is calibrated against how many items an "
                       "owner can chase in a day, which is an external "
                       "constraint and not a property of the data; the parts "
                       "that leave P1 land in P2, which has an SLA, and the "
                       "value that moves with them is the size of the bet")}


def value_triage(incidents: list, *, top_n: int = 10) -> dict:
    """How much of the value at risk sits in the top N incidents by value.

    The reason this is the right question. Grouping by supplier was the fix the
    README named, and building it showed the queue is still four times too long
    -- because the alerts are not concentrated on a few suppliers (the top five
    are a fifth of them, across nearly fifty) and because a large share of the
    parts are genuinely already below their lead time, which no threshold can
    argue away.

    So the queue is not too long because the policy mis-ranks; it is too long
    because the situation is bad. The only honest lever left is to stop
    pretending everything urgent can be worked, and say what the top N covers.
    If ten items carry most of the value, "work these ten today, the rest on a
    schedule" is a defensible policy. If they carry a tenth of it, the answer is
    more buyers, and the analysis should say so rather than tuning a threshold
    until the table looks acceptable.
    """
    p1 = sorted([i for i in incidents if i.severity == "P1"],
                key=lambda i: -i.value_at_risk)
    total_all = sum(i.value_at_risk for i in incidents) or 1e-9
    total_p1 = sum(i.value_at_risk for i in p1) or 1e-9
    head = p1[:top_n]
    head_value = sum(i.value_at_risk for i in head)
    cum, k80 = 0.0, None
    for n, i in enumerate(p1, 1):
        cum += i.value_at_risk
        if k80 is None and cum >= 0.80 * total_p1:
            k80 = n
    return {
        "n_p1_incidents": len(p1), "top_n": top_n,
        "top_n_value": head_value,
        "top_n_share_of_p1_value": head_value / total_p1,
        "top_n_share_of_all_value": head_value / total_all,
        "top_n_parts": sum(i.n_parts for i in head),
        "parts_in_p1": sum(i.n_parts for i in p1),
        "incidents_for_80pct_of_p1_value": k80,
        "p1_value": total_p1, "all_value": total_all,
        "verdict": ("workable: a top-N queue carries most of the exposure"
                    if head_value / total_p1 >= 0.6 else
                    "not workable by triage: the exposure is spread across more "
                    "incidents than one owner can work, and the answer is "
                    "capacity rather than a threshold"),
    }
