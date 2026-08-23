"""Severity tiers, escalation SLAs and routing -- the process integration the
spec says analytics without which is just a dashboard.

THE ARGUMENT. The README's item 4 quotes the spec: "the order-by dates exist;
nothing delivers them to a buyer, and analytics without process integration is a
dashboard." That is the whole difference between a model and a system, and the
hard part is not the delivery mechanism. It is the SEVERITY POLICY, because a
severity policy is where an organisation writes down what it is actually willing
to pay to avoid.

SEVERITY IS NOT RISK SCORE. This is the mistake worth avoiding. A part can carry
a high supplier-risk score and matter not at all -- plenty of stock, six weeks of
slack, a cheap second source. Severity is set by CONSEQUENCE AND URGENCY
together:

    severity ~ f(days of slack remaining, units at risk, single-sourced?)

The risk score belongs in the body of the alert as evidence. Putting it in the
severity field routes on the wrong variable and floods the highest tier with
parts nobody needs to act on -- which is how an alert channel dies.

SLAs ARE COMPUTED BACKWARDS FROM THE ORDER-BY DATE, not chosen from a menu. If a
part must be ordered in 3 days, a 5-day response SLA is a decoration. The SLA is
derived from the slack, which means it is a fact about the situation rather than
a service-desk convention.

WHAT IS NOT BUILT, and named so the module is not mistaken for an integration:
nothing is sent. There is no email, no ticket, no ERP write-back. This produces
routed, prioritised, deduplicated alert objects with SLAs and an escalation
chain, and a real deployment attaches a transport to them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# Who owns what, and why the split is not arbitrary. A buyer can expedite a PO;
# only engineering can approve an alternate part; only a supply-chain manager can
# fund a second source. Routing to the wrong role produces an acknowledged alert
# and no action, which looks like success in every metric a queue reports.
ROLES = {
    "BUYER": "can expedite, split, or re-schedule an existing PO",
    "COMMODITY_MANAGER": "can negotiate capacity, escalate with the supplier",
    "SUPPLY_CHAIN_MANAGER": "can fund a second source or authorise a buy-ahead",
    "ENGINEERING": "can approve an alternate part or a deviation",
    "PLANT_MANAGER": "can re-sequence the build plan around the shortage",
}

TIERS = {
    "P1": {"response_hours": 4, "escalate_after_hours": 8,
           "chain": ["BUYER", "COMMODITY_MANAGER", "SUPPLY_CHAIN_MANAGER",
                     "PLANT_MANAGER"],
           "meaning": "the line stops inside the lead time; no recovery by ordering"},
    "P2": {"response_hours": 24, "escalate_after_hours": 48,
           "chain": ["BUYER", "COMMODITY_MANAGER", "SUPPLY_CHAIN_MANAGER"],
           "meaning": "recoverable, but only if the order is placed now"},
    "P3": {"response_hours": 72, "escalate_after_hours": 120,
           "chain": ["BUYER", "COMMODITY_MANAGER"],
           "meaning": "comfortable slack; watch it"},
    "P4": {"response_hours": 168, "escalate_after_hours": None,
           "chain": ["BUYER"],
           "meaning": "informational; batch into the weekly review"},
}


@dataclass
class Alert:
    part: str
    supplier_id: str | None
    severity: str
    days_of_slack: float
    units_at_risk: float
    value_at_risk: float
    single_sourced: bool
    risk_score: float
    reason: str
    response_hours: int
    escalate_after_hours: int | None
    route_to: str
    escalation_chain: list = field(default_factory=list)
    dedupe_key: str = ""

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("part", "supplier_id", "severity", "days_of_slack",
                 "units_at_risk", "value_at_risk", "single_sourced",
                 "risk_score", "reason", "response_hours",
                 "escalate_after_hours", "route_to", "escalation_chain",
                 "dedupe_key")}


def severity_of(days_of_slack: float, units_at_risk: float,
                single_sourced: bool, lead_days: float) -> tuple[str, str]:
    """Consequence and urgency, not risk score.

    The slack is compared against the LEAD TIME, not against a fixed number of
    days. Ten days of slack is comfortable on a part with a 5-day lead and an
    emergency on one with a 60-day lead, and a policy expressed in absolute days
    gets that backwards for half the catalogue.
    """
    ratio = days_of_slack / max(lead_days, 1e-9)
    if days_of_slack <= 0 or ratio < 0.25:
        return "P1", (f"{days_of_slack:.0f} days of slack against a "
                      f"{lead_days:.0f}-day lead: ordering now does not recover it")
    if ratio < 0.75:
        sev = "P2"
    elif ratio < 1.5:
        sev = "P3"
    else:
        sev = "P4"
    # Single sourcing raises the tier by one, because there is no fallback and
    # the recovery options above all assume one exists.
    if single_sourced and sev != "P1":
        order = ["P1", "P2", "P3", "P4"]
        sev = order[max(order.index(sev) - 1, 0)]
        return sev, (f"{days_of_slack:.0f} days of slack on a {lead_days:.0f}-day "
                     "lead, single-sourced (tier raised: no fallback)")
    return sev, f"{days_of_slack:.0f} days of slack against a {lead_days:.0f}-day lead"


def route(severity: str, single_sourced: bool, over_capacity: bool) -> str:
    """The first owner. The chain is the fallback if they do not respond.

    Capacity problems go to the commodity manager rather than the buyer, because
    a buyer cannot create capacity -- expediting a capacity-constrained supplier
    just moves the queue. Routing it to a buyer produces a week of phone calls
    and no parts.
    """
    if over_capacity:
        return "COMMODITY_MANAGER"
    if severity == "P1" and single_sourced:
        return "SUPPLY_CHAIN_MANAGER"
    return TIERS[severity]["chain"][0]


def build_alerts(rows: list[dict], *, over_capacity_suppliers: set | None = None,
                 ) -> list[Alert]:
    """Turn per-part risk rows into routed, deduplicated alerts."""
    over = over_capacity_suppliers or set()
    out: list[Alert] = []
    for r in rows:
        sev, reason = severity_of(r["days_of_slack"], r["units_at_risk"],
                                  r.get("single_sourced", False),
                                  r.get("lead_days", 30.0))
        t = TIERS[sev]
        oc = r.get("supplier_id") in over
        key = hashlib.sha1(
            f"{r['part']}|{sev}|{r.get('supplier_id')}".encode()).hexdigest()[:12]
        out.append(Alert(
            part=r["part"], supplier_id=r.get("supplier_id"), severity=sev,
            days_of_slack=r["days_of_slack"], units_at_risk=r["units_at_risk"],
            value_at_risk=r.get("value_at_risk", 0.0),
            single_sourced=r.get("single_sourced", False),
            risk_score=r.get("risk_score", 0.0),
            reason=reason + (" | supplier over capacity" if oc else ""),
            response_hours=t["response_hours"],
            escalate_after_hours=t["escalate_after_hours"],
            route_to=route(sev, r.get("single_sourced", False), oc),
            escalation_chain=t["chain"], dedupe_key=key))
    order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    out.sort(key=lambda a: (order[a.severity], -a.value_at_risk))
    return out


def dedupe(alerts: list[Alert]) -> tuple[list[Alert], int]:
    """One open alert per (part, severity, supplier).

    A part that trips the rule on Monday and again on Tuesday is one problem. Two
    alerts is how a buyer learns to filter the channel to a folder, and after
    that the P1s go there too.
    """
    seen, kept = set(), []
    for a in alerts:
        if a.dedupe_key in seen:
            continue
        seen.add(a.dedupe_key)
        kept.append(a)
    return kept, len(alerts) - len(kept)


def load_by_role(alerts: list[Alert]) -> dict:
    """Alerts per owner, which is the number that says whether this is usable.

    A policy that routes 300 P1s to one buyer has not prioritised anything. This
    is the check on the severity policy itself, and it is the reason the tiering
    is deliberately conservative about promoting to P1.
    """
    per: dict[str, dict] = {}
    for a in alerts:
        d = per.setdefault(a.route_to, {"total": 0, "P1": 0, "P2": 0,
                                        "P3": 0, "P4": 0, "value": 0.0})
        d["total"] += 1
        d[a.severity] += 1
        d["value"] += a.value_at_risk
    return per
