"""The supply base, built from real public data plus a few clearly simulated layers.

REAL (from data/raw, see data_loaders.py):
  * BOM structure, part costs, product demand and every purchased part's
    lead-time distribution: Willems (2008) chain 24, power-driven hand tools.
    17 finished products, 31 in-house sub-assemblies, 209 purchased parts.
  * Supplier behaviour: USAID SCMS delivery history. Each supplier is a real
    SCMS vendor with its real lead times, real on-time rate against the
    scheduled date, real manufacturing sites and real order history.

ASSUMED (stated, not data):
  * Quantity per parent is 1 on every BOM line (Willems has no quantities).
  * Which SCMS vendor supplies which Willems part: matched by lead-time band,
    see `data_loaders.map_vendors_to_parts`. The two datasets are unrelated.

SIMULATED (seeded, documented rule, labelled as simulated everywhere):
  * On-hand stock and open orders. No public dataset has them.
  * Minimum order quantities.
  * The ORIGINAL promise date for the OTIF trap. SCMS keeps one scheduled date
    per line; nothing public keeps the original next to the reschedule.
  * Planted disruptions, used only to score the deterioration detector. They
    are planted into a "scoring sandbox": order histories resampled from each
    real vendor's own lead times, so the baseline is real and the disruption is
    the only thing we added.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import data_loaders as DL

WEEKS = 13          # the materials-planning standard horizon; see README
SEED = 20260819
SANDBOX_YEARS = 3
RECENT_DAYS = 365      # detector: recent window
BASELINE_DAYS = 1095   # detector: baseline starts this many days back


@dataclass
class Supplier:
    """A real SCMS vendor. Every number here is measured from its history."""
    supplier_id: str
    name: str
    n_lines: int
    lead_mean_days: float
    lead_sd_days: float
    lead_p50_days: float
    lead_p95_days: float
    on_time_vs_scheduled: float
    exact_on_scheduled_date: float
    late_rate: float
    sole_source_share: float
    sites: list
    lines_per_year: float
    fat_tail: bool                      # P95 / P50 lead time >= 2.5
    rel_late: np.ndarray = field(default_factory=lambda: np.zeros(1))
    habitual_rescheduler: bool = False  # SIMULATED trait, for the OTIF trap only


@dataclass
class Component:
    part: str
    level: int                  # 0 product, 1 in-house sub-assembly, 2 purchased
    supplier_id: str | None
    single_sourced: bool
    unit_cost: float
    on_hand: float              # SIMULATED for purchased parts
    moq: float                  # SIMULATED
    depth: int = 0              # Willems relDepth
    lead_values: tuple = (0.0,)
    lead_probs: tuple = (1.0,)
    lead_mean: float = 0.0
    lead_sd: float = 0.0
    lead_p95: float = 0.0
    has_distribution: bool = False


@dataclass
class BomLine:
    parent: str
    child: str
    qty_per: float              # always 1.0: Willems has no quantities


@dataclass
class PurchaseOrder:
    """An OPEN order (simulated). Receipt history lives in `Delivery`."""
    po_id: str
    part: str
    supplier_id: str
    qty: float
    promise_day: float
    original_promise_day: float
    received_day: float | None = None


@dataclass
class Delivery:
    """One vendor delivery line. Days are relative to day 0 = latest PO date."""
    supplier_id: str
    order_day: float
    promise_day: float          # latest promise: the SCMS scheduled date (real)
    original_promise_day: float  # SIMULATED for habitual reschedulers
    received_day: float
    source: str                 # "real" or "sandbox"
    site: str = ""


@dataclass
class SupplyBase:
    suppliers: dict[str, Supplier]
    components: dict[str, Component]
    bom: list[BomLine]
    pos: list[PurchaseOrder]
    products: list[str]
    demand: dict[str, np.ndarray]           # product -> units per week (plan)
    demand_sd_day: dict[str, float]         # product -> std dev of daily demand
    history: list[Delivery] = field(default_factory=list)   # real SCMS lines
    sandbox: list[Delivery] = field(default_factory=list)   # scoring sandbox
    planted_disruptions: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    _kids: dict = field(default_factory=dict, repr=False)
    _parents: dict = field(default_factory=dict, repr=False)
    _req: dict | None = field(default=None, repr=False)

    def index(self) -> None:
        self._kids = defaultdict(list)
        self._parents = defaultdict(list)
        for b in self.bom:
            self._kids[b.parent].append(b)
            self._parents[b.child].append(b)
        self._req = None

    def children(self, parent: str) -> list[BomLine]:
        return self._kids.get(parent, [])

    def parents_of(self, child: str) -> list[BomLine]:
        return self._parents.get(child, [])


def discrete_quantile(values, probs, q: float) -> float:
    order = np.argsort(values)
    v = np.asarray(values, float)[order]
    c = np.cumsum(np.asarray(probs, float)[order])
    return float(v[min(int(np.searchsorted(c, q - 1e-12)), len(v) - 1)])


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(seed: int = SEED, chain: int = DL.CHAIN) -> SupplyBase:
    """Load the real data from data/raw and build the supply base."""
    if not DL.raw_data_present():
        raise FileNotFoundError(
            "data/raw is missing. Run `python download_data.py` first.")
    rows, arcs = DL.read_willems_xls(DL.WILLEMS_XLS, chain)
    lines, log = DL.clean_scms(DL.read_scms_csv(DL.SCMS_CSV))
    return build_from(DL.chain_structure(rows, arcs), lines, seed=seed,
                      meta={"chain": chain, "scms_cleaning": log})


def build_from(chain: dict, scms_lines: list[dict], seed: int = SEED,
               min_vendor_lines: int = DL.MIN_VENDOR_LINES,
               meta: dict | None = None) -> SupplyBase:
    """Pure constructor: a parsed Willems chain + cleaned SCMS lines."""
    rng = np.random.default_rng(seed)

    # ---- suppliers: real SCMS vendors ------------------------------------
    vt = DL.vendor_table(scms_lines, min_vendor_lines)
    names = sorted(vt, key=lambda v: (vt[v]["lead_p50"], v))
    suppliers: dict[str, Supplier] = {}
    name_to_id = {}
    for i, v in enumerate(names):
        s = vt[v]
        sid = f"V{i + 1:02d}"
        name_to_id[v] = sid
        mine = [x for x in scms_lines if x["vendor"] == v]
        rel = np.array([max(x["late_days"], 0.0) / x["sched_lead_days"]
                        if x["sched_lead_days"] > 0 else 0.0 for x in mine])
        suppliers[sid] = Supplier(
            supplier_id=sid, name=v, n_lines=s["n_lines"],
            lead_mean_days=s["lead_mean"], lead_sd_days=s["lead_sd"],
            lead_p50_days=s["lead_p50"], lead_p95_days=s["lead_p95"],
            on_time_vs_scheduled=s["on_time_vs_scheduled"],
            exact_on_scheduled_date=s["exact_on_scheduled_date"],
            late_rate=s["late_rate"], sole_source_share=s["sole_source_share"],
            sites=s["sites"], lines_per_year=s["lines_per_year"],
            fat_tail=bool(s["lead_p95"] / max(s["lead_p50"], 1e-9) >= 2.5),
            rel_late=rel,
            # SIMULATED: a third of vendors habitually move the promise date.
            habitual_rescheduler=(i % 3 == 0))

    # ---- components: real Willems parts, sub-assemblies, products ---------
    pi = chain["part_info"]
    mapping = DL.map_vendors_to_parts(
        {p: pi[p]["lead_mean"] for p in chain["parts"]},
        {name_to_id[v]: vt[v]["lead_p50"] for v in names})
    components: dict[str, Component] = {}
    for p in chain["products"]:
        components[p] = Component(p, 0, None, False,
                                  chain["product_cost"].get(p, 0.0), 0.0, 0.0)
    for s in chain["subs"]:
        components[s] = Component(s, 1, None, False, 0.0, 0.0, 0.0,
                                  depth=chain["sub_depth"].get(s, 0))
    for p in chain["parts"]:
        info = pi[p]
        sid = mapping.get(p)
        sup = suppliers.get(sid)
        components[p] = Component(
            p, 2, sid,
            single_sourced=bool(sup and sup.sole_source_share >= 0.5),
            unit_cost=info["unit_cost"], on_hand=0.0,
            moq=float(rng.choice([1, 25, 50, 100, 250])),      # SIMULATED
            depth=info["depth"], lead_values=info["lead_values"],
            lead_probs=info["lead_probs"], lead_mean=info["lead_mean"],
            lead_sd=info["lead_sd"],
            lead_p95=discrete_quantile(info["lead_values"], info["lead_probs"], 0.95),
            has_distribution=info["has_distribution"])

    # Willems arcs run component -> the stage that consumes it, so the
    # destination is the parent.
    bom = [BomLine(parent=b, child=a, qty_per=1.0) for a, b in chain["bom"]]
    products = list(chain["products"])
    demand = {p: np.full(WEEKS, float(round(chain["demand_avg"][p] * 7)))
              for p in products}
    base = SupplyBase(suppliers, components, bom, [], products, demand,
                      dict(chain["demand_sd"]), meta=dict(meta or {}))
    base.index()

    # ---- real delivery history -------------------------------------------
    kept = [x for x in scms_lines if x["vendor"] in name_to_id]
    day0 = max(x["po_date"] for x in kept) if kept else None
    for x in sorted(kept, key=lambda x: (x["po_date"], x["vendor"], x["lead_days"])):
        sid = name_to_id[x["vendor"]]
        od = float((x["po_date"] - day0).days)
        base.history.append(Delivery(
            sid, od, od + x["sched_lead_days"], od + x["sched_lead_days"],
            od + x["lead_days"], "real", x["site"]))
    _simulate_original_promises(base, rng)

    _seed_inventory_and_orders(base, rng)
    _build_sandbox(base, rng)
    _plant_disruptions(base, rng)

    base.meta.update({
        "day0": day0.isoformat() if day0 else None,
        "vendor_mapping_rule": "lead-time band (rank quantile), an assumption",
        "parts_with_lead_distribution": sum(
            1 for c in components.values() if c.level == 2 and c.has_distribution),
        "chain_classes": chain.get("classes"),
        "demand_stages_split": chain.get("demand_stages_split", 0),
        "part_arcs_not_into_manufacturing": chain.get(
            "part_arcs_not_into_manufacturing", 0),
        "scms_vendors_kept": len(suppliers),
        "scms_lines_kept": len(base.history),
    })
    return base


def _simulate_original_promises(base: SupplyBase, rng) -> None:
    """SIMULATED. SCMS stores one scheduled date; we treat it as the LATEST
    promise. For the third of vendors marked as habitual reschedulers, half of
    their lines are given an original promise that was earlier by 10-50% of
    the quoted lead time. Everyone else's original promise equals the latest.
    """
    for d in base.history:
        s = base.suppliers[d.supplier_id]
        if s.habitual_rescheduler and rng.random() < 0.5:
            quoted = d.promise_day - d.order_day
            d.original_promise_day = d.promise_day - quoted * float(rng.uniform(0.1, 0.5))


def _seed_inventory_and_orders(base: SupplyBase, rng) -> None:
    """SIMULATED on-hand stock and open orders, set against the real plan.

    Rule (unchanged from the first version of this project so results are
    comparable): each purchased part gets a COVERAGE of its 13-week requirement.
    12% of parts are drawn short (coverage 0.45-0.95), the rest comfortable
    (1.05-1.9). 35-75% of the covered quantity is on hand today; the rest is
    split over 0-3 open orders arriving on days 2-91. A part with no open order
    and not enough stock is short by construction.
    """
    total_req = defaultdict(float)
    req = unit_requirements(base)
    for prod in base.products:
        units = float(base.demand[prod].sum())
        for part, q in req[prod].items():
            total_req[part] += q * units

    n = 0
    for part, c in base.components.items():
        if c.level != 2:
            continue
        orders = []
        for _ in range(int(rng.integers(0, 4))):
            n += 1
            arrive = float(rng.uniform(2, WEEKS * 7))
            orders.append(PurchaseOrder(f"PO-{n:05d}", part, c.supplier_id, 0.0,
                                        promise_day=arrive,
                                        original_promise_day=arrive))
        r = total_req.get(part, 0.0)
        coverage = (float(rng.uniform(0.45, 0.95)) if rng.random() < 0.12
                    else float(rng.uniform(1.05, 1.9)))
        share = float(rng.uniform(0.35, 0.75))
        c.on_hand = float(np.floor(r * coverage * share))
        remaining = max(0.0, r * coverage - c.on_hand)
        for po in orders:
            po.qty = float(np.ceil(remaining / len(orders)))
        base.pos.extend(orders)


def _build_sandbox(base: SupplyBase, rng) -> None:
    """Three years of order history per vendor, resampled from its REAL lines.

    Order count = the vendor's real orders per year x 3. Order dates are
    uniform over the last 1,095 days; each order takes a real (lead, scheduled
    lead) pair drawn from that vendor's history. No drift exists here unless we
    plant it, which is what makes precision and recall scoreable.

    Three years, not two, because real SCMS vendors order rarely (median about
    ten lines a year): a 120-day recent window, which the simulated version
    used, leaves most vendors with too few orders to judge at all.
    """
    by = defaultdict(list)
    for d in base.history:
        by[d.supplier_id].append(d)
    for sid in sorted(base.suppliers):
        s = base.suppliers[sid]
        real = by.get(sid, [])
        if not real:
            continue
        n = max(1, int(round(SANDBOX_YEARS * s.lines_per_year)))
        idx = rng.integers(0, len(real), size=n)
        days = np.sort(-rng.uniform(1, SANDBOX_YEARS * 365, size=n))
        for od, i in zip(days, idx):
            r = real[int(i)]
            lead = r.received_day - r.order_day
            quoted = r.promise_day - r.order_day
            base.sandbox.append(Delivery(sid, float(od), float(od + quoted),
                                         float(od + quoted), float(od + lead),
                                         "sandbox", r.site))


def _plant_disruptions(base: SupplyBase, rng) -> None:
    """SIMULATED disruptions, planted into the sandbox only.

      lead_time_doubling -- every order after onset takes one extra median lead
      tail_blowout       -- 35% of orders after onset take 1.2-2.4 extra median
                            leads; the median barely moves, the P95 explodes
    Onset falls 200-300 days before day 0, inside the detector's one-year
    recent window, so most of that window's orders are affected.
    """
    ids = sorted(base.suppliers)
    chosen = rng.choice(ids, size=min(6, len(ids)), replace=False)
    for i, sid in enumerate(chosen):
        kind = "lead_time_doubling" if i % 2 == 0 else "tail_blowout"
        s = base.suppliers[str(sid)]
        onset = float(rng.uniform(-300, -200))
        for d in base.sandbox:
            if d.supplier_id != sid or d.order_day < onset:
                continue
            if kind == "lead_time_doubling":
                d.received_day += s.lead_p50_days
            elif rng.random() < 0.35:
                d.received_day += s.lead_p50_days * float(rng.uniform(1.2, 2.4))
        base.planted_disruptions.append(
            {"supplier_id": str(sid), "kind": kind, "onset_day": onset})


# ---------------------------------------------------------------------------
# BOM
# ---------------------------------------------------------------------------

def explode(base: SupplyBase, product: str, qty: float) -> dict[str, float]:
    """Total quantity of every component needed to build `qty` of `product`.

    Multi-level: a sub-assembly reached along two paths contributes its
    children twice. Getting this wrong understates exactly the shared parts.
    """
    need: dict[str, float] = {}

    def walk(parent: str, mult: float) -> None:
        for b in base.children(parent):
            need[b.child] = need.get(b.child, 0.0) + b.qty_per * mult
            walk(b.child, b.qty_per * mult)

    walk(product, qty)
    return need


def unit_requirements(base: SupplyBase) -> dict[str, dict[str, float]]:
    """Per-unit component requirement for each product. Computed once."""
    if base._req is None:
        base._req = {p: explode(base, p, 1.0) for p in base.products}
    return base._req


def part_daily_demand(base: SupplyBase) -> dict[str, float]:
    """Units per day of each component, from the plan through the BOM."""
    out = defaultdict(float)
    req = unit_requirements(base)
    for p in base.products:
        per_day = float(np.mean(base.demand[p])) / 7.0
        for part, q in req[p].items():
            out[part] += q * per_day
    return dict(out)


def part_daily_demand_sd(base: SupplyBase) -> dict[str, float]:
    """Std dev of daily component demand, from the real product demand sd.
    Products are treated as independent (Willems gives no correlations)."""
    var = defaultdict(float)
    req = unit_requirements(base)
    for p in base.products:
        sd = base.demand_sd_day.get(p, 0.0)
        for part, q in req[p].items():
            var[part] += (q * sd) ** 2
    return {k: v ** 0.5 for k, v in var.items()}
