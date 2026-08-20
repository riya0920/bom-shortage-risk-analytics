"""A simulated manufacturing supply base with multi-level BOMs and ground truth.

What makes this manufacturing supply chain rather than generic supply chain: the
BOM. A $2 fastener at 100% availability-criticality stops a $2M machine, and the
only way to see that is to propagate component risk through a bill of materials to
END-ITEM BUILDABILITY. Supplier on-time-delivery percentages cannot answer the one
question that matters, which is CAN WE BUILD THE PLAN.

Generated with ground truth throughout:
  * 4 end products with multi-level BOMs (~200 components, shared across products --
    shared components are where the shortage maths gets interesting)
  * ~50 suppliers, some single-sourced and flagged as such
  * per supplier-part lead-time distributions with a mean AND a variance, some
    with fat tails, because the variance is what actually bites
  * open purchase orders with promise dates and (for history) actual receipts
  * quality rejection rates that reduce effective supply
  * planted disruptions: specific suppliers whose lead time doubles at a known
    week, so early-warning precision and recall are scoreable
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WEEKS = 13  # the materials-planning standard horizon; see README for why


@dataclass
class Supplier:
    supplier_id: str
    name: str
    lead_mean_days: float
    lead_sd_days: float
    fat_tail: bool
    quality_reject_rate: float
    financial_score: float          # 0-1, higher = healthier; a proxy, not a rating


@dataclass
class Component:
    part: str
    level: int
    supplier_id: str | None         # None = manufactured in-house (sub-assembly)
    single_sourced: bool
    unit_cost: float
    on_hand: float
    moq: float


@dataclass
class BomLine:
    parent: str
    child: str
    qty_per: float


@dataclass
class PurchaseOrder:
    po_id: str
    part: str
    supplier_id: str
    qty: float
    promise_day: float
    original_promise_day: float     # kept separately: see the OTIF trap in README
    received_day: float | None


@dataclass
class SupplyBase:
    suppliers: dict[str, Supplier]
    components: dict[str, Component]
    bom: list[BomLine]
    pos: list[PurchaseOrder]
    products: list[str]
    demand: dict[str, np.ndarray]   # product -> units required per week
    planted_disruptions: list[dict] = field(default_factory=list)

    def children(self, parent: str) -> list[BomLine]:
        return [b for b in self.bom if b.parent == parent]

    def parents_of(self, child: str) -> list[BomLine]:
        return [b for b in self.bom if b.child == child]


def build(seed: int = 20260819, n_suppliers: int = 50) -> SupplyBase:
    rng = np.random.default_rng(seed)

    suppliers: dict[str, Supplier] = {}
    for i in range(n_suppliers):
        sid = f"SUP-{i+1:03d}"
        fat = rng.random() < 0.25
        mean = float(rng.uniform(14, 75))
        suppliers[sid] = Supplier(
            supplier_id=sid, name=f"Supplier {i+1:03d}",
            lead_mean_days=mean,
            # Variance scaled to the mean, plus a heavier tail for a quarter of
            # them. A supplier with a 30-day mean and a 25-day sd is a completely
            # different planning problem from one with a 30-day mean and a 3-day
            # sd, and an OTD percentage cannot tell them apart.
            lead_sd_days=float(mean * rng.uniform(0.08, 0.45)),
            fat_tail=fat,
            quality_reject_rate=float(rng.beta(1.4, 90)),
            financial_score=float(np.clip(rng.beta(6, 2), 0, 1)),
        )

    products = ["PRD-A", "PRD-B", "PRD-C", "PRD-D"]
    components: dict[str, Component] = {}
    bom: list[BomLine] = []

    # Level 0 = end products, level 1 = sub-assemblies (made in house),
    # level 2 = purchased parts. Shared components are created on purpose.
    shared_subs = [f"SUB-S{i+1:02d}" for i in range(8)]
    shared_parts = [f"PRT-S{i+1:03d}" for i in range(30)]

    for p in products:
        components[p] = Component(p, 0, None, False, 0.0, 0.0, 0.0)

    def make_part(part: str, level: int) -> Component:
        sid = str(rng.choice(list(suppliers)))
        single = rng.random() < 0.3
        cost = float(np.exp(rng.normal(2.0, 1.4)))
        return Component(part, level, sid, single, cost,
                         on_hand=float(rng.integers(0, 900)),
                         moq=float(rng.choice([1, 25, 50, 100, 250])))

    for s in shared_subs:
        components[s] = Component(s, 1, None, False, 0.0, float(rng.integers(0, 60)), 0.0)
    for p in shared_parts:
        components[p] = make_part(p, 2)

    idx = 0
    for pi, prod in enumerate(products):
        # 3-5 sub-assemblies per product, some shared
        n_sub = int(rng.integers(5, 9))
        subs = []
        for k in range(n_sub):
            if k < 2 and rng.random() < 0.7:
                sub = str(rng.choice(shared_subs))
            else:
                sub = f"SUB-{pi+1}{k+1:02d}"
                components[sub] = Component(sub, 1, None, False, 0.0,
                                            float(rng.integers(0, 60)), 0.0)
            subs.append(sub)
            bom.append(BomLine(prod, sub, float(rng.integers(1, 4))))

        for sub in set(subs):
            if any(b.parent == sub for b in bom):
                continue  # already exploded (shared sub-assembly)
            n_parts = int(rng.integers(10, 22))
            for _ in range(n_parts):
                if rng.random() < 0.45:
                    part = str(rng.choice(shared_parts))
                else:
                    idx += 1
                    part = f"PRT-{idx:04d}"
                    components[part] = make_part(part, 2)
                if not any(b.parent == sub and b.child == part for b in bom):
                    bom.append(BomLine(sub, part, float(rng.integers(1, 7))))

    # Open purchase orders + receipt history for lead-time analytics.
    pos: list[PurchaseOrder] = []
    n = 0
    for part, c in components.items():
        if c.level != 2 or c.supplier_id is None:
            continue
        s = suppliers[c.supplier_id]
        # history: 8-20 received POs
        for _ in range(int(rng.integers(8, 21))):
            n += 1
            # Spread order history from ~2 years ago to ~2 weeks ago. The first
            # version drew from -120 to -700 days, which left the "recent" window
            # used by the deterioration detector completely EMPTY -- so every
            # supplier was skipped for insufficient data and the detector scored
            # precision 0 / recall 0 while looking like it ran.
            order_day = float(-rng.uniform(15, 700))
            lead = _draw_lead(rng, s)
            received = order_day + lead
            original_promise = order_day + s.lead_mean_days

            # RESCHEDULING. Some suppliers, when they are going to be late, call
            # ahead and move the promise date -- and in most ERP configurations
            # that overwrites the promise field in place, so the original is gone.
            # Scored against the reschedule they look perfect; scored against what
            # they first committed to they look like what they are.
            #
            # `reschedules` is a supplier trait here: roughly a third of them do it
            # habitually. Without this the OTIF gap is identically zero for
            # everyone and the definitional trap cannot be demonstrated, which is
            # what the first version of this generator produced.
            habitual_rescheduler = (int(s.supplier_id.split("-")[1]) % 3) == 0
            latest_promise = original_promise
            if habitual_rescheduler and received > original_promise:
                # Move the promise to the date they now intend to hit -- which
                # they then hit. Scored against it, the delivery is on time.
                latest_promise = received + float(rng.uniform(0.0, 1.0))
            pos.append(PurchaseOrder(
                f"PO-{n:05d}", part, s.supplier_id, float(rng.integers(50, 600)),
                promise_day=latest_promise,
                original_promise_day=original_promise,
                received_day=received))
        # open orders arriving inside the horizon
        for _ in range(int(rng.integers(0, 4))):
            n += 1
            arrive = float(rng.uniform(2, WEEKS * 7))
            pos.append(PurchaseOrder(
                f"PO-{n:05d}", part, s.supplier_id, float(rng.integers(50, 900)),
                promise_day=arrive,
                original_promise_day=arrive - float(rng.choice([0, 0, 0, 7, 14])),
                received_day=None))

    demand = {p: np.array([float(rng.integers(20, 70)) for _ in range(WEEKS)])
              for p in products}

    base = SupplyBase(suppliers, components, bom, pos, products, demand)

    # ------------------------------------------------------------------
    # Calibrate supply to the plan, instead of drawing it out of the air.
    #
    # The first version set on-hand from U(0, 900) and PO quantities from
    # U(50, 900) with no reference to what the BOM actually needs. A product
    # needing 15 of a part per unit, at 45 units/week for 13 weeks, needs ~8,800
    # of that part -- so every part was short, buildability came out at 4 units
    # across the whole horizon, all three allocation policies tied at 0% fill, and
    # the percentile fan was a flat line at zero. A model in which nothing can be
    # built cannot demonstrate anything about shortages.
    #
    # Supply is now set as a COVERAGE FRACTION of the 13-week requirement, drawn so
    # that most parts are comfortable and a minority genuinely gate the plan --
    # which is what a real materials position looks like and what makes the
    # shortage attribution in buildability.py have something to attribute.
    # ------------------------------------------------------------------
    total_req: dict[str, float] = {}
    for prod in products:
        per_unit = explode(base, prod, 1.0)
        units = float(demand[prod].sum())
        for part, qty in per_unit.items():
            total_req[part] = total_req.get(part, 0.0) + qty * units

    open_by_part: dict[str, list[PurchaseOrder]] = {}
    for po in pos:
        if po.received_day is None:
            open_by_part.setdefault(po.part, []).append(po)

    for part, req in total_req.items():
        c = components.get(part)
        if c is None or c.level != 2:
            continue
        # Coverage: an explicit mixture rather than a single skewed draw.
        #
        # Tuning note worth keeping, because it IS the BOM-propagation lesson.
        # The first attempt drew coverage ~ Beta(5,2)*1.35, i.e. a median of about
        # 0.98 of the 13-week requirement per part. That sounds close to adequate.
        # End-item buildability came out at 11-33% fill, because a product needs
        # EVERY one of its ~60-100 parts: if each part independently has a ~50%
        # chance of being under-covered, the chance that all of them are adequate
        # is indistinguishable from zero. Part-level coverage does not translate
        # into end-item coverage, and that non-linearity is the entire reason this
        # analysis has to run through the BOM instead of over a supplier scorecard.
        #
        # So the generator now plants a MINORITY of genuinely short parts against a
        # comfortable majority, which is what a real materials position looks like
        # and what leaves the shortage attribution something specific to attribute.
        if rng.random() < 0.12:
            coverage = float(rng.uniform(0.45, 0.95))   # planted short part
        else:
            coverage = float(rng.uniform(1.05, 1.9))
        on_hand_share = float(rng.uniform(0.35, 0.75))
        c.on_hand = float(np.floor(req * coverage * on_hand_share))
        remaining = max(0.0, req * coverage - c.on_hand)
        orders = open_by_part.get(part, [])
        if orders:
            each = remaining / len(orders)
            for po in orders:
                po.qty = float(np.ceil(each))
        elif remaining > 0:
            # No open order and not enough on hand: this part is a shortage driver
            # by construction, which is exactly the case worth surfacing.
            pass

    _plant_disruptions(base, rng)
    return base


def _draw_lead(rng: np.random.Generator, s: Supplier) -> float:
    """Lead time realisation. Fat-tailed suppliers get an occasional long one."""
    v = rng.normal(s.lead_mean_days, s.lead_sd_days)
    if s.fat_tail and rng.random() < 0.12:
        v += abs(rng.normal(0, s.lead_mean_days * 0.9))
    return float(max(1.0, v))


def _plant_disruptions(base: SupplyBase, rng: np.random.Generator) -> None:
    """Plant known deteriorations so alerting can be scored.

    Two flavours, because they are detected by different statistics:
      lead_time_doubling -- the mean moves. Any monitor catches this eventually.
      tail_blowout       -- the MEAN barely moves and the P95 explodes. Only a
                            monitor watching the distribution sees it, which is
                            the whole argument for not tracking averages.
    """
    ids = list(base.suppliers)
    chosen = rng.choice(ids, size=6, replace=False)
    for i, sid in enumerate(chosen):
        kind = "lead_time_doubling" if i % 2 == 0 else "tail_blowout"
        s = base.suppliers[sid]
        # Onset must fall INSIDE the detector's recent window (last 120 days),
        # otherwise the disruption contaminates the baseline it is being compared
        # against and the ratio collapses toward 1.0 -- a planted signal that
        # cannot be detected by construction, which tests nothing.
        onset = float(rng.uniform(-100, -55))
        for po in base.pos:
            if po.supplier_id != sid or po.received_day is None:
                continue
            if po.promise_day - s.lead_mean_days < onset:
                continue
            if kind == "lead_time_doubling":
                po.received_day += s.lead_mean_days
            elif rng.random() < 0.35:
                po.received_day += s.lead_mean_days * rng.uniform(1.2, 2.4)
        base.planted_disruptions.append(
            {"supplier_id": sid, "kind": kind, "onset_day": onset})


def explode(base: SupplyBase, product: str, qty: float) -> dict[str, float]:
    """Total quantity of every component needed to build `qty` of `product`.

    Multi-level: a shared sub-assembly used by two products contributes its
    children twice. Getting this wrong understates requirements on exactly the
    components that are shared, which are the ones that go short.
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
    return {p: explode(base, p, 1.0) for p in base.products}
