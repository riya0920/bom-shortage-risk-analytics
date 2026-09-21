"""Loaders and cleaning for the two real public datasets.

1. Willems (2008) supply chains. Sean P. Willems, "Real-world multiechelon
   supply chains used for inventory optimization", Manufacturing & Service
   Operations Management 10(1):19-23, 2008. Each chain NN has two sheets:
     NN_SD  one row per stage: stageCost, relDepth, stageClassification
            (Part / Manuf / Trans / Dist / Retail), avgDemand, stdDevDemand,
            stageTime (mean), stdDev stageTime, and for some stages a discrete
            lead-time distribution stageTime_k with probability stageTime_%_k
     NN_LL  one row per arc: sourceStage -> destinationStage
   What it does NOT have, and how this project handles it:
     * no quantity-per on the arcs -> every BOM line is 1 per parent
     * no supplier names           -> suppliers come from SCMS (below)
     * no inventory or open orders -> simulated in supply.py, labelled
   The sheets do not state units. We read stage times as days and demand as
   units per day, the way the paper's safety-stock models use them.

2. USAID SCMS Delivery History (US-government open data, 10,324 shipment
   lines, 2006-2015). We use the vendor, the three dates (PO sent to vendor,
   scheduled delivery, delivered to client), the product group and the
   manufacturing site.

Everything here is pure Python plus `xlrd` (only for the .xls workbook), so
the cleaning and mapping rules can be tested on small in-memory fixtures.
"""
from __future__ import annotations

import csv
import datetime as dt
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WILLEMS_XLS = RAW / "MSOM-06-038-R2 Data Set in Excel.xls"
SCMS_CSV = RAW / "SCMS_Delivery_History_Dataset_20150929.csv"

CHAIN = 24                 # power-driven hand tools; see README for the choice
MIN_VENDOR_LINES = 10      # vendors with fewer clean lines are left out
IN_HOUSE_VENDOR = "SCMS from RDC"


def raw_data_present() -> bool:
    return WILLEMS_XLS.exists() and SCMS_CSV.exists()


# ---------------------------------------------------------------------------
# Willems
# ---------------------------------------------------------------------------

def read_willems_xls(path=WILLEMS_XLS, chain: int = CHAIN):
    """(stage rows as dicts, arcs as (source, destination)) for one chain."""
    import xlrd
    book = xlrd.open_workbook(str(path))
    sd = book.sheet_by_name(f"{chain:02d}_SD")
    head = [str(h).strip() for h in sd.row_values(0)]
    rows = [dict(zip(head, sd.row_values(r))) for r in range(1, sd.nrows)]
    ll = book.sheet_by_name(f"{chain:02d}_LL")
    arcs = [(str(a).strip(), str(b).strip())
            for a, b in (ll.row_values(r)[:2] for r in range(1, ll.nrows))]
    return rows, arcs


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def lead_distribution(row: dict) -> tuple[tuple, tuple]:
    """(values, probabilities) of a stage's lead time.

    Stages with a stageTime_k distribution get it as given. Stages without
    one have a single fixed lead time in the data, and we keep it fixed
    rather than inventing spread for it.
    """
    vals, probs = [], []
    for k in range(1, 7):
        v, p = _num(row.get(f"stageTime_{k}")), _num(row.get(f"stageTime_%_{k}"))
        if v is not None and p is not None and p > 0:
            vals.append(v)
            probs.append(p)
    if vals:
        s = sum(probs)
        return tuple(vals), tuple(p / s for p in probs)
    return (float(_num(row.get("stageTime")) or 0.0),), (1.0,)


def chain_structure(rows: list[dict], arcs: list[tuple]) -> dict:
    """Turn a Willems chain into products, a BOM, demand and part data.

    Products are the manufacturing stages with no manufacturing stage after
    them (the finished goods). A product's demand is the sum of the demand at
    the stages downstream of it (Dist / Retail), found by walking through
    non-manufacturing stages. If a demand stage could be reached from more
    than one product its demand is split evenly; we count how often.
    """
    stage = {str(r["Stage Name"]).strip(): r for r in rows}
    cls = {k: str(r.get("stageClassification", "")).strip() for k, r in stage.items()}
    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in arcs:
        if a in stage and b in stage:
            succ[a].append(b)
            pred[b].append(a)

    manuf = [k for k in stage if cls[k] == "Manuf"]
    products = sorted(k for k in manuf
                      if not any(cls[c] == "Manuf" for c in succ[k]))
    parts = sorted(k for k in stage if cls[k] == "Part")
    subs = sorted(k for k in manuf if k not in products)

    # demand: each demand stage -> the nearest manufacturing ancestors
    def makers_of(k):
        out, stack, seen = set(), [k], set()
        while stack:
            x = stack.pop()
            for p in pred[x]:
                if p in seen:
                    continue
                seen.add(p)
                if cls[p] == "Manuf":
                    out.add(p)
                elif cls[p] != "Part":
                    stack.append(p)
        return out

    avg = defaultdict(float)
    var = defaultdict(float)
    split = 0
    for k, r in stage.items():
        m = _num(r.get("avgDemand"))
        if m is None or cls[k] == "Manuf":
            continue
        sd = _num(r.get("stdDevDemand")) or 0.0
        makers = sorted(makers_of(k))
        if len(makers) > 1:
            split += 1
        for p in makers:
            avg[p] += m / len(makers)
            var[p] += (sd / len(makers)) ** 2

    bom = [(a, b) for a, b in arcs
           if a in stage and b in stage and cls[b] == "Manuf"
           and cls[a] in ("Part", "Manuf")]
    dropped_arcs = sum(1 for a, b in arcs
                       if a in stage and b in stage and cls[a] == "Part"
                       and cls[b] != "Manuf")

    part_info = {}
    for p in parts:
        r = stage[p]
        vals, probs = lead_distribution(r)
        mean = sum(v * q for v, q in zip(vals, probs))
        sd = (sum(q * (v - mean) ** 2 for v, q in zip(vals, probs))) ** 0.5
        part_info[p] = {
            "unit_cost": float(_num(r.get("stageCost")) or 0.0),
            "depth": int(_num(r.get("relDepth")) or 0),
            "lead_values": vals, "lead_probs": probs,
            "lead_mean": mean, "lead_sd": sd,
            "has_distribution": len(vals) > 1,
            "stage_time": float(_num(r.get("stageTime")) or mean),
        }
    return {
        "products": products, "subs": subs, "parts": parts,
        "bom": bom, "part_info": part_info,
        "demand_avg": {p: avg.get(p, 0.0) for p in products},
        "demand_sd": {p: var.get(p, 0.0) ** 0.5 for p in products},
        "product_cost": {p: float(_num(stage[p].get("stageCost")) or 0.0)
                         for p in products},
        "sub_depth": {s: int(_num(stage[s].get("relDepth")) or 0) for s in subs},
        "n_stages": len(stage), "n_arcs": len(arcs),
        "classes": {c: sum(1 for k in cls if cls[k] == c) for c in sorted(set(cls.values()))},
        "demand_stages_split": split,
        "part_arcs_not_into_manufacturing": dropped_arcs,
    }


# ---------------------------------------------------------------------------
# SCMS
# ---------------------------------------------------------------------------

def read_scms_csv(path=SCMS_CSV) -> list[dict]:
    with open(path, encoding="latin-1", newline="") as f:
        return list(csv.DictReader(f))


_DATE_FORMATS = ("%d-%b-%y", "%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y")


def parse_date(s) -> dt.date | None:
    s = (s or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None     # "Date Not Captured", "Pre-PQ Process", blanks ...


def clean_scms(rows: list[dict]) -> tuple[list[dict], dict]:
    """Keep vendor lines with a real, measurable lead time. Returns (lines, log).

    Rules, in order, each counted in the log:
      1. Fulfill Via == "Direct Drop". The other value, "From RDC", is SCMS's
         own regional warehouse shipping from stock; it is recorded under the
         pseudo-vendor "SCMS from RDC" and has no supplier lead time.
      2. All three dates parse (PO sent to vendor, scheduled, delivered).
      3. Lead time (delivered - PO sent) > 0 days. Zero means the PO and the
         delivery carry the same date, which is not a lead time we can use;
         negative is impossible.
      4. Scheduled date not before the PO date.
    """
    log = {"rows": len(rows)}
    direct = [r for r in rows if (r.get("Fulfill Via") or "").strip() == "Direct Drop"
              and (r.get("Vendor") or "").strip() != IN_HOUSE_VENDOR]
    log["direct_drop"] = len(direct)
    out, no_date, zero, neg_sched = [], 0, 0, 0
    for r in direct:
        po = parse_date(r.get("PO Sent to Vendor Date"))
        sc = parse_date(r.get("Scheduled Delivery Date"))
        dl = parse_date(r.get("Delivered to Client Date"))
        if po is None or sc is None or dl is None:
            no_date += 1
            continue
        lead = (dl - po).days
        if lead <= 0:
            zero += 1
            continue
        sched = (sc - po).days
        if sched < 0:
            neg_sched += 1
            continue
        out.append({
            "vendor": r["Vendor"].strip(), "po_date": po,
            "sched_date": sc, "delivered_date": dl,
            "lead_days": float(lead), "sched_lead_days": float(sched),
            "late_days": float((dl - sc).days),
            "product_group": (r.get("Product Group") or "").strip(),
            "molecule": (r.get("Molecule/Test Type") or "").strip(),
            "site": (r.get("Manufacturing Site") or "").strip(),
            "mode": (r.get("Shipment Mode") or "").strip(),
        })
    log.update(dropped_missing_date=no_date, dropped_lead_not_positive=zero,
               dropped_scheduled_before_po=neg_sched, kept=len(out),
               vendors=len({x["vendor"] for x in out}))
    return out, log


def vendor_table(lines: list[dict], min_lines: int = MIN_VENDOR_LINES) -> dict:
    """Per-vendor real statistics for vendors with enough clean lines."""
    import numpy as np
    by = defaultdict(list)
    for x in lines:
        by[x["vendor"]].append(x)
    # A vendor is a sole source for a product if no other vendor in the file
    # shipped that molecule / test type.
    vend_per_mol = defaultdict(set)
    for x in lines:
        vend_per_mol[x["molecule"]].add(x["vendor"])
    out = {}
    for v, xs in by.items():
        if len(xs) < min_lines:
            continue
        lead = np.array([x["lead_days"] for x in xs])
        late = np.array([x["late_days"] for x in xs])
        sched = np.array([x["sched_lead_days"] for x in xs])
        first = min(x["po_date"] for x in xs)
        last = max(x["po_date"] for x in xs)
        years = max((last - first).days / 365.25, 1.0)
        out[v] = {
            "n_lines": len(xs),
            "lead_mean": float(lead.mean()), "lead_sd": float(lead.std(ddof=1)),
            "lead_p50": float(np.percentile(lead, 50)),
            "lead_p95": float(np.percentile(lead, 95)),
            "on_time_vs_scheduled": float(np.mean(late <= 0)),
            "exact_on_scheduled_date": float(np.mean(late == 0)),
            "late_rate": float(np.mean(late > 0)),
            "sched_lead_median": float(np.median(sched)),
            "sole_source_share": float(np.mean(
                [len(vend_per_mol[x["molecule"]]) == 1 for x in xs])),
            "sites": sorted({x["site"] for x in xs if x["site"]}),
            "first_po": first, "last_po": last,
            "lines_per_year": len(xs) / years,
        }
    return out


# ---------------------------------------------------------------------------
# the vendor -> part mapping (an ASSUMPTION)
# ---------------------------------------------------------------------------

def map_vendors_to_parts(part_lead: dict[str, float],
                         vendor_lead: dict[str, float]) -> dict[str, str]:
    """Assign each Willems part to one SCMS vendor by lead-time band.

    THIS IS AN ASSUMPTION, NOT DATA. The SCMS vendors sell medicines and test
    kits, not power-tool parts, and neither dataset links to the other. The
    rule only keeps the ordering sensible: parts are sorted by their Willems
    mean lead time, vendors by their real median SCMS lead time, and a part at
    the q-th quantile of part lead times goes to the vendor at the q-th
    quantile of vendor lead times. So long-lead parts go to slow vendors and
    short-lead parts to fast ones, and every vendor gets a contiguous band.

    Deterministic: ties are broken by name, no random numbers.
    """
    ps = sorted(part_lead, key=lambda p: (part_lead[p], p))
    vs = sorted(vendor_lead, key=lambda v: (vendor_lead[v], v))
    if not vs:
        return {}
    n_p, n_v = len(ps), len(vs)
    return {p: vs[min(n_v - 1, (i * n_v) // n_p)] for i, p in enumerate(ps)}
