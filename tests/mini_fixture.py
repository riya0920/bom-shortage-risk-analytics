"""A tiny hand-made supply chain in the SAME raw formats as the real data.

This is a TEST FIXTURE, not data. It exists so the loaders, the cleaning rules,
the mapping and the engine can be tested in CI, where data/raw is not present
(the real files are downloaded, never committed).

Willems-format stages: 3 products (M1-M3), 3 sub-assemblies (S1-S3, S3 inside
S1 so the BOM is multi-level, S1 shared by M1 and M2), 12 purchased parts.
SCMS-format rows: 8 vendors x 40 Direct Drop lines, plus junk rows that the
cleaning must drop.
"""
from __future__ import annotations

import datetime as dt

import numpy as np


def willems_rows():
    rows = []

    def stage(name, cls, cost=0.0, depth=0, t=1.0, dist=None, avg="", sd=""):
        r = {"Stage Name": name, "stageCost": cost, "relDepth": depth,
             "stageClassification": cls, "avgDemand": avg, "stdDevDemand": sd,
             "stageTime": t, "stdDev stageTime": ""}
        for k in range(1, 7):
            r[f"stageTime_{k}"] = ""
            r[f"stageTime_%_{k}"] = ""
        if dist:
            for k, (v, p) in enumerate(dist, 1):
                r[f"stageTime_{k}"] = v
                r[f"stageTime_%_{k}"] = p
            r["stageTime"] = sum(v * p for v, p in dist)
        rows.append(r)

    for i, m in enumerate(("M1", "M2", "M3")):
        stage(m, "Manuf", cost=5.0 + i, depth=1, t=2.0)
    for s in ("S1", "S2", "S3"):
        stage(s, "Manuf", depth=2, t=2.0)
    leads = [3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 30]
    for i, L in enumerate(leads):
        dist = [(L, 0.6), (L + 3, 0.4)] if i % 2 == 0 else None
        stage(f"P{i + 1:02d}", "Part", cost=0.5 + i, depth=3, t=float(L), dist=dist)
    stage("D1", "Dist", t=4.0, avg=10.0, sd=4.0)
    stage("D2", "Dist", t=4.0, avg=6.0, sd=3.0)
    stage("D3", "Dist", t=4.0, avg=3.0, sd=2.0)

    arcs = [("M1", "D1"), ("M2", "D2"), ("M3", "D3"),
            ("S1", "M1"), ("S1", "M2"), ("S2", "M3"), ("S3", "S1"),
            ("P01", "S1"), ("P02", "S1"), ("P03", "S3"), ("P04", "S3"),
            ("P05", "S2"), ("P06", "S2"), ("P07", "M1"), ("P08", "M2"),
            ("P09", "M3"), ("P10", "M3"), ("P11", "S2"), ("P12", "M1"),
            ("P01", "M3")]
    return rows, arcs


def scms_rows(seed: int = 3):
    rng = np.random.default_rng(seed)
    rows = []
    start = dt.date(2012, 1, 1)
    medians = [20, 35, 50, 70, 90, 120, 150, 200]
    for v, med in enumerate(medians):
        for k in range(40):
            po = start + dt.timedelta(days=int(k * 27 + rng.integers(0, 10)))
            sched = int(med * rng.uniform(0.7, 1.3))
            lead = sched + (int(rng.integers(1, 20)) if rng.random() < 0.15 else 0)
            fmt = "%d-%b-%y" if k % 2 else "%m/%d/%Y"
            rows.append({
                "Fulfill Via": "Direct Drop", "Vendor": f"Vendor {chr(65 + v)}",
                "PO Sent to Vendor Date": po.strftime(fmt),
                "Scheduled Delivery Date": (po + dt.timedelta(days=sched)).strftime(fmt),
                "Delivered to Client Date": (po + dt.timedelta(days=lead)).strftime(fmt),
                "Product Group": "ARV",
                "Molecule/Test Type": f"mol-{v}" if v < 2 else "shared-mol",
                "Manufacturing Site": f"Site {v % 3}", "Shipment Mode": "Air"})
    # junk the cleaning must drop
    rows.append({"Fulfill Via": "From RDC", "Vendor": "SCMS from RDC",
                 "PO Sent to Vendor Date": "N/A - From RDC",
                 "Scheduled Delivery Date": "1-Jun-13",
                 "Delivered to Client Date": "1-Jun-13"})
    rows.append({"Fulfill Via": "Direct Drop", "Vendor": "Vendor A",
                 "PO Sent to Vendor Date": "Date Not Captured",
                 "Scheduled Delivery Date": "1-Jun-13",
                 "Delivered to Client Date": "1-Jun-13"})
    rows.append({"Fulfill Via": "Direct Drop", "Vendor": "Vendor A",
                 "PO Sent to Vendor Date": "1-Jun-13",
                 "Scheduled Delivery Date": "1-Jun-13",
                 "Delivered to Client Date": "1-Jun-13"})          # zero lead
    for i in range(3):                                            # too-small vendor
        rows.append({"Fulfill Via": "Direct Drop", "Vendor": "Tiny Vendor",
                     "PO Sent to Vendor Date": "1-Jan-13",
                     "Scheduled Delivery Date": "1-Mar-13",
                     "Delivered to Client Date": "1-Mar-13",
                     "Molecule/Test Type": "x", "Manufacturing Site": "Site 9"})
    return rows


def build_mini():
    import data_loaders as DL
    import supply
    rows, arcs = willems_rows()
    lines, log = DL.clean_scms(scms_rows())
    base = supply.build_from(DL.chain_structure(rows, arcs), lines,
                             meta={"chain": "mini", "scms_cleaning": log})
    # Make one shared part clearly short so the plan is constrained somewhere.
    base.components["P01"].on_hand = 0.0
    base.pos = [po for po in base.pos if po.part != "P01"]
    return base
