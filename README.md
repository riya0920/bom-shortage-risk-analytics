# DATA-3 — Manufacturing Supply Chain & Shortage Risk Analytics

**Status: ~50% slice.** BOM propagation to buildability, Monte Carlo fan charts,
shortage attribution with order-by dates, the OTIF definitional trap, and
distribution-based supplier deterioration detection are built. The weekly
materials-review pack, expedite costing, and any real data are not.

```bash
python run_supply.py
python run_supply.py --quick
python run_supply.py --report-only
```

~60 seconds. Writes [docs/RESULTS.md](docs/RESULTS.md) and `out/results.json`.

Scale: 4 end products, 32 sub-assemblies, **260 purchased parts**, 442 BOM lines,
50 suppliers, 68 single-sourced parts, 13-week horizon.

## Why this is manufacturing supply chain and not supply chain

The BOM. Supplier on-time-delivery percentages cannot answer the only question a
materials manager has, which is **can we build the plan** — and a $2 fastener at
100% availability-criticality stops a $2M machine exactly as hard as a casting
does.

`explode()` walks the multi-level BOM so a shared sub-assembly used by two
products contributes its children twice. Getting that wrong understates
requirements on precisely the components that are shared, which are the ones that
go short.

## The finding that justifies the whole approach

While tuning the generator I set per-part coverage to a median of ~0.98 of the
13-week requirement — which sounds nearly adequate. **End-item buildability came
out at 11–33% fill.**

That is not a bug, it is the point. A product needs *every* one of its ~60–100
parts. If each part independently has a ~50% chance of being under-covered, the
chance that all of them are adequate is indistinguishable from zero. **Part-level
coverage does not translate into end-item coverage**, and that non-linearity is
the entire reason this analysis has to run through the BOM rather than over a
supplier scorecard. The generator now plants a minority of genuinely short parts
against a comfortable majority, which is what a real materials position looks like.

## Buildability as a distribution

| week | demand | P10 | P50 | P90 |
|---|---|---|---|---|
| 1 | 34 | 34 | 34 | 34 |
| 2 | 36 | 11 | 36 | 36 |
| 3 | 41 | 0 | 15 | 41 |
| … | | | | |

"P50 says 120" invites a commitment. "P10 84, P50 120, P90 140, and the gating
part is PRT-0041 from a single source" invites a decision. A point estimate on a
supply plan is a forecast pretending to be a fact.

**On the calibration table** in RESULTS.md §2: those percentages are *self-consistency*
checks — the percentiles are computed from the same simulations they are scored
against, so they verify arithmetic, not the model. Genuine calibration would need
the fan compared against realised outcomes, which requires simulating a future
this project does not run. That is stated in the report rather than left for a
reader to assume otherwise.

## Order-by dates, computed from P95

The shortage-driver table carries an **order-by date**, and it is computed from
the supplier's **P95** lead time, not the mean. Ordering a gating part to its mean
lead time means arriving late half the time, which for the part that stops the
line is not a plan.

That one substitution is the difference between an alert that is a fact — "you
will be short in week 7" — and an alert that is a decision — "order by day −3 or
the line stops in week 7". In the current run **5 of the 10 top drivers are
already past their order-by date**, which is the honest state of a materials
position nobody has been watching.

## The OTIF definitional trap, demonstrated

On-time-in-full against *which* promise?

| supplier | OTIF vs original promise | OTIF vs latest reschedule | gap |
|---|---|---|---|
| SUP-036 | 35.7% | **100.0%** | **+64.3 pts** |
| SUP-009 | 38.5% | **100.0%** | +61.5 |
| SUP-003 | 42.9% | **100.0%** | +57.1 |
| SUP-006 | 42.9% | **100.0%** | +57.1 |

A supplier who is chronically late but always calls ahead scores **100%** against
their reschedules and 36% against what they originally committed to. The 100% is
the number suppliers prefer, and it is the ERP default in a lot of configurations
because the promise field gets overwritten in place — the original is simply gone.

The scorecard here uses the **original** promise, and reports the gap as its own
metric: a direct measure of how much a supplier is managing their score rather
than their delivery.

The generator had to be changed to show this. In the first version, received POs
had `original_promise == latest_promise`, so the gap was identically zero for
every supplier and the trap was undemonstrable. Roughly a third of suppliers are
now habitual reschedulers.

## Deterioration caught by distributions, not averages

Three independent triggers — mean shift, P95 blowout, variance increase — because
they catch different things.

**Precision 0.43, recall 0.50** against 6 planted disruptions. That is a modest
detector and it is reported as one rather than tuned until it looks good:

| disruption type | planted | caught | caught by the P95 trigger alone |
|---|---|---|---|
| lead_time_doubling | 3 | 1 | 0 |
| tail_blowout | 3 | 2 | **1** |

The `tail_blowout` row is the argument. Those suppliers have a mean lead time that
has barely moved and a P95 that has doubled — they will stop a line, and every
mean-based monitor and every OTD percentage says they are fine. One of them is
caught by the P95 trigger *and nothing else*.

The precision of 0.43 means roughly one false alarm for every true one. In a
weekly materials review that is arguably acceptable — a flagged supplier costs a
phone call — but it is the alarm-economics tradeoff this portfolio keeps running
into, and a threshold sweep to characterise it properly is not built.

## Allocation when a shared component is short

| policy | total P50 units | worst product fill |
|---|---|---|
| even (proportional) | 912 | **28.3%** |
| margin-weighted | 912 | 26.5% |
| contractual priority | 912 | 26.5% |

Someone has to decide who gets the part, and that is a business policy rather than
an arithmetic fact. The job of the analysis is to make the tradeoff explicit and
quantified — not to pick.

The "even" policy had to be rewritten to earn its name: the first version consumed
inventory in list order and called it even, which meant the first product took
everything it wanted and the last starved at a 7.7% fill rate. An allocation
policy that depends on the order of a Python list is not a policy.

## Why 13 weeks

It covers most purchased-part lead times plus a planning cycle, so an order placed
today still lands inside the window. Beyond it, forecast uncertainty starts to
dominate supply uncertainty and the analysis is answering a sales question wearing
a materials hat.

## Built in the second pass — see [docs/EXTENSIONS.md](docs/EXTENSIONS.md)

`python extend.py` — four gaps this README previously named:

- **Expedite options with costs.** The first build identified shortage drivers and
  stopped. The number that decides an expedite is the premium against the margin of
  the units it unblocks — and roughly half the options **do not recover the
  runway**, i.e. the days saved are fewer than the days already lost past the
  order-by date. Air freight on a part that needed ordering six weeks ago buys a
  faster arrival of something still too late.
- **Multi-tier risk.** Dual-sourcing is worthless if both sources buy from the same
  sub-tier. The top tier-2 source sits behind **63 parts and all four products** —
  a concentration a single-tier risk score cannot see.
- **Genuine calibration.** RESULTS.md checked the percentiles against the same
  simulations they came from and said so. This draws independent futures and asks
  how often the fan contained them: **~98% inside P10–P90 against a target of
  80%** — the band is too wide, which is reported rather than tuned away.
- **The weekly materials pack**, generated to `out/materials_review_pack.md`.

## What is NOT built (the other 50%)

1. **No real data.** Everything is `src/supply.py`. No ERP extract, no real BOM,
   no real supplier history.
2. **No weekly materials-review pack.** The spec asks for a generated document
   with shortage drivers, supplier risk movers, buildability fan charts and
   recommended expedites *with cost estimates*. The ingredients are in
   `results.json`; the pack and the expedite costing are not built.
3. **No charts.** "Fan chart" is a table of P10/P50/P90 in markdown.
4. **No alert routing, severity tiers, or escalation SLAs.** The order-by dates
   exist; nothing delivers them to a buyer, and the spec is explicit that
   analytics without process integration is a dashboard.
5. **No threshold sweep on the deterioration detector**, so the precision/recall
   tradeoff is a single measured point rather than a curve.
6. **No MOQ, no lot-sizing, no safety-stock policy.** `Component.moq` exists and
   nothing reads it, which means the recommended order quantities would be wrong.
7. **No supplier capacity constraints.** A supplier can ship unlimited quantity;
   in reality the constraint is often the supplier's line, not their lead time.
8. **No calibration against realised outcomes.** See §2 of the report.
9. **The financial score is invented.** `financial_score` is a Beta draw standing
   in for a credit rating. It is a placeholder in a risk model, and a risk model
   with a placeholder input should be read as a demonstration of the *structure*,
   not as a risk assessment.

## Layout

```
src/supply.py              generator: BOMs, suppliers, lead-time distributions, PO history,
                           planted disruptions, and supply calibrated to the requirement
src/buildability.py        BOM explosion, weekly buildability, allocation policies,
                           Monte Carlo fan, shortage drivers with order-by dates
src/supplier_analytics.py  OTIF both ways, deterioration triggers, composite risk
run_supply.py              orchestration; writes docs/RESULTS.md
```
