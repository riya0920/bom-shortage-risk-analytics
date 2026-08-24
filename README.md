# DATA-3 — Manufacturing Supply Chain & Shortage Risk Analytics

**Status: complete.** BOM propagation to buildability, Monte Carlo fan charts,
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

## Completed in the third pass — see [docs/COMPLETION.md](docs/COMPLETION.md)

```bash
python complete.py    # ~40 s; writes COMPLETION.md and out/supply_dashboard.html
```

- **Safety stock, with the term everyone forgets.** The remembered formula covers
  demand variability; the full one covers lead-time variability too. Across
  260 purchased parts **97% of the
  variance is supply-side**, so the demand-only version understates safety stock
  by **6.9×**. And `z` is a *normal*
  quantile: on fat-tailed suppliers an empirical quantile of simulated lead-time
  demand asks for **2.48×** the
  parametric number, against 1.61× on
  well-behaved ones.
- **MOQ, lot sizing and capacity.** `Component.moq` existed and nothing read it.
  Applying it, **0 of 260 parts get
  raised to their minimum, forcing $6,351,226 of inventory
  nobody wanted.** That figure counts *only* what the minimum imposed — EOQ
  excess is a deliberate purchase and is reported separately, because conflating
  them overstates the one number here a supplier would check.
- **Supplier capacity**, which the README's item 7 said did not exist:
  2 suppliers over-committed. The distinction
  changes the remedy completely — a lead-time problem is solved by ordering
  earlier, and a capacity problem **is not solved by ordering earlier at all**.
- **The detector threshold sweep**, turning one measured point into a curve. Best
  F1 at scale 1.40 (precision 0.60, recall
  0.50), and an ablation showing the **P95 trigger carries the
  detection** — a detector watching only the average misses half of it.
- **Alert routing with severity tiers and SLAs.** Severity is consequence ×
  urgency, *not* risk score: measured correlation between the two is
  **-0.03**. Slack is judged against the
  lead time rather than an absolute number of days, because ten days is
  comfortable on a 5-day lead and an emergency on a 60-day one.
- **The invented financial score, replaced.** `financial_score` was a Beta draw,
  and that cannot be fixed by inventing a better distribution. Three *observable*
  signals — lead-time inflation, promise slippage against the original promise,
  and quality drift — predict the planted disruptions at **AUROC
  0.839** where the invented score sits at
  0.327, i.e. chance, which is exactly what a Beta draw
  should do.
- **Charts** at `out/supply_dashboard.html`, self-contained. The buildability fan
  is the point: a point forecast is the least useful number available, because
  the decision is how much cover to buy and that is set by the width of the band.

### Two things the checks caught that I would otherwise have shipped

**The variance trigger is dead weight.** The ablation shows removing it leaves
recall unchanged while precision improves — it contributes false positives and no
detections. I would have kept all three triggers on the reasoning in the
docstring; the measurement says otherwise, and it is now recommended for removal.

**The severity tiering does not survive its own load check.**
114 parts land in P1 and
221
alerts route to a single buyer. That is a list, not a prioritised queue. The
cause is that severity is computed per *part* while the remedy is usually per
*supplier* — one late supplier puts every part it ships into P1 at once — and the
fix (group by supplier, route one item with its parts list) is **not built**.
Reporting the alert count without the load-by-role table would have hidden it.

## Built in the fourth pass — see [docs/INCIDENTS_AND_TRANSPORT.md](docs/INCIDENTS_AND_TRANSPORT.md)

```bash
python run_pass4.py    # ~25 s
```

### First: every number in this project was irreproducible

`supply.build()` contained `for sub in set(subs):`, and that loop makes `rng`
calls. **Python salts string hashing per process** (PEP 456), so a set of strings
iterates in a different order in every interpreter, the random draws are consumed
in a different order, and the supply base differs on every run. Two runs of the
same script gave a value at risk of **656,531 and 1,059,890** — a 61% swing with
no input changed.

It survived three passes because it is deterministic *within* one process: three
builds in one interpreter agree exactly, three runs of the same script do not,
and nothing in a normal test session looks at the second case. The test that
catches it now shells out to three subprocesses. Fixed with `dict.fromkeys`.
This is the fourth instance of this family of bug across the nine projects — the
other three were `hash()` used as an RNG seed.

### Severity per supplier, and the diagnosis that was half right

An incident is one **decision** with one owner, not a summary: a supplier that
has slipped puts every part it ships at risk at once, and the answer to all of
them is the same phone call. One incident per *(supplier, severity)*, because a
supplier with two P1 parts and nine P4s is two conversations on two clocks.

| | alerts | incidents |
|---|---:|---:|
| items | 260 | **108** (58% fewer) |
| P1 items | 109 | **36** |
| busiest owner | BUYER 218 | BUYER **82** |
| worst P1 queue | BUYER 67 | SUPPLY_CHAIN_MANAGER **26** |
| parts covered | 260 | 260 |
| value at risk | 848,104 | 848,104 |

The last two rows are what keeps it honest: grouping changes the **count** an
owner sees and not the work, so parts and value are carried through unchanged.

**And it still fails the load check.** The stated diagnosis was that *one late
supplier puts every part it ships into P1 at once*. That happens, and grouping is
worth 58% — but the alerts are not concentrated: the top
five suppliers are **19%** of them across
48 suppliers. Sweeping the P1 threshold does not save it
either: even at a slack ratio of 0.01 the worst queue is
13, above the limit of 10. There is a
floor and it is correct — **54 of 260 parts already
have zero or negative slack**, and no threshold can argue that away.

> The queue is not too long because the policy mis-ranks. It is too long because
> the situation is bad.

That is a different finding from the one this project expected to make, and
tuning the threshold until the table looked acceptable would have buried it. The
one honest lever left is to say what a top-N queue covers: the top 10
P1 incidents by value carry **91%** of the
P1 value at risk, and 8 incidents carry 80%
of it.

### A transport

The receiver is a real HTTP server on a real socket — mocking it would test the
code that calls the transport, which is not the part that goes wrong.

- **Urgent is sent, routine is digested.** 108 incidents become
  72 messages: 70 individual P1/P2
  items and 2 digests. Forty separate messages saying
  *look at it on Friday* is how a channel gets filtered to a folder — after
  which the P1s go there too.
- **At-least-once, and the receiver deduplicates.** A sender that marks before
  the POST loses messages; one that marks after re-sends them. There is no third
  option. The case that matters is the crash window — deliver, then die before
  marking: the row stays `{'PENDING': 1}`, restart re-sends it, the
  receiver sees **2 requests**, records
  **1** and suppresses
  **1** on the idempotency key. An ordinary
  retry never produces a duplicate, so it never tests this.
- **A dead letter is not a failure to hide.** Against an endpoint that never
  answers: `{'DEAD': 2}`, **2 still in the outbox**.
  Dropping them makes the transport look reliable and makes a P1 disappear.
- **Rate limiting is part of the alert policy, not the plumbing.** With a budget
  of 5 per recipient, **10 sent and
  62 deferred**, nothing dropped, and the queue is drained in
  severity order so the budget goes to P1s rather than to whatever was enqueued
  first.

## What is NOT built

1. **No real data.** Everything is `src/supply.py`. No ERP extract, no real BOM,
   no real supplier history — so every number here is a statement about the
   generator, which is now at least a generator that reproduces.
2. **The load check still fails, and grouping was not enough.** The remaining
   answer is capacity or a top-N policy, not another threshold. Nothing here
   implements the top-N policy; it measures what one would cover.
3. **Supplier capacity is assumed, not sourced.** It is not in the dataset, so it
   is assigned as a multiple of current commitment and flagged as an assumption
   everywhere it is used.
4. **The distress proxy is still a proxy.** Its weights are stated rather than
   fitted, because there are no realised bankruptcies here to fit against —
   inventing some would be the same mistake the Beta draw was.
5. **No calibration against realised outcomes over time.** The risk bands were
   measured at 97.9% against an 80% target in pass 2, which means the band is too
   wide, and fixing it properly needs outcomes this simulation does not run long
   enough to produce.
6. **The transport has one sink and no scheduler.** A webhook, and a `drain()`
   that a caller drives on a clock it supplies. No email, no ticketing system, no
   ERP write-back, and nothing runs it at 06:00 — the outbox, the retry policy
   and the dead-letter state are the parts that would survive attaching any of
   those; the sink is the part that would be replaced.
7. **Escalation is a chain in a dict, not a timer.** `escalate_after_hours` is
   carried on every incident and nothing counts down or reassigns.

## Layout

```
src/supply.py              generator: BOMs, suppliers, lead-time distributions, PO history,
                           planted disruptions, and supply calibrated to the requirement
src/buildability.py        BOM explosion, weekly buildability, allocation policies,
                           Monte Carlo fan, shortage drivers with order-by dates
src/supplier_analytics.py  OTIF both ways, deterioration triggers, composite risk
src/routing.py             severity tiers, SLAs, routing and the escalation chain
src/incidents.py           per-supplier grouping, the load check, threshold calibration
src/transport.py           durable outbox, retries, dead letters, rate limits, a real sink
run_supply.py              orchestration; writes docs/RESULTS.md
run_pass4.py               incidents and the transport; writes the pass-4 doc
```
