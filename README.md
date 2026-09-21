# BOM Shortage Risk Analytics

**Live dashboard: https://riya0920.github.io/bom-shortage-risk-analytics/**

Can a factory build its 13-week production plan with the parts it has and the
parts on order? And if not, which parts will stop the line, which suppliers are
to blame, and who needs to act first?

## What we did

A factory makes 4 products. Each product is built from a multi-level bill of
materials (BOM): 32 sub-assemblies and **260 purchased parts** from **50
suppliers**, 68 of them from a single source. We built a tool that answers the
questions a materials manager asks every week:

1. **How much of the plan can we build?** Week by week, per product.
2. **What is stopping it?** Which parts gate production, and the last day to
   order them.
3. **Which suppliers can we trust?** On-time scores that can't be gamed, and an
   early warning when a supplier starts slipping.
4. **Who acts first?** Turn hundreds of part alerts into a short list for the
   right person.

The data is simulated so every answer can be checked against a known truth: we
know each supplier's real lead-time distribution, and we planted 6 supplier
disruptions for the detector to find.

## How we did it

- **BOM explosion.** Walk the multi-level BOM so a sub-assembly shared by two
  products counts its parts twice. Getting this wrong understates exactly the
  shared parts, which are the ones that go short.
- **Monte Carlo buildability.** 300 simulations, each drawing a new lead time for
  every open order. Result: buildable units per week as a range (P10 / P50 / P90),
  not one number. A product needs *every* one of its parts, so a small shortage in
  one cheap part stops the whole product.
- **Shortage drivers with order-by dates.** Rank parts by how often they were the
  one that stopped production. The order-by date uses the supplier's **P95** lead
  time (a bad-case delivery), not the average.
- **Supplier analytics.** On-time-in-full (OTIF) scored against the *original*
  promise and against the latest reschedule. A deterioration detector with three
  triggers (average up, P95 up, variance up), scored against the planted
  disruptions.
- **Planning extras.** Safety stock that includes lead-time variation, expedite
  options with costs, hidden tier-2 suppliers, and three policies for splitting a
  short part between products.
- **Alerts.** Severity tiers with response times, grouping by supplier, and a
  delivery layer (outbox, retries, rate limits) tested against a real local HTTP
  server.

Python, NumPy, SciPy, scikit-learn. 58 tests, run in CI.

## What we found

| Question | Finding |
|---|---|
| Can we build the plan? | **No.** In the median case only **25%** of planned units can be built (613 of 2,417). Shortfalls start in week 3, once today's stock runs out. |
| What is stopping it? | **6 of the top 10** shortage parts are already past their order-by date. 54 of 260 parts have zero or negative slack today. $848K of output value is at risk. |
| Are on-time scores honest? | **No.** Suppliers who reschedule early score **100%** against the rescheduled date but only **36-46%** against what they first promised. |
| Does the early warning work? | Partly. It catches **3 of 6** planted disruptions. At the best threshold precision is 60%. Most of the catches come from the **P95 trigger**: without it, recall drops from 50% to 33%. The variance trigger adds only false alarms. |
| Is safety stock right? | The common textbook formula (demand variation only) understates it **6.9x**, because 97% of the variation here comes from suppliers, not demand. |
| Does expediting help? | Only sometimes: **11 of 24** expedite options still arrive after the part was needed. |
| Is the risk hidden deeper? | Yes. One tier-2 supplier sits behind **63 parts and all 4 products**, which a tier-1 view cannot see. |
| Can people act on the alerts? | Grouping 260 part alerts into **108 supplier incidents** cut the queue by 58%. It is still too long: one owner has 26 urgent items against a limit of 10. The queue isn't long because of bad ranking; the parts really are short. The top 10 incidents hold **91%** of the value at risk. |

Two problems we found and fixed along the way:

- **Results changed between runs.** A loop over a Python `set` of strings used
  random numbers, and string hashing changes per process, so two runs of the same
  script gave value-at-risk figures 61% apart. Fixed, with a test that runs the
  script in three separate processes.
- **The risk band is too wide.** Checked against fresh simulated futures, 97% of
  outcomes land inside the P10-P90 band (should be 80%). We report this instead of
  tuning it away.

## What we decided, and why

1. **Plan with a range, not a single number.** How much safety cover to buy
   depends on the bad case, and a single number hides it.
2. **Order to the P95 lead time.** Ordering to the average means arriving late
   half the time, and for a part that stops the line that is not a plan.
3. **Score suppliers on the original promise.** It is the only version they
   cannot improve by rescheduling. The gap between the two scores is tracked as its
   own warning sign.
4. **Keep the P95 trigger, drop the variance trigger.** Measured, not assumed: P95
   does most of the detecting; variance only adds false alarms.
5. **Group alerts by supplier, then work a top-10 list.** One late supplier is one
   phone call, not twelve alerts. The top 10 incidents cover 91% of the exposure.
6. **Leave allocation to the business.** Who gets a short part (even split,
   highest margin, contract priority) is a business call. We show the trade-off;
   the worst-off product gets about 20% under all three.

**Limits:** the data is simulated, so what carries over is the method and the
size of the effects, not the exact numbers. Supplier capacity is assumed, and the
alert delivery has no scheduler or real email/ticket system.

## How to run it

```bash
pip install -r requirements.txt

python run_supply.py        # ~40s  buildability, shortage drivers, suppliers -> docs/RESULTS.md
python extend.py            # ~25s  expedites, tier-2 risk, calibration     -> docs/EXTENSIONS.md
python complete.py          # ~30s  safety stock, detector sweep, alerts     -> docs/COMPLETION.md
python run_pass4.py         # ~20s  supplier incidents and alert delivery    -> docs/INCIDENTS_AND_TRANSPORT.md
python build_dashboard.py   #       rebuilds docs/index.html from out/*.json

python -m pytest tests -q   # 58 tests
```

Every number above comes from the files in `out/`. The full write-ups are in
[`docs/`](docs/), and a weekly materials-review pack is generated at
[`out/materials_review_pack.md`](out/materials_review_pack.md).

## Code layout

```
src/supply.py              simulated supply base: BOMs, suppliers, lead times, orders, planted disruptions
src/buildability.py        BOM explosion, weekly buildability, Monte Carlo, allocation, order-by dates
src/supplier_analytics.py  OTIF both ways, deterioration detector, supplier risk score
src/inventory.py           safety stock, lot sizing, supplier capacity
src/routing.py             severity tiers, response times, routing
src/incidents.py           grouping alerts by supplier, the load check
src/transport.py           alert delivery: outbox, retries, dead letters, rate limits
```
