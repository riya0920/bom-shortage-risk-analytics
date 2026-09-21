# BOM Shortage Risk Analytics

**Live dashboard: https://riya0920.github.io/bom-shortage-risk-analytics/**

Can a factory build its 13-week production plan with the parts it has and the
parts on order? If not, which parts will stop the line, which suppliers can we
trust, and who needs to act first?

A bill of materials (BOM) is the recipe for a product: which parts and which
sub-assemblies go into it, level by level.

## What we did

We took a **real published supply chain** for power hand tools and **real
supplier delivery records**, and built the tool a materials manager uses every
week:

1. **How much of the plan can we build?** Week by week, per product, as a range.
2. **What is stopping it?** Which parts gate production, and the last day to
   order them.
3. **Which suppliers can we trust?** On-time scores that can't be gamed, and an
   early warning when a supplier starts slipping.
4. **Who acts first?** Turn part alerts into a short list for the right person.

### What is real, what is assumed, what is simulated

| | What | Source |
|---|---|---|
| **Real** | BOM structure, part costs, product demand (mean and spread), each part's lead-time distribution | Willems (2008), chain 24, power-driven hand tools: 17 products, 31 sub-assemblies, 209 purchased parts, 1,168 BOM lines |
| **Real** | Supplier lead times, on-time rate against the scheduled date, order history, the factories behind each vendor | USAID SCMS Delivery History (US-government open data): 4,126 delivery lines from 27 vendors, 2006-2015 |
| **Assumed** | 1 of each part per parent | Willems has no quantities |
| **Assumed** | Which vendor supplies which part | The two datasets are unrelated. Parts and vendors are both ranked by lead time and matched band by band (fast parts to fast vendors, 7-8 parts each) |
| **Simulated** | Stock on hand, open orders, minimum order sizes | No public dataset has them. Seeded rule: 12% of parts are short (45-95% of their 13-week need covered), the rest have 105-190%, 35-75% of it on hand |
| **Simulated** | The *original* promise date | SCMS keeps one scheduled date per line; nothing public keeps the original next to the reschedule |
| **Simulated** | 6 planted supplier disruptions | Used only to score the early-warning detector, planted into order histories resampled from each real vendor's own lead times |

**Why chain 24.** Of the 38 Willems chains, it has the most finished products
that share parts (17 products; 143 of 209 parts feed two or more, 44 feed all
17) and real demand for each product. 114 of its parts have a discrete
lead-time distribution; the other 95 have one fixed lead time, which we keep
fixed. Chain 26 (aircraft engines) has a distribution on every part but only
two demand points; chain 20 has 74 parts and 25 distributions.

## How we did it

- **BOM explosion.** Walk the multi-level BOM so a sub-assembly used along two
  paths counts its parts twice. Getting this wrong understates exactly the shared
  parts.
- **Monte Carlo buildability.** 300 simulations. Each open order lands at its
  promise date moved by a draw from the part's real lead-time distribution plus
  its vendor's real lateness. Result: buildable units per week as P10 / P50 / P90.
- **Shortage drivers with order-by dates.** Rank parts by how often they stopped
  a product. Order-by date = first short week minus the part's **P95** lead time.
- **Supplier analytics on real SCMS history.** On-time against the scheduled
  date, the share of deliveries landing exactly on it, a composite risk score
  with visible weights, and a deterioration detector (mean, P95 and variance
  triggers). The detector is scored in a sandbox with planted disruptions, then
  run on the real history.
- **Planning extras.** Safety stock with both demand and lead-time variation,
  expedite options, tier-2 factories (real SCMS manufacturing sites), and three
  policies for splitting a short part between products.
- **Alerts.** Severity tiers, grouping by supplier, and a delivery layer
  (outbox, retries, rate limits) tested against a real local HTTP server.

Python, NumPy, SciPy, scikit-learn, xlrd. 84 tests, run in CI (the 6 real-data
tests skip there, because the data is downloaded, not committed).

## What we found

| Question | Finding |
|---|---|
| Can we build the plan? | **No.** In the median case **41%** of planned units can be built (65,841 of 159,731). P10 39%, P90 43%. Shortfalls start in week 4. |
| What is stopping it? | Quantity, not timing. **0 of the top 10** shortage parts are past their order-by date: real part lead times are 3-55 days (median 9), so there is still time to order. 82 of 209 parts run out inside 13 weeks. |
| Are on-time scores honest? | The real data points to no. **89% of 4,126 SCMS deliveries land exactly on the scheduled date** (94% on or before), and 17 of 27 vendors hit the exact date over 90% of the time. That is what a date updated to match the delivery looks like. With a simulated original promise the gap reaches **+75 points**. |
| Does the early warning work? | Partly. At the default setting precision is **0.40**, recall **0.33**; best F1 at 1.10x gives precision 0.40, recall 0.67. Real vendors order about ten times a year: a 120-day window could judge only 11 of 27 of them, a one-year window judges 23. On the real history **37 of 96 vendor-years** get flagged (no ground truth to check). |
| Is safety stock right? | Only **25%** of the variance comes from lead times; real demand swings more. Leaving lead-time spread out understates safety stock just **1.29x**. |
| Does expediting help? | **0 of 24** expedite options are needed. Every top shortage part can still be ordered at its normal lead time. |
| Is the risk hidden deeper? | Yes, in real data: **27 of 59** manufacturing sites ship through more than one vendor. GSK Mississauga sits behind 4 vendors a buyer would treat as separate. |
| Can people act on the alerts? | At the simulated stock level nothing is urgent (0 P1). Grouping 209 part alerts gives 30 supplier incidents. With stock cut to x0.25, 44 urgent alerts become 16 incidents and the worst urgent queue falls from 41 to 15, still above the limit of 10. |

### What changed when the simulated data was replaced with real data

| Finding | Simulated version | Real data |
|---|---|---|
| Plan buildable (median) | 25% | 41% |
| Top shortage parts already too late | 6 of 10 | 0 of 10 |
| Share of safety-stock variance from suppliers | 97% (textbook formula 6.9x too low) | 25% (1.29x) |
| Expedites that arrive too late | 11 of 24 | 0 of 24 needed |
| Urgent items on the busiest owner | 26 | 0 (15 only when stock is cut to x0.25) |
| Detector: the P95 trigger carries detection | yes | no single trigger does; only the variance trigger adds false alarms |
| P10-P90 band holds the outcome (target 80%) | 97% | 99% |

The simulator had lead times of 14-75 days. The real chain's are 3-55, so the
problem moves from *timing* to *quantity*.

Three problems found along the way:

- **The BOM was read upside down at first.** Willems arcs run from a part to the
  stage that uses it. Read the other way, every product had an empty BOM and the
  plan came out 100% buildable. A test now checks the direction.
- **A safety-stock bug the simulated data hid.** The empirical version multiplied
  one day's demand by the lead time, so demand spread grew with L instead of
  sqrt(L). At the simulated demand spread (CV 0.25) it barely showed; at the real
  one (CV about 0.95) it doubled the answer. Fixed, with a test.
- **The detector's window was too short for real vendors.** 120 days worked on
  simulated data with frequent orders. Real vendors order too rarely, so the
  window is now one year.

## What we decided, and why

1. **Order more, not faster.** 59% of the plan is short, yet no top part is past
   its order-by date. The fix is order quantity; paying for speed buys nothing.
2. **Keep ordering to the P95 lead time.** It comes from each part's real
   distribution; ordering to the average means arriving late about half the time
   on any part whose lead time varies.
3. **Don't trust on-time scores against the scheduled date.** 89% exact-date hits
   is a moved date, not a punctual vendor. Keep the first promise and score
   against it.
4. **Judge vendors over a year, not 120 days.** Otherwise most real vendors
   can't be judged at all.
5. **Group alerts by supplier.** When stock is tight it cuts the worst urgent
   queue from 41 to 15. It is still over the limit, so it needs a top-N rule too.
6. **Leave allocation to the business.** Even split, highest value first and
   fixed priority leave the worst product at 23-25%; the trade-off is shown, not
   chosen.

**Limits:** stock, open orders, minimum orders and the original promise dates are
simulated, and the vendor-to-part match is an assumption, so the buildability,
alert and incident numbers show the method, not a real plant's position.
Supplier capacity is assumed. The real-history detector flags have no ground
truth. SCMS vendors ship medicines and test kits, not tool parts.

## How to run it

```bash
pip install -r requirements.txt
python download_data.py     # fetches both datasets into data/raw/ (no login; gitignored)

python run_supply.py        # ~20s  buildability, shortage drivers, suppliers -> docs/RESULTS.md
python extend.py            # ~10s  expedites, tier-2 sites, calibration     -> docs/EXTENSIONS.md
python complete.py          # ~10s  safety stock, detector sweep, alerts     -> docs/COMPLETION.md
python run_pass4.py         # ~20s  supplier incidents and alert delivery    -> docs/INCIDENTS_AND_TRANSPORT.md
python build_dashboard.py   #       rebuilds docs/index.html from out/*.json

python -m pytest tests -q   # 84 tests (6 need data/raw and skip without it)
```

`download_data.py` gets:

- Willems workbook: https://seanwillems.com/wp-content/uploads/2020/11/MSOM_Data_Set_Willems_InExcel.zip
- SCMS delivery history: https://raw.githubusercontent.com/ColbyRobinson/Supply-Chain-Shipment-Delay-Prediction/HEAD/data/SCMS_Delivery_History_Dataset_20150929.csv

Every number above comes from the files in `out/`. The full write-ups are in
[`docs/`](docs/), and a weekly materials-review pack is generated at
[`out/materials_review_pack.md`](out/materials_review_pack.md).

**Data credits.** Sean P. Willems, "Real-world multiechelon supply chains used
for inventory optimization", *Manufacturing & Service Operations Management*
10(1):19-23, 2008 (open to researchers who cite the paper). USAID Supply Chain
Management System (SCMS) Delivery History Dataset, US-government open data.

## Code layout

```
download_data.py           downloads the two public datasets into data/raw/
src/data_loaders.py        Willems and SCMS loaders, cleaning rules, the vendor-to-part mapping
src/supply.py              builds the supply base: real BOM + real vendors + simulated stock and orders
src/buildability.py        BOM explosion, weekly buildability, Monte Carlo, allocation, order-by dates
src/supplier_analytics.py  on-time both ways, deterioration detector, real-history scan, risk score, per-part rows
src/inventory.py           safety stock, lot sizing, supplier capacity
src/routing.py             severity tiers, response times, routing
src/incidents.py           grouping alerts by supplier, the load check
src/transport.py           alert delivery: outbox, retries, dead letters, rate limits
tests/mini_fixture.py      a tiny hand-made chain in the real file formats, so tests run without data
```
