"""Pass 4: severity per supplier, and an alert transport.

README items 2 and 3. Writes docs/INCIDENTS_AND_TRANSPORT.md and out/pass4.json.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import incidents as IN            # noqa: E402
import routing as RT              # noqa: E402
import supplier_analytics as SA   # noqa: E402
import supply                     # noqa: E402
import transport as TP            # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"
DOCS = ROOT / "docs"


def risk_rows(base) -> list:
    risk = {r["supplier_id"]: r for r in SA.risk_score(base)}
    rows = []
    for part, c in sorted(base.components.items()):
        if c.supplier_id is None:
            continue
        s = base.suppliers[c.supplier_id]
        weekly = sum(supply.explode(base, p, 1.0).get(part, 0.0)
                     * float(np.mean(base.demand[p])) for p in base.products)
        d_day = weekly / 7.0
        if d_day <= 0:
            continue
        slack = c.on_hand / d_day - s.lead_mean_days
        rows.append({"part": part, "supplier_id": c.supplier_id,
                     "days_of_slack": slack, "lead_days": s.lead_mean_days,
                     "units_at_risk": max(-slack, 0.0) * d_day,
                     "value_at_risk": max(-slack, 0.0) * d_day * c.unit_cost,
                     "single_sourced": c.single_sourced,
                     "risk_score": risk.get(c.supplier_id, {}).get(
                         "risk_score", 0.0)})
    return rows


def stage_incidents() -> dict:
    base = supply.build()
    rows = risk_rows(base)
    alerts, _ = RT.dedupe(RT.build_alerts(rows))
    inc = IN.group_by_supplier(alerts)
    lc = IN.load_check(alerts, inc)
    lc.pop("load_before", None)
    lc.pop("load_after", None)
    return {
        "load_check": lc,
        "concentration": IN.concentration(alerts),
        "calibration": IN.calibrate(rows),
        "value_triage": IN.value_triage(inc),
        "already_short": sum(1 for r in rows if r["days_of_slack"] <= 0),
        "n_rows": len(rows),
        "load_before": RT.load_by_role(alerts),
        "load_after": IN.load_by_role(inc),
        "sample_incident": max(inc, key=lambda i: i.n_parts).as_dict(),
        "_incidents": inc,
    }


def stage_transport(inc: list) -> dict:
    OUT.mkdir(exist_ok=True)
    out: dict = {}

    # 1. happy path, and what the digest does to the message count
    db = OUT / "pass4_outbox.db"
    db.unlink(missing_ok=True)
    ob = TP.Outbox(db)
    rx = TP.CapturingReceiver()
    msgs = TP.prepare(inc)
    try:
        r = TP.send_all(ob, TP.WebhookSink(rx.url), msgs, now=0.0,
                        per_recipient_limit=15)
        out["messages_prepared"] = len(msgs)
        out["incidents"] = len(inc)
        out["urgent_messages"] = sum(1 for m in msgs
                                     if m["severity"] in TP.URGENT)
        out["digest_messages"] = sum(1 for m in msgs
                                     if m["severity"] not in TP.URGENT)
        out["first_pass"] = {k: r[k] for k in
                             ("queued", "sent", "failed", "deferred", "dead")}
        out["received"] = len(rx.delivered)
        out["counts"] = r["counts"]

        # 2. re-running the whole job must not re-send anything
        r2 = TP.send_all(ob, TP.WebhookSink(rx.url), msgs, now=1.0,
                         per_recipient_limit=15)
        out["rerun"] = {"duplicate_enqueues": r2["duplicate_enqueues"],
                        "sent": r2["sent"]}
        out["receiver_duplicates_after_rerun"] = rx.duplicates
    finally:
        rx.close()
        ob.close()

    # 3. a receiver that fails, then recovers
    db2 = OUT / "pass4_outbox_retry.db"
    db2.unlink(missing_ok=True)
    ob2 = TP.Outbox(db2)
    rx2 = TP.CapturingReceiver(fail_first=3)
    sink2 = TP.WebhookSink(rx2.url)
    try:
        small = msgs[:2]
        first = TP.send_all(ob2, sink2, small, now=0.0, max_attempts=4,
                            base_backoff_s=1.0, per_recipient_limit=15)
        hist = TP.drain(ob2, sink2, start=1.0, step_s=4.0, rounds=6,
                        max_attempts=4, base_backoff_s=1.0,
                        per_recipient_limit=15)
        out["retry"] = {
            "messages": len(small), "first_pass_failed": first["failed"],
            "receiver_failed_first": 3,
            "rounds": hist["rounds"], "counts": hist["counts"],
            "delivered": len(rx2.delivered),
            "receiver_saw_requests": rx2.requests,
            "receiver_duplicates": rx2.duplicates,
            "dead_letters": hist["dead_letters"]}
    finally:
        rx2.close()
        ob2.close()

    # 4. a receiver that never recovers -> dead letters, kept not dropped
    db3 = OUT / "pass4_outbox_dead.db"
    db3.unlink(missing_ok=True)
    ob3 = TP.Outbox(db3)
    dead_sink = TP.WebhookSink("http://127.0.0.1:9/never")
    try:
        TP.send_all(ob3, dead_sink, msgs[:2], now=0.0, max_attempts=3,
                    base_backoff_s=0.5)
        h = TP.drain(ob3, dead_sink, start=0.5, step_s=4.0, rounds=8,
                     max_attempts=3, base_backoff_s=0.5)
        out["dead_letter"] = {"counts": h["counts"],
                              "dead_letters": h["dead_letters"],
                              "still_in_outbox": sum(h["counts"].values())}
    finally:
        ob3.close()

    # 4b. the crash window: the POST succeeded and the process died before the
    # outbox was marked. This is the case at-least-once EXISTS for, and it is
    # the only one that exercises receiver-side dedupe -- an ordinary retry
    # never produces a duplicate, because the failed send was never delivered.
    db5 = OUT / "pass4_outbox_crash.db"
    db5.unlink(missing_ok=True)
    ob5 = TP.Outbox(db5)
    rx5 = TP.CapturingReceiver()
    sink5 = TP.WebhookSink(rx5.url)
    try:
        one = msgs[:1]
        for m in one:
            ob5.enqueue(m["recipient"], m["severity"], m["payload"], now=0.0)
        row = ob5.due(0.0)[0]
        # deliver, then "crash" -- do not mark it sent
        sink5.send(row["recipient"], row["severity"], row["idem_key"],
                   json.loads(row["body"]))
        after_crash = dict(ob5.counts())
        # restart: the row is still PENDING, so it is sent again
        r5 = TP.send_all(ob5, sink5, [], now=1.0)
        out["crash_window"] = {
            "state_after_crash": after_crash,
            "resent_on_restart": r5["sent"],
            "receiver_requests": rx5.requests,
            "receiver_distinct_deliveries": len(rx5.delivered),
            "receiver_duplicates_suppressed": rx5.duplicates,
            "final_counts": ob5.counts()}
    finally:
        rx5.close()
        ob5.close()

    # 5. rate limiting: one recipient, more messages than the budget
    db4 = OUT / "pass4_outbox_rate.db"
    db4.unlink(missing_ok=True)
    ob4 = TP.Outbox(db4)
    rx4 = TP.CapturingReceiver()
    try:
        limit = 5
        r4 = TP.send_all(ob4, TP.WebhookSink(rx4.url), msgs, now=0.0,
                         per_recipient_limit=limit)
        per_r: dict = {}
        for d in rx4.delivered:
            k = d["body"]["recipient"]
            per_r[k] = per_r.get(k, 0) + 1
        out["rate_limit"] = {
            "limit_per_recipient": limit, "messages": len(msgs),
            "sent": r4["sent"], "deferred": r4["deferred"],
            "delivered_per_recipient": per_r,
            "max_to_one_recipient": max(per_r.values()) if per_r else 0,
            "nothing_dropped": r4["sent"] + r4["deferred"] == len(msgs)}
    finally:
        rx4.close()
        ob4.close()
    return out


def report(d: dict) -> str:
    L: list[str] = []
    A = L.append
    ic, tr = d["incidents"], d["transport"]
    lc, con, cal, vt = (ic["load_check"], ic["concentration"],
                        ic["calibration"], ic["value_triage"])

    A("# Incidents, a transport, and a diagnosis that was half right\n")
    A(f"The last two buildable items on this project's list. Generated by "
      f"`run_pass4.py` in {d['elapsed_s']:.0f} s.\n")

    A("## 0. First: every number in this project was irreproducible\n")
    A("`supply.build()` contained `for sub in set(subs):`, and the loop body "
      "makes `rng` calls. **Python salts string hashing per process** (PEP 456), "
      "so a set of strings iterates in a different order in every interpreter — "
      "and the random draws are then consumed in a different order, producing a "
      "different supply base on every run. Two runs of the same script gave a "
      "value at risk of 656,531 and 1,059,890: a 61% swing with no input "
      "changed.\n")
    A("It survived three passes because **it is deterministic within one "
      "process**. Three builds in one interpreter agree exactly; three runs of "
      "the same script do not, and nothing in a normal test session looks at "
      "the second case. Fixed with `dict.fromkeys`, which deduplicates in "
      "insertion order. This is the fourth instance of the same family of bug "
      "across these nine projects — the other three were `hash()` used as a "
      "seed.\n")
    A("Every figure below comes from the fixed generator and is identical "
      "across runs; the figures in the earlier passes did not and were not.\n")

    A("\n## 1. Severity per supplier — the fix the README named\n")
    A("An incident is not a summary of alerts. It is one **decision** with one "
      "owner: a supplier that has slipped puts every part it ships at risk at "
      "once, and the response to all of them is the same phone call. The parts "
      "become the incident's contents, which is where they were always useful. "
      "One incident per *(supplier, severity)* rather than per supplier, "
      "because a supplier with two P1 parts and nine P4s is two conversations "
      "on two clocks.\n")
    A("| | alerts | incidents |")
    A("|---|---:|---:|")
    A(f"| items | {lc['alerts']} | **{lc['incidents']}** ({lc['reduction_pct']:.0f}% fewer) |")
    A(f"| P1 items | {lc['p1_alerts']} | **{lc['p1_incidents']}** |")
    A(f"| busiest owner | {lc['busiest_before']['role']} {lc['busiest_before']['items']} | "
      f"{lc['busiest_after']['role']} **{lc['busiest_after']['items']}** |")
    A(f"| worst P1 queue | {lc['worst_p1_before']['role']} {lc['worst_p1_before']['items']} | "
      f"{lc['worst_p1_after']['role']} **{lc['worst_p1_after']['items']}** |")
    A(f"| parts covered | {lc['parts_covered_before']} | {lc['parts_covered_after']} |")
    A(f"| value at risk | {lc['value_before']:,.0f} | {lc['value_after']:,.0f} |")
    A("\nThe last two rows are the ones that keep this honest: grouping changes "
      "the **count** an owner sees and not the work, so the parts and the value "
      "are carried through unchanged and the comparison is visibly like for "
      "like.\n")

    A(f"\n### And it still fails the load check\n")
    A(f"Against a limit of {lc['per_owner_limit']} open items per owner and "
      f"{lc['p1_limit']} P1s — judgements, stated as such — the queue passes "
      f"**neither** before nor after: `passes_before={lc['passes_before']}`, "
      f"`passes_after={lc['passes_after']}`. The busiest owner still has "
      f"{lc['busiest_after']['items']} items and the worst P1 queue still has "
      f"{lc['worst_p1_after']['items']}.\n")
    A("**The stated diagnosis was half right.** It said *one late supplier puts "
      "every part it ships into P1 at once*. That happens, and grouping is worth "
      f"{lc['reduction_pct']:.0f}%. But the alerts are not concentrated: the top "
      f"five suppliers are **{con['top_share'] * 100:.0f}%** of them across "
      f"{con['n_suppliers']} suppliers.\n")
    A("| supplier | alerts | of which P1 | parts | share |")
    A("|---|---:|---:|---:|---:|")
    for r in con["top"]:
        A(f"| {r['supplier_id']} | {r['alerts']} | {r['P1']} | {r['n_parts']} | "
          f"{r['share_of_alerts'] * 100:.1f}% |")

    A("\n### So the threshold was swept, and it does not save it either\n")
    A("Tightening the slack ratio that fires P1, everything demoted landing in "
      "P2 (which has an SLA rather than none):\n")
    A("| P1 slack ratio | P1 alerts | P1 incidents | worst P1 queue | within limit? |")
    A("|---:|---:|---:|---:|:--:|")
    for r in cal["sweep"]:
        A(f"| {r['p1_ratio']:.2f} | {r['p1_alerts']} | {r['p1_incidents']} | "
          f"{r['worst_p1_owner']} {r['worst_p1_items']} | "
          f"{'yes' if r['p1_passes'] else 'no'} |")
    floor = cal["sweep"][-1]
    A(f"\nEven at a ratio of {floor['p1_ratio']:.2f} the worst queue is "
      f"{floor['worst_p1_items']}, still above {cal['p1_limit']}. There is a "
      f"floor, and it is correct: **{ic['already_short']} of {ic['n_rows']} "
      "parts have zero or negative slack** — they are already below their lead "
      "time, ordering now does not recover them, and no ratio threshold can or "
      "should argue that away.\n")
    A("**The queue is not too long because the policy mis-ranks. It is too long "
      "because the situation is bad.** That is a different finding from the one "
      "this project expected to make, and tuning the threshold until the table "
      "looked acceptable would have buried it.\n")

    A("\n### The only honest lever left\n")
    A(f"Stop pretending everything urgent can be worked, and say what a top-N "
      f"queue covers. The top {vt['top_n']} P1 incidents by value carry "
      f"**{vt['top_n_share_of_p1_value'] * 100:.0f}% of the P1 value at risk** "
      f"({vt['top_n_value']:,.0f} of {vt['p1_value']:,.0f}) across "
      f"{vt['top_n_parts']} of {vt['parts_in_p1']} parts; "
      f"{vt['incidents_for_80pct_of_p1_value']} incidents carry 80% of it.\n")
    A(f"> {vt['verdict']}\n")

    A("\n## 2. A transport\n")
    A("Alerts were objects, and the README said a deployment attaches a "
      "transport. Attaching one is where the problems are, and none of them are "
      "about the protocol. The receiver is a real HTTP server on a real socket — "
      "mocking it would test the code that calls the transport, which is not the "
      "part that goes wrong.\n")

    A(f"\n### Urgent is sent, routine is digested\n")
    A(f"{tr['incidents']} incidents become **{tr['messages_prepared']} "
      f"messages**: {tr['urgent_messages']} individual P1/P2 items and "
      f"{tr['digest_messages']} digests. The response to a P4 is *look at it on "
      "Friday*, and forty separate messages saying that is how a channel gets "
      "filtered to a folder — after which the P1s go there too.\n")

    A("\n### At-least-once, and the receiver deduplicates\n")
    rt = tr["retry"]
    A("A sender that marks a message sent *before* the POST loses it when the "
      "POST fails; one that marks *after* will re-send when it crashes in "
      "between. There is no third option, so this marks after and every message "
      "carries an idempotency key the receiver deduplicates on.\n")
    A(f"- Receiver rejects its first {rt['receiver_failed_first']} requests with "
      f"a 503. First pass: **{rt['first_pass_failed']} failures**, nothing "
      "marked sent.")
    A(f"- Retried on a backoff clock: all {rt['messages']} messages delivered, "
      f"receiver saw **{rt['receiver_saw_requests']} requests** for "
      f"{rt['delivered']} distinct deliveries and "
      f"**{rt['receiver_duplicates']} duplicates**, final states "
      f"`{rt['counts']}`.")
    A(f"- Re-running the entire job re-enqueues nothing: "
      f"**{tr['rerun']['duplicate_enqueues']} duplicate enqueues rejected** by "
      f"the UNIQUE key, {tr['rerun']['sent']} additional sends, and the receiver "
      f"has seen {tr['receiver_duplicates_after_rerun']} duplicates.\n")
    cw = tr["crash_window"]
    A(f"\n**And the ordinary retry never produces a duplicate**, because a "
      f"failed send was never delivered — so receiver-side dedupe is untested "
      f"by it. The case it exists for is the crash window: deliver the message, "
      f"then die before marking the outbox. State after the crash: "
      f"`{cw['state_after_crash']}` — still pending. On restart it is sent again "
      f"({cw['resent_on_restart']} send), the receiver sees "
      f"**{cw['receiver_requests']} requests** and records "
      f"**{cw['receiver_distinct_deliveries']} delivery**, suppressing "
      f"**{cw['receiver_duplicates_suppressed']}** on the idempotency key. "
      "That is the whole contract, and it only works because both halves are "
      "there.\n")
    A("The key is a SHA-256 of recipient plus canonical payload — deliberately "
      "not `hash()`, which is the bug found at the top of this document and "
      "which here would mean every restart re-sending everything.\n")

    A("\n### A dead letter is not a failure to hide\n")
    dl = tr["dead_letter"]
    A(f"Against an endpoint that never answers, the messages exhaust their "
      f"retries and land in `DEAD`: `{dl['counts']}`, "
      f"**{dl['still_in_outbox']} still in the outbox**. Dropping them would "
      "make the transport look reliable and make a P1 disappear; retrying "
      "forever turns one broken endpoint into an outage of the whole queue.\n")

    A("\n### Rate limiting is part of the alert policy, not the plumbing\n")
    rl = tr["rate_limit"]
    A(f"The load check says how many items an owner can absorb. A transport that "
      f"ignores it delivers all of them at 06:00, and the queue that was too "
      f"long on paper is now too long in somebody's inbox. With a budget of "
      f"{rl['limit_per_recipient']} per recipient: **{rl['sent']} sent, "
      f"{rl['deferred']} deferred**, nothing dropped "
      f"(`{rl['nothing_dropped']}`), and no recipient received more than "
      f"**{rl['max_to_one_recipient']}**.\n")
    A("The queue is drained in severity order, so the budget is spent on P1s "
      "rather than on whatever was enqueued first, and what exceeds it is "
      "deferred visibly — a silently held P1 is worse than a noisy one.\n")
    return "\n".join(L) + "\n"


def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    ic = stage_incidents()
    print("  incidents done")
    inc = ic.pop("_incidents")
    d = {"incidents": ic, "transport": stage_transport(inc)}
    print("  transport done")
    d["elapsed_s"] = time.time() - t0
    (OUT / "pass4.json").write_text(json.dumps(d, indent=2, default=str),
                                    encoding="utf-8")
    (DOCS / "INCIDENTS_AND_TRANSPORT.md").write_text(report(d), encoding="utf-8")
    print(f"wrote docs/INCIDENTS_AND_TRANSPORT.md in {d['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
