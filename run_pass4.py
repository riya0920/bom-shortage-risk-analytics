"""Incidents per supplier, and an alert transport.

Writes docs/INCIDENTS_AND_TRANSPORT.md and out/pass4.json.

The per-part rows come from `supplier_analytics.part_risk_rows`: real plan,
real lead times, real unit costs, SIMULATED stock and open orders. Because the
stock is simulated, the load check is also run with stock scaled down (x0.5,
x0.25) to show how the queue behaves when the position is tighter.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import incidents as IN            # noqa: E402
import routing as RT              # noqa: E402
import supplier_analytics as SA   # noqa: E402
import supply                     # noqa: E402
import transport as TP            # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"
DOCS = ROOT / "docs"
STOCK_SCALES = (1.0, 0.5, 0.25)
STRESS_SCALE = 0.25     # the case the transport is exercised on


def risk_rows(base, stock_scale: float = 1.0) -> list:
    return SA.part_risk_rows(base, stock_scale)


def incidents_for(base, stock_scale: float) -> dict:
    rows = risk_rows(base, stock_scale)
    alerts, _ = RT.dedupe(RT.build_alerts(rows))
    inc = IN.group_by_supplier(alerts)
    lc = IN.load_check(alerts, inc)
    load_before, load_after = lc.pop("load_before"), lc.pop("load_after")
    return {
        "stock_scale": stock_scale,
        "load_check": lc,
        "concentration": IN.concentration(alerts),
        "calibration": IN.calibrate(rows),
        "value_triage": IN.value_triage(inc),
        "already_short": sum(1 for r in rows if r["days_of_slack"] <= 0),
        "runs_out_in_horizon": sum(1 for r in rows
                                   if r["runout_day"] < supply.WEEKS * 7),
        "n_rows": len(rows),
        "load_before": load_before,
        "load_after": load_after,
        "sample_incident": max(inc, key=lambda i: i.n_parts).as_dict(),
        "_incidents": inc,
    }


def stage_incidents() -> dict:
    base = supply.build()
    runs = {s: incidents_for(base, s) for s in STOCK_SCALES}
    out = {k: v for k, v in runs[1.0].items()}
    out["sensitivity"] = [{
        "stock_scale": s,
        "p1_alerts": r["load_check"]["p1_alerts"],
        "p1_incidents": r["load_check"]["p1_incidents"],
        "alerts": r["load_check"]["alerts"],
        "incidents": r["load_check"]["incidents"],
        "worst_p1_before": r["load_check"]["worst_p1_before"]["items"],
        "worst_p1_after": r["load_check"]["worst_p1_after"]["items"],
        "busiest_after": r["load_check"]["busiest_after"]["items"],
        "passes_after": r["load_check"]["passes_after"],
        "already_short": r["already_short"],
        "value_at_risk": r["load_check"]["value_before"],
        "top10_share_of_p1_value": r["value_triage"]["top_n_share_of_p1_value"],
        "incidents_for_80pct": r["value_triage"]["incidents_for_80pct_of_p1_value"],
        "top_supplier_share": r["concentration"]["top_share"],
    } for s, r in runs.items()]
    stress = runs[STRESS_SCALE]
    out["stress"] = {k: v for k, v in stress.items() if k != "_incidents"}
    out["_stress_incidents"] = stress["_incidents"]
    return out


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
    lc = ic["load_check"]
    st = ic["stress"]
    slc, scon, scal, svt = (st["load_check"], st["concentration"],
                            st["calibration"], st["value_triage"])

    A("# Incidents and a transport\n")
    A(f"Generated by `run_pass4.py` in {d['elapsed_s']:.0f} s. Per-part rows use the "
      "real plan, real lead times and real unit costs, with **simulated** stock and "
      "open orders.\n")

    A("## 1. Severity per supplier\n")
    A("An incident is one decision with one owner: a supplier that has slipped "
      "puts every part it ships at risk at once, and the response is one phone "
      "call. One incident per *(supplier, severity)*, because a supplier with two "
      "P1 parts and nine P4s is two conversations on two clocks.\n")
    A("### At the simulated stock level\n")
    A("| | alerts | incidents |")
    A("|---|---:|---:|")
    A(f"| items | {lc['alerts']} | **{lc['incidents']}** ({lc['reduction_pct']:.0f}% fewer) |")
    A(f"| P1 items | {lc['p1_alerts']} | {lc['p1_incidents']} |")
    A(f"| busiest owner | {lc['busiest_before']['role']} {lc['busiest_before']['items']} | "
      f"{lc['busiest_after']['role']} **{lc['busiest_after']['items']}** |")
    A(f"| parts covered | {lc['parts_covered_before']} | {lc['parts_covered_after']} |")
    A(f"| value at risk | {lc['value_before']:,.0f} | {lc['value_after']:,.0f} |")
    A(f"\n{ic['runs_out_in_horizon']} of {ic['n_rows']} parts run out inside 13 weeks "
      f"and {ic['already_short']} are already past the point where a normal order "
      "could arrive in time. **Nothing is P1.** Real lead times are 3-55 days, so "
      "every part that will run out can still be ordered in time. The simulated "
      "version had 14-75 day lead times and put 26 urgent items on one owner; "
      "that finding does not survive real lead times. The grouped queue still "
      f"fails the load check on total size ({lc['busiest_after']['items']} items for "
      f"one owner against a limit of {lc['per_owner_limit']}), but those are routine "
      "P3/P4 items that go out as a weekly digest.\n")

    A("### When stock is tighter (sensitivity)\n")
    A("The stock level is simulated, so the same check is run with on-hand stock "
      "scaled down. Open orders and everything real stay the same.\n")
    A("| on-hand stock | P1 alerts | P1 incidents | worst P1 queue (per part) | "
      "worst P1 queue (per supplier) | busiest owner | passes? | top-10 share of P1 value |")
    A("|---:|---:|---:|---:|---:|---:|:--:|---:|")
    for r in ic["sensitivity"]:
        A(f"| x{r['stock_scale']:.2f} | {r['p1_alerts']} | {r['p1_incidents']} | "
          f"{r['worst_p1_before']} | {r['worst_p1_after']} | {r['busiest_after']} | "
          f"{'yes' if r['passes_after'] else 'no'} | "
          f"{100 * r['top10_share_of_p1_value']:.0f}% |")
    A(f"\nAt x{STRESS_SCALE:.2f} stock the grouping does real work: "
      f"{slc['p1_alerts']} P1 part alerts become {slc['p1_incidents']} P1 "
      f"incidents, and the worst urgent queue goes from "
      f"{slc['worst_p1_before']['items']} to {slc['worst_p1_after']['items']} "
      f"against a limit of {slc['p1_limit']}. Grouping helps by as much as alerts "
      f"are concentrated: the top five vendors carry "
      f"{scon['top_share'] * 100:.0f}% of alerts across {scon['n_suppliers']} vendors.\n")
    A("| vendor | alerts | of which P1 | parts | share |")
    A("|---|---:|---:|---:|---:|")
    for r in scon["top"]:
        A(f"| {r['supplier_id']} | {r['alerts']} | {r['P1']} | {r['n_parts']} | "
          f"{r['share_of_alerts'] * 100:.1f}% |")
    A("\nTightening the slack ratio that fires P1 (stress case):\n")
    A("| P1 slack ratio | P1 alerts | P1 incidents | worst P1 queue | within limit? |")
    A("|---:|---:|---:|---:|:--:|")
    for r in scal["sweep"]:
        A(f"| {r['p1_ratio']:.2f} | {r['p1_alerts']} | {r['p1_incidents']} | "
          f"{r['worst_p1_owner']} {r['worst_p1_items']} | "
          f"{'yes' if r['p1_passes'] else 'no'} |")
    A(f"\nTop-{svt['top_n']} triage in the stress case: the top {svt['top_n']} P1 "
      f"incidents carry **{svt['top_n_share_of_p1_value'] * 100:.0f}%** of the P1 "
      f"value at risk; {svt['incidents_for_80pct_of_p1_value']} incidents carry 80%.\n")
    A(f"> {svt['verdict']}\n")

    A("\n## 2. A transport\n")
    A(f"Exercised on the stress-case incidents (stock x{STRESS_SCALE:.2f}), because "
      "at the base stock level there is nothing urgent to send. The receiver is a "
      "real HTTP server on a local socket.\n")
    A(f"{tr['incidents']} incidents become **{tr['messages_prepared']} messages**: "
      f"{tr['urgent_messages']} individual P1/P2 items and {tr['digest_messages']} "
      "digests. Routine items go out as one digest per owner and severity.\n")
    rt = tr["retry"]
    A("**At-least-once, and the receiver deduplicates.** The sender marks a "
      "message sent only after the POST succeeds, and every message carries an "
      "idempotency key.\n")
    A(f"- Receiver rejects its first {rt['receiver_failed_first']} requests with a "
      f"503. First pass: {rt['first_pass_failed']} failures, nothing marked sent.")
    A(f"- Retried on a backoff clock: all {rt['messages']} messages delivered, "
      f"receiver saw {rt['receiver_saw_requests']} requests for {rt['delivered']} "
      f"deliveries and {rt['receiver_duplicates']} duplicates.")
    A(f"- Re-running the whole job: {tr['rerun']['duplicate_enqueues']} duplicate "
      f"enqueues rejected, {tr['rerun']['sent']} extra sends, "
      f"{tr['receiver_duplicates_after_rerun']} duplicates at the receiver.")
    cw = tr["crash_window"]
    A(f"- Crash window (delivered, then died before marking): state after crash "
      f"`{cw['state_after_crash']}`, re-sent on restart ({cw['resent_on_restart']}), "
      f"receiver saw {cw['receiver_requests']} requests, recorded "
      f"{cw['receiver_distinct_deliveries']} delivery and suppressed "
      f"{cw['receiver_duplicates_suppressed']} duplicate.")
    dl = tr["dead_letter"]
    A(f"- Endpoint that never answers: messages land in `DEAD` "
      f"(`{dl['counts']}`), {dl['still_in_outbox']} kept in the outbox, none dropped.")
    rl = tr["rate_limit"]
    A(f"- Rate limit of {rl['limit_per_recipient']} per recipient: {rl['sent']} sent, "
      f"{rl['deferred']} deferred, nothing dropped ({rl['nothing_dropped']}), max "
      f"{rl['max_to_one_recipient']} to one recipient, urgent first.\n")
    A("The generator and the idempotency key are both checked for being identical "
      "across separate processes (Python salts string hashing per process, which "
      "once made every number in this project differ run to run).\n")
    return "\n".join(L) + "\n"


def main() -> None:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    ic = stage_incidents()
    print("  incidents done")
    ic.pop("_incidents")
    stress_inc = ic.pop("_stress_incidents")
    d = {"incidents": ic, "transport": stage_transport(stress_inc)}
    print("  transport done")
    d["elapsed_s"] = time.time() - t0
    (OUT / "pass4.json").write_text(json.dumps(d, indent=2, default=str),
                                    encoding="utf-8")
    (DOCS / "INCIDENTS_AND_TRANSPORT.md").write_text(report(d), encoding="utf-8")
    print(f"wrote docs/INCIDENTS_AND_TRANSPORT.md in {d['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
