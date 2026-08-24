"""Pass 4: incidents, the transport, and the reproducibility bug.

The most important test here is the boring one: two builds in two PROCESSES
agree. That is the bug that made every number in this project irreproducible for
three passes, and it survived because it is deterministic inside one process --
so a test that builds twice in one interpreter would have passed throughout.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import incidents as IN            # noqa: E402
import routing as RT              # noqa: E402
import supply                     # noqa: E402
import transport as TP            # noqa: E402


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def test_the_generator_is_deterministic_across_processes():
    """`for sub in set(subs)` drove rng calls, and set iteration over strings
    varies between processes (PEP 456). Same seed, different supply base, every
    run. An in-process check cannot see it."""
    prog = (f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
            "import supply\n"
            "b = supply.build()\n"
            "print(sum(c.on_hand for c in b.components.values()), "
            "len(b.components), len(b.bom))\n")
    outs = {subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, check=True).stdout.strip()
            for _ in range(3)}
    assert len(outs) == 1, f"build() differs across processes: {outs}"


def test_no_rng_is_driven_by_set_iteration():
    """The shape of the bug, not just this instance."""
    src = (SRC / "supply.py").read_text(encoding="utf-8")
    assert "for sub in set(" not in src
    assert "dict.fromkeys(subs)" in src


# ---------------------------------------------------------------------------
# incidents
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    import numpy as np
    import supplier_analytics as SA
    base = supply.build()
    risk = {r["supplier_id"]: r for r in SA.risk_score(base)}
    out = []
    for part, c in sorted(base.components.items()):
        if c.supplier_id is None:
            continue
        s = base.suppliers[c.supplier_id]
        weekly = sum(supply.explode(base, p, 1.0).get(part, 0.0)
                     * float(np.mean(base.demand[p])) for p in base.products)
        d = weekly / 7.0
        if d <= 0:
            continue
        slack = c.on_hand / d - s.lead_mean_days
        out.append({"part": part, "supplier_id": c.supplier_id,
                    "days_of_slack": slack, "lead_days": s.lead_mean_days,
                    "units_at_risk": max(-slack, 0.0) * d,
                    "value_at_risk": max(-slack, 0.0) * d * c.unit_cost,
                    "single_sourced": c.single_sourced,
                    "risk_score": risk.get(c.supplier_id, {}).get("risk_score", 0.0)})
    return out


@pytest.fixture(scope="module")
def alerts(rows):
    a, _ = RT.dedupe(RT.build_alerts(rows))
    return a


@pytest.fixture(scope="module")
def incs(alerts):
    return IN.group_by_supplier(alerts)


def test_grouping_loses_no_part_and_no_value(alerts, incs):
    """The check that keeps the reduction honest -- a report that leads with
    "260 became 108" while parts vanish has substituted a metric for the work."""
    lc = IN.load_check(alerts, incs)
    assert lc["parts_covered_after"] == lc["parts_covered_before"]
    assert lc["value_after"] == pytest.approx(lc["value_before"])
    assert lc["incidents"] < lc["alerts"]


def test_one_incident_per_supplier_and_severity(incs):
    keys = [(i.supplier_id, i.severity) for i in incs if i.supplier_id]
    assert len(keys) == len(set(keys))


def test_severities_are_not_collapsed_onto_the_worst(incs):
    """A supplier with a P1 and some P4s is two conversations on two clocks.
    Collapsing them forces the whole group onto the P1 SLA -- the original
    problem, arrived at from the other direction."""
    multi = {}
    for i in incs:
        if i.supplier_id:
            multi.setdefault(i.supplier_id, set()).add(i.severity)
    assert any(len(v) > 1 for v in multi.values())
    for i in incs:
        assert i.response_hours == RT.TIERS[i.severity]["response_hours"]


def test_an_alert_without_a_supplier_stays_individual():
    a = RT.build_alerts([
        {"part": "P-1", "supplier_id": None, "days_of_slack": -3.0,
         "lead_days": 30.0, "units_at_risk": 5.0, "value_at_risk": 50.0},
        {"part": "P-2", "supplier_id": None, "days_of_slack": -4.0,
         "lead_days": 30.0, "units_at_risk": 6.0, "value_at_risk": 60.0}])
    inc = IN.group_by_supplier(a)
    assert len(inc) == 2
    assert all(i.supplier_id is None and i.n_parts == 1 for i in inc)


def test_the_load_check_still_fails_after_grouping(alerts, incs):
    """The finding. Grouping was the fix the README named and it is worth
    roughly half the count -- and the queue is still four times too long."""
    lc = IN.load_check(alerts, incs)
    assert lc["passes_before"] is False
    assert lc["passes_after"] is False
    assert lc["reduction_pct"] > 40


def test_alerts_are_not_concentrated_on_a_few_suppliers(alerts):
    """Why grouping could not have been sufficient: the stated diagnosis
    assumed concentration, and there is not much."""
    c = IN.concentration(alerts)
    assert c["n_suppliers"] > 30
    assert c["top_share"] < 0.35


def test_the_threshold_sweep_has_a_floor_and_it_is_correct(rows):
    """Parts already at or below zero slack cannot be demoted by a slack-ratio
    threshold, and should not be: ordering now does not recover them."""
    cal = IN.calibrate(rows)
    tight = cal["sweep"][-1]
    already_short = sum(1 for r in rows if r["days_of_slack"] <= 0)
    assert already_short > 0
    assert tight["p1_alerts"] >= already_short * 0.5
    assert cal["first_passing_p1"] is None


def test_calibrate_restores_the_severity_function(rows):
    before = RT.severity_of
    IN.calibrate(rows, ratios=(0.25, 0.1))
    assert RT.severity_of is before


def test_value_triage_reports_a_verdict_rather_than_a_number(incs):
    vt = IN.value_triage(incs, top_n=10)
    assert 0.0 <= vt["top_n_share_of_p1_value"] <= 1.0
    assert vt["incidents_for_80pct_of_p1_value"] >= 1
    assert "workable" in vt["verdict"]


# ---------------------------------------------------------------------------
# the transport
# ---------------------------------------------------------------------------

@pytest.fixture
def outbox(tmp_path):
    ob = TP.Outbox(tmp_path / "o.db")
    yield ob
    ob.close()


def test_the_idempotency_key_is_stable_across_processes():
    """Not `hash()`. Here that bug would mean every restart re-sending
    everything -- the same family as the generator bug above."""
    payload = {"b": 2, "a": [1, 2, 3]}
    prog = (f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
            "import transport as TP\n"
            f"print(TP.Outbox.key('BUYER', {payload!r}))\n")
    outs = {subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, check=True).stdout.strip()
            for _ in range(3)}
    assert len(outs) == 1


def test_key_ignores_dict_ordering_but_not_content():
    a = TP.Outbox.key("BUYER", {"x": 1, "y": 2})
    b = TP.Outbox.key("BUYER", {"y": 2, "x": 1})
    c = TP.Outbox.key("BUYER", {"x": 1, "y": 3})
    d = TP.Outbox.key("MANAGER", {"x": 1, "y": 2})
    assert a == b and a != c and a != d


def test_enqueue_is_idempotent(outbox):
    p = {"kind": "incident", "key": "abc"}
    assert outbox.enqueue("BUYER", "P1", p)["queued"] is True
    assert outbox.enqueue("BUYER", "P1", p)["queued"] is False
    assert outbox.counts() == {TP.PENDING: 1}


def test_delivery_happens_over_a_real_socket(outbox):
    rx = TP.CapturingReceiver()
    try:
        r = TP.send_all(outbox, TP.WebhookSink(rx.url),
                        [{"recipient": "BUYER", "severity": "P1",
                          "payload": {"key": "k1"}}])
        assert r["sent"] == 1
        assert len(rx.delivered) == 1
        assert rx.delivered[0]["body"]["recipient"] == "BUYER"
        assert rx.delivered[0]["body"]["payload"]["key"] == "k1"
    finally:
        rx.close()


def test_a_failing_receiver_is_retried_and_then_recovers(outbox):
    rx = TP.CapturingReceiver(fail_first=2)
    sink = TP.WebhookSink(rx.url)
    try:
        first = TP.send_all(outbox, sink,
                            [{"recipient": "BUYER", "severity": "P1",
                              "payload": {"key": "k"}}],
                            now=0.0, max_attempts=4, base_backoff_s=1.0)
        assert first["sent"] == 0 and first["failed"] == 1
        h = TP.drain(outbox, sink, start=1.0, step_s=4.0, rounds=5,
                     max_attempts=4, base_backoff_s=1.0)
        assert h["counts"].get(TP.SENT) == 1
        assert rx.requests == 3 and len(rx.delivered) == 1
    finally:
        rx.close()


def test_the_crash_window_is_the_case_dedupe_exists_for(outbox):
    """An ordinary retry never delivers twice -- the failed send never arrived.
    The duplicate comes from succeeding and then dying before marking."""
    rx = TP.CapturingReceiver()
    sink = TP.WebhookSink(rx.url)
    try:
        outbox.enqueue("BUYER", "P1", {"key": "k"}, now=0.0)
        row = outbox.due(0.0)[0]
        sink.send(row["recipient"], row["severity"], row["idem_key"],
                  json.loads(row["body"]))          # delivered
        assert outbox.counts() == {TP.PENDING: 1}   # ... then we "crash"
        TP.send_all(outbox, sink, [], now=1.0)      # restart re-sends
        assert rx.requests == 2
        assert len(rx.delivered) == 1
        assert rx.duplicates == 1
    finally:
        rx.close()


def test_exhausted_retries_become_dead_letters_and_stay(outbox):
    sink = TP.WebhookSink("http://127.0.0.1:9/never")
    TP.send_all(outbox, sink, [{"recipient": "BUYER", "severity": "P1",
                                "payload": {"key": "k"}}],
                now=0.0, max_attempts=2, base_backoff_s=0.1)
    TP.drain(outbox, sink, start=0.2, step_s=1.0, rounds=4, max_attempts=2,
             base_backoff_s=0.1)
    dead = outbox.dead_letters()
    assert len(dead) == 1
    assert dead[0]["last_error"]
    assert sum(outbox.counts().values()) == 1, "a dead letter was dropped"


def test_rate_limiting_defers_and_never_drops(outbox):
    rx = TP.CapturingReceiver()
    try:
        msgs = [{"recipient": "BUYER", "severity": "P2",
                 "payload": {"key": f"k{i}"}} for i in range(9)]
        r = TP.send_all(outbox, TP.WebhookSink(rx.url), msgs, now=0.0,
                        per_recipient_limit=4)
        assert r["sent"] == 4
        assert r["deferred"] == 5
        assert r["sent"] + r["deferred"] == len(msgs)
        assert len(rx.delivered) == 4
    finally:
        rx.close()


def test_the_budget_is_spent_on_the_urgent_items_first(outbox):
    rx = TP.CapturingReceiver()
    try:
        msgs = ([{"recipient": "BUYER", "severity": "P4",
                  "payload": {"key": f"low{i}"}} for i in range(5)]
                + [{"recipient": "BUYER", "severity": "P1",
                    "payload": {"key": f"hi{i}"}} for i in range(2)])
        TP.send_all(outbox, TP.WebhookSink(rx.url), msgs, now=0.0,
                    per_recipient_limit=2)
        sent = [d["body"]["severity"] for d in rx.delivered]
        assert sent == ["P1", "P1"], sent
    finally:
        rx.close()


def test_routine_severities_are_digested_and_urgent_are_not(incs):
    msgs = TP.prepare(incs)
    kinds = {m["payload"]["kind"] for m in msgs}
    assert kinds == {"incident", "digest"}
    for m in msgs:
        if m["severity"] in TP.URGENT:
            assert m["payload"]["kind"] == "incident"
        else:
            assert m["payload"]["kind"] == "digest"
    n_routine = sum(1 for i in incs if i.severity not in TP.URGENT)
    n_digests = sum(1 for m in msgs if m["payload"]["kind"] == "digest")
    assert n_digests < n_routine


def test_a_digest_accounts_for_every_incident_it_replaces(incs):
    msgs = TP.prepare(incs)
    covered = sum(m["payload"]["n_incidents"] for m in msgs
                  if m["payload"]["kind"] == "digest")
    assert covered == sum(1 for i in incs if i.severity not in TP.URGENT)
