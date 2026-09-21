"""Incidents, the transport, and cross-process reproducibility.

The incident tests use hand-made per-part rows (mixed severities over six
suppliers), so they test the grouping logic rather than one data set's numbers.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))

import incidents as IN            # noqa: E402
import routing as RT              # noqa: E402
import transport as TP            # noqa: E402


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def test_the_supply_base_is_deterministic_across_processes():
    """Python salts string hashing per process (PEP 456), so any random draw
    driven by set iteration differs between runs while agreeing inside one
    interpreter. Only a cross-process check can see it."""
    prog = (f"import sys; sys.path[:0] = [{str(SRC)!r}, {str(ROOT / 'tests')!r}]\n"
            "from mini_fixture import build_mini\n"
            "b = build_mini()\n"
            "print(sum(c.on_hand for c in b.components.values()), len(b.pos), "
            "sum(d.received_day for d in b.sandbox), b.planted_disruptions)\n")
    outs = {subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, check=True).stdout.strip()
            for _ in range(3)}
    assert len(outs) == 1, f"build differs across processes: {outs}"


def test_no_rng_is_driven_by_set_iteration():
    """The shape of the bug, not just one instance."""
    for name in ("supply.py", "data_loaders.py", "buildability.py"):
        src = (SRC / name).read_text(encoding="utf-8")
        assert "in set(" not in src, name


# ---------------------------------------------------------------------------
# incidents
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    out = []
    slacks = [-5.0, 2.0, 8.0, 14.0, 25.0, 60.0]
    for s in range(6):
        for k, slack in enumerate(slacks[: 3 + s % 4]):
            out.append({"part": f"S{s}-P{k}", "supplier_id": f"SUP{s}",
                        "days_of_slack": slack + s, "lead_days": 20.0,
                        "units_at_risk": 10.0 * (k + 1),
                        "value_at_risk": 100.0 * (k + 1) * (s + 1),
                        "single_sourced": (s + k) % 5 == 0, "risk_score": 0.1 * s})
    return out


@pytest.fixture(scope="module")
def alerts(rows):
    a, _ = RT.dedupe(RT.build_alerts(rows))
    return a


@pytest.fixture(scope="module")
def incs(alerts):
    return IN.group_by_supplier(alerts)


def test_grouping_loses_no_part_and_no_value(alerts, incs):
    """A report that leads with "N became M" while parts vanish has swapped a
    metric for the work."""
    lc = IN.load_check(alerts, incs)
    assert lc["parts_covered_after"] == lc["parts_covered_before"]
    assert lc["value_after"] == pytest.approx(lc["value_before"])
    assert lc["incidents"] < lc["alerts"]


def test_one_incident_per_supplier_and_severity(incs):
    keys = [(i.supplier_id, i.severity) for i in incs if i.supplier_id]
    assert len(keys) == len(set(keys))


def test_severities_are_not_collapsed_onto_the_worst(incs):
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


def test_grouping_never_lengthens_the_worst_queue(alerts, incs):
    lc = IN.load_check(alerts, incs)
    assert lc["worst_p1_after"]["items"] <= lc["worst_p1_before"]["items"]
    assert lc["busiest_after"]["items"] <= lc["busiest_before"]["items"]


def test_concentration_shares_are_proper(alerts):
    c = IN.concentration(alerts)
    assert c["n_suppliers"] == 6
    assert 0 < c["top_share"] <= 1
    assert sum(r["alerts"] for r in c["top"]) <= len(alerts)


def test_tightening_the_p1_ratio_never_adds_p1s_and_keeps_the_floor(rows):
    """Parts at or below zero slack cannot be demoted by a ratio threshold:
    ordering now does not recover them."""
    cal = IN.calibrate(rows)
    p1 = [r["p1_alerts"] for r in cal["sweep"]]
    assert p1 == sorted(p1, reverse=True)
    already_short = sum(1 for r in rows if r["days_of_slack"] <= 0)
    assert already_short > 0
    assert cal["sweep"][-1]["p1_alerts"] >= already_short


def test_calibrate_restores_the_severity_function(rows):
    before = RT.severity_of
    IN.calibrate(rows, ratios=(0.25, 0.1))
    assert RT.severity_of is before


def test_value_triage_reports_a_verdict_rather_than_a_number(incs):
    vt = IN.value_triage(incs, top_n=10)
    assert 0.0 <= vt["top_n_share_of_p1_value"] <= 1.0
    assert vt["incidents_for_80pct_of_p1_value"] >= 1
    assert "workable" in vt["verdict"]


def test_value_triage_with_no_p1_says_so():
    a = RT.build_alerts([{"part": "X", "supplier_id": "S", "days_of_slack": 90.0,
                          "lead_days": 10.0, "units_at_risk": 0.0,
                          "value_at_risk": 0.0}])
    vt = IN.value_triage(IN.group_by_supplier(a))
    assert vt["n_p1_incidents"] == 0 and "no P1" in vt["verdict"]


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
