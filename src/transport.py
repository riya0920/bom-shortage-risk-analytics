"""An alert transport: a durable outbox, real delivery, and the failures.

The README said alerts were "objects, not messages ... a deployment attaches a
transport". Attaching one is where the interesting problems are, and none of them
are about the protocol:

  AT-LEAST-ONCE IS THE ONLY GUARANTEE AVAILABLE, so the receiver has to be able
  to tolerate a repeat. A sender that crashes between "the POST succeeded" and
  "mark it sent" will send again; a sender that marks first will lose the message
  when the POST fails. There is no third option, so this marks AFTER, sends an
  idempotency key with every message, and the receiver deduplicates on it. That
  is the contract, and the test drives it by failing a delivery mid-flight.

  A DEAD LETTER IS NOT A FAILURE TO HIDE. After the retries are exhausted the
  message goes to a dead-letter state and STAYS in the outbox. Dropping it makes
  the transport look reliable and makes a P1 disappear; retrying forever turns
  one broken endpoint into an outage of the whole queue.

  RATE LIMITING IS PART OF THE ALERT POLICY, NOT THE PLUMBING. The load check in
  `incidents.py` says how many items an owner can absorb. A transport that
  ignores that will happily deliver all of them at 06:00, and the queue that was
  too long on paper is now too long in somebody's inbox. So the sender has a
  per-recipient budget per window, and what exceeds it is DEFERRED rather than
  dropped -- and the deferral is visible, because a silently held P1 is worse
  than a noisy one.

  URGENT IS SENT, ROUTINE IS DIGESTED. P1 and P2 go one message each. P3 and P4
  are batched into one digest per recipient per window, because the response to
  a P4 is "look at it on Friday" and forty separate messages saying that is how a
  channel gets filtered to a folder -- after which the P1s go there too.

The receiver is a real HTTP server on a real socket. Mocking the transport would
test the code that calls the transport, which is not the part that goes wrong.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import sqlite3
import threading
import urllib.error
import urllib.request

PENDING, SENT, DEAD, DEFERRED = "PENDING", "SENT", "DEAD", "DEFERRED"

SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    idem_key      TEXT NOT NULL UNIQUE,
    recipient     TEXT NOT NULL,
    severity      TEXT NOT NULL,
    body          TEXT NOT NULL,
    state         TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    next_attempt  REAL NOT NULL DEFAULT 0,
    last_error    TEXT,
    created_at    REAL NOT NULL,
    sent_at       REAL
);
CREATE INDEX IF NOT EXISTS ix_outbox_due ON outbox (state, next_attempt);
"""


class Outbox:
    """Durable queue. The UNIQUE idempotency key is what makes enqueue safe to
    repeat -- a run that crashes halfway and restarts must not double-send, and
    an application-level "have I seen this" check has a race in it that a UNIQUE
    constraint does not."""

    def __init__(self, path):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @staticmethod
    def key(recipient: str, payload: dict) -> str:
        """Stable across processes: content plus recipient, hashed.

        Deliberately NOT `hash()`, which Python salts per process (PEP 456) --
        the same bug this project's generator had, and here it would mean every
        restart re-sending everything.
        """
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(f"{recipient}|{blob}".encode()).hexdigest()[:24]

    def enqueue(self, recipient: str, severity: str, payload: dict,
                now: float = 0.0) -> dict:
        k = self.key(recipient, payload)
        try:
            self.conn.execute(
                "INSERT INTO outbox (idem_key, recipient, severity, body, "
                "state, next_attempt, created_at) VALUES (?,?,?,?,?,?,?)",
                (k, recipient, severity,
                 json.dumps(payload, sort_keys=True, default=str),
                 PENDING, now, now))
            self.conn.commit()
            return {"queued": True, "idem_key": k}
        except sqlite3.IntegrityError:
            return {"queued": False, "idem_key": k, "why": "already queued"}

    def due(self, now: float, limit: int = 100) -> list:
        return list(self.conn.execute(
            "SELECT * FROM outbox WHERE state IN (?,?) AND next_attempt <= ? "
            "ORDER BY CASE severity WHEN 'P1' THEN 0 WHEN 'P2' THEN 1 "
            "WHEN 'P3' THEN 2 ELSE 3 END, id LIMIT ?",
            (PENDING, DEFERRED, now, limit)))

    def counts(self) -> dict:
        return {r["state"]: r["n"] for r in self.conn.execute(
            "SELECT state, COUNT(*) n FROM outbox GROUP BY state")}

    def dead_letters(self) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM outbox WHERE state=?", (DEAD,))]

    def mark_sent(self, row_id: int, now: float) -> None:
        self.conn.execute(
            "UPDATE outbox SET state=?, sent_at=?, attempts=attempts+1 "
            "WHERE id=?", (SENT, now, row_id))
        self.conn.commit()

    def mark_failed(self, row_id: int, err: str, now: float,
                    max_attempts: int, base_backoff_s: float) -> str:
        row = self.conn.execute("SELECT attempts FROM outbox WHERE id=?",
                                (row_id,)).fetchone()
        attempts = row["attempts"] + 1
        if attempts >= max_attempts:
            self.conn.execute(
                "UPDATE outbox SET state=?, attempts=?, last_error=? WHERE id=?",
                (DEAD, attempts, err[:400], row_id))
            state = DEAD
        else:
            self.conn.execute(
                "UPDATE outbox SET state=?, attempts=?, last_error=?, "
                "next_attempt=? WHERE id=?",
                (PENDING, attempts, err[:400],
                 now + base_backoff_s * (2 ** (attempts - 1)), row_id))
            state = PENDING
        self.conn.commit()
        return state

    def defer(self, row_id: int, until: float, why: str) -> None:
        self.conn.execute(
            "UPDATE outbox SET state=?, next_attempt=?, last_error=? WHERE id=?",
            (DEFERRED, until, why, row_id))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# sinks
# ---------------------------------------------------------------------------

class WebhookSink:
    """POSTs JSON to a URL. Raises on anything that is not 2xx."""

    def __init__(self, url: str, timeout: float = 3.0):
        self.url = url
        self.timeout = timeout
        self.attempts = 0

    def send(self, recipient: str, severity: str, idem_key: str,
             payload: dict) -> None:
        self.attempts += 1
        body = json.dumps({"recipient": recipient, "severity": severity,
                           "payload": payload}, default=str).encode()
        req = urllib.request.Request(
            self.url, data=body,
            headers={"Content-Type": "application/json",
                     # The key travels with the message. A receiver that cannot
                     # see it cannot deduplicate, and at-least-once delivery
                     # without receiver-side dedupe is just duplicates.
                     "Idempotency-Key": idem_key,
                     "X-Alert-Severity": severity})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if not 200 <= r.status < 300:
                raise RuntimeError(f"HTTP {r.status}")


class CapturingReceiver:
    """A real HTTP endpoint that records deliveries and deduplicates on the key.

    `fail_first` makes the first N requests fail with a 503, which is how the
    retry path and the at-least-once contract get exercised rather than
    asserted.
    """

    def __init__(self, fail_first: int = 0):
        self.delivered: list = []
        self.seen_keys: set = set()
        self.duplicates = 0
        self.requests = 0
        self.fail_first = int(fail_first)
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                outer.requests += 1
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                if outer.requests <= outer.fail_first:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    return
                key = self.headers.get("Idempotency-Key")
                if key in outer.seen_keys:
                    outer.duplicates += 1
                else:
                    outer.seen_keys.add(key)
                    outer.delivered.append(
                        {"key": key, "body": json.loads(raw or b"{}")})
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"OK")
                self.close_connection = True

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/alerts"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


# ---------------------------------------------------------------------------
# the sender
# ---------------------------------------------------------------------------

URGENT = ("P1", "P2")


def prepare(incidents: list, *, window_s: float = 86400.0,
            now: float = 0.0) -> list:
    """Turn incidents into messages: urgent individually, routine digested."""
    msgs = []
    digests: dict = {}
    for inc in incidents:
        d = inc.as_dict() if hasattr(inc, "as_dict") else dict(inc)
        if d["severity"] in URGENT:
            msgs.append({"recipient": d["route_to"], "severity": d["severity"],
                         "payload": {"kind": "incident", **d}})
        else:
            digests.setdefault((d["route_to"], d["severity"]), []).append(d)
    for (rcpt, sev), items in sorted(digests.items()):
        msgs.append({
            "recipient": rcpt, "severity": sev,
            "payload": {"kind": "digest", "severity": sev,
                        "window_s": window_s, "n_incidents": len(items),
                        "n_parts": sum(i["n_parts"] for i in items),
                        "value_at_risk": sum(i["value_at_risk"] for i in items),
                        "incidents": [i["key"] for i in items]}})
    return msgs


def send_all(outbox: Outbox, sink, messages: list, *, now: float = 0.0,
             max_attempts: int = 4, base_backoff_s: float = 1.0,
             per_recipient_limit: int = 15,
             window_s: float = 86400.0) -> dict:
    """Enqueue, then drain what is due within each recipient's budget."""
    queued = duplicates = 0
    for m in messages:
        r = outbox.enqueue(m["recipient"], m["severity"], m["payload"], now=now)
        queued += 1 if r["queued"] else 0
        duplicates += 0 if r["queued"] else 1

    sent = failed = dead = deferred = 0
    budget: dict = {}
    for row in outbox.due(now):
        used = budget.get(row["recipient"], 0)
        if used >= per_recipient_limit:
            # Deferred, never dropped -- and P1 first, because `due()` orders by
            # severity, so the budget is spent on the urgent items rather than
            # on whatever happened to be enqueued first.
            outbox.defer(row["id"], now + window_s,
                         f"per-recipient budget of {per_recipient_limit} spent")
            deferred += 1
            continue
        try:
            sink.send(row["recipient"], row["severity"], row["idem_key"],
                      json.loads(row["body"]))
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            state = outbox.mark_failed(row["id"], str(e), now, max_attempts,
                                       base_backoff_s)
            failed += 1
            dead += 1 if state == DEAD else 0
            continue
        outbox.mark_sent(row["id"], now)
        budget[row["recipient"]] = used + 1
        sent += 1
    return {"queued": queued, "duplicate_enqueues": duplicates, "sent": sent,
            "failed": failed, "dead": dead, "deferred": deferred,
            "counts": outbox.counts()}


def drain(outbox: Outbox, sink, *, start: float = 0.0, step_s: float = 4.0,
          rounds: int = 6, **kw) -> dict:
    """Keep retrying on a clock until nothing is due. Returns the history."""
    hist = []
    now = start
    for _ in range(rounds):
        r = send_all(outbox, sink, [], now=now, **kw)
        hist.append({"t": now, **{k: r[k] for k in
                                  ("sent", "failed", "dead", "deferred")}})
        now += step_s
        if not outbox.due(now):
            break
    return {"rounds": hist, "counts": outbox.counts(),
            "dead_letters": len(outbox.dead_letters())}
