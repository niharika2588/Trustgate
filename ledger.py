"""
SIMULATED ledger for the demo.

This is NOT IBM Blockchain. It is a local, append-only, hash-chained table (SQLite) that behaves like the
on-chain part of TrustGate: only the SHA-256 of the documents and the trust score are recorded, never the
documents themselves. Every entry contains the hash of the previous entry, so editing an old row breaks the chain.

To connect a real network later, keep the same two methods (record, find) in a class that calls
Hyperledger Fabric / IBM Blockchain (chaincode `RecordShipment(shipmentId, hash, score)`) and set
LEDGER_BACKEND accordingly in app.py.
"""
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone

GENESIS = "0" * 64


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_hashes(ship):
    docs = ship["documents"]
    per_doc = {k: sha256(canonical(v)) for k, v in sorted(docs.items())}
    return per_doc, sha256(canonical(docs))


class LocalLedger:
    backend_name = "SIMULATED local hash-chained ledger (SQLite)"

    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS ledger(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, shipment_id TEXT, shipment_hash TEXT,
                trust_score INTEGER, source TEXT, prev_hash TEXT, entry_hash TEXT);
            CREATE TABLE IF NOT EXISTS documents(          -- off-chain store: original documents stay here
                shipment_id TEXT PRIMARY KEY, json TEXT, doc_hashes TEXT);
            CREATE TABLE IF NOT EXISTS reviews(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, shipment_id TEXT, json TEXT, result TEXT,
                status TEXT DEFAULT 'pending', note TEXT);
            """)
            self.db.commit()

    # ---------------------------------------------------------------- ledger
    def _last_hash(self):
        r = self.db.execute("SELECT entry_hash FROM ledger ORDER BY id DESC LIMIT 1").fetchone()
        return r["entry_hash"] if r else GENESIS

    def record(self, shipment_id, shipment_hash, trust_score, source):
        with self.lock:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            prev = self._last_hash()
            entry_hash = sha256(f"{prev}|{shipment_id}|{shipment_hash}|{trust_score}|{source}|{ts}")
            cur = self.db.execute(
                "INSERT INTO ledger(ts,shipment_id,shipment_hash,trust_score,source,prev_hash,entry_hash) VALUES(?,?,?,?,?,?,?)",
                (ts, shipment_id, shipment_hash, trust_score, source, prev, entry_hash))
            self.db.commit()
            return self.get(cur.lastrowid)

    def get(self, entry_id):
        r = self.db.execute("SELECT * FROM ledger WHERE id=?", (entry_id,)).fetchone()
        return dict(r) if r else None

    def find(self, shipment_id):
        r = self.db.execute("SELECT * FROM ledger WHERE shipment_id=? ORDER BY id DESC LIMIT 1", (shipment_id,)).fetchone()
        return dict(r) if r else None

    def entries(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM ledger ORDER BY id")]

    def verify_chain(self):
        prev = GENESIS
        for e in self.entries():
            expect = sha256(f"{prev}|{e['shipment_id']}|{e['shipment_hash']}|{e['trust_score']}|{e['source']}|{e['ts']}")
            if e["prev_hash"] != prev or e["entry_hash"] != expect:
                return False, e["id"]
            prev = e["entry_hash"]
        return True, None

    # ---------------------------------------------------------------- off-chain document store
    def store_document(self, shipment_id, ship, doc_hashes):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO documents VALUES(?,?,?)",
                            (shipment_id, canonical(ship), json.dumps(doc_hashes)))
            self.db.commit()

    def get_document(self, shipment_id):
        r = self.db.execute("SELECT * FROM documents WHERE shipment_id=?", (shipment_id,)).fetchone()
        return (json.loads(r["json"]), json.loads(r["doc_hashes"])) if r else (None, None)

    def committed_shipments(self):
        """Documents in the order they were recorded (used to rebuild the seller/invoice history at start-up)."""
        q = "SELECT d.json FROM ledger l JOIN documents d ON d.shipment_id=l.shipment_id ORDER BY l.id"
        return [json.loads(r["json"]) for r in self.db.execute(q)]

    # ---------------------------------------------------------------- human review queue
    def add_review(self, shipment_id, ship, result):
        with self.lock:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            cur = self.db.execute("INSERT INTO reviews(ts,shipment_id,json,result) VALUES(?,?,?,?)",
                                  (ts, shipment_id, canonical(ship), json.dumps(result)))
            self.db.commit()
            return cur.lastrowid

    def reviews(self, status=None):
        q = "SELECT * FROM reviews" + (" WHERE status=?" if status else "") + " ORDER BY id"
        rows = self.db.execute(q, (status,) if status else ()).fetchall()
        out = []
        for r in rows:
            res = json.loads(r["result"])
            out.append({"id": r["id"], "ts": r["ts"], "shipment_id": r["shipment_id"], "status": r["status"],
                        "note": r["note"], "trust_score": res["trust_score"], "reasons": res["reasons"]})
        return out

    def get_review(self, review_id):
        r = self.db.execute("SELECT * FROM reviews WHERE id=?", (review_id,)).fetchone()
        return dict(r) if r else None

    def set_review(self, review_id, status, note):
        with self.lock:
            self.db.execute("UPDATE reviews SET status=?, note=? WHERE id=?", (status, note, review_id))
            self.db.commit()

    def reset(self):
        with self.lock:
            self.db.executescript("DELETE FROM ledger; DELETE FROM documents; DELETE FROM reviews; "
                                  "DELETE FROM sqlite_sequence;")
            self.db.commit()
