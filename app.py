"""
TrustGate API + demo UI.   Run:  uvicorn app:app --host 0.0.0.0 --port 8000

Flow for POST /api/validate:
    documents -> features + rule checks -> XGBoost -> trust score (0-100) -> decision gate
        score >= 80  : APPROVE -> SHA-256 hash + score written to the (simulated) ledger, documents stay off-chain
        50 <= s < 80 : REVIEW  -> goes to the human review queue, nothing written yet
        score < 50   : REJECT  -> nothing written, reasons returned
"""
import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

import extractor                                        # document upload -> shipment JSON (additive, optional deps)
from ledger import LocalLedger, document_hashes
from scoring import DOC_KEYS, Scorer

BASE = Path(__file__).parent
DB_PATH = os.environ.get("TRUSTGATE_DB", str(BASE / "trustgate.db"))

app = FastAPI(title="TrustGate", description="Validate shipment documents before they reach the ledger")
scorer = Scorer(BASE / "model")
ledger = LocalLedger(DB_PATH)          # swap point: a class with the same methods that talks to IBM Blockchain
lock = threading.RLock()


def rebuild_history():
    h = scorer.fresh_history()
    for ship in ledger.committed_shipments():      # shipments already recorded on the ledger count as history
        h.update(ship)
    return h


history = rebuild_history()


def check_shape(ship):
    if not isinstance(ship, dict) or "documents" not in ship or not isinstance(ship["documents"], dict):
        raise HTTPException(422, 'body must look like {"shipment_id": "...", "documents": {...}}')
    missing = [k for k in DOC_KEYS if k not in ship["documents"]]
    if missing:
        raise HTTPException(422, f"documents missing: {', '.join(missing)}")
    for k in ("invoice", "purchase_order", "packing_list"):
        items = ship["documents"][k].get("items")
        if not isinstance(items, list) or not items:
            raise HTTPException(422, f"{k}.items must be a non-empty list")
    ship.setdefault("shipment_id", ship["documents"]["invoice"].get("shipment_id") or "UNKNOWN")


def commit(ship, trust_score, source):
    per_doc, ship_hash = document_hashes(ship)
    ledger.store_document(ship["shipment_id"], ship, per_doc)
    entry = ledger.record(ship["shipment_id"], ship_hash, trust_score, source)
    history.update(ship)
    return entry


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": scorer.engine, "trees": scorer.model.n_trees, "ledger": ledger.backend_name,
            "thresholds": {"approve_min": scorer.approve_min, "review_min": scorer.review_min}}


@app.get("/api/samples")
def samples():
    return json.load(open(BASE / "samples" / "index.json", encoding="utf-8"))


@app.get("/api/samples/{sample_id}")
def sample(sample_id: str):
    p = BASE / "samples" / f"{Path(sample_id).name}.json"
    if not p.exists():
        raise HTTPException(404, "unknown sample")
    return json.load(open(p, encoding="utf-8"))


@app.post("/api/validate")
async def validate(request: Request):
    try:
        ship = await request.json()
    except Exception:
        raise HTTPException(422, "body is not valid JSON")
    check_shape(ship)
    with lock:
        try:
            result = scorer.score(ship, history)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(422, f"could not read the documents ({type(e).__name__}: {e})")
        per_doc, ship_hash = document_hashes(ship)
        result.update({"shipment_id": ship["shipment_id"], "shipment_hash": ship_hash, "doc_hashes": per_doc,
                       "ledger_entry": None, "review_id": None})
        if result["decision"] == "approve":
            result["ledger_entry"] = commit(ship, result["trust_score"], "auto")
        elif result["decision"] == "review":
            result["review_id"] = ledger.add_review(ship["shipment_id"], ship, result)
        result["ledger_backend"] = ledger.backend_name
    return result


@app.get("/api/ledger")
def get_ledger():
    ok, broken = ledger.verify_chain()
    return {"backend": ledger.backend_name, "chain_valid": ok, "broken_at": broken, "entries": ledger.entries()}


@app.get("/api/ledger/{shipment_id}/document")
def stored_document(shipment_id: str):
    ship, _ = ledger.get_document(shipment_id)
    if ship is None:
        raise HTTPException(404, "no document stored for this shipment (off-chain store)")
    return ship


@app.post("/api/verify")
async def verify(request: Request):
    """Tamper check: recompute the hash of the documents you hold and compare with the ledger record."""
    body = await request.json()
    ship = body.get("document")
    sid = body.get("shipment_id") or (ship or {}).get("shipment_id")
    if not isinstance(ship, dict) or "documents" not in ship or not sid:
        raise HTTPException(422, "send {shipment_id, document: {documents: ...}}")
    entry = ledger.find(sid)
    if not entry:
        return {"shipment_id": sid, "on_ledger": False, "match": False, "message": "no ledger record for this shipment"}
    per_doc, ship_hash = document_hashes(ship)
    _, original_docs = ledger.get_document(sid)
    changed = [k for k, v in per_doc.items() if original_docs and original_docs.get(k) != v]
    match = ship_hash == entry["shipment_hash"]
    return {"shipment_id": sid, "on_ledger": True, "match": match, "ledger_hash": entry["shipment_hash"],
            "computed_hash": ship_hash, "changed_documents": changed,
            "message": "documents are identical to what was recorded" if match else "TAMPERED: hash does not match the ledger"}


@app.get("/api/review")
def review_queue():
    return ledger.reviews()


@app.post("/api/review/{review_id}")
async def review_decide(review_id: int, request: Request):
    body = await request.json()
    action = body.get("action")
    if action not in ("approve", "reject"):
        raise HTTPException(422, "action must be approve or reject")
    with lock:
        r = ledger.get_review(review_id)
        if not r:
            raise HTTPException(404, "unknown review id")
        if r["status"] != "pending":
            raise HTTPException(409, f"already {r['status']}")
        ledger.set_review(review_id, "approved" if action == "approve" else "rejected", body.get("note", ""))
        entry = None
        if action == "approve":
            ship = json.loads(r["json"])
            entry = commit(ship, json.loads(r["result"])["trust_score"], "human reviewer")
    return {"review_id": review_id, "status": "approved" if action == "approve" else "rejected", "ledger_entry": entry}


@app.post("/api/reset")
def reset():
    global history
    with lock:
        ledger.reset()
        history = rebuild_history()
    return {"status": "demo reset"}


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


# ------------------------------------------------------------------ document upload (additive, see extractor.py)
try:                                                    # python-multipart only ships with requirements-extract.txt
    from fastapi import File, UploadFile
    import multipart                                   # noqa: F401
    _HAVE_MULTIPART = True
except Exception:
    _HAVE_MULTIPART = False


@app.get("/api/extract/capabilities")
def extract_capabilities():
    """Which document formats this environment can read (used by the UI to disable the file picker)."""
    try:
        return extractor.capabilities()
    except Exception as e:
        return {"error": "capabilities unavailable: %s" % type(e).__name__, "limits": {}}


if _HAVE_MULTIPART:
    @app.post("/api/extract")
    async def extract(files: list[UploadFile] = File(...)):
        """Extract shipment JSON from uploaded documents. Does NOT validate and does NOT touch the ledger."""
        if not files:
            raise HTTPException(422, "no files were uploaded (field name: files)")
        uploads = []
        try:
            for uf in files:
                data = await uf.read()                 # bytes are parsed, never imported or executed
                uploads.append({"name": uf.filename or "upload", "data": data})
            try:
                return extractor.extract_files(uploads)
            except extractor.ExtractError as e:
                raise HTTPException(e.status, e.message)
            except MemoryError:
                raise HTTPException(413, "upload too large to process in memory")
            except Exception as e:
                raise HTTPException(422, "could not read the documents (%s)" % type(e).__name__)
        finally:
            uploads.clear()                             # drop the buffers as soon as we are done
else:
    @app.post("/api/extract")
    async def extract():
        raise HTTPException(503, "document upload is not installed on this server: "
                                 "pip install -r requirements-extract.txt")
