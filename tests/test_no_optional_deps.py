"""Prove the base install (requirements.txt only) still works: no optional extractor dependency."""
import sys
import types
import os

BLOCKED = {"pypdf", "docx", "openpyxl", "PIL", "pytesseract", "pypdfium2", "pdf2image", "anthropic",
           "multipart"}


class Blocker:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] in BLOCKED else None

    def load_module(self, name):
        raise ImportError("blocked for this test: " + name)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked for this test: " + name)
        return None


sys.meta_path.insert(0, Blocker())

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extractor  # noqa: E402

caps = extractor.capabilities()
print("capabilities with nothing installed:", caps)
assert caps["pdf"] is False and caps["docx"] is False and caps["ocr"] is False
assert caps["txt"] and caps["csv"] and caps["json"]

ship = ("COMMERCIAL INVOICE\nInvoice No.: INV-1\nDate: 2026-06-15\nSeller: A Ltd\nBuyer: B Ltd\n"
        "Total Amount: 1,000.00\n")
res = extractor.extract_files([{"name": "invoice.txt", "data": ship.encode()}])
inv = res["shipment"]["documents"]["invoice"]
print("plain TXT still extracted:", inv["doc_number"], inv["date"], inv["seller"], inv["total_amount"])
assert inv["doc_number"] == "INV-1" and inv["total_amount"] == 1000.0

# the API must still import and answer 503 on /api/extract
from fastapi.testclient import TestClient  # noqa: E402
import app as app_module  # noqa: E402

c = TestClient(app_module.app)
assert c.get("/api/extract/capabilities").status_code == 200
r = c.post("/api/extract", files=[("files", ("x.txt", b"hello", "text/plain"))])
print("POST /api/extract without python-multipart ->", r.status_code, r.json())
assert r.status_code == 503
assert c.get("/api/health").status_code == 200
assert c.get("/api/samples").status_code == 200
r = c.post("/api/validate", json={"shipment_id": "x", "documents": {}})
assert r.status_code == 422
print("existing routes unaffected: OK")
print("PASS test_i_base_install_without_optional_deps")