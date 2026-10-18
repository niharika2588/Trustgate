"""
TrustGate document-upload test suite.

    python -m pytest tests/test_extract.py -v        (needs pytest)
    python tests/test_extract.py                     (standalone, prints the accuracy tables)

a) clean_1 -> PDF invoice + DOCX purchase order + XLSX packing list + PNG bill of lading (OCR, skipped
   when Tesseract is absent) -> extracted JSON must match samples/clean_1.json field by field
b) three more samples (approve / review / reject) + CSV and TXT variants
c) round trip: extracted JSON -> POST /api/validate gives the same decision and trust score
d) tamper: invoice total +1 in the uploaded PDF must survive extraction and change the validation
e) bad inputs: empty, corrupt PDF, .exe renamed .pdf, 25 MB, unsupported extension, unrecognisable file
f) the existing routes and the original UI flow still work
g) model/, scoring.py, ledger.py, samples/, requirements.txt, run.sh are byte identical to the baseline
"""
import io
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import extractor            # noqa: E402
import make_files           # noqa: E402

OCR = extractor.capabilities()["ocr"]
LOW = extractor.LOW_CONFIDENCE


# ------------------------------------------------------------------ helpers
def sample(sample_id):
    with open(os.path.join(BASE, "samples", sample_id + ".json"), encoding="utf-8") as fh:
        return json.load(fh)["shipment"]


def flatten(ship):
    """{'invoice.doc_number': v, 'invoice.items[1].qty': v, ...}"""
    out = {}
    out["shipment_id"] = ship.get("shipment_id")
    for dk, doc in (ship.get("documents") or {}).items():
        for k, v in doc.items():
            if k == "items":
                for i, item in enumerate(v or [], start=1):
                    for ik, iv in (item or {}).items():
                        out["%s.items[%d].%s" % (dk, i, ik)] = iv
            else:
                out["%s.%s" % (dk, k)] = v
    return out


def same(a, b):
    if a is None or b is None:
        return a is b or a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= max(1e-6, abs(float(b)) * 1e-9)
    return str(a).strip() == str(b).strip()


def upload(client, files):
    payload = [("files", (name, data, "application/octet-stream")) for name, data in files]
    return client.post("/api/extract", files=payload)


def four_files(ship, formats):
    """formats = {"invoice": "pdf", ...} -> list of (name, bytes)"""
    out = []
    for doc_type, fmt in formats.items():
        name = {"pdf": "invoice.pdf", "docx": "purchase_order.docx", "xlsx": "packing_list.xlsx",
                "csv": "packing_list.csv", "txt": "invoice.txt", "png": "bill_of_lading.png"}[fmt]
        if doc_type in ("packing_list", "bill_of_lading"):
            name = doc_type + "." + fmt
        elif fmt in ("csv", "txt", "png", "pdf"):
            name = doc_type + "." + fmt
        if doc_type == "bill_of_lading" and fmt == "png":
            data = make_files.make_png(ship, doc_type)
        elif fmt == "xlsx":
            data = make_files.make_xlsx(ship, [doc_type])
        else:
            data = getattr(make_files, "make_" + fmt)(ship, doc_type)
        out.append((name, data))
    return out


def client():
    from fastapi.testclient import TestClient
    os.environ.setdefault("TRUSTGATE_DB", os.path.join(BASE, "tests", "_test_ledger.db"))
    import app as app_module
    return TestClient(app_module.app), app_module


def report(title, rows, ok, bad):
    print("\n== %s: %d/%d fields exact (%.1f%%)" % (title, ok, ok + bad, 100.0 * ok / max(ok + bad, 1)))
    for k, e, a, s in rows:
        if s == "DIFF":
            print("   DIFF %-42s expected=%r extracted=%r" % (k, e, a))
    return bad == 0


def accuracy_table(title, expected, got, skip_prefix=()):
    """Field-by-field accuracy, optionally skipping document prefixes that were not uploaded."""
    exp, act = flatten(expected), flatten(got)
    keys = sorted(set(exp) | set(act))
    rows, ok, bad = [], 0, 0
    for k in keys:
        if k == "shipment_id" or k.endswith(".items") or any(k.startswith(p) for p in skip_prefix):
            continue
        good = same(exp.get(k), act.get(k))
        rows.append((k, exp.get(k), act.get(k), "ok" if good else "DIFF"))
        ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
    return rows, ok, bad


# ------------------------------------------------------------------ a) accuracy on clean_1
def test_a_accuracy_clean_1():
    ship = sample("clean_1")
    formats = {"invoice": "pdf", "purchase_order": "docx", "packing_list": "xlsx"}
    files = four_files(ship, formats)
    skip = ()
    if OCR:
        files.append(("bill_of_lading.png", make_files.make_png(ship, "bill_of_lading")))
        print("\n[a] OCR available: the scanned bill_of_lading.png is uploaded too")
    else:
        skip = ("bill_of_lading.",)
        print("\n[a] OCR NOT available (no Tesseract binary + pytesseract): the scanned bill of lading")
        print("    (PNG) is SKIPPED as specified, and the bill_of_lading fields are excluded from the table.")
    c, _ = client()
    res = upload(c, files)
    assert res.status_code == 200, res.text
    body = res.json()
    rows, ok, bad = accuracy_table("a) clean_1", ship, body["shipment"], skip)
    assert report("a) clean_1  [pdf + docx + xlsx" + (" + png/ocr" if OCR else ", png skipped") + "]", rows, ok, bad)
    assert [m for m in body["missing"] if not m.startswith(skip)] == [], body["missing"]
    if OCR:
        assert body["files"][-1]["method"] == "ocr"


# ------------------------------------------------------------------ b) other samples + csv/txt
SAMPLES_B = [("clean_2", "approve"), ("borderline_review", "review"), ("fake_seller_or_hs", "reject")]
FORMATS_B = [
    {"invoice": "csv", "purchase_order": "txt", "packing_list": "xlsx", "bill_of_lading": "docx"},
    {"invoice": "pdf", "purchase_order": "docx", "packing_list": "csv", "bill_of_lading": "txt"},
    {"invoice": "txt", "purchase_order": "csv", "packing_list": "docx", "bill_of_lading": "pdf"},
]


def test_b_accuracy_other_samples():
    c, _ = client()
    for (sid, kind), formats in zip(SAMPLES_B, FORMATS_B):
        ship = sample(sid)
        files = four_files(ship, formats)
        res = upload(c, files)
        assert res.status_code == 200, res.text
        body = res.json()
        rows, ok, bad = accuracy_table("b) %s" % sid, ship, body["shipment"])
        report("b) %-17s (%-7s) %s" % (sid, kind, "+".join(formats.values())), rows, ok, bad)
        assert bad == 0, "%s: %d field(s) differ" % (sid, bad)
        # the null fields of a sample must stay null - the extractor must never invent them
        exp, act = flatten(ship), flatten(body["shipment"])
        for k, v in exp.items():
            if v is None:
                assert k in body["missing"], "%s is null in the sample but was not reported as missing" % k


# ------------------------------------------------------------------ c) round trip through /api/validate
def test_c_round_trip_validation():
    c, _ = client()
    for sid, _kind in [("clean_1", None)] + SAMPLES_B:
        ship = sample(sid)
        res = upload(c, four_files(ship, FORMATS_B[SAMPLES_B.index((sid, _kind))] if sid != "clean_1"
                                   else {"invoice": "pdf", "purchase_order": "docx", "packing_list": "xlsx",
                                         "bill_of_lading": "txt"}))
        assert res.status_code == 200, res.text
        extracted = res.json()["shipment"]
        c.post("/api/reset")
        a = c.post("/api/validate", json=ship).json()
        c.post("/api/reset")
        b = c.post("/api/validate", json=extracted).json()
        same_verdict = a["decision"] == b["decision"]
        same_score = a["trust_score"] == b["trust_score"]
        print("\n[c] %-17s original: %-7s %3d   extracted: %-7s %3d   %s"
              % (sid, a["decision"], a["trust_score"], b["decision"], b["trust_score"],
                 "identical" if same_verdict and same_score else "DIFFERENT"))
        if not (same_verdict and same_score):
            print("      features differing:",
                  {k: (a["features"].get(k), b["features"].get(k))
                   for k in a["features"] if a["features"].get(k) != b["features"].get(k)})
        assert same_verdict and same_score, "round trip changed the decision/score for " + sid


# ------------------------------------------------------------------ d) tamper
def test_d_tampered_invoice_total():
    ship = sample("clean_1")
    c, _ = client()
    clean_pdf = make_files.make_pdf(ship, "invoice")
    tampered = dict(ship)
    tampered["documents"] = json.loads(json.dumps(ship["documents"]))
    tampered["documents"]["invoice"]["total_amount"] = float(ship["documents"]["invoice"]["total_amount"]) + 1
    tampered_pdf = make_files.make_pdf(tampered, "invoice")
    assert clean_pdf != tampered_pdf

    res = upload(c, [("invoice.pdf", tampered_pdf),
                     ("purchase_order.docx", make_files.make_docx(ship, "purchase_order")),
                     ("packing_list.xlsx", make_files.make_xlsx(ship, ["packing_list"])),
                     ("bill_of_lading.txt", make_files.make_txt(ship, "bill_of_lading"))])
    assert res.status_code == 200, res.text
    got = res.json()["shipment"]["documents"]["invoice"]["total_amount"]
    print("\n[d] invoice total printed 2,228.00 -> extracted %r (must be 2228.0, not normalised back to 2227.0)"
          % got)
    assert got == 2228.0, "the tampered total was normalised away: %r" % got

    c.post("/api/reset")
    a = c.post("/api/validate", json=tampered).json()
    c.post("/api/reset")
    b = c.post("/api/validate", json=res.json()["shipment"]).json()
    c.post("/api/reset")
    clean = c.post("/api/validate", json=ship).json()
    assert a["decision"] == b["decision"] and a["trust_score"] == b["trust_score"]
    assert b["features"]["grand_err_rel"] > 0, b["features"]
    assert clean["features"]["grand_err_rel"] == 0
    print("[d] validation reflects it: extracted JSON scores %s/%s (clean copy %s/%s); "
          "grand_err_rel %s vs 0" % (b["decision"], b["trust_score"], clean["decision"],
                                     clean["trust_score"], b["features"]["grand_err_rel"]))
    assert b["features"] != clean["features"], "the tampered invoice did not change the model features"
    if (b["decision"], b["trust_score"]) == (clean["decision"], clean["trust_score"]):
        print("[d] note: +1 on a 2,227 invoice is a 0.04%% error, below the 0.1%% r_calc threshold, so the "
              "decision\n    stays the same by design - the value itself and the features did change.")

    # a larger edit must move the verdict
    big = dict(ship)
    big["documents"] = json.loads(json.dumps(ship["documents"]))
    big["documents"]["invoice"]["total_amount"] = float(ship["documents"]["invoice"]["total_amount"]) + 100
    res2 = upload(c, [("invoice.pdf", make_files.make_pdf(big, "invoice")),
                      ("purchase_order.docx", make_files.make_docx(ship, "purchase_order")),
                      ("packing_list.xlsx", make_files.make_xlsx(ship, ["packing_list"])),
                      ("bill_of_lading.txt", make_files.make_txt(ship, "bill_of_lading"))])
    got2 = res2.json()["shipment"]["documents"]["invoice"]["total_amount"]
    c.post("/api/reset")
    v2 = c.post("/api/validate", json=res2.json()["shipment"]).json()
    c.post("/api/reset")
    v2_ref = c.post("/api/validate", json=big).json()
    print("[d] +100 variant: extracted %r -> %s/%s (same JSON as the hand made one: %s), rules: %s"
          % (got2, v2["decision"], v2["trust_score"],
             (v2["decision"], v2["trust_score"]) == (v2_ref["decision"], v2_ref["trust_score"]),
             [f["rule"] for f in v2["rule_flags"]]))
    assert got2 == 2327.0
    assert (v2["decision"], v2["trust_score"]) == (v2_ref["decision"], v2_ref["trust_score"])


# ------------------------------------------------------------------ e) bad input
def test_e_bad_inputs():
    c, _ = client()
    ship = sample("clean_1")
    big = b"x" * (21 * 1024 * 1024)
    cases = [
        ("empty file", [("empty.pdf", b"")], 422),
        ("corrupt pdf", [("broken.pdf", b"%PDF-1.4\nthis is not a pdf at all")], 422),
        (".exe renamed .pdf", [("invoice.pdf", b"MZ\x90\x00" + b"\x00" * 200)], 415),
        ("21 MB file", [("huge.pdf", big)], 413),
        ("unsupported extension", [("archive.zip", b"PK\x03\x04rubbish")], 415),
        ("html file", [("page.pdf", b"<!doctype html><html><body>hello</body></html>")], 415),
    ]
    for label, files, want in cases:
        r = upload(c, files)
        detail = (r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json")
                  else r.text)
        print("\n[e] %-24s -> HTTP %s  %s" % (label, r.status_code, str(detail)[:110]))
        assert r.status_code == want, "%s: expected %s got %s" % (label, want, r.status_code)

    # a readable file that contains no document at all -> 200 with everything reported as missing
    r = upload(c, [("notes.txt", b"Dear team,\n\nThe container is delayed, see you on Friday.\n\nRegards")])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["files"][0]["doc_type"] == "unclassified", body["files"][0]
    assert len(body["missing"]) > 20, body["missing"]
    print("[e] %-24s -> HTTP 200, doc_type=unclassified, %d missing fields reported"
          % ("no recognisable document", len(body["missing"])))

    # too many files
    r = upload(c, [("n%d.txt" % i, b"hello") for i in range(11)])
    assert r.status_code == 413, r.status_code
    print("[e] %-24s -> HTTP %s  %s" % ("11 files", r.status_code, str(r.json().get("detail"))[:110]))

    # server still alive
    assert c.get("/api/health").status_code == 200
    print("[e] server still running: OK")


# ------------------------------------------------------------------ f) existing routes
def test_f_existing_routes():
    c, _ = client()
    assert c.get("/").status_code == 200 and "TrustGate" in c.get("/").text
    idx = c.get("/api/samples").json()
    assert idx and all("id" in s for s in idx)
    one = c.get("/api/samples/clean_1").json()["shipment"]
    assert c.post("/api/reset").status_code == 200

    good = c.post("/api/validate", json=one).json()
    assert good["decision"] == "approve" and good["ledger_entry"], good
    led = c.get("/api/ledger").json()
    assert led["chain_valid"] and led["entries"], led
    doc = c.get("/api/ledger/SHP-004001/document").json()
    v = c.post("/api/verify", json={"shipment_id": "SHP-004001", "document": doc}).json()
    assert v["match"] is True, v

    # review flow
    review_sample = sample("borderline_review")
    r1 = c.post("/api/validate", json=review_sample).json()
    assert r1["decision"] == "review" and r1["review_id"], r1
    q = c.get("/api/review").json()
    assert any(x["id"] == r1["review_id"] and x["status"] == "pending" for x in q), q
    dec = c.post("/api/review/%d" % r1["review_id"], json={"action": "approve"}).json()
    assert dec["status"] == "approved" and dec["ledger_entry"], dec

    # tamper path in the UI
    doc["documents"]["invoice"]["total_amount"] = float(doc["documents"]["invoice"]["total_amount"]) + 1
    v2 = c.post("/api/verify", json={"shipment_id": "SHP-004001", "document": doc}).json()
    assert v2["match"] is False and "invoice" in v2["changed_documents"], v2

    assert c.post("/api/validate", content=b"{bad").status_code == 422
    assert c.get("/api/samples/nope").status_code == 404
    caps = c.get("/api/extract/capabilities").json()
    assert "pdf" in caps and "ocr" in caps, caps
    c.post("/api/reset")
    assert c.get("/api/ledger").json()["entries"] == []
    print("\n[f] existing routes: samples, validate, review, ledger, verify, reset, index -> all OK")
    print("[f] capabilities: " + json.dumps({k: caps[k] for k in ("pdf", "docx", "xlsx", "csv", "ocr", "legacy_doc")}))


# ------------------------------------------------------------------ h) other formats and edge cases
def test_h_formats_and_edges():
    c, _ = client()
    ship = sample("clean_1")

    # one TXT file holding all four documents (split by headings)
    bundle = []
    for dt in ("invoice", "purchase_order", "packing_list", "bill_of_lading"):
        bundle += ["", ""] + make_files.text_lines(ship, dt)
    res = upload(c, [("shipment_bundle.txt", ("\n".join(bundle)).encode("utf-8"))])
    assert res.status_code == 200, res.text
    body = res.json()
    rows, ok, bad = accuracy_table("4 documents in one TXT", ship, body["shipment"])
    assert report("h) one TXT with 4 documents (split by headings)", rows, ok, bad), body["warnings"]
    assert body["files"][0]["documents"] == ["invoice", "purchase_order", "packing_list", "bill_of_lading"]

    # JSON that already matches the schema -> passed through unchanged
    res = upload(c, [("export.json", json.dumps(ship).encode("utf-8"))])
    assert res.json()["shipment"]["documents"]["invoice"]["doc_number"] == "INV-104001"
    print("\n[h] JSON passthrough: kept every value, method=%s"
          % res.json()["files"][0]["method"])

    # XML
    xml = ("<invoice><doc_number>INV-104001</doc_number><date>2026-06-15</date>"
           "<seller>Apex Overseas Pvt Ltd</seller><buyer>Coral Foods Ltd</buyer>"
           "<total_amount>2227.00</total_amount></invoice>").encode()
    res = upload(c, [("invoice.xml", xml)])
    inv = res.json()["shipment"]["documents"]["invoice"]
    print("[h] XML: doc_number=%r date=%r total=%r" % (inv["doc_number"], inv["date"], inv["total_amount"]))
    assert inv["doc_number"] == "INV-104001" and inv["total_amount"] == 2227.0

    # RTF is parsed without LibreOffice
    rtf = (b"{\\rtf1\\ansi\\deff0 {\\fonttbl{\\f0 Courier;}}\\fs18 "
           b"COMMERCIAL INVOICE\\par Invoice No.: INV-104001\\par Date: 2026-06-15\\par }")
    res = upload(c, [("invoice.rtf", rtf)])
    inv = res.json()["shipment"]["documents"]["invoice"]
    print("[h] RTF (built-in reader, no LibreOffice): doc_number=%r date=%r" % (inv["doc_number"], inv["date"]))
    assert inv["doc_number"] == "INV-104001"

    # .doc without LibreOffice -> clear message, not a crash
    res = upload(c, [("old.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100)])
    detail = res.json().get("detail", "")
    print("[h] legacy .doc without LibreOffice: HTTP %s - %s" % (res.status_code, str(detail)[:150]))
    assert res.status_code == 422 and "LibreOffice" in str(detail)

    # scanned PDF: the page has no text layer, so OCR is used
    if OCR:
        from PIL import Image
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        cv = canvas.Canvas(buf, pagesize=A4)
        png = io.BytesIO(make_files.make_png(ship, "invoice", width=1654))
        iw, ih = Image.open(png).size
        scale = min(A4[0] / iw, A4[1] / ih)                      # keep the aspect ratio: stretched scans OCR badly
        cv.drawImage(ImageReader(png), (A4[0] - iw * scale) / 2, (A4[1] - ih * scale) / 2,
                     width=iw * scale, height=ih * scale)
        cv.showPage()
        cv.save()
        res = upload(c, [("scanned_invoice.pdf", buf.getvalue())])
        inv = res.json()["shipment"]["documents"]["invoice"]
        print("[h] scanned PDF -> method=%s, doc_number=%r, total=%r"
              % (res.json()["files"][0]["method"], inv["doc_number"], inv["total_amount"]))
        assert res.json()["files"][0]["method"] == "ocr"
        assert inv["doc_number"] == "INV-104001" and inv["total_amount"] == 2227.0

    # 51 page PDF -> warning, only 50 pages read
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as cvmod
    buf = io.BytesIO()
    cv = cvmod.Canvas(buf, pagesize=A4)
    for i in range(51):
        cv.setFont("Helvetica", 10)
        cv.drawString(50, 780, "COMMERCIAL INVOICE page %d" % (i + 1))
        cv.drawString(50, 760, "Invoice No.: INV-104001")
        cv.showPage()
    cv.save()
    res = upload(c, [("long_invoice.pdf", buf.getvalue())])
    assert res.status_code == 200
    assert any("only the first 50" in w for w in res.json()["warnings"]), res.json()["warnings"]
    assert res.json()["files"][0]["pages"] == 50
    print("[h] 51 page PDF: read 50 pages and warned about it")

    # path traversal in the filename + derived shipment id
    res = upload(c, [("../../etc/../../passwd.txt", make_files.make_txt(ship, "invoice"))])
    assert res.status_code == 200
    assert "/" not in res.json()["files"][0]["name"] and "\\" not in res.json()["files"][0]["name"]
    print("[h] filename sanitised to %r" % res.json()["files"][0]["name"])

    stripped = json.loads(json.dumps(ship))
    for dk in stripped["documents"]:
        stripped["documents"][dk].pop("shipment_id", None)
    res = upload(c, four_files(stripped, {"invoice": "txt"}))
    sid = res.json()["shipment"]["shipment_id"]
    assert sid.startswith("SHP-") and len(sid) == 12, sid
    assert any("generated" in w for w in res.json()["warnings"]), res.json()["warnings"]
    print("[h] no shipment id in the files -> %s generated + warned" % sid)


# padded (space aligned) columns instead of pipes - a different PDF table layout
    res = upload(c, [("invoice_padded.pdf", make_files.make_pdf(ship, "invoice", pad=True)),
                     ("purchase_order.txt", ("\n".join(make_files.text_lines(ship, "purchase_order", pad=True)))
                      .encode("utf-8")),
                     ("packing_list.xlsx", make_files.make_xlsx(ship, ["packing_list"])),
                     ("bill_of_lading.docx", make_files.make_docx(ship, "bill_of_lading"))])
    rows, ok, bad = accuracy_table("padded columns", ship, res.json()["shipment"])
    assert report("h) space aligned columns (no pipes)", rows, ok, bad), res.json()["warnings"]

    # g) protected files unchanged
def test_g_protected_files_unchanged():
    import hashlib
    baseline = json.load(open(os.path.join(BASE, "tests", "baseline_checksums.json"), encoding="utf-8"))
    changed = []
    for rel, want in sorted(baseline.items()):
        got = hashlib.sha256(open(os.path.join(BASE, rel), "rb").read()).hexdigest()
        if got != want:
            changed.append(rel)
    print("\n[g] %d protected files checked (model/, samples/, scoring.py, ledger.py, "
          "requirements.txt, run.sh): %s" % (len(baseline), "all unchanged" if not changed else changed))
    assert not changed, changed


if __name__ == "__main__":
    failures = 0
    for fn in (test_a_accuracy_clean_1, test_b_accuracy_other_samples, test_c_round_trip_validation,
               test_d_tampered_invoice_total, test_e_bad_inputs, test_f_existing_routes,
               test_h_formats_and_edges, test_g_protected_files_unchanged):
        try:
            fn()
            print("PASS %s" % fn.__name__)
        except AssertionError as e:
            failures += 1
            print("FAIL %s: %s" % (fn.__name__, e))
    for junk in ("tests/_test_ledger.db", "t.xlsx"):
        try:
            os.remove(os.path.join(BASE, junk))
        except OSError:
            pass
    print("\n%s" % ("ALL TESTS PASSED" if not failures else "%d TEST(S) FAILED" % failures))
    sys.exit(1 if failures else 0)