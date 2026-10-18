"""
TrustGate document extractor: real files -> the exact shipment JSON that scoring.py reads.

Fully offline and deterministic (label regexes + table parsing). If the optional environment
variable ANTHROPIC_API_KEY is set, extractor_llm.py may fill fields the rules could not read
(clearly reported as method "llm"). Nothing is ever invented: unreadable fields stay None and are
listed in "missing", and mismatches between documents are preserved exactly as printed.

Result shape:
    {"shipment": {...target JSON...}, "confidence": {"doc.field": 0..1},
     "missing": ["doc.field"], "warnings": [str], "files": [{name, detected_type, doc_type, pages, method}]}
"""
import hashlib
import os
import re
import shutil
import tempfile

import extractor_fields as F
import extractor_readers as R

MAX_FILES = 10
MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 50
MAX_TOTAL_BYTES = 60 * 1024 * 1024
LOW_CONFIDENCE = 0.6

DOC_KEYS = ("purchase_order", "invoice", "packing_list", "bill_of_lading")
ITEM_KEYS = {
    "purchase_order": ["product_id", "hs_code", "description", "qty", "unit_price", "total"],
    "invoice": ["product_id", "hs_code", "description", "qty", "unit_price", "total"],
    "packing_list": ["product_id", "hs_code", "description", "qty", "net_weight_kg"],
    "bill_of_lading": [],
}
# same required fields the model scores (scoring.REQ)
REQUIRED = {
    "purchase_order": ["doc_number", "date", "buyer", "seller"],
    "invoice": ["doc_number", "date", "seller", "buyer", "total_amount"],
    "packing_list": ["doc_number", "gross_weight_kg", "total_packages"],
    "bill_of_lading": ["doc_number", "consignee", "port_of_loading", "ship_on_board_date", "container_no"],
}
EXTRA_FIELDS = {
    "purchase_order": ["currency", "incoterm", "shipment_id", "total_amount"],
    "invoice": ["po_number", "currency", "incoterm", "shipment_id"],
    "packing_list": ["date", "invoice_number", "shipment_id"],
    "bill_of_lading": ["invoice_number", "shipper", "port_of_discharge", "delivery_eta", "vessel",
                       "total_packages", "gross_weight_kg", "shipment_id"],
}
ORDER = {
    "purchase_order": ["doc_type", "doc_number", "date", "seller", "buyer", "currency", "incoterm", "items",
                       "shipment_id", "total_amount"],
    "invoice": ["doc_type", "doc_number", "date", "po_number", "seller", "buyer", "currency", "incoterm",
                "items", "shipment_id", "total_amount"],
    "packing_list": ["doc_type", "doc_number", "date", "invoice_number", "items", "total_packages",
                     "gross_weight_kg", "shipment_id"],
    "bill_of_lading": ["doc_type", "doc_number", "invoice_number", "shipper", "consignee", "port_of_loading",
                       "port_of_discharge", "ship_on_board_date", "delivery_eta", "vessel", "container_no",
                       "total_packages", "gross_weight_kg", "shipment_id"],
}
METHOD_FACTOR = {"text": 1.0, "table": 0.95, "ocr": 0.8, "json": 1.0, "llm": 0.5}


class ExtractError(Exception):
    """Carries the HTTP status the API should answer with."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def capabilities():
    caps = R.capabilities()
    caps["formats"] = R.ALLOWED_EXTENSIONS
    caps["llm"] = _llm_available()
    caps["limits"] = {"max_files": MAX_FILES, "max_bytes_per_file": MAX_BYTES, "max_pages_per_pdf": MAX_PAGES,
                      "allowed_extensions": R.ALLOWED_EXTENSIONS}
    return caps


def _llm_available():
    try:
        import extractor_llm
        return extractor_llm.available()
    except Exception:
        return False


# ------------------------------------------------------------------ one document
def _blank_doc(doc_type):
    doc = {k: None for k in ORDER[doc_type]}
    doc["doc_type"] = doc_type
    doc["items"] = []
    return doc


def _clean_field(field, raw, conf, method_factor):
    """Normalise one raw label value into the type the model expects. Returns (value, confidence)."""
    if raw is None:
        return None, 0.0
    conf = round(min(max(conf * method_factor, 0.0), 1.0), 2)
    if field in ("date", "ship_on_board_date", "delivery_eta"):
        val, dconf = F.parse_date(raw)
        return val, (0.0 if val is None else round(min(conf * dconf, 1.0), 2))
    if field in ("qty", "total_packages"):
        val = F.as_int(F.parse_number(raw))
        return (None if val is None else F.as_int(val)), (0.0 if val is None else conf)
    if field in ("unit_price", "total", "total_amount", "gross_weight_kg", "net_weight_kg"):
        val = F.parse_number(raw)
        return (None if val is None else round(float(val), 4)), (0.0 if val is None else conf)
    val = F.clean_value(raw)
    if val and field in ("hs_code",):
        digits = re.sub(r"\D", "", val)
        val = digits or val
    if val and len(val) > 120:
        val = val[:120]
    return val, (0.0 if not val else conf)


def _extract_section(section, filename):
    """section = {"doc_type", "lines"} -> (doc, confidence, warnings)"""
    doc_type = section["doc_type"]
    lines = section["lines"]
    method = "table" if any(R.TAB in l or "|" in l for l in lines) else "text"
    method = "ocr" if section.get("method") == "ocr" else method
    factor = METHOD_FACTOR.get(method, 1.0)
    doc = _blank_doc(doc_type)
    conf, warnings = {}, []

    labels = F.FIELD_LABELS[doc_type]
    deny = F.FIELD_DENY.get(doc_type, {})
    for field, patterns in labels.items():
        raw, pconf = F.find_label_value(lines, patterns, deny.get(field))
        value, c = _clean_field(field, raw, pconf, factor)
        doc[field] = value
        if value is not None:
            conf["%s.%s" % (doc_type, field)] = c

    if doc_type in ("purchase_order", "invoice"):
        cur = F.parse_currency(lines)
        inc = F.parse_incoterm(lines)
        if cur:
            doc["currency"], conf["%s.currency" % doc_type] = cur, round(factor * 0.9, 2)
        if inc:
            doc["incoterm"], conf["%s.incoterm" % doc_type] = inc, round(factor * 0.85, 2)
    if doc_type == "bill_of_lading" and not doc.get("container_no"):
        c_no = F.parse_container(lines)
        if c_no:
            doc["container_no"], conf["%s.container_no" % doc_type] = c_no, round(factor * 0.8, 2)

    items = F.parse_items(lines, doc_type)
    doc["items"] = items
    if items:
        keys = ITEM_KEYS[doc_type]
        filled = sum(1 for it in items for k in keys if it.get(k) is not None)
        conf["%s.items" % doc_type] = round(factor * (filled / float(len(items) * max(len(keys), 1))), 2)
        for idx, it in enumerate(items, start=1):
            for k in keys:
                if it.get(k) is None:
                    warnings.append("%s item %d: %s not found" % (doc_type, idx, k))
    elif doc_type in ("purchase_order", "invoice", "packing_list"):
        warnings.append("%s: no line-item table found (no column header row with Qty)" % doc_type)
    return doc, conf, warnings, method


def _merge_doc(target, doc, conf, warnings, src, out_conf, out_warnings, seen):
    """First document of a type wins; later ones are reported, never merged silently."""
    doc_type = target["doc_type"]
    if doc_type in seen:
        out_warnings.append("%s: a second document of this type was found in %s and ignored "
                            "(the first one, from %s, is kept)" % (doc_type, src, seen[doc_type]))
        return
    seen[doc_type] = src
    merged = {}
    for k in ORDER[doc_type]:
        merged[k] = doc["items"] if k == "items" else doc.get(k)
    target.clear()
    target.update(merged)
    out_conf.update(conf)
    for w in warnings:
        out_warnings.append(w)


# ------------------------------------------------------------------ whole run
def extract_files(files):
    """files: list of {"name": str, "data": bytes}. Returns the result dict, raises ExtractError."""
    if not files:
        raise ExtractError(422, "no files were uploaded")
    if len(files) > MAX_FILES:
        raise ExtractError(413, "too many files: %d, the limit is %d per upload" % (len(files), MAX_FILES))
    for f in files:
        if len(f["data"]) > MAX_BYTES:
            raise ExtractError(413, "%s is %.1f MB, the limit is %d MB per file"
                               % (f["name"], len(f["data"]) / 1048576.0, MAX_BYTES // 1048576))

    tmpdir = tempfile.mkdtemp(prefix="trustgate_extract_")
    try:
        return _extract_in(files, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)          # temp files are always removed


def _extract_in(files, tmpdir):
    documents, confidence, missing, warnings = ({k: _blank_doc(k) for k in DOC_KEYS}, {}, [], [])
    file_reports, seen, all_lines = [], {}, []

    for idx, f in enumerate(files):
        name = _sanitise(f["name"])
        data = f["data"]
        if not data:
            raise ExtractError(422, "%s is empty (0 bytes)" % name)
        ext = os.path.splitext(name)[1].lower().lstrip(".")
        if ext not in R.EXT_TYPES:
            raise ExtractError(415, "unsupported file type '%s'. Allowed: %s"
                               % (ext or "(no extension)", ", " + ", ".join(R.ALLOWED_EXTENSIONS)))
        kind, evidence = R.detect_type(name, data)
        if kind is None:
            raise ExtractError(415, "%s is not a supported document (%s). Allowed: %s"
                               % (name, evidence, ", ".join(R.ALLOWED_EXTENSIONS)))
        path = os.path.join(tmpdir, "f%02d_%s.%s" % (idx, _safe_name(name), ext))
        with open(path, "wb") as fh:
            fh.write(data)

        result = R.read_file(path, name, kind, MAX_PAGES)
        if not result.pages and result.structured is None:
            # the container was recognised but nothing could be read out of it -> unreadable upload
            raise ExtractError(422, "%s could not be read (%s)" %
                               (name, "; ".join(result.warnings) or "no text, no pages, not valid %s" % kind))
        warnings.extend("%s: %s" % (name, w) for w in result.warnings)

        report = {"name": name, "detected_type": kind, "detection": evidence, "doc_type": None,
                  "pages": len(result.pages), "method": result.method,
                  "documents": [], "error": None}
        if result.structured is not None:
            _apply_structured(result.structured, documents, confidence, warnings, name, report)
            file_reports.append(report)
            for k in report["documents"]:
                seen[k] = name
            continue
        if not result.pages:
            warnings.append("%s: no readable text (%s)" % (name, "; ".join(result.warnings) or "empty"))
            report["error"] = "no readable text"
            file_reports.append(report)
            continue

        lines, meta, cursor = [], [], 0
        for page in result.pages:
            meta.append(page.meta(cursor))
            lines.extend(page.lines)
            cursor += len(page.lines)
        all_lines.extend(lines)
        sections = F.split_documents(lines, name, meta)
        if len(sections) > 1:
            warnings.append("%s: %d documents found in one file" % (name, len(sections)))
        for sec in sections:
            sec["method"] = result.method
            page_methods = [p.get("method") for p in sec.get("pages") or []]
            if page_methods and any(m == "ocr" for m in page_methods):
                sec["method"] = "ocr"
            doc_type = sec["doc_type"]
            if not doc_type:
                report["documents"].append("unclassified")
                warnings.append("%s: document could not be classified - ignored (not guessed)" % name)
                continue
            report["documents"].append(doc_type)
            doc, conf, sw, method = _extract_section(sec, name)
            if _llm_available():
                doc, conf, llm_w = _llm_pass(doc, doc_type, sec, conf)
                warnings.extend("%s: %s" % (name, w) for w in llm_w)
            _merge_doc(documents[doc_type], doc, conf, sw, name, confidence, warnings, seen)
            if report["doc_type"] is None:
                report["doc_type"], report["method"] = doc_type, method
        if report["doc_type"] is None:
            report["doc_type"] = "unclassified"
        file_reports.append(report)

    _finalise(documents, confidence, missing, warnings, all_lines)
    return {"shipment": {"shipment_id": documents["_shipment_id"], "documents": {k: documents[k] for k in DOC_KEYS}},
            "confidence": confidence, "missing": missing, "warnings": warnings, "files": file_reports}


def _llm_pass(doc, doc_type, sec, conf):
    import extractor_llm
    text = "\n".join(sec["lines"])[:24000]
    if extractor_llm.available() and any(doc.get(k) in (None, "") for k in REQUIRED[doc_type]):
        fills = extractor_llm.fill_missing(doc_type, doc, text)
        warnings = []
        for k, v in fills.items():
            if doc.get(k) in (None, ""):
                val, c = _clean_field(k, v, 0.5, 1.0)
                if val is not None:
                    doc[k] = val
                    conf["%s.%s" % (doc_type, k)] = METHOD_FACTOR["llm"]
                    warnings.append("%s.%s was filled by the optional LLM pass, not by a rule" % (doc_type, k))
        if not fills:
            warnings = []
        return doc, conf, warnings
    return doc, conf, []


def _apply_structured(ship, documents, confidence, warnings, name, report):
    docs = ship.get("documents") or {}
    for k in DOC_KEYS:
        if not isinstance(docs.get(k), dict):
            continue
        src = docs[k]
        if k in report["documents"]:
            warnings.append("%s: %s already found in another file - ignored" % (name, k))
            continue
        report["documents"].append(k)
        for f in list(ORDER[k]) + ["items"]:
            if f in src:
                documents[k][f] = src[f]
        for f in ORDER[k]:
            if f != "items" and documents[k].get(f) is not None:
                confidence["%s.%s" % (k, f)] = 1.0
        if documents[k].get("items"):
            confidence["%s.items" % k] = 1.0
        report["doc_type"] = report["doc_type"] or k
    if ship.get("shipment_id"):
        documents["_shipment_id"] = ship["shipment_id"]
        confidence["shipment_id"] = 1.0


def _finalise(documents, confidence, missing, warnings, all_lines=None):
    doc_numbers = [documents[k].get("doc_number") for k in DOC_KEYS if documents[k].get("doc_number")]

    # 1. shipment id: from the documents, otherwise derived (and flagged)
    sid = documents.get("_shipment_id")
    derived = False
    if not sid and all_lines:
        sid = F.parse_shipment_id(all_lines, doc_numbers)
        if sid:
            confidence["shipment_id"] = 0.75
    if not sid:
        for k in DOC_KEYS:
            if documents[k].get("shipment_id"):
                sid = documents[k]["shipment_id"]
                confidence["shipment_id"] = 0.9
                break
    if not sid:
        stem = hashlib.sha256("|".join(sorted(str(n) for n in doc_numbers)).encode("utf-8")).hexdigest()[:8].upper()
        sid = "SHP-" + stem
        derived = True
        confidence["shipment_id"] = 0.3
        warnings.append("no shipment id in the documents: generated %s from a hash of the document "
                        "numbers - edit it if you have the real one" % sid)
    documents["_shipment_id"] = sid

    # 2. report what is still unreadable, and normalise the key order
    for k in DOC_KEYS:
        doc = documents[k]
        doc.setdefault("doc_type", k)
        doc.setdefault("items", [])
        doc["shipment_id"] = sid
        for field in REQUIRED[k] + EXTRA_FIELDS[k]:
            if field == "items":
                continue
            if doc.get(field) in (None, ""):
                if derived and field == "shipment_id":
                    continue
                missing.append("%s.%s" % (k, field))
                confidence.pop("%s.%s" % (k, field), None)
        if not doc.get("items") and k != "bill_of_lading":
            missing.append("%s.items" % k)
        for idx, it in enumerate(doc["items"], start=1):
            for field in ITEM_KEYS[k]:
                if it.get(field) is None:
                    missing.append("%s.items[%d].%s" % (k, idx, field))
        documents[k] = {f: doc.get(f) for f in ORDER[k]}
        documents[k]["shipment_id"] = sid
    seen = []
    for m in missing:                                   # keep the first occurrence, preserve order
        if m not in seen:
            seen.append(m)
    missing[:] = seen


def _sanitise(name):
    """Strip any path and unsafe characters from an uploaded filename."""
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ ()\-]", "_", name).strip(" .")
    return (name or "upload")[:120]


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9._\-]", "_", os.path.splitext(name)[0])[:60] or "file"