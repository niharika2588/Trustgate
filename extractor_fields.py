"""
Deterministic, offline field extraction for TrustGate shipment documents.

No network, no ML: label regexes + table/column heuristics + date/number normalisation.
Nothing is ever invented: a value that cannot be read stays None and is reported as missing.
"""
import re

DOC_TYPES = ("purchase_order", "invoice", "packing_list", "bill_of_lading")

# ------------------------------------------------------------------ small value parsers
_THOUSANDS = ".,'’ "


def clean_value(v):
    if v is None:
        return None
    v = str(v).replace("\u00a0", " ").strip()
    v = re.sub(r"^[\s:;#=\-–—.]+", "", v)
    v = re.sub(r"[\s.,;:]+$", "", v)
    return v or None


def parse_number(raw):
    """'1,037.00' -> 1037.0 ; 'USD 2 227' -> 2227 ; '1.037,00' -> 1037.0 ; '(12.50)' -> -12.5"""
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d.,]", "", s.replace(" ", "").replace("\u00a0", ""))
    if not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        grouped = re.fullmatch(r"\d{1,3}(,\d{3})+", s)          # 1,037 / 1,037,000
        if grouped or len(tail) == 3 or not head:
            s = s.replace(",", "")
        else:
            s = head + "." + tail                              # European decimal comma
    elif s.count(".") > 1:
        s = s.replace(".", "")                                 # 1.037.000 style grouping
    try:
        val = float(s)
    except ValueError:
        return None
    return -val if neg and val > 0 else val


def as_int(val):
    if val is None:
        return None
    return int(round(val)) if abs(val - round(val)) < 1e-9 else val


_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_MONTHS.update({m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "June", "July", "August",
     "September", "October", "November", "December"], 1) if m not in _MONTHS})


def _y(m):
    y = int(m)
    if y < 100:
        y += 2000 if y < 70 else 1900
    return y


def parse_date(raw):
    """Returns (YYYY-MM-DD, confidence) - confidence drops when the D/M vs M/D order was ambiguous."""
    if raw is None:
        return None, 0.0
    s = clean_value(str(raw)) or ""
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s, flags=re.I)
    s = re.sub(r"[,.]?\s*\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?.*$", "", s, flags=re.I)   # drop clock time / "T00:00"
    s = s.replace("年", "-").replace("月", "-").replace("日", "")
    s = s.strip(" .,-/")

    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)                        # ISO-ish
    if m:
        return _mk_date(_y(m.group(1)), int(m.group(2)), int(m.group(3))), 0.95
    m = re.search(r"(\d{1,2})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{2,4})", s)          # 15/06/2026
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        conf = 0.95
        if a > 12 and b <= 12:
            d, mo = a, b
        elif b > 12 and a <= 12:
            d, mo = b, a
        else:                                                                       # genuinely ambiguous
            d, mo, conf = a, b, 0.75
        return _mk_date(_y(m.group(3)), mo, d), conf
    m = re.search(r"(\d{1,2})\s*[-/. ]\s*([A-Za-z]{3,})\s*[-/. ]\s*(\d{2,4})", s)  # 15-Jun-2026
    if m and m.group(2)[:3].lower() in _MONTHS:
        return _mk_date(_y(m.group(3)), _MONTHS[m.group(2)[:3].lower()], int(m.group(1))), 0.9
    m = re.search(r"([A-Za-z]{3,})\.?\s+(\d{1,2}),?\s+(\d{2,4})", s)                # Jun 15, 2026
    if m and m.group(1)[:3].lower() in _MONTHS:
        return _mk_date(_y(m.group(3)), _MONTHS[m.group(1)[:3].lower()], int(m.group(2))), 0.9
    m = re.search(r"(\d{1,2})\s+([A-Za-z]{3,})\.?\s+(\d{2,4})", s)                 # 15 June 2026
    if m and m.group(2)[:3].lower() in _MONTHS:
        return _mk_date(_y(m.group(3)), _MONTHS[m.group(2)[:3].lower()], int(m.group(1))), 0.9
    return None, 0.0


def _mk_date(y, m, d):
    if not (1900 <= y <= 2200 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    try:
        from datetime import date
        return date(y, m, d).isoformat()
    except ValueError:
        return None


CURRENCY_SYMBOLS = {"$": "USD", "us$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR", "₩": "KRW"}
ISO_RE = re.compile(r"\b(USD|EUR|GBP|JPY|CNY|RMB|INR|AUD|CAD|CHF|SGD|HKD|AED|SAR|SEK|NOK|DKK|ZAR|BRL|MXN|KRW)\b", re.I)
ISO_MAP = {"RMB": "CNY"}

CURRENCY_LABEL = re.compile(r"(currency|amounts?\s+in|in\s+(?:usd|eur|gbp))", re.I)


def parse_currency(lines):
    for line in lines:
        if CURRENCY_LABEL.search(line):
            tail = line.split(":", 1)[1] if ":" in line else line
            m = ISO_RE.search(tail)
            if m:
                return _iso(m.group(1))
            m = re.search(r"[$\u20ac\u00a3\u00a5\u20b9]", tail)
            if m:
                return CURRENCY_SYMBOLS.get(m.group(0), None)
    for line in lines:
        m = ISO_RE.search(line)
        if m:
            return _iso(m.group(1))
    for line in lines:
        for sym, code in CURRENCY_SYMBOLS.items():
            if sym in line:
                return code
    return None


def _iso(code):
    code = code.upper()
    return ISO_MAP.get(code, code)


INCOTERMS = ("EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP", "DDU")
_INCOTERM_RE = re.compile(r"\b(" + "|".join(INCOTERMS) + r")\b", re.I)


def parse_incoterm(lines):
    for line in lines:
        if re.search(r"(incoterm|delivery terms|terms of delivery|trade terms)", line, re.I):
            m = _INCOTERM_RE.search(line.split(":", 1)[-1] if ":" in line else line)
            if m:
                return m.group(1).upper()
    for line in lines:
        m = _INCOTERM_RE.search(line)
        if m:
            return m.group(1).upper()
    return None


CONTAINER_RE = re.compile(r"\b([A-Z]{3}\s?\d{6,7})\b")


def parse_container(lines):
    for line in lines:
        if re.search(r"(container|equipment|cntr)", line, re.I):
            m = CONTAINER_RE.search(line.replace(" ", ""))
            if m:
                return re.sub(r"\s+", "", m.group(1))
            v = clean_value(line.split(":", 1)[-1] if ":" in line else "")
            if v and re.search(r"\d{5,}", v):
                return re.sub(r"\s+", "", v)
    for line in lines:
        m = CONTAINER_RE.search(line.replace(" ", ""))
        if m:
            return re.sub(r"\s+", "", m.group(1))
    return None


SHIPMENT_RE = re.compile(r"(\d[\d\-\s]{3,})")


def parse_shipment_id(lines, doc_numbers):
    """Explicit 'Shipment No' wins; otherwise the most common trailing digit group of the document numbers."""
    for line in lines:
        if re.search(r"(shipment|shpt|consignment|transport)\s*(id|no|number|ref|#)", line, re.I):
            v = clean_value(line.split(":", 1)[-1] if ":" in line else "")
            if v and re.search(r"\d{4,}", v):
                m = SHIPMENT_RE.search(v)
                digits = re.sub(r"\D", "", m.group(1)) if m else ""
                return ("SHP-" + digits) if digits else v
    groups = {}
    for n in doc_numbers:
        m = re.search(r"(\d{3,})\s*$", str(n or "").strip())
        if m:
            groups[m.group(1)] = groups.get(m.group(1), 0) + 1
    if not groups:
        return None
    best = max(groups.items(), key=lambda kv: kv[1])
    if best[1] < 2:                                     # a single number is not enough of a signal
        return None
    return "SHP-" + best[0]


# ------------------------------------------------------------------ classification
KEYWORDS = {
    "invoice": [("commercial invoice", 6), ("proforma invoice", 6), ("pro forma invoice", 6), ("tax invoice", 6),
                ("invoice no", 5), ("invoice number", 5), ("invoice date", 4), ("inv no", 4), ("amount due", 2),
                ("invoice", 2)],
    "purchase_order": [("purchase order", 6), ("purchase order no", 6), ("purchase order number", 6), ("po no", 5),
                       ("po number", 5), ("p.o. no", 5), ("requisition", 3), ("to order of", 3), ("order number", 2),
                       ("purchase order", 2)],
    "packing_list": [("packing list", 6), ("packing slip", 6), ("packing list no", 6), ("packing list number", 6),
                     ("pl no", 4), ("cartons", 3), ("total packages", 4), ("number of packages", 4),
                     ("net weight", 3), ("gross weight", 3), ("packing list", 2)],
    "bill_of_lading": [("bill of lading", 6), ("b/l no", 6), ("bl no", 6), ("b/l number", 6), ("waybill", 4),
                       ("master bill", 5), ("consignee", 4), ("port of loading", 4), ("port of discharge", 4),
                       ("on board", 3), ("container no", 3), ("shipper", 2), ("vessel", 2)],
}
NAME_HINTS = {"invoice": ("invoice", "inv", "factuur"), "purchase_order": ("purchase", "po", "order"),
              "packing_list": ("packing", "pl", "packlist"),
              "bill_of_lading": ("lading", "bol", "bl", "waybill", "b_l")}
TITLE_RE = {t: re.compile(r"^\W{0,3}(" + "|".join(sorted((re.escape(k) for k, _ in v), key=len, reverse=True)) +
                          r")\W{0,3}$", re.I) for t, v in KEYWORDS.items()}


def score_doc_type(lines, filename=""):
    """Keyword score per document type. Returns {doc_type: score}."""
    text = "\n".join(lines)
    low = text.lower()
    name = re.sub(r"[^a-z]+", " ", filename.lower())
    scores = {}
    for t, kws in KEYWORDS.items():
        s = 0
        for kw, weight in kws:
            n = low.count(kw)
            if n:
                s += weight * (1 + min(n - 1, 2) * 0.25)
        for hint in NAME_HINTS[t]:
            if re.search(r"\b" + re.escape(hint) + r"\b", name):
                s += 6                       # the filename alone is enough to name the document type
                break
        scores[t] = s
    return scores


def split_documents(lines, filename="", pages_meta=None):
    """Split a flat line list into logical documents (by title headings). Falls back to one document."""
    marks = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or len(s) > 60:
            continue
        best, score = None, 0
        for t, rx in TITLE_RE.items():
            if rx.match(s):
                sc = score_doc_type([s]).get(t, 0)
                if sc > score:
                    best, score = t, sc
        if best:
            marks.append((i, best))
    # drop a mark that repeats the same type too close to the previous one (page headers)
    clean = []
    for i, t in marks:
        if clean and clean[-1][1] == t and i - clean[-1][0] < 25:
            continue
        clean.append((i, t))
    if len(clean) >= 2:
        bounds, doc_types = [], []
        for k, (i, t) in enumerate(clean):
            end = clean[k + 1][0] if k + 1 < len(clean) else len(lines)
            bounds.append((i, end))
            doc_types.append(t)
    else:
        sc = score_doc_type(lines, filename)
        top = max(sc, key=lambda k: sc[k])
        bounds, doc_types = [(0, len(lines))], [top if sc[top] >= 6 else None]
    out = []
    for (i, j), t in zip(bounds, doc_types):
        seg = lines[i:j]
        pg = None
        if pages_meta is not None:
            pg = [p for p in pages_meta if p.get("start", 0) <= i < p.get("end", j + 1)]
        out.append({"doc_type": t, "lines": seg, "pages": pg or []})
    return out


# ------------------------------------------------------------------ label-based fields
# each entry: field -> list of (label regex, confidence)
FIELD_LABELS = {
    "purchase_order": {
        "doc_number": [(r"purchase\s*order\s*(no|number|#|ref)", .95), (r"\bp\.?\s*o\.?\s*(no|number|#)", .95),
                       (r"\bpo\s*(no|number|#)\b", .95), (r"\border\s*(no|number|#)\b", .6)],
        "date": [(r"purchase\s*order\s*date", .95), (r"order\s*date", .95), (r"\bdate\b", .7), (r"\bdated\b", .8)],
        "po_number": [(r"purchase\s*order\s*(no|number|#)", .95), (r"\bp\.?\s*o\.?\s*(no|number|#)", .95),
                      (r"\bpo\s*(no|number|#)\b", .95)],
        "seller": [(r"\bseller\b", .95), (r"\bsupplier\b", .95), (r"\bvendor\b", .95), (r"\bexporter\b", .8),
                   (r"\bsold\s+by\b", .95), (r"\bfrom\b", .4)],
        "buyer": [(r"\bbuyer\b", .95), (r"\bpurchaser\b", .95), (r"\bbill\s+to\b", .9), (r"\bsold\s+to\b", .95),
                  (r"\bto\s+order\s+of\b", .9), (r"\bfor\s+the\s+account\s+of\b", .8), (r"\bcustomer\b", .7)],
        "total_amount": [(r"grand\s*total", .95), (r"total\s*(amount|value)", .95), (r"order\s*value", .9),
                         (r"total\s*price", .85), (r"\btotal\b", .7)],
    },
    "invoice": {
        "doc_number": [(r"invoice\s*(no|number|#|ref)", .95), (r"commercial\s*invoice\s*(no|number|#)", .95),
                       (r"tax\s*invoice\s*(no|number|#)", .95), (r"\binv\s*(no|number|#)\b", .95), (r"\bref\b", .4)],
        "date": [(r"invoice\s*date", .95), (r"date\s*of\s*issue", .95), (r"issue\s*date", .95),
                 (r"date\s*of\s*invoice", .95), (r"\bdate\b", .7), (r"\bdated\b", .8)],
        "po_number": [(r"purchase\s*order\s*(no|number|#)", .95), (r"\bp\.?\s*o\.?\s*(no|number|#)", .95),
                      (r"\bpo\s*(no|number|#)\b", .95), (r"your\s*order\s*(no|number)", .8),
                      (r"reference\s*(po|order)", .8)],
        "seller": [(r"\bseller\b", .95), (r"\bsupplier\b", .95), (r"\bvendor\b", .9), (r"\bexporter\b", .85),
                   (r"\bsold\s+by\b", .95), (r"\bfrom\b", .4)],
        "buyer": [(r"\bbuyer\b", .95), (r"\bpurchaser\b", .95), (r"\bbill\s+to\b", .9), (r"\bsold\s+to\b", .95),
                  (r"\bcustomer\b", .8), (r"\bto\s*order\s+of\b", .8)],
        "total_amount": [(r"grand\s*total", .95), (r"invoice\s*total", .95), (r"total\s*(amount|value)", .95),
                         (r"amount\s*due", .8), (r"balance\s*due", .7), (r"\btotal\b", .7)],
    },
    "packing_list": {
        "doc_number": [(r"packing\s*(list|slip)\s*(no|number|#|ref)", .95), (r"\bpl\s*(no|number|#)\b", .95),
                       (r"\bpack\s*list\s*(no|number|#)", .95), (r"document\s*(no|number)", .6)],
        "date": [(r"packing\s*(list|slip)\s*date", .95), (r"\bdate\b", .7), (r"\bdated\b", .8)],
        "invoice_number": [(r"invoice\s*(no|number|#)", .95), (r"inv\s*(no|number|#)", .95),
                           (r"commercial\s*invoice\s*(no|number|#)", .95)],
        "total_packages": [(r"total\s*(packages|cartons|pallets|drums|boxes)", .95),
                           (r"(no\.?|number)\s*of\s*(packages|cartons|pallets|boxes)", .95),
                           (r"total\s*packages", .95), (r"\bpackages\b", .6), (r"\bcartons\b", .6),
                           (r"\bpallets\b", .6), (r"\bboxes\b", .5)],
        "gross_weight_kg": [(r"total\s*gross\s*weight", .95), (r"gross\s*weight(\s*\(\s*kg\s*\))?", .95),
                            (r"gross\s*wt", .9)],
    },
    "bill_of_lading": {
        "doc_number": [(r"b\s*/?\s*l\s*(no|number|#)", .95), (r"bill\s*of\s*lading\s*(no|number|#)", .95),
                       (r"\bbol\s*(no|number|#)", .95), (r"master\s*bill\s*(no|number|#)", .95),
                       (r"waybill\s*(no|number|#)", .9), (r"\bbl\s*(no|number|#)\b", .9)],
        "invoice_number": [(r"invoice\s*(no|number|#)", .95), (r"inv\s*(no|number|#)", .95)],
        "shipper": [(r"\bshipper\b", .95), (r"\bexporter\b", .8), (r"\bshipped\s+by\b", .9)],
        "consignee": [(r"\bconsignee\b", .95), (r"\bnotify\s*party\b", .6)],
        "port_of_loading": [(r"port\s*of\s*loading", .95), (r"port\s*of\s*embarkation", .95),
                            (r"loading\s*port", .95), (r"\bpol\b", .9), (r"\bfrom\s*port\b", .9)],
        "port_of_discharge": [(r"port\s*of\s*discharge", .95), (r"discharge\s*port", .95),
                              (r"destination\s*port", .95), (r"port\s*of\s*destination", .95), (r"\bpod\b", .9)],
        "ship_on_board_date": [(r"(shipped\s*)?on\s*board(\s*date)?", .95), (r"laden\s*on\s*board", .95),
                               (r"date\s*of\s*shipment", .9), (r"sailing\s*date", .8)],
        "delivery_eta": [(r"(estimated\s*)?(time\s*of\s*)?arrival(\s*date)?", .95), (r"\beta\b", .9),
                         (r"delivery\s*date", .9), (r"estimated\s*delivery", .9)],
        "vessel": [(r"vessel(\s*name)?", .95), (r"ship\s*name", .9), (r"\bcarrier\b", .5)],
        "container_no": [(r"container\s*(no|number|#)", .95), (r"equipment\s*(no|number)", .9),
                         (r"container", .6)],
        "total_packages": [(r"total\s*(packages|cartons|pallets)", .95), (r"no\.?\s*of\s*packages", .95),
                           (r"\bpackages\b", .6), (r"\bcartons\b", .6), (r"\bpallets\b", .6)],
        "gross_weight_kg": [(r"total\s*gross\s*weight", .95), (r"gross\s*weight(\s*\(\s*kg\s*\))?", .95),
                            (r"gross\s*wt", .9)],
    },
}
# generic fallback: the machine readable field name itself ("doc_number", "total_amount", ...) so that
# XML exports and template driven documents are read too. Deliberately a low confidence match.
for _dt, _fields in FIELD_LABELS.items():
    for _f in _fields:
        _generic = (r"(?<![A-Za-z])" + _f.replace("_", r"[ _\-]") + r"(?![A-Za-z])", 0.8)
        if not any(p == _generic[0] for p, _c in _fields[_f]):
            _fields[_f] = _fields[_f] + [_generic]

# fields that must never be filled from a generic /D/M order that would clearly be another field
FIELD_DENY = {"invoice": {"seller": [r"\bbuyer\b", r"bill\s+to", r"sold\s+to", r"\bpurchaser\b", r"\bcustomer\b"],
                          "buyer": [r"\bseller\b", r"\bsupplier\b", r"sold\s+by", r"\bexporter\b", r"\bvendor\b"]},
              "purchase_order": {"seller": [r"\bbuyer\b", r"bill\s+to", r"sold\s+to", r"\bpurchaser\b"],
                                 "buyer": [r"\bseller\b", r"\bsupplier\b", r"sold\s+by", r"\bexporter\b",
                                           r"\bvendor\b"]}}

LABEL_SPLIT = re.compile(r"^\s*(?P<label>[^:=]{2,60}?)\s*(?P<sep>[:=#])\s*(?P<value>.*)$")


def labelled_pairs(lines):
    """Yield (label, value) for 'Label: value' style lines, joining a value that wrapped to the next line."""
    out = []
    for i, line in enumerate(lines):
        m = LABEL_SPLIT.match(line)
        if m:
            value = m.group("value").strip()
            if not value and i + 1 < len(lines) and not LABEL_SPLIT.match(lines[i + 1]):
                value = lines[i + 1].strip()
            out.append((m.group("label"), value, lines[i]))
        else:
            out.append((None, None, line))
    return out


def find_label_value(lines, patterns, deny=None):
    """First label regex that matches a line; returns (raw value, confidence)."""
    deny = [re.compile(d, re.I) for d in (deny or [])]
    for label, value, raw in labelled_pairs(lines):
        if not label or value is None:
            continue
        for rx, conf in patterns:
            if re.search(rx, label, re.I) and not any(d.search(label) for d in deny):
                v = clean_value(value)
                if v:
                    return v, conf
    # label on its own line, value on the next one ("Total Amount:" then the number)
    for i, line in enumerate(lines):
        s = line.strip().rstrip(":=")
        if not s or len(s) > 45 or i + 1 >= len(lines):
            continue
        if "\t" in s or "|" in s or len(split_cells(line)) > 1:
            continue                                     # this row is already split: it is a label/value row
        for rx, conf in patterns:
            if re.search(rx, s, re.I) and not any(d.search(s) for d in deny):
                v = clean_value(lines[i + 1])
                if v and not re.search(r"\d{2,}:\d{2}", v):
                    return v, conf - 0.1
    # two column key/value sheet: "Invoice No." in cell 0, the value in cell 1 (CSV/TSV/XLSX export)
    for line in lines:
        cells = split_cells(line)
        if len(cells) < 2:
            continue
        if sum(1 for c in cells if _match_column(c)) >= 3:
            continue                                     # this is an item header row, not a label row
        key, value = clean_value(cells[0]), clean_value(" ".join(cells[1:]))
        if not key or not value or len(key) > 60 or len(value) > 80:
            continue
        for rx, conf in patterns:
            if re.search(rx, key, re.I) and not any(d.search(key) for d in deny):
                return value, conf - 0.05
    return None, 0.0


# ------------------------------------------------------------------ line items (tables)
ITEM_COLUMNS = {
    "index": [r"^(item|sr|line|seq|no|#|s\.?\s?no)\.?$", r"^(item|line)\s*(no|number|#)$", r"^s\.?n$"],
    "product_id": [r"product\s*(code|id|no|number|#)", r"^product$", r"^sku$", r"article\s*(no|number|code)",
                   r"^article$", r"item\s*(code|id)", r"material\s*(no|code)", r"part\s*(no|number)"],
    "hs_code": [r"hs\s*(code|no|number)", r"^hsn?$", r"harmoniz?ed", r"tariff\s*(code|no)", r"commodity\s*code"],
    "unit_price": [r"unit\s*price", r"unit\s*cost", r"^rate$", r"^price$", r"price\s*/\s*unit", r"unit\s*value",
                   r"unit\s*price\s*\(\w+\)"],
    "total": [r"^total$", r"line\s*total", r"^amount$", r"total\s*amount", r"^value$", r"extended",
              r"net\s*amount", r"total\s*value", r"amount\s*\(?\w+\)?"],
    "net_weight_kg": [r"net\s*weight", r"net\s*wt", r"n\.?\s*w\.?", r"^weight(\s*\(?\s*kg\s*\)?)?$", r"^kg$"],
    "qty": [r"^qty$", r"^quantity$", r"q'ty", r"\bpcs\b", r"\bpieces\b", r"^units?$", r"qty\s*\(\w+\)"],
    "description": [r"description", r"product\s*name", r"^goods$", r"commodity", r"item(\s*desc.*)?",
                    r"\bnarration\b", r"particulars", r"designation"],
}
ITEM_KEYS = ("product_id", "hs_code", "description", "qty", "unit_price", "total", "net_weight_kg")
NUMERIC_KEYS = ("qty", "unit_price", "total", "net_weight_kg")
WEIGHT_KEYS = ("net_weight_kg",)

# Exact (normalised) header text -> role. Checked before the regexes so that a column called
# "Unit Price" can never be read as a quantity just because it contains the word "unit".
EXACT_COLUMNS = {
    "item": "index", "item no": "index", "item number": "index", "no": "index", "sr no": "index",
    "line": "index", "line no": "index", "seq": "index",
    "product": "product_id", "product code": "product_id", "product id": "product_id",
    "product no": "product_id", "product number": "product_id", "sku": "product_id", "article": "product_id",
    "article no": "product_id", "item code": "product_id", "item id": "product_id", "material": "product_id",
    "material no": "product_id", "part no": "product_id", "part number": "product_id",
    "hs": "hs_code", "hs code": "hs_code", "hs no": "hs_code", "hsn": "hs_code", "hsn code": "hs_code",
    "harmonized code": "hs_code", "harmonised code": "hs_code", "tariff code": "hs_code",
    "commodity code": "hs_code", "commodity no": "hs_code",
    "description": "description", "product name": "description", "goods": "description",
    "commodity": "description", "item description": "description", "narration": "description",
    "particulars": "description", "designation": "description",
    "qty": "qty", "quantity": "qty", "q'ty": "qty", "pcs": "qty", "pieces": "qty", "units": "qty",
    "unit": "qty", "unit qty": "qty", "quantity pcs": "qty", "qty pcs": "qty",
    "unit price": "unit_price", "unit cost": "unit_price", "price": "unit_price", "rate": "unit_price",
    "unit value": "unit_price", "price unit": "unit_price",
    "total": "total", "amount": "total", "line total": "total", "total amount": "total", "value": "total",
    "extended": "total", "net amount": "total", "total value": "total",
    "net weight": "net_weight_kg", "net wt": "net_weight_kg", "n w": "net_weight_kg",
    "weight": "net_weight_kg", "kg": "net_weight_kg", "net weight kg": "net_weight_kg",
    "weight kg": "net_weight_kg",
}
_FALLBACK_ORDER = ("index", "product_id", "hs_code", "unit_price", "total", "net_weight_kg", "qty", "description")


def _normalise_header(cell):
    c = str(cell).lower()
    c = re.sub(r"\(.*?\)", " ", c)                    # drop "(kg)" / "(USD)" style brackets
    c = re.sub(r"[^a-z0-9#\s]", " ", c)
    c = re.sub(r"\bnw\b", "n w", c)
    return re.sub(r"\s+", " ", c).strip()


def _match_column(cell):
    c = _normalise_header(cell)
    if not c:
        return None
    exact = EXACT_COLUMNS.get(c)
    if exact:
        return exact
    for key in _FALLBACK_ORDER:
        for p in ITEM_COLUMNS[key]:
            if re.search(p, c):
                return key
    return None


def _is_hs(value):
    s = re.sub(r"\D", "", str(value or ""))
    return 4 <= len(s) <= 10


def map_row(cells, keys, line_no=None):
    """Map one table row onto the item roles. Returns a dict (possibly with None values) or None."""
    cells = [c for c in cells if c is not None]
    if not any(str(c).strip() for c in cells):
        return None
    filled = {}
    # 1. exact positional match (numbers are normalised here)
    if keys and len(cells) == len(keys):
        for k, c in zip(keys, cells):
            cell = str(c).strip()
            if k in NUMERIC_KEYS:
                v = parse_number(cell)
                if v is None:
                    return None
                filled[k] = as_int(v) if (k == "qty" and float(v).is_integer()) else float(v)
            else:
                filled[k] = cell
        return filled
    # 2. content driven mapping
    for pos, raw in enumerate(cells):
        cell = str(raw).strip()
        if not cell:
            continue
        low = cell.lower()
        if "total" in low and not any(ch.isdigit() for ch in cell) and pos < len(cells) - 1:
            continue                                   # "Total" caption row
        if "index" in keys and "index" not in filled and line_no is not None and cell.isdigit() \
                and int(cell) == line_no:
            filled["index"] = cell
            continue
        if "index" in keys and "index" not in filled and cell.isdigit() and line_no is not None \
                and int(cell) <= 3 and len(cells) >= 4:
            filled["index"] = cell
            continue
        num = parse_number(cell) if re.fullmatch(r"[\s$€£¥\u20b9,.\d()\-%]+", cell) else None
        if num is None and _is_hs(cell) and "hs_code" in keys and "hs_code" not in filled:
            filled["hs_code"] = re.sub(r"\D", "", cell)
            continue
        if num is not None:
            is_int = re.fullmatch(r"\s*[\$€£¥\u20b9]?\s*-?\(?\d{1,9}\)?\s*%?\s*", cell) and float(num).is_integer()
            for k in (("qty",) if is_int else NUMERIC_KEYS):
                if k in keys and k not in filled:
                    filled[k] = num
                    break
            else:
                for k in NUMERIC_KEYS:
                    if k in keys and k not in filled:
                        filled[k] = num
                        break
            continue
        # text cell
        if "product_id" in keys and "product_id" not in filled and re.fullmatch(r"[A-Za-z0-9._/\-]{1,20}", cell):
            filled["product_id"] = cell
            continue
        if "description" in keys and "description" not in filled:
            filled["description"] = cell
            continue
    return filled or None


def parse_items(lines, doc_type):
    """Find the item table(s) in a document and return a list of raw item dicts."""
    want_weight = doc_type == "packing_list"
    items = []
    header_keys, header_idx, n_seen = None, None, 0
    for i, line in enumerate(lines):
        cells = split_cells(line)
        if len(cells) < 3:
            continue
        roles = [_match_column(c) for c in cells]
        keys = [r for r in roles if r]
        if "qty" in keys and len(keys) >= 3 and (want_weight or "unit_price" in keys or "total" in keys):
            header_keys, header_idx = keys, i
            continue
        if header_keys is None:
            continue
        if i <= header_idx:
            continue
        cells2 = split_cells(line)
        n_seen += 1
        if len(cells2) < 2:
            continue
        row = map_row(cells2, header_keys, line_no=n_seen)
        if not row:
            continue
        if row.get("qty") is None and not row.get("description"):
            continue                                    # totals / subtotal / footer rows
        item = normalise_item(row, doc_type)
        if item:
            items.append(item)
        if len(items) > 500:
            break
    return items


def normalise_item(row, doc_type):
    item = {}
    pid = clean_value(row.get("product_id"))
    if pid and not re.fullmatch(r"[\d.]+", str(pid)) or (pid and len(str(pid)) <= 20 and not pid.isdigit()):
        item["product_id"] = str(pid)
    elif pid and str(pid).isdigit():
        item["product_id"] = str(int(str(pid)))          # keep it a STRING: scoring.py looks products up by string
    hs = clean_value(row.get("hs_code"))
    if hs:
        digits = re.sub(r"\D", "", hs)
        item["hs_code"] = digits if digits else hs
    desc = clean_value(row.get("description"))
    if desc:
        item["description"] = desc
    qty = row.get("qty")
    if isinstance(qty, (int, float)):
        item["qty"] = as_int(qty) if float(qty).is_integer() else qty
    up, tot = row.get("unit_price"), row.get("total")
    if isinstance(up, (int, float)):
        item["unit_price"] = round(float(up), 4)
    if isinstance(tot, (int, float)):
        item["total"] = round(float(tot), 4)
    nw = row.get("net_weight_kg")
    if isinstance(nw, (int, float)):
        item["net_weight_kg"] = round(float(nw), 4)
    if doc_type == "packing_list":
        return item if item.get("qty") is not None else None
    return item if (item.get("qty") is not None and (item.get("total") is not None or
                                                   item.get("unit_price") is not None)) else None


def split_cells(line):
    """Split a rendered table row into cells: tab > pipe > semicolon > 2+ spaces > single spaces."""
    s = line.rstrip()
    if not s.strip():
        return []
    for sep in ("\t", "|", ";"):
        if sep in s:
            cells = [c.strip() for c in s.split(sep)]
            if sum(1 for c in cells if c) >= 2:
                return cells
    if "  " in s:                                     # space aligned columns: runs of 2+ spaces are the gaps
        cells = [c for c in re.split(r"\s{2,}", s.strip()) if c]
        if len(cells) >= 2:
            return cells
    return s.strip().split()