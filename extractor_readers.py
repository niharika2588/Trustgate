"""
File readers for TrustGate document extraction.

Every optional dependency is imported lazily and inside try/except: a missing library or a missing
external binary (Tesseract, LibreOffice) downgrades the result to a clear warning, never a crash.
Uploaded bytes are only ever written to a private temp file and parsed - never imported or executed.
"""
import csv
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

TAB = "\t"          # internal separator used to render table rows into text lines

# ------------------------------------------------------------------ optional dependencies
_OPT = {}


def _have(mod):
    if mod in _OPT:
        return _OPT[mod]
    try:
        m = __import__(mod)
        _OPT[mod] = m
    except Exception:
        _OPT[mod] = None
    return _OPT[mod]


def has_pypdf():
    return _have("pypdf") is not None


def has_docx():
    return _have("docx") is not None


def has_openpyxl():
    return _have("openpyxl") is not None


def _find_tesseract():
    """Look for the Tesseract binary, including the default install paths on Windows and Linux/s390x."""
    exe = shutil.which("tesseract")
    if exe:
        return exe
    candidates = ["/usr/bin/tesseract", "/usr/local/bin/tesseract", "/snap/bin/tesseract",
                  "/opt/tesseract/bin/tesseract"]
    for env in ("ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            candidates.append(os.path.join(base, "Tesseract-OCR", "tesseract.exe"))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def tesseract_available():
    if "tesseract" in _OPT:
        return _OPT["tesseract"]
    pyt = _have("pytesseract")
    exe = _find_tesseract()
    if pyt is None or _have("PIL") is None or not exe:
        _OPT["tesseract"] = False
        return False
    try:
        pyt.pytesseract.tesseract_cmd = exe          # tesseract is often not on PATH on Windows
        pyt.get_tesseract_version()
        _OPT["tesseract"] = True
    except Exception:
        _OPT["tesseract"] = False
    return _OPT["tesseract"]


def libreoffice_available():
    if "soffice" in _OPT:
        return _OPT["soffice"]
    exe = None
    for name in ("soffice", "libreoffice"):
        exe = shutil.which(name)
        if exe:
            break
    if not exe and os.name == "nt":
        for name in (r"C:\Program Files\LibreOffice\program\soffice.exe",
                     r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"):
            if os.path.exists(name):
                exe = name
                break
    _OPT["soffice"] = exe
    return exe


def rasterize_pdf_available():
    return _have("pypdfium2") is not None or _have("pdf2image") is not None


# ------------------------------------------------------------------ type detection (extension AND magic bytes)
EXT_TYPES = {
    "pdf": "pdf", "docx": "docx", "doc": "legacy_doc", "rtf": "legacy_doc", "odt": "legacy_doc",
    "xlsx": "xlsx", "xlsm": "xlsx", "xls": "legacy_sheet", "ods": "legacy_sheet",
    "csv": "csv", "tsv": "csv", "txt": "txt", "md": "txt", "json": "json", "xml": "xml",
    "png": "image", "jpg": "image", "jpeg": "image", "tif": "image", "tiff": "image", "bmp": "image",
    "gif": "image", "webp": "image",
}
ALLOWED_EXTENSIONS = sorted(EXT_TYPES)


def detect_type(filename, data):
    """Returns (kind, evidence). Magic bytes win over the extension."""
    head = data[:4096]
    ext = os.path.splitext(filename or "")[1].lower().lstrip(".")
    ext_kind = EXT_TYPES.get(ext)
    if data[:5] == b"%PDF-" or b"%PDF-" in head[:1024]:
        return "pdf", "magic:%PDF"
    if data[:2] == b"MZ" or data[:4] == b"\x7fELF" or data[:4] in (b"\xca\xfe\xba\xbe", b"\xfe\xed\xfa\xce"):
        return None, "content:executable binary (not a document)"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image", "magic:PNG"
    if data[:3] == b"\xff\xd8\xff":
        return "image", "magic:JPEG"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image", "magic:GIF"
    if data[:2] in (b"BM",):
        return "image", "magic:BMP"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image", "magic:TIFF"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image", "magic:WEBP"
    if data[:2] == b"PK":
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
            names = set(zf.namelist())
            if "word/document.xml" in names:
                return "docx", "magic:zip(word/document.xml)"
            if any(n.startswith("xl/") for n in names):
                return "xlsx", "magic:zip(xl/)"
            if "mimetype" in names:
                mt = zf.read("mimetype")[:120].decode("utf-8", "ignore")
                if "opendocument.text" in mt:
                    return "legacy_doc", "magic:odt"
                if "opendocument.spreadsheet" in mt:
                    return "legacy_sheet", "magic:ods"
            if ext_kind == "docx":
                return "docx", "ext:.docx (zip without word/document.xml)"
            return "docx", "magic:zip"
        except Exception:
            return None, "magic:PK but not a readable zip"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        if ext_kind == "legacy_sheet":
            return "legacy_sheet", "magic:OLE2 + ext"
        return "legacy_doc", "magic:OLE2"
    if head[:5] == b"{\\rtf":
        return "legacy_doc", "magic:RTF"
    stripped = head.lstrip()[:1]
    if stripped in (b"{", b"["):
        try:
            import json
            json.loads(data.decode("utf-8", "ignore") or "{}")
            return "json", "content:JSON"
        except Exception:
            pass
    if stripped == b"<":
        low = data[:2048].decode("utf-8", "ignore").lower()
        if "<!doctype html" in low or "<html" in low:
            return None, "content:HTML (not a document format)"
        return "xml", "content:XML"
    if ext_kind in ("csv", "tsv"):
        return "csv", "ext:." + ext
    if ext_kind in ("txt", "md"):
        return "txt", "ext:." + ext
    if ext_kind:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                data.decode("cp1252")
                return ext_kind, "ext:." + ext + " (cp1252 text)"
            except Exception:
                return None, "ext:." + ext + " but not decodable text"
        return ext_kind, "ext:." + ext
    return None, "unknown"


# ------------------------------------------------------------------ page object
class Page:
    def __init__(self, index, lines, method, tables=0):
        self.index = index
        self.lines = lines
        self.method = method
        self.tables = tables

    def meta(self, start):
        return {"number": self.index, "method": self.method, "tables": self.tables, "start": start}


class ReadResult:
    def __init__(self, kind, method, pages, warnings, structured=None):
        self.kind = kind
        self.method = method
        self.pages = pages
        self.warnings = warnings
        self.structured = structured       # pre-parsed shipment dict for JSON passthrough


def _blank():
    return []


# ------------------------------------------------------------------ readers
def _decode_text(data):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", "replace")


def read_txt(path, data, max_pages):
    text = _decode_text(data)
    lines = [l.rstrip() for l in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return ReadResult("txt", "text", [Page(1, lines, "text")], [])


def read_json(path, data, max_pages):
    """A JSON file that already matches the schema is passed through untouched."""
    try:
        obj = __import__("json").loads(_decode_text(data))
    except Exception as e:
        return ReadResult("json", "text", [], ["file is not valid JSON (%s)" % type(e).__name__])
    ship = obj
    if isinstance(obj, dict) and isinstance(obj.get("shipment"), dict):
        ship = obj["shipment"]
    if isinstance(ship, dict) and isinstance(ship.get("documents"), dict):
        return ReadResult("json", "json", [], ["file already contains the TrustGate schema - passed through"],
                          structured=ship)
    lines = _decode_text(data).splitlines()
    return ReadResult("json", "text", [Page(1, lines, "text")], [])


def read_xml(path, data, max_pages):
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(data)
    except Exception as e:
        return ReadResult("xml", "text", [], ["file is not valid XML (%s)" % type(e).__name__])
    lines = []

    def walk(node, depth=0):
        tag = re.sub(r"^\{.*\}", "", node.tag)
        txt = (node.text or "").strip()
        tail = (node.tail or "").strip()
        if txt:
            lines.append(("%s: %s" % (tag, txt)) if len(lines) == 0 or True else txt)
        for ch in node:
            walk(ch, depth + 1)
        if tail:
            lines[-1] = (lines[-1] + " " + tail).strip() if lines else tail

    walk(root)
    return ReadResult("xml", "text", [Page(1, lines, "text")], [])


def read_csv(path, data, max_pages):
    text = _decode_text(data)
    sample = text[:4096]
    delim = ","
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        counts = {d: sample.count(d) for d in ",;\t|"}
        delim = max(counts, key=lambda d: counts[d])
        if counts[delim] == 0:
            delim = ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    lines = [TAB.join(c) for c in rows]
    return ReadResult("csv", "table", [Page(1, lines, "table", tables=1)],
                      [] if delim != "," else ["CSV files with commas inside cells should use ; or tab as separator"])


def read_xlsx(path, data, max_pages):
    openpyxl = _have("openpyxl")
    if openpyxl is None:
        return ReadResult("xlsx", "table", [], ["openpyxl is not installed - run: pip install -r requirements-extract.txt"])
    pages, warnings = [], []
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as e:
        return ReadResult("xlsx", "table", [], ["cannot read workbook (%s)" % type(e).__name__])
    for idx, ws in enumerate(wb.worksheets[:max_pages], start=1):
        lines = []
        for row in ws.iter_rows(values_only=True):
            if row is None:
                continue
            cells = ["" if c is None else _xlsx_cell(c) for c in row]
            while cells and cells[-1] == "":
                cells.pop()
            if cells:
                lines.append(TAB.join(cells))
        pages.append(Page(idx, lines, "table", tables=1))
    if len(wb.worksheets) > max_pages:
        warnings.append("workbook has %d sheets, only the first %d were read" % (len(wb.worksheets), max_pages))
    return ReadResult("xlsx", "table", pages, warnings)


def _xlsx_cell(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def read_docx(path, data, max_pages):
    docx = _have("docx")
    if docx is None:
        return ReadResult("docx", "text", [], ["python-docx is not installed - run: pip install -r requirements-extract.txt"])
    import docx as _d
    from docx.document import Document as _Doc
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    try:
        doc = _d.Document(path)
    except Exception as e:
        return ReadResult("docx", "text", [], ["cannot read document (%s)" % type(e).__name__])
    lines, tables, page = [], 0, Page(1, [], "text")
    body = doc.element.body
    for child in body.iterchildren():
        tag = re.sub(r"^\{.*\}", "", child.tag)
        if tag == "p":
            lines.append(Paragraph(child, doc).text.rstrip())
        elif tag == "tbl":
            t = Table(child, doc)
            tables += 1
            for row in t.rows:
                lines.append(TAB.join(c.text.strip().replace("\n", " ") for c in row.cells))
    page.lines, page.tables = lines, tables
    return ReadResult("docx", "table" if tables else "text", [page], [])


def read_pdf(path, data, max_pages):
    pypdf = _have("pypdf")
    if pypdf is None:
        return ReadResult("pdf", "text", [], ["pypdf is not installed - run: pip install -r requirements-extract.txt"])
    warnings, pages = [], []
    try:
        reader = pypdf.PdfReader(path)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return ReadResult("pdf", "text", [], ["PDF is password protected"])
        n = len(reader.pages)
    except Exception as e:
        return ReadResult("pdf", "text", [], ["cannot read PDF (%s: %s)" % (type(e).__name__, e)])
    if n > max_pages:
        warnings.append("PDF has %d pages, only the first %d were read" % (n, max_pages))
    ocr_ok = tesseract_available()
    for i, page in enumerate(reader.pages[:max_pages], start=1):
        try:
            txt = page.extract_text() or ""
        except Exception as e:
            txt = ""
            warnings.append("page %d: text extraction failed (%s)" % (i, type(e).__name__))
        lines = [l.rstrip() for l in txt.replace("\r\n", "\n").split("\n")]
        method = "text"
        if not "".join(lines).strip():
            if ocr_ok and rasterize_pdf_available():
                txt = _ocr_pdf_page(path, i)
                lines = [l.rstrip() for l in txt.split("\n")]
                method = "ocr"
                if "".join(lines).strip():
                    warnings.append("page %d has no text layer - OCR was used" % i)
                else:
                    warnings.append("page %d looks like a scan but OCR produced no text "
                                    "(low contrast, handwriting or rotated page?)" % i)
            else:
                warnings.append("page %d has no text layer (scanned image). Install OCR to read it: "
                                "pip install pytesseract pillow pypdfium2 + the Tesseract binary" % i)
        pages.append(Page(i, lines, method))
    return ReadResult("pdf", "ocr" if any(p.method == "ocr" for p in pages) else "text", pages, warnings)


def _ocr_pdf_page(pdf_path, number):
    """Rasterise one page of the PDF and read it with Tesseract. Returns '' when nothing can be read."""
    pyt = _have("pytesseract")
    try:
        pypdfium2 = _have("pypdfium2")
        if pypdfium2 is not None:
            doc = pypdfium2.PdfDocument(pdf_path)
            return pyt.image_to_string(doc[number - 1].render(scale=3).to_pil())
    except Exception:
        pass
    try:
        pdf2image = _have("pdf2image")
        if pdf2image is not None:
            img = pdf2image.convert_from_path(pdf_path, first_page=number, last_page=number, dpi=300)[0]
            return pyt.image_to_string(img)
    except Exception:
        pass
    return ""


def read_image(path, data, max_pages):
    from PIL import Image as PILImage
    if _have("PIL") is None:
        return ReadResult("image", "text", [], ["Pillow is not installed - run: pip install -r requirements-extract.txt"])
    try:
        img = PILImage.open(path)
        img = img.convert("L") if img.mode not in ("L", "RGB") else img
    except Exception as e:
        return ReadResult("image", "text", [], ["cannot read image (%s)" % type(e).__name__])
    if not tesseract_available():
        return ReadResult("image", "text", [],
                          ["no OCR available: install the Tesseract binary "
                           "(Windows: winget install UB-Mannheim.TesseractOCR / Ubuntu: apt install tesseract-ocr) "
                           "and run: pip install pytesseract pillow"])
    try:
        txt = _have("pytesseract").image_to_string(img)
    except Exception as e:
        return ReadResult("image", "text", [], ["OCR failed (%s)" % type(e).__name__])
    lines = [l.rstrip() for l in txt.replace("\r\n", "\n").split("\n")]
    if not "".join(lines).strip():
        return ReadResult("image", "ocr", [], ["OCR found no text in this image"])
    return ReadResult("image", "ocr", [Page(1, lines, "ocr")], [])


def rtf_to_text(path):
    """Very small, dependency-free RTF text extractor (enough for label: value documents)."""
    raw = open(path, "rb").read().decode("latin-1", "replace")
    raw = re.sub(r"\{\\\*\\[a-zA-Z]+-?[0-9]*[^{}]*\}", "", raw)      # font tables, destinations
    raw = re.sub(r"\\'[0-9a-fA-F]{2}", "", raw)
    raw = raw.replace("\\par", "\n").replace("\\line", "\n").replace("\\tab", "\t")
    raw = re.sub(r"\\[a-zA-Z]+-?[0-9]* ?", "", raw)
    raw = raw.replace("{", "").replace("}", "")
    return raw


def read_legacy(path, data, kind, max_pages):
    """Legacy .doc/.rtf/.odt (and .xls/.ods). RTF is parsed in-process; the rest need LibreOffice."""
    if os.path.splitext(path)[1].lower() == ".rtf":
        txt = rtf_to_text(path)
        lines = [l.rstrip() for l in txt.replace("\r\n", "\n").split("\n")]
        return ReadResult("legacy_doc", "text", [Page(1, lines, "text")], ["RTF parsed with the built-in reader"])
    soffice = libreoffice_available()
    if not soffice:
        return ReadResult(kind, "text", [],
                          ["%s files need LibreOffice, which is not installed. Install it "
                           "(Windows: https://www.libreoffice.org/download ; Ubuntu/s390x: apt install libreoffice-writer) "
                           "or convert the file to .docx / .pdf / .xlsx and upload that."
                           % os.path.splitext(path)[1].lstrip(".").upper()])
    target = "csv" if kind == "legacy_sheet" else "docx"
    outdir = tempfile.mkdtemp(prefix="trustgate_lo_")
    try:
        proc = subprocess.run([soffice, "--headless", "--norestore", "--convert-to", target, "--outdir", outdir, path],
                              capture_output=True, timeout=120)
        produced = [os.path.join(outdir, f) for f in os.listdir(outdir)]
        if not produced:
            return ReadResult(kind, "text", [], ["LibreOffice could not convert this file (%s)"
                                                 % (proc.stderr or b"").decode("utf-8", "ignore")[:120].strip()])
        data2 = open(produced[0], "rb").read()
        r = read_docx(produced[0], data2, max_pages) if target == "docx" else read_csv(produced[0], data2, max_pages)
        r.warnings.insert(0, "legacy file converted with LibreOffice")
        return r
    except Exception as e:
        return ReadResult(kind, "text", [], ["LibreOffice conversion failed (%s: %s)" % (type(e).__name__, e)])
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


# ------------------------------------------------------------------ dispatcher
def read_file(path, filename, kind, max_pages=50):
    with open(path, "rb") as fh:
        data = fh.read()
    if kind == "txt":
        return read_txt(path, data, max_pages)
    if kind == "json":
        return read_json(path, data, max_pages)
    if kind == "xml":
        return read_xml(path, data, max_pages)
    if kind == "csv":
        return read_csv(path, data, max_pages)
    if kind == "xlsx":
        return read_xlsx(path, data, max_pages)
    if kind == "docx":
        return read_docx(path, data, max_pages)
    if kind == "pdf":
        return read_pdf(path, data, max_pages)
    if kind == "image":
        return read_image(path, data, max_pages)
    if kind in ("legacy_doc", "legacy_sheet"):
        return read_legacy(path, data, kind, max_pages)
    return ReadResult(kind, "text", [], ["no reader for this format"])


def capabilities():
    return {
        "pdf": has_pypdf(),
        "docx": has_docx(),
        "xlsx": has_openpyxl(),
        "xls": libreoffice_available() is not None,
        "csv": True,
        "txt": True,
        "json": True,
        "xml": True,
        "images": _have("PIL") is not None and tesseract_available(),
        "image_read": _have("PIL") is not None,
        "ocr": tesseract_available(),
        "ocr_pdf": tesseract_available() and rasterize_pdf_available(),
        "legacy_doc": libreoffice_available() is not None,
        "rtf": True,
    }