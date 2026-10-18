"""
Test helpers: build REAL document files (PDF / DOCX / XLSX / CSV / TXT / PNG) from a sample shipment.

Development-only dependencies (reportlab, python-docx, openpyxl, Pillow) are used here and never by
the application itself.
"""
import io
import os

SEP = "\t"          # what the extractor uses internally; rendered as "|" or aligned spaces in text formats


def _num(v):
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _money(v, grouped=True):
    if v is None:
        return None
    s = "%.2f" % float(v)
    if grouped:
        whole, _, dec = s.partition(".")
        neg = whole.startswith("-")
        whole = whole.lstrip("-")
        whole = "{:,}".format(int(whole))
        s = ("-" if neg else "") + whole + ("." + dec if dec else "")
    return s


def doc_blocks(ship, doc_type):
    """-> (title, [(label, value)], [column headers], [item rows as strings])"""
    d = ship["documents"][doc_type]
    title = {"invoice": "COMMERCIAL INVOICE", "purchase_order": "PURCHASE ORDER",
             "packing_list": "PACKING LIST", "bill_of_lading": "BILL OF LADING"}[doc_type]
    labels, cols, rows = [], [], []

    if doc_type == "invoice":
        labels = [("Seller", d.get("seller")), ("Buyer", d.get("buyer")),
                  ("Invoice No.", d.get("doc_number")), ("Invoice Date", d.get("date")),
                  ("PO Number", d.get("po_number")), ("Shipment No", d.get("shipment_id")),
                  ("Currency", d.get("currency")), ("Incoterm", d.get("incoterm"))]
        cols = ["Product Code", "HS Code", "Description", "Qty", "Unit Price", "Amount"]
        for it in d.get("items") or []:
            rows.append([it.get("product_id"), it.get("hs_code"), it.get("description"),
                         _num(it.get("qty")), _money(it.get("unit_price")), _money(it.get("total"))])
        totals = [("Total Amount (%s)" % d.get("currency"), _money(d.get("total_amount")))]
    elif doc_type == "purchase_order":
        labels = [("Seller", d.get("seller")), ("Buyer", d.get("buyer")),
                  ("Purchase Order No.", d.get("doc_number")), ("Date", d.get("date")),
                  ("Shipment No", d.get("shipment_id")), ("Currency", d.get("currency")),
                  ("Incoterm", d.get("incoterm"))]
        cols = ["Product Code", "HS Code", "Description", "Qty", "Unit Price", "Amount"]
        for it in d.get("items") or []:
            rows.append([it.get("product_id"), it.get("hs_code"), it.get("description"),
                         _num(it.get("qty")), _money(it.get("unit_price")), _money(it.get("total"))])
        totals = [("Total Amount (%s)" % d.get("currency"), _money(d.get("total_amount")))]
    elif doc_type == "packing_list":
        labels = [("Packing List No.", d.get("doc_number")), ("Date", d.get("date")),
                  ("Invoice No.", d.get("invoice_number")), ("Shipment No", d.get("shipment_id")),
                  ("Total Packages", _num(d.get("total_packages"))),
                  ("Gross Weight (KG)", _num(d.get("gross_weight_kg")))]
        cols = ["Product Code", "HS Code", "Description", "Qty", "Net Weight (KG)"]
        for it in d.get("items") or []:
            rows.append([it.get("product_id"), it.get("hs_code"), it.get("description"),
                         _num(it.get("qty")), _num(it.get("net_weight_kg"))])
        totals = []
    else:
        labels = [("B/L No.", d.get("doc_number")), ("Invoice No.", d.get("invoice_number")),
                  ("Shipper", d.get("shipper")), ("Consignee", d.get("consignee")),
                  ("Port of Loading", d.get("port_of_loading")),
                  ("Port of Discharge", d.get("port_of_discharge")),
                  ("Date Shipped on Board", d.get("ship_on_board_date")),
                  ("Estimated Time of Arrival", d.get("delivery_eta")),
                  ("Vessel", d.get("vessel")), ("Container No.", d.get("container_no")),
                  ("Total Packages", _num(d.get("total_packages"))),
                  ("Gross Weight (KG)", _num(d.get("gross_weight_kg"))),
                  ("Shipment No", d.get("shipment_id"))]
        totals = []
    labels = [(k, v) for k, v in labels if v not in (None, "")]
    totals = [(k, v) for k, v in totals if v not in (None, "")]
    return title, labels, cols, rows, totals


def text_lines(ship, doc_type, sep="|", pad=False):
    """Render a document as plain lines (used for TXT / CSV and inside PDF)."""
    title, labels, cols, rows, totals = doc_blocks(ship, doc_type)
    out = [title, ""]
    for k, v in labels:
        out.append("%s: %s" % (k, v))
    out.append("")
    if cols:
        if pad:
            widths = [max([len(str(c))] + [len(str(r[i])) for r in rows]) for i, c in enumerate(cols)]
            out.append("  ".join(str(c).ljust(w) for c, w in zip(cols, widths)))
            for r in rows:
                out.append("  ".join(str(x).ljust(w) for x, w in zip(r, widths)))
        else:
            out.append(sep.join(cols))
            for r in rows:
                out.append(sep.join(str(x) for x in r))
        out.append("")
    for k, v in totals:
        out.append("%s: %s" % (k, v))
    return out


# ------------------------------------------------------------------ PDF (text layer)
def make_pdf(ship, doc_type, pad=False, overrides=None):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 790
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, text_lines(ship, doc_type, pad=pad)[0])
    y -= 24
    c.setFont("Courier", 9)
    for line in text_lines(ship, doc_type, pad=pad)[1:]:
        if overrides and line in overrides:
            line = overrides[line]
        c.drawString(45, y, line)
        y -= 13
        if y < 60:
            c.showPage()
            c.setFont("Courier", 9)
            y = 790
    c.showPage()
    c.save()
    return buf.getvalue()


# ------------------------------------------------------------------ DOCX
def make_docx(ship, doc_type):
    import docx
    buf = io.BytesIO()
    d = docx.Document()
    title, labels, cols, rows, totals = doc_blocks(ship, doc_type)
    d.add_heading(title, level=1)
    for k, v in labels:
        d.add_paragraph("%s: %s" % (k, v))
    if cols:
        t = d.add_table(rows=1, cols=len(cols))
        t.style = "Table Grid"
        for c, name in zip(t.rows[0].cells, cols):
            c.text = str(name)
        for r in rows:
            cells = t.add_row().cells
            for c, x in zip(cells, r):
                c.text = str(x)
        d.add_paragraph("")
    for k, v in totals:
        d.add_paragraph("%s: %s" % (k, v))
    d.save(buf)
    return buf.getvalue()


# ------------------------------------------------------------------ XLSX (two sheets = two documents)
def make_xlsx(ship, doc_types):
    import openpyxl
    wb = openpyxl.Workbook()
    first = True
    for doc_type in doc_types:
        title, labels, cols, rows, totals = doc_blocks(ship, doc_type)
        ws = wb.active if first else wb.create_sheet()
        ws.title = doc_type
        first = False
        ws.append([title])
        for k, v in labels:
            ws.append([k, v])
        ws.append([])
        if cols:
            ws.append(cols)
            num_idx = {i for i, c in enumerate(cols)
                       if str(c).split(" (")[0] in ("Qty", "Unit Price", "Amount", "Net Weight", "Total",
                                                    "Total Packages", "Gross Weight")}
            for r in rows:
                ws.append([None if x is None else (float(str(x).replace(",", "")) if i in num_idx and re_num(x)
                                                   else str(x)) for i, x in enumerate(r)])
            ws.append([])
        for k, v in totals:
            ws.append([k, v])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def re_num(x):
    try:
        float(str(x))
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ CSV
def make_csv(ship, doc_type):
    title, labels, cols, rows, totals = doc_blocks(ship, doc_type)
    lines = [[title]]
    for k, v in labels:
        lines.append([k, v])
    if cols:
        lines.append([])
        lines.append(list(cols))
        lines.extend([[None if x is None else x for x in r] for r in rows])
        lines.append([])
    for k, v in totals:
        lines.append([k, v])
    out = []
    for r in lines:
        out.append(",".join('"' + str(c) + '"' if c is not None else "" for c in r))
    return ("\n".join(out) + "\n").encode("utf-8")


# ------------------------------------------------------------------ TXT / PNG
def make_txt(ship, doc_type):
    return ("\n".join(text_lines(ship, doc_type)) + "\n").encode("utf-8")


def make_png(ship, doc_type, width=1400):
    """A 'scanned' page: plain image, no text layer. width=1654 gives A4 at 200 dpi."""
    from PIL import Image, ImageDraw, ImageFont
    lines = text_lines(ship, doc_type)
    size = max(16, round(width / 55.0))
    lh = round(size * 1.35)
    img = Image.new("L", (width, 80 + lh * len(lines)), 255)
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        dr.text((40, 40 + lh * i), line, fill=0, font=font)
    img.info["dpi"] = (200, 200)           # reportlab keeps the resolution when embedding into a PDF
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


WRITERS = {"pdf": make_pdf, "docx": make_docx, "xlsx": make_xlsx, "csv": make_csv, "txt": make_txt, "png": make_png}