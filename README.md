# TrustGate – deploy on IBM LinuxONE

## 1. Copy and run
    scp trustgate_app.zip <user>@<linuxone-host>:~
    ssh <user>@<linuxone-host>
    unzip trustgate_app.zip && cd trustgate_app
    uname -m            # s390x = LinuxONE
    python3 --version   # needs 3.9+  (sudo dnf/apt install python3 python3-venv if missing)
    ./run.sh 8000

Open http://<host>:8000  (or from your laptop: `ssh -L 8000:localhost:8000 <user>@<host>` then http://localhost:8000).
If the port is blocked by the firewall, use the SSH tunnel.

## 2. How the scoring works without xgboost
`scoring.py` reads `model/trustgate_xgb.json` with a pure-Python tree walker, so scoring needs no numpy/xgboost
(no s390x wheels needed). If `pip install xgboost` works, it is used only for SHAP-based reasons.

## 3. If FastAPI/pydantic fails to install on s390x
`pip install` of pydantic-core may try to compile Rust. Fix: `sudo dnf install gcc rust cargo` (or apt equivalent) and rerun,
or ask the organisers for a Python/Docker image. Last resort: wrap the same functions in Flask (4 routes).

## 4. Use your own Colab model
Replace `model/trustgate_xgb.json` and `model/model_meta.json` with the files exported from the notebook
(feature order must match). Restart.

## 5. Ledger
`ledger.py` is a SIMULATED hash-chained SQLite ledger (only SHA-256 + trust score recorded; documents stay in the
off-chain table). To use IBM Blockchain / Hyperledger Fabric later, write a class with the same methods
(`record`, `find`, `entries`, `verify_chain`) that calls chaincode `RecordShipment(shipmentId, hash, score)`,
and instantiate it in `app.py` instead of `LocalLedger`.

## 6. Demo flow
Pick a sample -> Validate. Approve (>=80) writes hash+score to the ledger; Review (50-79) goes to the human queue;
Reject (<50) writes nothing. "Tamper check" shows a changed document no longer matches the ledger hash.
Env `TRUSTGATE_DB` sets the DB path; "Reset demo" clears it.

## 7. Document upload (optional)

Instead of pasting JSON by hand you can upload the real documents (PDF, Word, Excel, CSV, TXT, images).
The server reads them, fills the JSON box, and you press **Validate & submit** as usual.
Extraction never validates and never writes to the ledger - it only produces
`{shipment, confidence, missing, warnings, files}`.

### Install

The base app still installs and runs with `requirements.txt` only; upload then answers `503`.
To enable it (all pure-Python wheels, so this also works on s390x / IBM LinuxONE):

    pip install -r requirements-extract.txt

`requirements-extract.txt` installs: `python-multipart` (required), `pypdf` (PDF text layer),
`python-docx` (.docx), `openpyxl` (.xlsx/.xlsm). CSV, TSV, TXT, MD, JSON and XML need nothing.

### Optional OCR (scanned PDFs and images)

    pip install pytesseract pillow pypdfium2        # or pdf2image + poppler-utils
    # Tesseract binary:
    winget install UB-Mannheim.TesseractOCR         # Windows
    sudo apt install tesseract-ocr                   # Ubuntu / LinuxONE
    export PATH=$PATH:/usr/local/bin

Without it, a scanned page returns the warning "page N has no text layer (scanned image) ..." instead of failing.

### Optional legacy .doc / .xls / .odt

`.rtf` is parsed by a built-in reader. `.doc`, `.xls` and `.ods` need LibreOffice:

    # Windows: https://www.libreoffice.org/download/  (soffice.exe is found automatically)
    sudo apt install libreoffice-writer libreoffice-calc      # Ubuntu / LinuxONE

Without it the upload is refused with "convert the file to .docx / .pdf / .xlsx and upload that".

### Commands

    curl -F "files=@invoice.pdf" -F "files=@po.docx" -F "files=@pl.xlsx" \
         -F "files=@bol.pdf"  http://localhost:8000/api/extract
    curl http://localhost:8000/api/extract/capabilities

Limits per request: 10 files, 20 MB per file, 50 pages per PDF, extension allow-list.
Errors are JSON `{"detail": "..."}`: `413` too large / too many, `415` unsupported type, `422` unreadable.

### Optional LLM fallback

If the environment variable `ANTHROPIC_API_KEY` is set, fields the rules could not read are offered to
Claude (`pip install anthropic`) and clearly reported with confidence `0.5` and a warning.
It never overwrites a value a rule already read, and nothing is ever sent anywhere when the variable is unset.

### Tests

    pip install httpx pytest reportlab python-docx openpyxl pillow
    python tests/test_extract.py          # prints the accuracy tables, or: pytest tests/test_extract.py -v

The extractor is rule based and deterministic: it never invents a value. Unreadable fields stay `null`
and are listed in `missing`; mismatches between documents are passed through exactly as printed so the
model can detect them.
