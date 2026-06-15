# ArtefyCraft OCR Backend

FastAPI HTTP service for OCR document processing. Called from n8n workflows.

**Endpoints:**
- `GET  /` — service info
- `GET  /health` — health check
- `POST /process-master` — classify + extract single document
- Other family-specific endpoints (telecom, digital invoice, scanned invoice, receipt image)

**Tech:** Python 3.11 + FastAPI + Tesseract OCR (with Latvian lang pack) + Poppler + pdfplumber.

**Deploy:** Railway auto-detects Dockerfile. Set env vars: `MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.

This service is consumed only by the ArtefyCraft multi-tenant n8n on Railway. Independent from the Woodenlays deployment.
