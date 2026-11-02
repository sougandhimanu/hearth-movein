# Hearth Move-In Guardian

Helps renters understand a lease before they sign and document a room's condition before they move in, so they can protect their deposit.
Built with Muse Spark and SAM 3.1 for the Meta Global AI Developer Hackathon.

> Not legal advice. Flags are questions to ask the landlord, each backed by an exact quote from the lease.

## What it does

### 1. Lease checker (Muse Spark)
Upload a lease as a PDF, photo or `.txt` file and choose the language you want explanations in. The agent:
- extracts the key terms (rent, deposit, dates, fees, renewal, entry, repairs), each with an exact quote;
- flags risky or unusual clauses, explains them in your language and suggests a question to ask;
- saves reminders for deadlines such as rent due dates, the move-in inspection window and the renewal notice;
- drafts a polite email to the landlord about the flagged items.

A flag is kept only if its quote is found word for word in the lease. That check runs in code, not in the model.

### 2. Move-in condition report (Muse Spark + SAM 3.1)
Upload photos of each room, optionally linked to a lease. The agent:
- scans each photo for visible damage (stains, cracks, scratches, peeling paint, mold...);
- outlines each defect with SAM 3.1, retrying with a simpler word if nothing is found;
- has Muse Spark check each outline and keeps only the ones it confirms;
- saves a timestamped report with outlined photos and the lease's inspection deadline.

### Vault
Leases, reports, reminders and the full agent chats (every tool call each agent made) are stored locally in SQLite. Anything can be deleted.

## Quick start
```
pip install -r requirements.txt
echo MODEL_API_KEY=your-key > .env
uvicorn app.main:app --reload
```
Open http://localhost:8000.

Without a key the app runs in **mock mode** with fake model responses (shown by a banner), so you can try the whole flow offline.

To try it with sample data, upload `samples/sample_lease.txt` or `samples/sample_lease.pdf`, then the `samples/room_*.jpg` photos. To regenerate the samples, run `python scripts/make_samples.py`.

### Settings
Set these in `.env`. Only `MODEL_API_KEY` is required.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_API_KEY` | none | Meta Model API key. Never commit it. |
| `SPARK_MODEL` | `muse-spark-1.3` | Model for reasoning and vision. |
| `SAM_MODEL` | `sam-3.1` | Model for outlining damage. |
| `META_BASE_URL` | `https://api.meta.ai/v1` | API base URL. |
| `MOCK_MODE` | off | Set to `1` to force offline mode. |
| `DATA_DIR` | `./data` | Where the Vault is stored. |

## How it works
```
app/
  main.py             FastAPI routes and Vault endpoints
  lease_agent.py      lease checker: term extraction, then a tool-calling agent loop
  condition_agent.py  condition report: scan, outline, check, record
  meta.py             Meta API client (Chat Completions, Responses API, SAM 3.1)
  mock.py             offline stand-in for the API
  overlay.py          draws outline boxes on photos
  db.py               Vault (SQLite and stored files)
static/index.html     single-page UI
```

Both agents are tool-calling loops over Chat Completions. The model chooses which tool to call, and the code runs it and enforces the rules:
- **Lease:** `flag_clause` rejects any quote not found in the lease text; `set_reminder` stores a date only if it is a valid `YYYY-MM-DD`.
- **Condition:** `record_finding` attaches an outline only if `check_outline` confirmed it.

Image analysis (Muse Spark vision and SAM 3.1) runs inside the tools, so images and tool calling are never mixed in one request.

## Tests
```
pytest
```
Runs offline in mock mode.

## Data and privacy
Everything stays on your machine in `data/`, which is git-ignored. The sample lease and photos are invented. For personal documents, use the standard model, not the "contributor" variants.
