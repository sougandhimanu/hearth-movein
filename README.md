# Hearth Move-In Guardian

Two agents for renters, built for the Meta Global AI Developer Hackathon:

1. **Lease checker** (Muse Spark): reads a lease (PDF, photo or text), extracts terms with exact quotes, flags risky clauses, explains them in the renter's language, sets reminders and drafts a message to the landlord.
   A flag is kept only if its quote is found verbatim in the lease text (checked in code, not by the model).
2. **Move-in condition report** (Muse Spark + SAM 3.1): finds damage in room photos, outlines it with SAM 3.1, has Spark verify each outline, retries with a simpler noun if SAM returns nothing, and saves a timestamped report. It is linked to the lease's inspection deadline.

Everything is saved in a **Vault** (SQLite + files) with the **agent chats** (what each agent did) and delete buttons.
Not legal advice: flags are questions to ask the landlord.

## Run
```
pip install -r requirements.txt
cp .env.example .env        # add MODEL_API_KEY (never commit it)
uvicorn app.main:app --reload
```
Open http://localhost:8000. Without a key the app runs in **MOCK MODE** (clearly bannered) so you can try the flow offline.
Samples: `python scripts/make_samples.py`, then upload `samples/sample_lease.txt` (or the PDF) and the `room_*.jpg` photos.
Tests: `pytest` (mock mode, 9 tests).

## Architecture
`app/lease_agent.py`, `app/condition_agent.py`: tool-calling loops over Chat Completions (text only). Vision and SAM calls are wrapped as tools, so images and tool calling are never mixed in one request.
`app/meta.py`: Meta API client (base URL `https://api.meta.ai/v1`, `MODEL_API_KEY`). `app/mock.py`: offline model. `app/db.py`: Vault.

## Not verified against the live API (test these first)
- Responses API shapes: `text.format` json_schema, `input_image` / `input_file` parts (assumed OpenAI-compatible).
- SAM 3.1 with defect words ("stain", "crack"), and the exact box record format (parser accepts positional and key=value).
- SAM masks are not decoded, so only boxes are drawn (the docs mention a `meta-sam-parser` package for masks).
- Accuracy on messy lease photos and non-English leases.

## Hackathon notes
Public repo, new work only, synthetic data only (the sample lease and photos are invented). Use the standard model, not "contributor" variants, for personal documents.
