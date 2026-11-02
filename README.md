# Hearth Move-In Guardian

Helps renters keep their deposit. Understand your lease before you sign, document your room when you move in, and prove what was already there when you move out.
Built with Muse Spark and SAM 3.1 for the Meta Global AI Developer Hackathon.

> Not legal advice. Flags are questions to ask the landlord, each backed by an exact quote from the lease.

## What it does

### 1. Lease checker
Upload a lease (PDF, photo or `.txt`) and choose your language. The app:
- pulls out the key terms: rent, deposit, dates, fees, renewal, entry, repairs;
- flags risky or unusual clauses, explains them in your language and suggests what to ask;
- sets reminders for deadlines like rent, the move-in inspection window and the renewal notice;
- drafts a polite email to the landlord.

Every flag quotes the lease word for word, and the app checks that quote against the lease itself.

### 2. Move-in condition report
Upload a photo of each room. The app finds visible damage (stains, cracks, scratches, peeling paint, mold...), outlines it on the photo, double-checks each outline, and saves a timestamped report linked to your lease's inspection deadline.

### 3. Move-out check
Upload move-out photos taken from the same spots and match them to your move-in report. The app compares each pair and marks every piece of damage as **already there at move-in**, **worse**, **new** or **unclear**. It then drafts an email asking the landlord not to deduct for anything that was already there. Damage is only marked "already there" when the side-by-side comparison confirms it.

### Works in your language
Explanations, reminders and reports can be written in the language you choose (for example Hindi, Telugu, Tamil, Spanish or Japanese), and leases written in those languages can be checked too.

### Vault
Everything you check is saved on your computer, never in this project folder: leases, reports, reminders and a log of what each agent did. You can open or delete anything.

## Setup
You need **Python 3.10 or newer** (tested on 3.12).

```
git clone https://github.com/sougandhimanu/hearth-movein.git
cd hearth-movein
python -m venv .venv
```
On macOS / Linux, use `python3` wherever this guide says `python` if `python` is not found.

Activate the environment:
- Windows (Command Prompt): `.venv\Scripts\activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`. If it says running scripts is disabled, first run `Set-ExecutionPolicy -Scope Process Bypass`, which only affects this window.
- macOS / Linux: `source .venv/bin/activate`

Install:
```
python -m pip install -r requirements.txt
```

Then choose one of the two ways to run it.

## Run with the real API
Uses Muse Spark and SAM 3.1, so you need a Meta Model API key from [dev.meta.ai](https://dev.meta.ai).

1. Copy `.env.example` to `.env`:
   - Windows: `copy .env.example .env`
   - macOS / Linux: `cp .env.example .env`
2. Open `.env` in a text editor and replace `your-meta-model-api-key` with your key:
   ```
   MODEL_API_KEY=your-key
   ```
3. Start the app:
   ```
   python -m uvicorn app.main:app
   ```
4. Open http://localhost:8000. If port 8000 is already in use, add `--port 8001` and open http://localhost:8001 instead. No mock-mode banner at the top means your key was found.

Each check takes 1-2 minutes while the agents work. `.env.example` also lists optional settings, such as `SPARK_MODEL` and `SAM_MODEL` to choose models.

## Run in mock mode (no API key)
Runs offline with a fake model, to try the app without a key. Results are canned (the same damage and clauses every time), and a banner at the top says mock mode is on.

1. Skip the `.env` file, or leave the placeholder key in it unchanged.
2. Start the app:
   ```
   python -m uvicorn app.main:app
   ```
3. Open http://localhost:8000. Each check finishes in a second.

To force mock mode even with a key in `.env`, add the line `MOCK_MODE=1` to it.

## Try it with the sample files
Use the files in the `samples` folder, in either mode:
1. **Lease checker:** upload `sample_lease.pdf`. Optionally type another language, such as Hindi or Telugu.
2. **Move-in report:** upload `room_living.jpg` and `room_bedroom.jpg`, and link the lease.
3. **Move-out check:** choose that move-in report, upload the same two photos, and keep the suggested matches. Because nothing changed, every defect is marked **already there at move-in**, and an email to the landlord is drafted.

## How it works
Each feature is an agent: the model decides which tool to use next, and the code runs the tool and enforces the rules, so the model cannot skip a check.
Muse Spark reads leases and photos; SAM 3.1 outlines the damage.

```
app/
  main.py             web server and Vault endpoints
  lease_agent.py      lease checker
  condition_agent.py  move-in reports and move-out checks
  meta.py             Meta API client (Muse Spark, SAM 3.1)
  mock.py             offline stand-in for the API
  db.py               Vault storage
static/index.html     the web page
```

## Tests
```
python -m pytest
```
Runs offline in mock mode; no API key needed.

The sample lease and photos are invented. For personal documents, use the standard model, not the "contributor" variants.
