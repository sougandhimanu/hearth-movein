"""Settings, read from the environment (a .env file is loaded if present)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))

BASE_URL = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
API_KEY = os.getenv("MODEL_API_KEY", "")
SPARK_MODEL = os.getenv("SPARK_MODEL", "muse-spark-1.3")
SAM_MODEL = os.getenv("SAM_MODEL", "sam-3.1")

# Mock mode: explicit, or automatic when there is no API key.
MOCK_MODE = os.getenv("MOCK_MODE", "").lower() in ("1", "true", "yes") or not API_KEY

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PHOTOS = 8
MAX_IMAGE_SIDE = 1600
LEASE_MAX_TURNS = 14
CONDITION_MAX_TURNS = 24
