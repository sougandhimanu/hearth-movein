"""Settings, read from the environment (a .env file is loaded if present)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_env_file(path: Path) -> str:
    """Windows PowerShell's `echo ... > .env` writes UTF-16, which would otherwise be read as an empty file."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", "replace")


try:
    import io

    from dotenv import load_dotenv

    if (ROOT / ".env").exists():
        load_dotenv(stream=io.StringIO(_read_env_file(ROOT / ".env")))
except ImportError:  # python-dotenv is optional
    pass


def _user_data_dir() -> Path:
    """Per-user app data folder, outside the code, so personal leases and photos never sit in the repo."""
    if os.name == "nt":
        return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "hearth"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "hearth"
    return Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "hearth"


DATA_DIR = Path(os.getenv("DATA_DIR") or _user_data_dir())

BASE_URL = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
API_KEY = os.getenv("MODEL_API_KEY", "")
SPARK_MODEL = os.getenv("SPARK_MODEL", "muse-spark-1.3")
SAM_MODEL = os.getenv("SAM_MODEL", "sam-3.1")

# Mock mode: explicit, or automatic when there is no API key.
MOCK_MODE = os.getenv("MOCK_MODE", "").lower() in ("1", "true", "yes") or not API_KEY

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PHOTOS = 8
MAX_IMAGE_SIDE = 1600
LEASE_MAX_TURNS = 20
CONDITION_MAX_TURNS = 24
