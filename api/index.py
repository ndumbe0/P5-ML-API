import sys
from pathlib import Path

# Vercel deploys api/index.py as the function entry; the FastAPI app lives
# one level up in main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app  # noqa: E402

handler = app
