"""One-time script to download and save the embedding model to a local directory.

Run once from the project root:

    python creation/without-schema/download_models.py

The model is saved to  models/all-MiniLM-L6-v2/  (relative to the project root).
Subsequent pipeline runs load it from disk — no network calls required.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root is two levels up from this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download(model_name: str = "all-MiniLM-L6-v2") -> Path:
    from sentence_transformers import SentenceTransformer

    dest = MODELS_DIR / model_name
    if dest.exists():
        print(f"Model already present at: {dest}")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{model_name}' from HuggingFace …")

    model = SentenceTransformer(model_name)
    model.save(str(dest))

    print(f"Saved to: {dest}")
    return dest


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "all-MiniLM-L6-v2"
    download(model_name)

