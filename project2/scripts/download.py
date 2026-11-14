"""Downloads the first N photos (Nikon D7000, Indoor) listed in the RAISE CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lle.download import download_raise_nef  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="RAISE_258.csv")
    ap.add_argument("--out", default="data/nef")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    download_raise_nef(a.csv, a.out, limit=a.limit, workers=a.workers)


if __name__ == "__main__":
    main()
