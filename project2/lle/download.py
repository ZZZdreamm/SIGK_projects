"""Downloading NEF files of the RAISE dataset (based on the code from the assignment).

Changes compared to the version from the instructions:
* writing to a temporary ``.part`` file and renaming only after the full download
  (an interrupted download does not leave a corrupted file that would later be "skipped"),
* retries and ``raise_for_status``,
* parallel downloads (the server is slow for a single connection),
* device / keyword filter (Nikon D7000, Indoor) - the provided CSV already
  contains only such photos, but the filter protects against using another file.
"""
from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def read_raise_rows(csv_path, limit=None, device="Nikon D7000", keyword="Indoor"):
    """Returns the first ``limit`` CSV rows matching the filter (in file order)."""
    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if device and row.get("Device", "").strip() != device:
                continue
            if keyword and keyword.lower() not in row.get("Keywords", "").lower():
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def nef_filename(row) -> str:
    filename = row["File"]
    if not filename.upper().endswith(".NEF"):
        filename += ".NEF"
    return filename


def _download_one(i, n, row, output_dir, retries):
    nef_url = row["NEF"]
    filename = nef_filename(row)
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"[{i + 1}/{n}] Skipping {filename} - already exists.", flush=True)
        return
    for attempt in range(1, retries + 1):
        try:
            print(f"[{i + 1}/{n}] Downloading {filename}...", flush=True)
            tmp = filepath + ".part"
            with requests.get(nef_url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(tmp, "wb") as out_file:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        out_file.write(chunk)
            os.replace(tmp, filepath)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  {filename}: attempt {attempt} failed: {e}", flush=True)
            time.sleep(5 * attempt)
    print(f"  giving up on {filename}", flush=True)


def download_raise_nef(csv_path, output_dir, limit=None, retries=3, workers=6):
    """
    Downloads NEF files from the RAISE dataset based on a CSV file.
    The 'limit' parameter specifies the maximum number of files to process.
    Files are downloaded in parallel ('workers' connections) - the server is slow
    for a single connection.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = read_raise_rows(csv_path, limit=limit)
    with ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(rows):
            ex.submit(_download_one, i, len(rows), row, output_dir, retries)
    return rows
