#!/usr/bin/env python3
"""Resolve Freebase MIDs to human-readable names via Wikidata SPARQL.

One-time script that:
1. Reads all KG Parquet rows, extracts unique answer MIDs
2. Queries Wikidata SPARQL in batches of 200
3. Saves mid_to_name.json alongside the KG Parquet data
"""

import ast
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

KG_DIR = Path.home() / ".cache" / "lifelong_agent" / "database" / "knowledge_graph"
OUTPUT = KG_DIR / "mid_to_name.json"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
BATCH_SIZE = 200
MID_RE = re.compile(r"^m\.\w+$")


def _parse_list(value: object) -> list:
    """Parse answer_list which may be a Python-repr string, JSON, or already a list."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return [value] if value is not None else []
    # Try JSON first, then Python literal_eval (handles single-quoted lists)
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else [result]
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else [result]
    except (ValueError, SyntaxError):
        return [value]


def extract_mids(kg_dir: Path) -> set[str]:
    """Extract all unique Freebase MIDs from answer_list columns."""
    mids: set[str] = set()
    for pf in sorted(kg_dir.rglob("*.parquet")):
        df = pd.read_parquet(pf)
        for answer_list in df["answer_list"]:
            parsed = _parse_list(answer_list)
            for a in parsed:
                a_str = str(a)
                if MID_RE.match(a_str):
                    mids.add(a_str)
    return mids


def mid_to_freebase_id(mid: str) -> str:
    """Convert m.02h8b9t -> /m/02h8b9t for Wikidata P646."""
    return "/" + mid.replace(".", "/")


def freebase_id_to_mid(fid: str) -> str:
    """Convert /m/02h8b9t -> m.02h8b9t."""
    return fid.lstrip("/").replace("/", ".")


def query_wikidata_batch(mids: list[str]) -> dict[str, str]:
    """Query Wikidata SPARQL for a batch of Freebase MIDs."""
    freebase_ids = [mid_to_freebase_id(m) for m in mids]
    values = " ".join(f'"{fid}"' for fid in freebase_ids)

    sparql = f"""
    SELECT ?mid ?label WHERE {{
        ?item wdt:P646 ?mid .
        ?item rdfs:label ?label .
        FILTER(LANG(?label) = "en")
        VALUES ?mid {{ {values} }}
    }}
    """

    params = urllib.parse.urlencode({
        "query": sparql,
        "format": "json",
    })
    url = f"{WIKIDATA_SPARQL}?{params}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "OpenJarvis/1.0 (https://github.com/open-jarvis/OpenJarvis)",
        "Accept": "application/sparql-results+json",
    })

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())

    results: dict[str, str] = {}
    for binding in data.get("results", {}).get("bindings", []):
        fid = binding["mid"]["value"]
        label = binding["label"]["value"]
        mid = freebase_id_to_mid(fid)
        results[mid] = label
    return results


def main() -> None:
    print(f"Reading KG data from {KG_DIR}")
    mids = extract_mids(KG_DIR)
    print(f"Found {len(mids)} unique Freebase MIDs to resolve")

    if not mids:
        print("No MIDs to resolve.")
        return

    mid_list = sorted(mids)
    resolved: dict[str, str] = {}

    for i in range(0, len(mid_list), BATCH_SIZE):
        batch = mid_list[i : i + BATCH_SIZE]
        print(f"  Querying batch {i // BATCH_SIZE + 1} ({len(batch)} MIDs)...")
        try:
            batch_results = query_wikidata_batch(batch)
            resolved.update(batch_results)
            print(f"    Resolved {len(batch_results)}/{len(batch)}")
        except Exception as e:
            print(f"    Error: {e}")

        if i + BATCH_SIZE < len(mid_list):
            time.sleep(1)  # rate limit

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(resolved, f, indent=2, ensure_ascii=False)

    pct = len(resolved) / len(mids)
    print(f"\nResolved {len(resolved)}/{len(mids)} MIDs ({pct:.0%})")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
