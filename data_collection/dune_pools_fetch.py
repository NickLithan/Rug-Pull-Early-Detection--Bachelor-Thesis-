import os
import time
import csv
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def reload_constants():
    global DUNE_API_KEY, BASE, HEADERS
    load_dotenv()
    DUNE_API_KEY = os.environ["DUNE_API_KEY"]
    BASE = "https://api.dune.com/api/v1"
    HEADERS = {"X-Dune-API-Key": DUNE_API_KEY}


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
    retry=retry_if_exception_type(requests.RequestException),
)
def fetch_page(query_id: int, limit: int, offset: int):
    url = f"{BASE}/query/{query_id}/results"
    r = requests.get(url, headers=HEADERS, params={"limit": limit, "offset": offset}, timeout=60)
    r.raise_for_status()
    j = r.json()
    rows = j.get("result", {}).get("rows", []) or []
    return rows


def download_all(fieldnames: list[str], query_id: int, out_csv: str, 
                 page_size=1000, sleep_s=0.2):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    reload_constants()

    offset = 0
    total = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        while True:
            try:
                rows = fetch_page(query_id, page_size, offset)
            except Exception as e:
                print(f"Failed to fetch page at offset {offset}: {e}")
                break
            if not rows:
                break

            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})

            total += len(rows)
            offset += len(rows)
            print(f"Fetched {len(rows)} rows (total {total})")
            time.sleep(sleep_s)

    if total == 0:
        raise RuntimeError("Request failed. Make sure the query is saved and you have access to it.")

    print(f"Wrote {total} rows → {out_csv}")


# example
if __name__ == "__main__":
    query_id = 6683313  # example of a Dune query ID
    fieldnames = ["token_mint", "pool_type", "pool_id", "first_trade_time", "has_pumpdotfun_history"]
    out_csv = "data/dune_raydium_wsol_launches_q4_2024.csv"

    download_all(fieldnames, query_id, out_csv=out_csv)
