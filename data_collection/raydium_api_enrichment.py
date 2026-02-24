import os
import csv
import json
import time
from dataclasses import dataclass, fields, asdict
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.auto import tqdm


API_BASE = "https://api-v3.raydium.io"
POOL_KEYS_BY_ID = f"{API_BASE}/pools/key/ids"

WSOL = "So11111111111111111111111111111111111111112"
AMMV4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
CPMM = "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C"

INPUT_CSV = "data/dune_raydium_wsol_launches_q4_2024.csv"
OUTPUT_CSV = "data/raydium_pools_enriched_q4_2024.csv"
CHECKPOINT_PATH = "data/raydium_enrich_checkpoint.json"

BATCH_SIZE = 50
SLEEP_S = 0.1

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "raydium-research/1.0", "Accept": "application/json"})


# Output schema
@dataclass
class EnrichedPoolRow:
    # From Dune
    token_mint: str
    pool_id: str
    pool_type: str  # program id (AMMv4 or CPMM)
    first_trade_time: str
    has_pumpdotfun_history: int

    # From Raydium
    token_vault: str
    quote_vault: str
    authority: str
    lp_mint: str
    token_decimals: Optional[int]
    token_program: str

    # AMMv4/OpenBook fields (empty for CPMM)
    open_orders: str
    target_orders: str
    market_id: str
    market_program_id: str
    market_authority: str
    market_base_vault: str
    market_quote_vault: str
    market_bids: str
    market_asks: str
    market_event_queue: str


@dataclass
class FailedPoolRow:
    pool_id: str
    token_mint: str
    reason: str
    detail: str


# Utils
def csv_fieldnames(dc_class) -> list[str]:
    return [f.name for f in fields(dc_class)]


class Checkpoint:
    def __init__(self, path: str):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}

    def save(self):
        with open(self.path + ".tmp", "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(self.path + ".tmp", self.path)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value):
        self.data[key] = value
        self.save()


# API
@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
    retry=retry_if_exception_type((requests.RequestException,)),
)
def fetch_pool_keys_batch(pool_ids: list[str]) -> dict[str, dict]:
    """Fetch pool keys for a batch of IDs. Returns {pool_id: keys_dict}."""
    r = SESSION.get(POOL_KEYS_BY_ID, params={"ids": ",".join(pool_ids)}, timeout=30)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and data.get("success") is False:
        raise RuntimeError(f"Raydium API error: {data.get('msg')}")

    # API returns list of pool key objects
    items = data.get("data", [])
    if isinstance(items, dict):
        items = items.get("data", [])

    return {k["id"]: k for k in items if k.get("id")}


def build_enriched_row(dune_row: dict, keys: dict) -> tuple[Optional[EnrichedPoolRow], Optional[FailedPoolRow]]:
    """Combine Dune row with Raydium keys. Returns (enriched, None) or (None, failed)."""
    pool_id = dune_row["pool_id"]
    token_mint = dune_row["token_mint"]
    has_pumpdotfun_history = int(dune_row["has_pumpdotfun_history"] == "True")

    k = keys.get(pool_id)
    if not k:
        return None, FailedPoolRow(pool_id, token_mint, "missing_from_api", "")

    # Extract mints
    mintA_raw = k.get("mintA", {})
    mintB_raw = k.get("mintB", {})
    mintA = mintA_raw.get("address", "") if isinstance(mintA_raw, dict) else str(mintA_raw or "")
    mintB = mintB_raw.get("address", "") if isinstance(mintB_raw, dict) else str(mintB_raw or "")

    # Validate mint pair matches {token_mint, WSOL}
    expected = {token_mint, WSOL}
    actual = {mintA, mintB}
    if actual != expected:
        return None, FailedPoolRow(pool_id, token_mint, "mint_mismatch", f"expected={expected}, actual={actual}")

    # Extract vaults
    vault = k.get("vault", {})
    vaultA = vault.get("A", "") if isinstance(vault, dict) else ""
    vaultB = vault.get("B", "") if isinstance(vault, dict) else ""

    if not vaultA or not vaultB:
        return None, FailedPoolRow(pool_id, token_mint, "missing_vault", f"vaultA={vaultA}, vaultB={vaultB}")

    # Assign token vs quote vault
    if mintA == WSOL:
        token_vault, quote_vault = vaultB, vaultA
        token_mint_info = mintB_raw
    else:
        token_vault, quote_vault = vaultA, vaultB
        token_mint_info = mintA_raw

    # Token metadata
    if isinstance(token_mint_info, dict):
        token_decimals = token_mint_info.get("decimals")
        token_program = token_mint_info.get("programId", "")
    else:
        token_decimals = None
        token_program = ""

    # LP mint
    lp_mint_raw = k.get("mintLp", {})
    lp_mint = lp_mint_raw.get("address", "") if isinstance(lp_mint_raw, dict) else str(lp_mint_raw or "")

    return EnrichedPoolRow(
        token_mint=token_mint,
        pool_id=pool_id,
        pool_type=dune_row["pool_type"],
        first_trade_time=dune_row["first_trade_time"],
        has_pumpdotfun_history=has_pumpdotfun_history,
        token_vault=token_vault,
        quote_vault=quote_vault,
        authority=k.get("authority", ""),
        lp_mint=lp_mint,
        token_decimals=token_decimals,
        token_program=token_program,
        open_orders=k.get("openOrders", ""),
        target_orders=k.get("targetOrders", ""),
        market_id=k.get("marketId", ""),
        market_program_id=k.get("marketProgramId", ""),
        market_authority=k.get("marketAuthority", ""),
        market_base_vault=k.get("marketBaseVault", ""),
        market_quote_vault=k.get("marketQuoteVault", ""),
        market_bids=k.get("marketBids", ""),
        market_asks=k.get("marketAsks", ""),
        market_event_queue=k.get("marketEventQueue", ""),
    ), None


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    checkpoint = Checkpoint(CHECKPOINT_PATH)

    # Load Dune input
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        dune_rows = list(csv.DictReader(f))

    print(f"Loaded {len(dune_rows)} pools from Dune")

    # Resume support
    batch_done = checkpoint.get("batch_done", -1)
    assert isinstance(batch_done, int) and batch_done >= -1, "Checkpoint storage failed"
    start_batch = batch_done + 1

    written_pools = set()
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            written_pools = {r["pool_id"] for r in csv.DictReader(f)}

    # Prepare output file
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    fieldnames = csv_fieldnames(EnrichedPoolRow)

    n_batches = (len(dune_rows) + BATCH_SIZE - 1) // BATCH_SIZE
    total_written = len(written_pools)
    failed_rows: list[FailedPoolRow] = []

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        with tqdm(total=len(dune_rows), desc="Fetching pool keys", unit="pool") as pbar:
            pbar.update(start_batch * BATCH_SIZE)

            for batch_i in range(start_batch, n_batches):
                batch = dune_rows[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
                batch = [r for r in batch if r["pool_id"] not in written_pools]

                if not batch:
                    checkpoint.set("batch_done", batch_i)
                    pbar.update(BATCH_SIZE)
                    continue

                pool_ids = [r["pool_id"] for r in batch]

                try:
                    keys = fetch_pool_keys_batch(pool_ids)
                except Exception as e:
                    print(f"\nBatch {batch_i} failed: {e}")
                    for r in batch:
                        failed_rows.append(FailedPoolRow(r["pool_id"], r["token_mint"], "api_error", str(e)))
                    checkpoint.set("batch_done", batch_i)
                    pbar.update(len(batch))
                    continue

                for dune_row in batch:
                    enriched, failed = build_enriched_row(dune_row, keys)
                    if enriched:
                        writer.writerow(asdict(enriched))
                        written_pools.add(enriched.pool_id)
                        total_written += 1
                    elif failed:
                        failed_rows.append(failed)

                f.flush()
                checkpoint.set("batch_done", batch_i)
                pbar.update(len(batch))
                pbar.set_postfix(written=total_written, failed=len(failed_rows))
                time.sleep(SLEEP_S)

    print(f"\nDone! Wrote {total_written} pools to {OUTPUT_CSV}")

    if failed_rows:
        failed_path = OUTPUT_CSV.replace(".csv", "_failed.csv")
        with open(failed_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_fieldnames(FailedPoolRow))
            w.writeheader()
            for row in failed_rows:
                w.writerow(asdict(row))
        print(f"Failed pools ({len(failed_rows)}) written to {failed_path}")


if __name__ == "__main__":
    main()
