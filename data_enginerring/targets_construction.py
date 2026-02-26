import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import Type, Optional
from targets_set import TargetsConstructor, TargetSet


def make_targets_table(target_constructors: list[Type[TargetsConstructor]],
                        out_csv: str, progress_desc: Optional[str] = None):
    """Calculates targets from target components tables."""

    directory = Path("data/target_components")
    output_path = Path(f"data/{out_csv}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_set = TargetSet(target_constructors)
    all_rows = []
    skipped = 0

    day_files = sorted(
        f for month_dir in sorted(directory.iterdir())
        if month_dir.is_dir()
        for f in sorted(month_dir.iterdir())
        if f.suffix == ".csv"
    )

    desc_str = "Days" if not progress_desc else f"[{progress_desc}] Days"
    for day_file in tqdm(day_files, desc=desc_str):
        table = pd.read_csv(str(day_file))

        for _, row in table.iterrows():
            token_mint = row["token_mint"]

            try:
                targets = target_set.calculate(row)
            except Exception as e:
                print(f"  WARN {token_mint}: {e}")
                skipped += 1
                continue

            targets["token_mint"] = token_mint
            all_rows.append(targets)

    result = pd.DataFrame(all_rows)
    meta_cols = ["token_mint"]
    target_cols = [c for c in result.columns if c not in meta_cols]
    result = result[meta_cols + target_cols]
    result.to_csv(output_path, index=False)

    print(f"Done: {len(result)} tokens with targets, {skipped} skipped -> {output_path}")
