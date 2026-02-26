from pathlib import Path
import pandas as pd

from features_construction import make_features_table
from feature_set import BENCHMARK_FEATURES, MICROSTRUCTURE_FEATURES
from targets_construction import make_targets_table
from targets_set import TARGETS


def main():
    features_benchmark_table_str = "features_benchmark.csv"
    features_expanded_table_str = "features_expanded.csv"
    targets_table_str = "targets.csv"
    dataset_benchmark_table_str = "dataset_benchmark.csv"
    dataset_expanded_table_str = "dataset_expanded.csv"

    features_benchmark_updated = False
    if not Path(f"data/features/{features_benchmark_table_str}").is_file():
        make_features_table(BENCHMARK_FEATURES, features_benchmark_table_str, progress_desc="Benchmark")
        features_benchmark_updated = True

    features_expanded_updated = False
    if not Path(f"data/features/{features_expanded_table_str}").is_file():
        make_features_table(BENCHMARK_FEATURES + MICROSTRUCTURE_FEATURES, features_expanded_table_str, "Expanded")
        features_expanded_updated = True

    targets_updated = False
    if not Path(f"data/{targets_table_str}").is_file():
        make_targets_table(TARGETS, targets_table_str, "Targets")
        targets_updated = True

    path_dataset_benchmark = Path(f"data/{dataset_benchmark_table_str}")
    if not path_dataset_benchmark.is_file() or features_benchmark_updated or targets_updated:
        features_benchmark = pd.read_csv(f"data/features/{features_benchmark_table_str}", parse_dates=["first_trade_time"])
        targets = pd.read_csv(f"data/{targets_table_str}")
        joined = features_benchmark.set_index("token_mint").join(targets.set_index("token_mint"), how="inner")
        joined.to_csv(str(path_dataset_benchmark))
        print(f"Constructed benchmark dataset: {str(path_dataset_benchmark)}")

    path_dataset_expanded = Path(f"data/{dataset_expanded_table_str}")
    if not path_dataset_expanded.is_file() or features_expanded_updated or targets_updated:
        features_expanded = pd.read_csv(f"data/features/{features_expanded_table_str}", parse_dates=["first_trade_time"])
        targets = pd.read_csv(f"data/{targets_table_str}")
        joined = features_expanded.set_index("token_mint").join(targets.set_index("token_mint"), how="inner")
        joined.to_csv(str(path_dataset_expanded))
        print(f"Constructed benchmark dataset: {str(path_dataset_expanded)}")


if __name__ == '__main__':
    main()
