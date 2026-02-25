from pathlib import Path

from features_from_deltas import make_features_table
from feature_set import BENCHMARK_FEATURES, MICROSTRUCTURE_FEATURES


def main():
    benchmark_table_str = "features_benchmark.csv"
    expanded_table_str = "features_expanded.csv"

    if not Path(f"data/features/{benchmark_table_str}").is_file():
        make_features_table(BENCHMARK_FEATURES, benchmark_table_str, progress_desc="Benchmark")

    if not Path(f"data/features/{expanded_table_str}").is_file():
        make_features_table(BENCHMARK_FEATURES + MICROSTRUCTURE_FEATURES, expanded_table_str, "Expanded")


if __name__ == '__main__':
    main()
