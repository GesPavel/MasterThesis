import argparse
import logging
from pathlib import Path

from dataset_generation.io_parsing import parse_rank_selection, parse_scenarios_selection
from dataset_generation.pc_pipeline import build_pc_features


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment processed dataset pickles with mmseqs2 protein-cluster (PC) features.",
    )
    parser.add_argument(
        "--rank",
        default="all",
        help="Taxon rank to process: family, genus, order, or all.",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Scenarios to process as a comma-separated list, or all.",
    )
    parser.add_argument(
        "--output-root",
        default="processed_data",
        help="Root directory containing the generated dataset splits (from generate_datasets.py).",
    )
    parser.add_argument(
        "--mmseqs-bin",
        default="D:/Programming/mmseqs/mmseqs.bat",
        help="Path to the mmseqs binary/wrapper.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_arg_parser()
    args = parser.parse_args()

    ranks = parse_rank_selection(args.rank)
    scenarios = parse_scenarios_selection(args.scenarios)
    output_root = Path(args.output_root)

    build_pc_features(
        ranks=ranks,
        scenarios=scenarios,
        output_root=output_root,
        mmseqs_bin=args.mmseqs_bin,
    )


if __name__ == "__main__":
    main()
