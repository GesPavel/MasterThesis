import argparse
import logging
from pathlib import Path

from dataset_generation.constants import DEFAULT_SCENARIO2_TEST_FRACTION, DEFAULT_SCENARIO3_TEST_FRACTION
from dataset_generation.io_parsing import parse_rank_selection, parse_scenarios_selection
from dataset_generation.pipeline import build_processed_datasets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate rank-aware processed datasets for scenarios 1-4."
    )
    parser.add_argument(
        "--rank",
        default="all",
        help="Taxon rank to generate: family, genus, order, or all.",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Scenarios to generate as a comma-separated list, or all.",
    )
    parser.add_argument(
        "--output-root",
        default="processed_data",
        help="Root directory for generated datasets.",
    )
    parser.add_argument(
        "--msl41-pkl",
        default="datasets/MSL41_complete.pkl",
        help="Primary MSL41 pickle path.",
    )
    parser.add_argument(
        "--msl41-fasta",
        default="datasets/MSL41.fasta",
        help="Primary MSL41 FASTA path.",
    )
    parser.add_argument(
        "--msl41-faa",
        default="datasets/MSL41.faa",
        help="Primary MSL41 FAA path.",
    )
    parser.add_argument(
        "--msl40-pkl",
        default="datasets/MSL40_complete.pkl",
        help="Secondary MSL40 pickle path used for scenario 4.",
    )
    parser.add_argument(
        "--additional-dir",
        default="datasets/additional",
        help="Directory containing the four dark-matter datasets.",
    )
    parser.add_argument(
        "--scenario2-test-fraction",
        type=float,
        default=DEFAULT_SCENARIO2_TEST_FRACTION,
        help="Hold-out fraction for Scenario 2.",
    )
    parser.add_argument(
        "--scenario3-test-fraction",
        type=float,
        default=DEFAULT_SCENARIO3_TEST_FRACTION,
        help="Hold-out fraction for Scenario 3.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used in stochastic splits.",
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
    msl41_pkl = Path(args.msl41_pkl)
    msl41_fasta = Path(args.msl41_fasta)
    msl41_faa = Path(args.msl41_faa) if args.msl41_faa else None
    msl40_pkl = Path(args.msl40_pkl)
    additional_dir = Path(args.additional_dir)

    build_processed_datasets(
        ranks=ranks,
        scenarios=scenarios,
        output_root=output_root,
        msl41_pkl=msl41_pkl,
        msl41_fasta=msl41_fasta,
        msl41_faa=msl41_faa,
        msl40_pkl=msl40_pkl,
        additional_dir=additional_dir,
        scenario2_test_fraction=args.scenario2_test_fraction,
        scenario3_test_fraction=args.scenario3_test_fraction,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
