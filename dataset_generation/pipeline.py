import logging
from pathlib import Path
from typing import Dict, Sequence

from dataset_generation.artifacts import SplitArtifacts, print_artifact_summary
from dataset_generation.constants import DEFAULT_SCENARIO2_TEST_FRACTION, DEFAULT_SCENARIO3_TEST_FRACTION, canonical_rank
from dataset_generation.scenarios import scenario1_split, scenario2_split, scenario3_split, scenario4_split

logger = logging.getLogger(__name__)


def build_processed_datasets(
    ranks: Sequence[str],
    scenarios: Sequence[str],
    output_root: Path,
    msl41_pkl: Path,
    msl41_fasta: Path,
    msl41_faa: Path | None,
    msl40_pkl: Path,
    additional_dir: Path,
    scenario2_test_fraction: float = DEFAULT_SCENARIO2_TEST_FRACTION,
    scenario3_test_fraction: float = DEFAULT_SCENARIO3_TEST_FRACTION,
    random_seed: int = 42,
) -> Dict[str, Dict[str, SplitArtifacts]]:
    results: Dict[str, Dict[str, SplitArtifacts]] = {}

    logger.info("Building processed datasets for rank(s)=%s, scenario(s)=%s", list(ranks), list(scenarios))

    for rank in ranks:
        rank_key = canonical_rank(rank)
        logger.info("=== Processing rank '%s' ===", rank_key)
        results[rank_key] = {}

        if "1" in scenarios:
            logger.info("--- rank '%s': running scenario 1 ---", rank_key)
            artifacts = scenario1_split(
                rank=rank_key,
                output_root=output_root,
                msl41_pkl=msl41_pkl,
                msl41_fasta=msl41_fasta,
                msl41_faa=msl41_faa,
                additional_dir=additional_dir,
            )
            results[rank_key]["1"] = artifacts
            print_artifact_summary("1", rank_key, artifacts)

        if "2" in scenarios:
            logger.info("--- rank '%s': running scenario 2 ---", rank_key)
            artifacts = scenario2_split(
                rank=rank_key,
                output_root=output_root,
                msl41_pkl=msl41_pkl,
                msl41_fasta=msl41_fasta,
                msl41_faa=msl41_faa,
                additional_dir=additional_dir,
                test_fraction=scenario2_test_fraction,
                random_seed=random_seed,
            )
            results[rank_key]["2"] = artifacts
            print_artifact_summary("2", rank_key, artifacts)

        if "3" in scenarios:
            logger.info("--- rank '%s': running scenario 3 ---", rank_key)
            artifacts = scenario3_split(
                rank=rank_key,
                output_root=output_root,
                msl41_pkl=msl41_pkl,
                msl41_fasta=msl41_fasta,
                msl41_faa=msl41_faa,
                additional_dir=additional_dir,
                test_fraction=scenario3_test_fraction,
                random_seed=random_seed,
            )
            results[rank_key]["3"] = artifacts
            print_artifact_summary("3", rank_key, artifacts)

        if "4" in scenarios:
            logger.info("--- rank '%s': running scenario 4 ---", rank_key)
            artifacts = scenario4_split(
                rank=rank_key,
                output_root=output_root,
                msl41_pkl=msl41_pkl,
                msl41_fasta=msl41_fasta,
                msl41_faa=msl41_faa,
                msl40_pkl=msl40_pkl,
                additional_dir=additional_dir,
            )
            results[rank_key]["4"] = artifacts
            print_artifact_summary("4", rank_key, artifacts)

        logger.info("=== Finished rank '%s' ===", rank_key)

    logger.info("All requested rank(s)/scenario(s) complete.")
    return results
