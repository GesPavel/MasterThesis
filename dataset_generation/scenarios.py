import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from dataset_generation.artifacts import SplitArtifacts, save_split_artifacts, scenario_output_dir
from dataset_generation.constants import DEFAULT_SCENARIO2_TEST_FRACTION, DEFAULT_SCENARIO3_TEST_FRACTION, ID_COL
from dataset_generation.dark_matter import discover_dark_matter_groups
from dataset_generation.io_utils import filter_unknown_taxa, load_dataframe, normalize_accession, subset_dataframe_by_accessions
from dataset_generation.taxon_bucketing import bucketed_taxon_holdout

logger = logging.getLogger(__name__)


def scenario1_split(
    rank: str,
    output_root: Path,
    msl41_pkl: Path,
    msl41_fasta: Path,
    msl41_faa: Path | None,
    additional_dir: Path,
) -> SplitArtifacts:
    """Scenario 1: in-distribution robustness.

    Train: the full MSL41 dataset.
    Test: mutated (dark-matter) versions of the same genomes used for training.
    """
    logger.info("Scenario 1 (rank=%s): starting", rank)
    df = filter_unknown_taxa(load_dataframe(msl41_pkl), rank)
    train_ids = set(df[ID_COL].map(normalize_accession).unique())
    logger.info("Scenario 1 (rank=%s): %d train genome(s)", rank, len(train_ids))
    dark_matter_groups = discover_dark_matter_groups(additional_dir)
    scenario_dir = scenario_output_dir(output_root, rank, "1")

    artifacts = save_split_artifacts(
        scenario_dir=scenario_dir,
        train_df=df,
        test_df=None,
        train_accessions=train_ids,
        test_accessions=None,
        main_fasta=msl41_fasta,
        main_faa=msl41_faa,
        dark_matter_groups=dark_matter_groups,
        test_label_prefix="test",
        dark_matter_accessions=train_ids,
    )
    logger.info("Scenario 1 (rank=%s): done", rank)
    return artifacts


def scenario2_split(
    rank: str,
    output_root: Path,
    msl41_pkl: Path,
    msl41_fasta: Path,
    msl41_faa: Path | None,
    additional_dir: Path,
    test_fraction: float = DEFAULT_SCENARIO2_TEST_FRACTION,
    random_seed: int = 42,
) -> SplitArtifacts:
    """Scenario 2: hold-out by genome.

    Train: a stratified subset of genomes from MSL41.
    Test: the remaining genomes (no genome overlap with train).
    Dark-matter test: mutated versions of the held-out test genomes only.
    """
    logger.info("Scenario 2 (rank=%s): starting", rank)
    df = filter_unknown_taxa(load_dataframe(msl41_pkl), rank)
    genomes = df[[ID_COL, rank]].drop_duplicates().reset_index(drop=True)

    taxon_counts = genomes[rank].value_counts()
    singleton_taxa = set(taxon_counts[taxon_counts < 2].index)
    singleton_genomes = genomes[genomes[rank].isin(singleton_taxa)]
    stratifiable_genomes = genomes[~genomes[rank].isin(singleton_taxa)]
    logger.info(
        "Scenario 2 (rank=%s): %d singleton taxa forced into train (%d genome(s))",
        rank, len(singleton_taxa), len(singleton_genomes),
    )

    if stratifiable_genomes.empty:
        raise ValueError("Scenario 2 cannot be built because no stratifiable genomes remain after singleton handling.")

    train_ids, test_ids = train_test_split(
        stratifiable_genomes[ID_COL],
        test_size=test_fraction,
        random_state=random_seed,
        stratify=stratifiable_genomes[rank],
    )

    train_ids = set(train_ids) | set(singleton_genomes[ID_COL].map(normalize_accession).tolist())
    test_ids = set(test_ids)
    logger.info(
        "Scenario 2 (rank=%s): %d train genome(s), %d test genome(s)",
        rank, len(train_ids), len(test_ids),
    )

    train_df = subset_dataframe_by_accessions(df, train_ids)
    test_df = subset_dataframe_by_accessions(df, test_ids)
    dark_matter_groups = discover_dark_matter_groups(additional_dir)
    scenario_dir = scenario_output_dir(output_root, rank, "2")

    artifacts = save_split_artifacts(
        scenario_dir=scenario_dir,
        train_df=train_df,
        test_df=test_df,
        train_accessions=train_ids,
        test_accessions=test_ids,
        main_fasta=msl41_fasta,
        main_faa=msl41_faa,
        dark_matter_groups=dark_matter_groups,
        test_label_prefix="test",
    )
    logger.info("Scenario 2 (rank=%s): done", rank)
    return artifacts


def scenario3_split(
    rank: str,
    output_root: Path,
    msl41_pkl: Path,
    msl41_fasta: Path,
    msl41_faa: Path | None,
    additional_dir: Path,
    test_fraction: float = DEFAULT_SCENARIO3_TEST_FRACTION,
    random_seed: int = 42,
) -> SplitArtifacts:
    """Scenario 3: hold-out by taxon (true open-set novelty).

    Train: genomes from a subset of taxa at ``rank``.
    Test: genomes from taxa held out entirely (no taxon overlap with train).
    Dark-matter test: mutated versions of the held-out test genomes only.
    """
    logger.info("Scenario 3 (rank=%s): starting", rank)
    df = filter_unknown_taxa(load_dataframe(msl41_pkl), rank)

    train_df, test_df, train_ids, test_ids = bucketed_taxon_holdout(
        df=df,
        rank=rank,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    dark_matter_groups = discover_dark_matter_groups(additional_dir)
    scenario_dir = scenario_output_dir(output_root, rank, "3")

    artifacts = save_split_artifacts(
        scenario_dir=scenario_dir,
        train_df=train_df,
        test_df=test_df,
        train_accessions=train_ids,
        test_accessions=test_ids,
        main_fasta=msl41_fasta,
        main_faa=msl41_faa,
        dark_matter_groups=dark_matter_groups,
        test_label_prefix="test",
    )
    logger.info("Scenario 3 (rank=%s): done", rank)
    return artifacts


def scenario4_split(
    rank: str,
    output_root: Path,
    msl41_pkl: Path,
    msl41_fasta: Path,
    msl41_faa: Path | None,
    msl40_pkl: Path,
    additional_dir: Path,
) -> SplitArtifacts:
    """Scenario 4: time-split by ICTV release.

    Train: genomes present in both MSL40 and MSL41 (the older release).
    Test: genomes newly appearing in MSL41 (absent from MSL40).
    Dark-matter test: mutated versions of the newly-appearing test genomes only.
    """
    logger.info("Scenario 4 (rank=%s): starting", rank)
    df_msl41 = filter_unknown_taxa(load_dataframe(msl41_pkl), rank)
    df_msl40 = load_dataframe(msl40_pkl)

    msl41_ids = set(df_msl41[ID_COL].map(normalize_accession).unique())
    msl40_ids = set(df_msl40[ID_COL].map(normalize_accession).unique())
    overlap_ids = msl41_ids & msl40_ids
    new_ids = msl41_ids - msl40_ids
    logger.info(
        "Scenario 4 (rank=%s): %d overlapping (train) genome(s), %d new (test) genome(s)",
        rank, len(overlap_ids), len(new_ids),
    )

    if not overlap_ids:
        raise ValueError("No overlapping genomes between MSL41 and MSL40.")
    if not new_ids:
        raise ValueError("No new genomes found in MSL41.")

    train_df = subset_dataframe_by_accessions(df_msl41, overlap_ids)
    test_df = subset_dataframe_by_accessions(df_msl41, new_ids)
    dark_matter_groups = discover_dark_matter_groups(additional_dir)
    scenario_dir = scenario_output_dir(output_root, rank, "4")

    artifacts = save_split_artifacts(
        scenario_dir=scenario_dir,
        train_df=train_df,
        test_df=test_df,
        train_accessions=overlap_ids,
        test_accessions=new_ids,
        main_fasta=msl41_fasta,
        main_faa=msl41_faa,
        dark_matter_groups=dark_matter_groups,
        test_label_prefix="test",
    )
    logger.info("Scenario 4 (rank=%s): done", rank)
    return artifacts
