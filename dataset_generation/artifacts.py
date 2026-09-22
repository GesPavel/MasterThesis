import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

from dataset_generation.constants import ID_COL
from dataset_generation.dark_matter import DarkMatterGroup
from dataset_generation.io_utils import ensure_dir, load_dataframe, normalize_accession, split_sequence_file, write_dataframe_split

logger = logging.getLogger(__name__)


def build_taxonomy_reference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Combine the taxonomy-bearing split frames into a single lookup keyed by
    normalized accession. Dark-matter genomes are mutated copies of these
    genomes and inherit their taxonomy from here.
    """
    parts = [train_df]
    if test_df is not None:
        parts.append(test_df)

    reference = pd.concat(parts, ignore_index=True).copy()
    reference[ID_COL] = reference[ID_COL].map(normalize_accession)
    return reference.drop_duplicates(subset=ID_COL)


def enrich_dark_matter_df(dm_df: pd.DataFrame, taxonomy_reference: pd.DataFrame) -> pd.DataFrame:
    """
    Repair a raw dark-matter dataframe so it is schema-compatible with the main
    test pickle:

      * strip the leading '>' FASTA-header artifact from the accession column;
      * attach the taxonomy / metadata columns (Family, Genus, ... ) that the
        mutated pickle lacks, sourced from the original genome it was derived
        from (matched on normalized accession).

    The dark-matter genome's own feature columns (hmms_hits, sequence, ...) are
    preserved -- only columns missing from ``dm_df`` are pulled from the reference.
    """
    dm_df = dm_df.copy()
    dm_df[ID_COL] = dm_df[ID_COL].map(normalize_accession)

    extra_cols = [c for c in taxonomy_reference.columns if c != ID_COL and c not in dm_df.columns]
    ref = taxonomy_reference[[ID_COL] + extra_cols]

    merged = dm_df.merge(ref, on=ID_COL, how="left")

    missing = merged[extra_cols[0]].isna().sum() if extra_cols else 0
    if missing:
        logger.warning("%d dark-matter genome(s) had no taxonomy match in the reference", missing)

    return merged


@dataclass(frozen=True)
class SplitArtifacts:
    pkl: Dict[str, Path]
    fasta: Dict[str, Path]
    faa: Dict[str, Path]


def scenario_output_dir(output_root: Path, rank: str, scenario: str) -> Path:
    return ensure_dir(output_root / rank / f"scenario{scenario}")


def format_dirs_for_scenario(scenario_dir: Path) -> Dict[str, Path]:
    return {
        "pkl": ensure_dir(scenario_dir / "pkl"),
        "fasta": ensure_dir(scenario_dir / "fasta"),
        "faa": ensure_dir(scenario_dir / "faa"),
    }


def save_split_artifacts(
    scenario_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    train_accessions: Iterable[str],
    test_accessions: Iterable[str] | None,
    main_fasta: Path,
    main_faa: Path | None,
    dark_matter_groups: Sequence[DarkMatterGroup],
    test_label_prefix: str = "test",
    dark_matter_accessions: Iterable[str] | None = None,
) -> SplitArtifacts:
    logger.info("Writing split artifacts to %s", scenario_dir)
    dirs = format_dirs_for_scenario(scenario_dir)

    pkl_paths: Dict[str, Path] = {}
    fasta_paths: Dict[str, Path] = {}
    faa_paths: Dict[str, Path] = {}

    pkl_paths["train"] = write_dataframe_split(train_df, dirs["pkl"] / "train.pkl")
    fasta_paths["train"] = split_sequence_file(main_fasta, dirs["fasta"] / "train.fasta", train_accessions)
    if main_faa is not None and main_faa.exists():
        faa_paths["train"] = split_sequence_file(main_faa, dirs["faa"] / "train.faa", train_accessions, protein_level=True)

    if test_df is not None and test_accessions is not None:
        pkl_paths[test_label_prefix] = write_dataframe_split(test_df, dirs["pkl"] / f"{test_label_prefix}.pkl")
        fasta_paths[test_label_prefix] = split_sequence_file(main_fasta, dirs["fasta"] / f"{test_label_prefix}.fasta", test_accessions)
        if main_faa is not None and main_faa.exists():
            faa_paths[test_label_prefix] = split_sequence_file(main_faa, dirs["faa"] / f"{test_label_prefix}.faa", test_accessions, protein_level=True)

    dm_accessions = dark_matter_accessions if dark_matter_accessions is not None else test_accessions

    taxonomy_reference = build_taxonomy_reference(train_df, test_df)

    for dm_group in dark_matter_groups:
        dm_label = f"{test_label_prefix}_{dm_group.label}"
        logger.info("Processing dark-matter group '%s' -> %s", dm_group.label, dm_label)
        if dm_group.pkl is not None:
            dm_df = load_dataframe(dm_group.pkl)
            dm_df = enrich_dark_matter_df(dm_df, taxonomy_reference)
            pkl_paths[dm_label] = write_dataframe_split(
                dm_df,
                dirs["pkl"] / f"{dm_label}.pkl",
                accession_ids=dm_accessions,
            )
        if dm_group.fasta is not None and dm_group.fasta.exists():
            fasta_paths[dm_label] = split_sequence_file(
                dm_group.fasta,
                dirs["fasta"] / f"{dm_label}.fasta",
                dm_accessions,
            )
        if dm_group.faa is not None and dm_group.faa.exists():
            faa_paths[dm_label] = split_sequence_file(
                dm_group.faa,
                dirs["faa"] / f"{dm_label}.faa",
                dm_accessions,
                protein_level=True,
            )

    logger.info(
        "Finished writing artifacts to %s (%d pkl, %d fasta, %d faa)",
        scenario_dir, len(pkl_paths), len(fasta_paths), len(faa_paths),
    )
    return SplitArtifacts(pkl=pkl_paths, fasta=fasta_paths, faa=faa_paths)


def print_artifact_summary(scenario: str, rank: str, artifacts: SplitArtifacts) -> None:
    print()
    print(f"=== Scenario {scenario} | rank={rank} ===")
    print(f"Pickle outputs: {len(artifacts.pkl)}")
    for name, path in sorted(artifacts.pkl.items()):
        print(f"  {name}: {path}")
    print(f"FASTA outputs: {len(artifacts.fasta)}")
    for name, path in sorted(artifacts.fasta.items()):
        print(f"  {name}: {path}")
    print(f"FAA outputs: {len(artifacts.faa)}")
    for name, path in sorted(artifacts.faa.items()):
        print(f"  {name}: {path}")
