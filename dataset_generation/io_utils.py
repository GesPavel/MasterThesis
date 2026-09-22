import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from Bio import SeqIO

from .constants import ID_COL

logger = logging.getLogger(__name__)


def normalize_accession(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lstrip(">")


def extract_genome_accession(protein_id: str) -> str:
    """
    FAA records are per-protein (prodigal-style: <genome_accession>_<protein_index>),
    e.g. "HQ641347.1_20" -> "HQ641347.1".
    """
    return protein_id.rsplit("_", 1)[0]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    logger.info("Loading dataframe from %s", path)
    df = pd.read_pickle(path)
    logger.info("Loaded %d row(s) from %s", len(df), path)
    return df


def filter_unknown_taxa(df: pd.DataFrame, rank: str) -> pd.DataFrame:
    if rank not in df.columns:
        raise KeyError(f"Taxonomy rank '{rank}' not found in dataframe columns: {list(df.columns)}")
    normalized_values = df[rank].astype(str).str.strip().str.lower()
    mask = df[rank].notna() & (normalized_values != "unknown")
    filtered = df[mask].copy()
    logger.info("Filtered unknown '%s' taxa: %d -> %d row(s)", rank, len(df), len(filtered))
    return filtered


def subset_dataframe_by_accessions(
    df: pd.DataFrame,
    accession_ids: Iterable[str],
    accession_col: str = ID_COL,
) -> pd.DataFrame:
    normalized_ids = {normalize_accession(value) for value in accession_ids}
    subset = df[df[accession_col].map(normalize_accession).isin(normalized_ids)].copy()
    return subset


def split_sequence_file(
    source_path: Path,
    destination_path: Path,
    accession_ids: Iterable[str] | None,
    protein_level: bool = False,
) -> Path:
    """
    ``protein_level=True`` matches records by their FAA-style genome accession
    (id minus the trailing "_<protein_index>") rather than an exact id match,
    since ``accession_ids`` are always genome-level while FAA records are per-protein.
    """
    if accession_ids is None:
        records = list(SeqIO.parse(str(source_path), "fasta"))
    else:
        normalized_ids = {normalize_accession(value) for value in accession_ids}

        def record_matches(record_id: str) -> bool:
            normalized = normalize_accession(record_id)
            if protein_level:
                normalized = extract_genome_accession(normalized)
            return normalized in normalized_ids

        records = [
            record
            for record in SeqIO.parse(str(source_path), "fasta")
            if record_matches(record.id)
        ]
    ensure_dir(destination_path.parent)
    SeqIO.write(records, str(destination_path), "fasta")
    logger.info("Wrote %d sequence record(s) from %s to %s", len(records), source_path, destination_path)
    return destination_path


def write_dataframe_split(
    df: pd.DataFrame,
    destination_path: Path,
    accession_ids: Iterable[str] | None = None,
) -> Path:
    ensure_dir(destination_path.parent)
    if accession_ids is None:
        df.to_pickle(destination_path)
        logger.info("Wrote %d row(s) to %s", len(df), destination_path)
    else:
        subset = subset_dataframe_by_accessions(df, accession_ids)
        subset.to_pickle(destination_path)
        logger.info("Wrote %d row(s) to %s", len(subset), destination_path)
    return destination_path
