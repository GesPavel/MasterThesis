import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd

from dataset_generation.constants import ID_COL
from dataset_generation.io_utils import ensure_dir, extract_genome_accession, normalize_accession

logger = logging.getLogger(__name__)

SEARCH_TSV_COLUMNS = [
    "query", "target", "pident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
]


def _build_command(mmseqs_bin: str, args: List[str]) -> List[str]:
    # CreateProcess can't launch .bat/.cmd files directly on Windows without
    # going through cmd.exe /c, even with shell=False.
    if os.name == "nt" and str(mmseqs_bin).lower().endswith((".bat", ".cmd")):
        return ["cmd", "/c", mmseqs_bin, *args]
    return [mmseqs_bin, *args]


def _run_mmseqs(mmseqs_bin: str, args: List[str], expected_output: Path) -> None:
    command = _build_command(mmseqs_bin, args)
    logger.info("Running: %s", " ".join(str(part) for part in command))
    result = subprocess.run(command, capture_output=True, text=True)

    # Some mmseqs wrappers (e.g. the Windows .bat launcher, which bootstraps a
    # busybox-provided bash that easy-cluster/easy-search need internally)
    # hardcode `exit /b 0` and report success even when mmseqs.exe itself
    # failed. The exit code alone can't be trusted, so also verify the
    # expected output file actually landed on disk.
    if result.returncode != 0 or not expected_output.exists():
        raise RuntimeError(
            f"mmseqs command did not produce expected output {expected_output} "
            f"(exit {result.returncode}): {' '.join(str(part) for part in command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    logger.info("mmseqs command completed successfully")


def run_easy_cluster(mmseqs_bin: str, faa_path: Path, result_prefix: Path, tmp_dir: Path) -> Path:
    ensure_dir(result_prefix.parent)
    ensure_dir(tmp_dir)
    cluster_tsv = result_prefix.parent / f"{result_prefix.name}_cluster.tsv"
    _run_mmseqs(mmseqs_bin, ["easy-cluster", str(faa_path), str(result_prefix), str(tmp_dir)], cluster_tsv)
    return cluster_tsv


def run_easy_search(mmseqs_bin: str, query_faa: Path, target_faa: Path, result_path: Path, tmp_dir: Path) -> Path:
    ensure_dir(result_path.parent)
    ensure_dir(tmp_dir)
    _run_mmseqs(mmseqs_bin, ["easy-search", str(query_faa), str(target_faa), str(result_path), str(tmp_dir)], result_path)
    return result_path


def parse_cluster_tsv(cluster_tsv_path: Path) -> Dict[str, str]:
    logger.info("Parsing cluster TSV %s", cluster_tsv_path)
    clusters_df = pd.read_csv(cluster_tsv_path, sep="\t", header=None, names=["pc_id", "protein_id"])
    protein_to_pc = dict(zip(clusters_df["protein_id"], clusters_df["pc_id"]))
    logger.info("Parsed %d protein-cluster assignment(s) into %d protein(s)", len(clusters_df), len(protein_to_pc))
    return protein_to_pc


def best_hits_from_search(search_tsv_path: Path) -> Dict[str, str]:
    logger.info("Parsing easy-search result %s", search_tsv_path)
    hits_df = pd.read_csv(search_tsv_path, sep="\t", header=None, names=SEARCH_TSV_COLUMNS)
    if hits_df.empty:
        logger.info("No hits found in %s", search_tsv_path)
        return {}

    best_idx = hits_df.sort_values(["evalue", "bits"], ascending=[True, False]).groupby("query").head(1).index
    best_hits_df = hits_df.loc[best_idx]
    best_hits = dict(zip(best_hits_df["query"], best_hits_df["target"]))
    logger.info("Resolved best hit for %d/%d unique quer(y/ies)", len(best_hits), hits_df["query"].nunique())
    return best_hits


def build_genome_pc_hits(protein_to_pc: Dict[str, str]) -> Dict[str, List[str]]:
    genome_to_pcs: Dict[str, set] = {}
    for protein_id, pc_id in protein_to_pc.items():
        genome_id = extract_genome_accession(protein_id)
        genome_to_pcs.setdefault(genome_id, set()).add(pc_id)
    return {genome_id: list(pcs) for genome_id, pcs in genome_to_pcs.items()}


def attach_pc_hits(df: pd.DataFrame, genome_to_pc_hits: Dict[str, List[str]]) -> pd.DataFrame:
    df = df.copy()
    accessions = df[ID_COL].map(normalize_accession)

    pc_hits = []
    missing_genomes = 0
    for accession in accessions:
        hits = genome_to_pc_hits.get(accession)
        if hits is None:
            hits = []
            missing_genomes += 1
        pc_hits.append(hits)

    df["pc_hits"] = pc_hits
    logger.info("Attached pc_hits to %d row(s); %d genome(s) had no PC assignment", len(df), missing_genomes)

    pc_counts = [len(x) for x in pc_hits]
    if pc_counts:
        logger.info(
            "PC statistics: mean=%.2f, median=%.2f, max=%d",
            sum(pc_counts) / len(pc_counts),
            pd.Series(pc_counts).median(),
            max(pc_counts),
        )
    return df


def augment_train_with_pc(
    train_faa: Path,
    train_pkl: Path,
    output_pkl: Path,
    work_dir: Path,
    mmseqs_bin: str,
) -> Dict[str, str]:
    logger.info("Augmenting train pickle with PC features: %s", train_pkl)
    cluster_tsv = run_easy_cluster(
        mmseqs_bin=mmseqs_bin,
        faa_path=train_faa,
        result_prefix=work_dir / "train",
        tmp_dir=work_dir / "tmp_cluster_train",
    )
    protein_to_pc = parse_cluster_tsv(cluster_tsv)
    genome_to_pc_hits = build_genome_pc_hits(protein_to_pc)

    df = pd.read_pickle(train_pkl)
    df = attach_pc_hits(df, genome_to_pc_hits)
    ensure_dir(output_pkl.parent)
    df.to_pickle(output_pkl)
    logger.info("Wrote PC-augmented train pickle to %s", output_pkl)
    return protein_to_pc


def augment_test_with_pc(
    test_faa: Path,
    test_pkl: Path,
    train_faa: Path,
    protein_to_pc: Dict[str, str],
    output_pkl: Path,
    work_dir: Path,
    mmseqs_bin: str,
    result_label: str,
) -> Path:
    logger.info("Augmenting test pickle with PC features: %s", test_pkl)
    search_tsv = run_easy_search(
        mmseqs_bin=mmseqs_bin,
        query_faa=test_faa,
        target_faa=train_faa,
        result_path=work_dir / f"search_{result_label}.tsv",
        tmp_dir=work_dir / f"tmp_search_{result_label}",
    )
    best_hits = best_hits_from_search(search_tsv)

    query_protein_to_pc = {
        query_protein: protein_to_pc[target_protein]
        for query_protein, target_protein in best_hits.items()
        if target_protein in protein_to_pc
    }
    genome_to_pc_hits = build_genome_pc_hits(query_protein_to_pc)

    df = pd.read_pickle(test_pkl)
    df = attach_pc_hits(df, genome_to_pc_hits)
    ensure_dir(output_pkl.parent)
    df.to_pickle(output_pkl)
    logger.info("Wrote PC-augmented test pickle to %s", output_pkl)
    return output_pkl
