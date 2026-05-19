import pickle
import random
from pathlib import Path

import pandas as pd
from Bio import SeqIO


def scenario4_split(
    new_edition_pickle_path,
    old_edition_pickle_path,
    new_edition_fasta_path,
    output_dir,
    taxonomy_rank="Family",
    random_seed=42,
):
    """
    Scenario 4:
    Temporal split by ICTV releases.

    Train:
        Genomes already present in MSL40.

    Test:
        Genomes newly appearing in MSL41.

    Constraints:
        - No accession overlap between train/test
        - Temporal split only
        - No dark matter augmentation here
    """

    random.seed(random_seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accession_col = "Accession"

    # ---------------------------------------------------
    # Load dataframes
    # ---------------------------------------------------
    with open(new_edition_pickle_path, "rb") as f:
        df_msl41 = pickle.load(f)

    with open(old_edition_pickle_path, "rb") as f:
        df_msl40 = pickle.load(f)

    # ---------------------------------------------------
    # Filter unknown taxa in MSL41
    # ---------------------------------------------------
    df_msl41 = df_msl41[df_msl41[taxonomy_rank].notna()]
    df_msl41 = df_msl41[df_msl41[taxonomy_rank] != "Unknown"]

    # ---------------------------------------------------
    # Extract accession sets
    # ---------------------------------------------------
    msl41_ids = set(df_msl41[accession_col].unique())
    msl40_ids = set(df_msl40[accession_col].unique())

    # ---------------------------------------------------
    # Temporal split
    # ---------------------------------------------------
    overlap_ids = msl41_ids & msl40_ids
    new_ids = msl41_ids - msl40_ids

    if len(overlap_ids) == 0:
        raise ValueError(
            "No overlapping genomes between MSL41 and MSL40."
        )

    if len(new_ids) == 0:
        raise ValueError(
            "No new genomes found in MSL41."
        )

    # ---------------------------------------------------
    # Build train/test dataframes
    # ---------------------------------------------------
    train_df = df_msl41[
        df_msl41[accession_col].isin(overlap_ids)
    ].copy()

    test_df = df_msl41[
        df_msl41[accession_col].isin(new_ids)
    ].copy()

    # ---------------------------------------------------
    # Save pickles
    # ---------------------------------------------------
    train_pickle = output_dir / "train.pkl"
    test_pickle = output_dir / "test.pkl"

    train_df.to_pickle(train_pickle)
    test_df.to_pickle(test_pickle)

    # ---------------------------------------------------
    # Load FASTA
    # ---------------------------------------------------
    fasta_records = list(
        SeqIO.parse(new_edition_fasta_path, "fasta")
    )

    # ---------------------------------------------------
    # Split FASTA
    # ---------------------------------------------------
    train_records = []
    test_records = []

    for record in fasta_records:

        record_id = record.id

        if record_id in overlap_ids:
            train_records.append(record)

        elif record_id in new_ids:
            test_records.append(record)

    # ---------------------------------------------------
    # Save FASTA
    # ---------------------------------------------------
    train_fasta = output_dir / "train.fasta"
    test_fasta = output_dir / "test.fasta"

    SeqIO.write(train_records, train_fasta, "fasta")
    SeqIO.write(test_records, test_fasta, "fasta")

    # ---------------------------------------------------
    # Taxonomy statistics
    # ---------------------------------------------------
    train_taxa = set(train_df[taxonomy_rank].unique())
    test_taxa = set(test_df[taxonomy_rank].unique())

    novel_taxa = test_taxa - train_taxa

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------
    print()
    print("=== Scenario 4 Split ===")
    print("Temporal split: MSL40 -> MSL41")
    print()

    print(f"Training genomes (present in MSL40): {len(overlap_ids)}")
    print(f"Testing genomes (new in MSL41): {len(new_ids)}")
    print()

    print(f"Train taxa: {len(train_taxa)}")
    print(f"Test taxa: {len(test_taxa)}")
    print(f"Novel taxa in test: {len(novel_taxa)}")
    print()

    print(f"Train pickle: {train_pickle}")
    print(f"Test pickle: {test_pickle}")
    print(f"Train FASTA:  {train_fasta}")
    print(f"Test FASTA:   {test_fasta}")


if __name__ == "__main__":

    scenario4_split(
        new_edition_pickle_path="raw_data/MSL41_complete.pkl",
        old_edition_pickle_path="raw_data/MSL40_complete.pkl",
        new_edition_fasta_path="raw_data/MSL41.fasta",
        output_dir="processed_data/genus/scenario4",
        taxonomy_rank="Genus",
        random_seed=42,
    )