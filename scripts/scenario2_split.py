# Scenario 2 Dataset Split Script

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from Bio import SeqIO


def scenario2_split(
    pickle_path,
    fasta_path,
    output_dir,
    taxonomy_rank="Family",
    test_fraction=0.3,
    random_seed=42,
):
    """
    Scenario 2:
    Hold-out by genome (same taxonomic universe).

    Train:
        Random subset of genomes.

    Test:
        Remaining genomes.

    Constraints:
        - No genome overlap between train/test.
        - Taxa MAY overlap.
        - Stratified by taxonomy rank.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # Load pickle dataframe
    # ---------------------------------------------------
    with open(pickle_path, "rb") as f:
        df = pickle.load(f)

    # ---------------------------------------------------
    # Basic filtering
    # ---------------------------------------------------
    df = df[df[taxonomy_rank].notna()]
    df = df[df[taxonomy_rank] != "Unknown"]

    accession_col = "Accession"

    genomes = (
        df[[accession_col, taxonomy_rank]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ---------------------------------------------------
    # Handle singleton taxa
    # ---------------------------------------------------
    taxon_counts = genomes[taxonomy_rank].value_counts()

    singleton_taxa = set(taxon_counts[taxon_counts < 2].index)

    singleton_genomes = genomes[
        genomes[taxonomy_rank].isin(singleton_taxa)
    ]

    stratifiable_genomes = genomes[
        ~genomes[taxonomy_rank].isin(singleton_taxa)
    ]

    print(f"Singleton taxa: {len(singleton_taxa)}")
    print(f"Singleton genomes forced into train: {len(singleton_genomes)}")

    # ---------------------------------------------------
    # Train/test genome split
    # ---------------------------------------------------
    train_ids, test_ids = train_test_split(
        stratifiable_genomes[accession_col],
        test_size=test_fraction,
        random_state=random_seed,
        stratify=stratifiable_genomes[taxonomy_rank],
    )

    # Add singleton genomes to train
    train_ids = set(train_ids) | set(singleton_genomes[accession_col])
    test_ids = set(test_ids)

    train_ids = set(train_ids)
    test_ids = set(test_ids)

    # ---------------------------------------------------
    # Split dataframe
    # ---------------------------------------------------
    train_df = df[df[accession_col].isin(train_ids)].copy()
    test_df = df[df[accession_col].isin(test_ids)].copy()

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
    fasta_records = list(SeqIO.parse(fasta_path, "fasta"))

    # ---------------------------------------------------
    # Split FASTA
    # ---------------------------------------------------
    train_records = []
    test_records = []

    for record in fasta_records:
        record_id = record.id

        if record_id in train_ids:
            train_records.append(record)

        elif record_id in test_ids:
            test_records.append(record)

    # ---------------------------------------------------
    # Save FASTA
    # ---------------------------------------------------
    train_fasta = output_dir / "train.fasta"
    test_fasta = output_dir / "test.fasta"

    SeqIO.write(train_records, train_fasta, "fasta")
    SeqIO.write(test_records, test_fasta, "fasta")

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------
    print("=== Scenario 2 Split ===")
    print(f"Total genomes: {len(genomes)}")
    print(f"Train genomes: {len(train_ids)}")
    print(f"Test genomes: {len(test_ids)}")
    print()
    print(f"Train pickle: {train_pickle}")
    print(f"Test pickle: {test_pickle}")
    print(f"Train FASTA:  {train_fasta}")
    print(f"Test FASTA:   {test_fasta}")


if __name__ == "__main__":

    scenario2_split(
        pickle_path="raw_data/joined/MSL40_hmm_pc.pkl",
        fasta_path="raw_data/Raw/MSL40.fasta",
        output_dir="processed_data/family/scenario2_joined",
        taxonomy_rank="Family",
        test_fraction=0.3,
        random_seed=42,
    )
