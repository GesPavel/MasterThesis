import pickle
import random
from pathlib import Path

import pandas as pd
from Bio import SeqIO


def scenario3_split(
    pickle_path,
    fasta_path,
    output_dir,
    taxonomy_rank="Family",
    test_fraction=0.2,
    random_seed=42,
):
    """
    Scenario 3:
    Hold-out by taxon (true open-set novelty).

    Train:
        Genomes from a subset of taxa.

    Test:
        Genomes from taxa completely absent from train.

    Constraints:
        - No taxon overlap between train/test
        - No genome overlap between train/test
        - Balanced taxon-size sampling using buckets
    """

    random.seed(random_seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accession_col = "Accession"

    # ---------------------------------------------------
    # Load pickle dataframe
    # ---------------------------------------------------
    with open(pickle_path, "rb") as f:
        df = pickle.load(f)

    # ---------------------------------------------------
    # Filter unknown taxa
    # ---------------------------------------------------
    df = df[df[taxonomy_rank].notna()]
    df = df[df[taxonomy_rank] != "Unknown"]

    total_genomes = len(df)

    print(f"Total genomes after filtering: {total_genomes}")

    # ---------------------------------------------------
    # Compute taxon sizes + buckets
    # ---------------------------------------------------
    taxon_sizes = (
        df.groupby(taxonomy_rank)
        .size()
        .sort_values(ascending=False)
    )

    total = taxon_sizes.sum()
    cum_frac = taxon_sizes.cumsum() / total

    large = taxon_sizes[cum_frac <= 0.2]
    medium = taxon_sizes[
        (cum_frac > 0.2) & (cum_frac <= 0.5)
    ]
    small = taxon_sizes[cum_frac > 0.5]

    print(f"Large taxa: {len(large)}")
    print(f"Medium taxa: {len(medium)}")
    print(f"Small taxa: {len(small)}")

    # ---------------------------------------------------
    # Select test taxa
    # ---------------------------------------------------
    target_size = int(total_genomes * test_fraction)

    selected_taxa = []
    current_size = 0

    bucket_plan = [
        (large, 0.3),
        (medium, 0.3),
        (small, 0.4),
    ]

    for bucket, fraction in bucket_plan:

        taxa_list = bucket.index.tolist()
        random.shuffle(taxa_list)

        bucket_target = int(target_size * fraction)
        bucket_size = 0

        for taxon in taxa_list:

            selected_taxa.append(taxon)

            size = taxon_sizes[taxon]

            current_size += size
            bucket_size += size

            if (
                bucket_size >= bucket_target
                or current_size >= target_size
            ):
                break

        if current_size >= target_size:
            break

    test_taxa = set(selected_taxa)
    train_taxa = set(df[taxonomy_rank]) - test_taxa

    # ---------------------------------------------------
    # Split dataframe
    # ---------------------------------------------------
    train_df = df[df[taxonomy_rank].isin(train_taxa)].copy()
    test_df = df[df[taxonomy_rank].isin(test_taxa)].copy()

    # ---------------------------------------------------
    # Extract accession IDs
    # ---------------------------------------------------
    train_ids = set(train_df[accession_col].unique())
    test_ids = set(test_df[accession_col].unique())

    # ---------------------------------------------------
    # Save pickles
    # ---------------------------------------------------
    train_pickle = output_dir / "scenario3_train.pkl"
    test_pickle = output_dir / "scenario3_test.pkl"

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
    train_fasta = output_dir / "scenario3_train.fasta"
    test_fasta = output_dir / "scenario3_test.fasta"

    SeqIO.write(train_records, train_fasta, "fasta")
    SeqIO.write(test_records, test_fasta, "fasta")

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------
    print()
    print("=== Scenario 3 Split ===")
    print(f"Taxonomy rank: {taxonomy_rank}")
    print()

    print(f"Total taxa: {len(taxon_sizes)}")
    print(f"Train taxa: {len(train_taxa)}")
    print(f"Test taxa: {len(test_taxa)}")
    print()

    print(f"Train genomes: {len(train_ids)}")
    print(f"Test genomes: {len(test_ids)}")
    print()

    print(f"Train pickle: {train_pickle}")
    print(f"Test pickle: {test_pickle}")
    print(f"Train FASTA:  {train_fasta}")
    print(f"Test FASTA:   {test_fasta}")


if __name__ == "__main__":

    scenario3_split(
        pickle_path="raw_data/MSL41_complete.pkl",
        fasta_path="raw_data/MSL41.fasta",
        output_dir="genus/scenario3",
        taxonomy_rank="Genus",
        test_fraction=0.3,
        random_seed=42,
    )