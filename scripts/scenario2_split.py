# Scenario 2 Dataset Split Script

import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from Bio import SeqIO


def _split_pickle_file(source_path, destination_path, accession_ids, accession_col="Accession"):
    with open(source_path, "rb") as f:
        df = pickle.load(f)

    if accession_col not in df.columns:
        raise KeyError(f"Column '{accession_col}' not found in {source_path}")

    subset_df = df[df[accession_col].isin(accession_ids)].copy()
    subset_df.to_pickle(destination_path)


def _split_sequence_file(source_path, destination_path, accession_ids):
    records = [
        record
        for record in SeqIO.parse(source_path, "fasta")
        if record.id in accession_ids
    ]
    SeqIO.write(records, destination_path, "fasta")


def _split_additional_testing_files(
    additional_testing_input_dir,
    output_dir,
    accession_ids,
    accession_col="Accession",
):
    if additional_testing_input_dir is None:
        return []

    additional_testing_input_dir = Path(additional_testing_input_dir)
    additional_output_dir = output_dir / "additional_testing"
    additional_output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []

    for source_path in sorted(additional_testing_input_dir.iterdir()):
        if not source_path.is_file():
            continue

        suffix = source_path.suffix.lower()
        destination_path = additional_output_dir / source_path.name

        if suffix in {".pkl", ".pickle"}:
            _split_pickle_file(
                source_path,
                destination_path,
                accession_ids,
                accession_col=accession_col,
            )
        elif suffix in {".fasta", ".fa", ".faa"}:
            _split_sequence_file(source_path, destination_path, accession_ids)
        else:
            continue

        written_files.append(destination_path)

    return written_files


def scenario2_split(
    pickle_path,
    fasta_path,
    output_dir,
    taxonomy_rank="Family",
    test_fraction=0.3,
    random_seed=42,
    faa_path=None,
    additional_testing_input_dir=None,
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
        - Optional FAA and additional testing files are split using the final
          test genomes.
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
    sequence_inputs = [fasta_path]
    if faa_path is not None:
        sequence_inputs.append(faa_path)

    sequence_outputs = []

    for sequence_input in sequence_inputs:
        source_path = Path(sequence_input)
        output_suffix = source_path.suffix or ".fasta"

        train_sequence_path = output_dir / f"train{output_suffix}"
        test_sequence_path = output_dir / f"test{output_suffix}"

        _split_sequence_file(source_path, train_sequence_path, train_ids)
        _split_sequence_file(source_path, test_sequence_path, test_ids)

        sequence_outputs.extend([train_sequence_path, test_sequence_path])

    additional_output_files = _split_additional_testing_files(
        additional_testing_input_dir,
        output_dir,
        test_ids,
        accession_col=accession_col,
    )

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

    for sequence_output in sequence_outputs:
        print(f"Sequence file: {sequence_output}")

    if additional_output_files:
        print(f"Additional testing files: {len(additional_output_files)}")
        for additional_file in additional_output_files:
            print(f"  {additional_file}")


if __name__ == "__main__":

    scenario2_split(
        pickle_path="raw_data/joined/MSL40_hmm_pc.pkl",
        fasta_path="raw_data/Raw/MSL40.fasta",
        output_dir="processed_data/family/scenario2_joined",
        taxonomy_rank="Family",
        test_fraction=0.3,
        random_seed=42,
        faa_path=None,
        additional_testing_input_dir=None,
    )
