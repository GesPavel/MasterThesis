import pickle
import random
from pathlib import Path

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


def scenario4_split(
    new_edition_pickle_path,
    old_edition_pickle_path,
    new_edition_fasta_path,
    output_dir,
    taxonomy_rank="Family",
    random_seed=42,
    new_edition_faa_path=None,
    additional_testing_input_dir=None,
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
        - Optional FAA and additional testing files are split using the final
          test genomes
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
    # Load sequence files
    # ---------------------------------------------------
    sequence_inputs = [new_edition_fasta_path]
    if new_edition_faa_path is not None:
        sequence_inputs.append(new_edition_faa_path)

    sequence_outputs = []

    for sequence_input in sequence_inputs:
        source_path = Path(sequence_input)
        output_suffix = source_path.suffix or ".fasta"

        train_sequence_path = output_dir / f"train{output_suffix}"
        test_sequence_path = output_dir / f"test{output_suffix}"

        _split_sequence_file(source_path, train_sequence_path, overlap_ids)
        _split_sequence_file(source_path, test_sequence_path, new_ids)

        sequence_outputs.extend([train_sequence_path, test_sequence_path])

    additional_output_files = _split_additional_testing_files(
        additional_testing_input_dir,
        output_dir,
        new_ids,
        accession_col=accession_col,
    )

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

    for sequence_output in sequence_outputs:
        print(f"Sequence file: {sequence_output}")

    if additional_output_files:
        print(f"Additional testing files: {len(additional_output_files)}")
        for additional_file in additional_output_files:
            print(f"  {additional_file}")


if __name__ == "__main__":

    scenario4_split(
        new_edition_pickle_path="raw_data/MSL41_complete.pkl",
        old_edition_pickle_path="raw_data/MSL40_complete.pkl",
        new_edition_fasta_path="raw_data/MSL41.fasta",
        output_dir="processed_data/genus/scenario4",
        taxonomy_rank="Genus",
        random_seed=42,
        new_edition_faa_path=None,
        additional_testing_input_dir=None,
    )