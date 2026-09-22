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


def scenario3_split(
    pickle_path,
    fasta_path,
    output_dir,
    taxonomy_rank="Family",
    test_fraction=0.2,
    random_seed=42,
    faa_path=None,
    additional_testing_input_dir=None,
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
        - Optional FAA and additional testing files are split using the final
          test genomes
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
    train_pickle = output_dir / "train.pkl"
    test_pickle = output_dir / "test.pkl"

    train_df.to_pickle(train_pickle)
    test_df.to_pickle(test_pickle)

    # ---------------------------------------------------
    # Load sequence files
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

    for sequence_output in sequence_outputs:
        print(f"Sequence file: {sequence_output}")

    if additional_output_files:
        print(f"Additional testing files: {len(additional_output_files)}")
        for additional_file in additional_output_files:
            print(f"  {additional_file}")


if __name__ == "__main__":

    scenario3_split(
        pickle_path="raw_data/MSL41_complete.pkl",
        fasta_path="raw_data/MSL41.fasta",
        output_dir="genus/scenario3",
        taxonomy_rank="Genus",
        test_fraction=0.3,
        random_seed=42,
        faa_path=None,
        additional_testing_input_dir=None,
    )