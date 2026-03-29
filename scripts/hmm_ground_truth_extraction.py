import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from dataset_processing import DATASET_PICKLE_PATH, get_dataset_filepath, get_intermediate_output_path
from dataset_processing.util import load_filtered_dataframe


# -----------------------------
# 1. Data preparation
# -----------------------------


def add_hmm_sets(df):
    """
    Convert hmms_hits lists into sets for fast intersection.
    """
    df["hmm_set"] = df["hmms_hits"].apply(lambda x: set(x) if isinstance(x, list) else set())
    return df


# -----------------------------
# 2. Similarity computation
# -----------------------------
def fast_similarity(set_i, set_j):
    """
    Compute simple overlap between two HMM sets.
    Used as a cheap proxy for similarity.
    """
    return len(set_i & set_j)


def compute_all_similarities(df, idx_i, indices):
    """
    Compute similarity of one genome to all others.

    Returns:
        List of tuples (idx_j, similarity), sorted descending.
    """
    set_i = df.loc[idx_i, "hmm_set"]

    sims = []
    for idx_j in indices:
        if idx_i == idx_j:
            continue

        set_j = df.loc[idx_j, "hmm_set"]
        sim = fast_similarity(set_i, set_j)

        sims.append((idx_j, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims


# -----------------------------
# 3. Pair generation
# -----------------------------
def generate_neighbor_pairs(df, idx_i, sims, rank, k_neighbors):
    """
    Generate top-k nearest neighbor pairs for one genome.

    These are informative (hard positives/negatives).
    """
    pairs = []

    tax_i = df.loc[idx_i, rank]
    name_i = df.loc[idx_i, "Accession"]

    for idx_j, _ in sims[:k_neighbors]:
        tax_j = df.loc[idx_j, rank]
        name_j = df.loc[idx_j, "Accession"]

        y = int(tax_i == tax_j)
        pairs.append((name_i, name_j, y))

    return pairs


def generate_random_pairs(df, idx_i, indices, rank, k_random):
    """
    Generate random pairs for diversity (easy negatives).
    """
    pairs = []

    tax_i = df.loc[idx_i, rank]
    name_i = df.loc[idx_i, "Accession"]

    random_js = random.sample(indices, k_random)

    for idx_j in random_js:
        if idx_i == idx_j:
            continue

        tax_j = df.loc[idx_j, rank]
        name_j = df.loc[idx_j, "Accession"]

        y = int(tax_i == tax_j)
        pairs.append((name_i, name_j, y))

    return pairs


# -----------------------------
# 4. Main builder
# -----------------------------
def build_pairs(
    df,
    output_file,
    rank="Family",
    k_neighbors=10,
    k_random=10,
):
    """
    Build pair dataset:
    - For each genome:
        - top-k similar genomes (informative)
        - k random genomes (diversity)
    - Save as CSV

    Output columns:
        genome1, genome2, y
    """
    indices = df.index.tolist()
    output_path = Path(output_file)

    if output_path.exists():
        output_path.unlink()

    buffer = []

    for idx_i in tqdm(indices):
        sims = compute_all_similarities(df, idx_i, indices)

        buffer.extend(
            generate_neighbor_pairs(df, idx_i, sims, rank, k_neighbors)
        )

        buffer.extend(
            generate_random_pairs(df, idx_i, indices, rank, k_random)
        )

    pairs_df = pd.DataFrame(buffer, columns=["genome1", "genome2", "y"])
    pairs_df.to_csv(output_file, index=False)


# -----------------------------
# 5. Entry point
# -----------------------------
def main():
    TAXONOMY_RANK = 'Family' # Genus, Order, etc.
    hmm_dataset_path = DATASET_PICKLE_PATH
    train_genome_set_path = get_dataset_filepath(TAXONOMY_RANK, 'train')
    output_path = get_intermediate_output_path(TAXONOMY_RANK, 'hmm', 'pairs')

    print("Loading and filtering data...")
    df = load_filtered_dataframe(
        hmm_dataset_path, train_genome_set_path
    )

    print("Preparing HMM sets...")
    df = add_hmm_sets(df)

    print("Building pairs...")
    build_pairs(
        df,
        output_path,
        rank="Family",
        k_neighbors=10,
        k_random=10,
    )

    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()