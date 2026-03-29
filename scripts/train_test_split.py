import os

import pandas as pd
import random

from dataset_processing import DATASET_PICKLE_PATH, get_dataset_filepath


def split_dataset(
        df,
        taxonomy_rank,
        novelty_fraction=0.10,
        random_test_fraction=0.15,
        random_seed=42
):
    random.seed(random_seed)

    # -----------------------------
    # 1. Filter Unknown families
    # -----------------------------
    df = df[df[taxonomy_rank].notna()]
    df = df[df[taxonomy_rank] != 'Unknown']
    total_genomes = len(df)

    print(f"    After filtering Unknown {taxonomy_rank}s, total genomes: {total_genomes}")

    # -----------------------------
    # 2. Compute taxons sizes + buckets
    # -----------------------------
    taxon_sizes = df.groupby(taxonomy_rank).size().sort_values(ascending=False)

    total = taxon_sizes.sum()
    cum_frac = taxon_sizes.cumsum() / total

    # Define buckets
    large = taxon_sizes[cum_frac <= 0.2]
    medium = taxon_sizes[(cum_frac > 0.2) & (cum_frac <= 0.5)]
    small = taxon_sizes[cum_frac > 0.5]

    print(f"    Buckets: large={len(large)}, medium={len(medium)}, small={len(small)}")

    # -----------------------------
    # 3. Select novelty taxons (bucket-aware)
    # -----------------------------
    target_size = int(total_genomes * novelty_fraction)

    selected_taxa = []
    current_size = 0

    # proportions you can tune
    bucket_plan = [
        (large, 0.3),
        (medium, 0.3),
        (small, 0.4),
    ]

    # Map each taxon → bucket label
    taxon_to_bucket = {}

    for t in large.index:
        taxon_to_bucket[t] = 'large'
    for t in medium.index:
        taxon_to_bucket[t] = 'medium'
    for t in small.index:
        taxon_to_bucket[t] = 'small'

    for bucket, fraction in bucket_plan:
        bucket_taxa = bucket.index.tolist()
        random.shuffle(bucket_taxa)

        bucket_target = int(target_size * fraction)

        bucket_size = 0

        for fam in bucket_taxa:
            selected_taxa.append(fam)
            taxon_size = taxon_sizes[fam]

            current_size += taxon_size
            bucket_size += taxon_size

            if bucket_size >= bucket_target or current_size >= target_size:
                break

        if current_size >= target_size:
            break
    # -----------------------------
    # 4. Build novelty test set
    # -----------------------------
    novelty_mask = df[taxonomy_rank].isin(selected_taxa)

    novelty_test = df[novelty_mask].copy()
    remaining = df[~novelty_mask].copy()
    print(
        f"    Selected {len(selected_taxa)} units of rank {taxonomy_rank} for novelty test, "
        f"total genomes in novelty set: {current_size} ({current_size / total_genomes:.2%})")

    # -----------------------------
    # 5. Stratified random test set
    # -----------------------------
    random_test = remaining.groupby(
        taxonomy_rank,
        group_keys=False
    ).sample(
        frac=random_test_fraction,
        random_state=random_seed
    )

    # -----------------------------
    # 6. Train set
    # -----------------------------
    train_set = remaining.drop(random_test.index)
    print(f"    Train set size: {len(train_set)} genomes")
    # -----------------------------
    # 7. Print sanity checks
    # -----------------------------
    print("Total genomes:", total_genomes)
    print("Novelty test:", len(novelty_test), f"({len(novelty_test) / total_genomes:.2%})")
    print("Random test:", len(random_test), f"({len(random_test) / total_genomes:.2%})")
    print("Train set:", len(train_set), f"({len(train_set) / total_genomes:.2%})")

    print(f"\nTotal number of taxa ranked as {taxonomy_rank} in novelty set:", len(selected_taxa))
    print("Overlap check:",
          len(set(novelty_test.index) & set(random_test.index)) == 0)
    # -----------------------------
    # 8. Reduce output columns
    # -----------------------------
    ID_COL = 'Accession'

    # novelty → include bucket
    novelty_out = novelty_test[[ID_COL, taxonomy_rank]].copy()
    novelty_out['bucket'] = novelty_out[taxonomy_rank].map(taxon_to_bucket)
    print(random_test.columns.tolist())
    # random + train → no bucket
    random_out = random_test[[ID_COL, taxonomy_rank]].copy()
    train_out = train_set[[ID_COL, taxonomy_rank]].copy()

    return novelty_out, random_out, train_out


def main():
    TAXONOMY_RANK = 'Family'  # Genus, Order, etc.

    output_path_novelty = get_dataset_filepath(TAXONOMY_RANK, 'novelty')
    output_path_random = get_dataset_filepath(TAXONOMY_RANK, 'random')
    output_path_train = get_dataset_filepath(TAXONOMY_RANK, 'train')

    # -----------------------------
    # Load data
    # -----------------------------
    print("Loading dataset from pickle...")
    df = pd.read_pickle(DATASET_PICKLE_PATH)

    # -----------------------------
    # Split
    # -----------------------------
    novelty_fraction = 0.10
    random_test_fraction = 0.15
    print("Splitting dataset with novelty fraction", f"{novelty_fraction:.2%}", "and random test fraction",
          f"{random_test_fraction:.2%}")

    novelty_test, random_test, train_set = split_dataset(df, TAXONOMY_RANK, novelty_fraction, random_test_fraction)

    # -----------------------------
    # Save CSVs
    # -----------------------------
    novelty_test.to_csv(output_path_novelty, index=False)
    random_test.to_csv(output_path_random, index=False)
    train_set.to_csv(output_path_train, index=False)

    print("\nSaved files:")
    print("-", output_path_novelty)
    print("-", output_path_random)
    print("-", output_path_train)


if __name__ == '__main__':
    main()
