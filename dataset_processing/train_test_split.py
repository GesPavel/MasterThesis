import pandas as pd
import random
from sklearn.model_selection import train_test_split as sk_train_test_split


def split_dataset(
    df,
    taxonomy_rank,
    novelty_fraction=0.10,
    random_test_fraction=0.15,
    true_train_fraction=0.7,
    calibration_val_fraction=0.15,
    novelty_fit_val_fraction=0.15,
    random_seed=42
):
    random.seed(random_seed)

    # -----------------------------
    # 1. Filter Unknown taxa
    # -----------------------------
    df = df[df[taxonomy_rank].notna()]
    df = df[df[taxonomy_rank] != 'Unknown']
    total_genomes = len(df)

    print(f"    After filtering Unknown {taxonomy_rank}s, total genomes: {total_genomes}")

    # -----------------------------
    # 2. Compute taxon sizes + buckets
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
    # 3. Select novelty taxa (bucket-aware)
    # -----------------------------
    target_size = int(total_genomes * novelty_fraction)

    selected_taxa = []
    current_size = 0

    bucket_plan = [
        (large, 0.3),
        (medium, 0.3),
        (small, 0.4),
    ]

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

        for taxon in bucket_taxa:
            selected_taxa.append(taxon)
            taxon_size = taxon_sizes[taxon]

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
        f"total genomes in novelty set: {current_size} ({current_size / total_genomes:.2%})"
    )

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
    print(f"    Initial train set size: {len(train_set)} genomes")

    # -----------------------------
    # 6.1. Further split the train set
    # -----------------------------
    ID_COL = 'Accession'
    
    assert abs(true_train_fraction + calibration_val_fraction + novelty_fit_val_fraction - 1.0) < 1e-6, \
        "Train/val/test fractions must sum to 1."

    all_train_genomes = train_set[ID_COL].unique()

    true_train_genomes, temp_genomes = sk_train_test_split(
        all_train_genomes,
        test_size=(1 - true_train_fraction),
        random_state=random_seed
    )

    relative_test_size = novelty_fit_val_fraction / (calibration_val_fraction + novelty_fit_val_fraction)

    calibration_val_genomes, model_fit_val_genomes = sk_train_test_split(
        temp_genomes,
        test_size=relative_test_size,
        random_state=random_seed
    )
    
    true_train = train_set[train_set[ID_COL].isin(true_train_genomes)]
    calibration_val = train_set[train_set[ID_COL].isin(calibration_val_genomes)]
    model_fit_val = train_set[train_set[ID_COL].isin(model_fit_val_genomes)]

    # -----------------------------
    # 7. Print sanity checks
    # -----------------------------
    print("Total genomes:", total_genomes)
    print("Novelty test:", len(novelty_test), f"({len(novelty_test) / total_genomes:.2%})")
    print("Random test:", len(random_test), f"({len(random_test) / total_genomes:.2%})")
    print("True train set:", len(true_train), f"({len(true_train) / total_genomes:.2%})")
    print("Calibration val set:", len(calibration_val), f"({len(calibration_val) / total_genomes:.2%})")
    print("Model fit val set:", len(model_fit_val), f"({len(model_fit_val) / total_genomes:.2%})")

    print(f"\nTotal number of taxa ranked as {taxonomy_rank} in novelty set:", len(selected_taxa))
    print("Overlap check:",
          len(set(novelty_test.index) & set(random_test.index)) == 0)

    # -----------------------------
    # 8. Reduce output columns
    # -----------------------------
    
    novelty_out = novelty_test[[ID_COL, taxonomy_rank]].copy()
    novelty_out['bucket'] = novelty_out[taxonomy_rank].map(taxon_to_bucket)

    random_out = random_test[[ID_COL, taxonomy_rank]].copy()
    train_out = true_train[[ID_COL, taxonomy_rank]].copy()
    calibration_val_out = calibration_val[[ID_COL, taxonomy_rank]].copy()
    model_fit_val_out = model_fit_val[[ID_COL, taxonomy_rank]].copy()

    return novelty_out, random_out, train_out, calibration_val_out, model_fit_val_out