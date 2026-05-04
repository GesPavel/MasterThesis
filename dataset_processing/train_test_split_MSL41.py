import pandas as pd
import random
from sklearn.model_selection import train_test_split as sk_train_test_split


def split_dataset_msl41(
    df_primary: pd.DataFrame,
    df_secondary: pd.DataFrame,
    taxonomy_rank: str,
    novelty_test_fraction: float,
    random_test_fraction: float,
    true_train_fraction: float,
    calibration_fraction: float,
    novelty_fit_fraction: float,
    random_seed: int = 42,
):
    random.seed(random_seed)

    ID_COL = "Accession"

    # -----------------------------
    # 0. Check fractions sum to 1
    # -----------------------------
    total_fraction = (
        novelty_test_fraction
        + random_test_fraction
        + true_train_fraction
        + calibration_fraction
        + novelty_fit_fraction
    )
    assert abs(total_fraction - 1.0) < 1e-6, "Fractions must sum to 1."

    # -----------------------------
    # 1. Filter unknown taxa in MSL41
    # -----------------------------
    df_primary = df_primary[df_primary[taxonomy_rank].notna()]
    df_primary = df_primary[df_primary[taxonomy_rank] != "Unknown"]

    # -----------------------------
    # 2. Find overlap (train) and new (test) genomes
    # -----------------------------
    primary_ids = set(df_primary[ID_COL].unique())
    secondary_ids = set(df_secondary[ID_COL].unique())
    overlap_ids = primary_ids & secondary_ids

    df_overlap = df_primary[df_primary[ID_COL].isin(overlap_ids)].copy()
    df_new = df_primary[~df_primary[ID_COL].isin(overlap_ids)].copy()

    if df_overlap.empty:
        raise ValueError("No overlapping genomes between MSL41 and MSL40.")

    total_genomes = len(df_primary)
    print(f"Total genomes after filtering (MSL41): {total_genomes}")
    print(f"Overlap genomes for training: {len(df_overlap)}")
    print(f"New genomes for testing: {len(df_new)}")

    # -----------------------------
    # 3. Compute taxon sizes + buckets (overlap only)
    # -----------------------------
    taxon_sizes = df_overlap.groupby(taxonomy_rank).size().sort_values(ascending=False)

    total = taxon_sizes.sum()
    cum_frac = taxon_sizes.cumsum() / total

    large = taxon_sizes[cum_frac <= 0.2]
    medium = taxon_sizes[(cum_frac > 0.2) & (cum_frac <= 0.5)]
    small = taxon_sizes[cum_frac > 0.5]

    # -----------------------------
    # 4. Select novelty taxa (overlap only)
    # -----------------------------
    target_size = int(len(df_overlap) * novelty_test_fraction)

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

            if bucket_size >= bucket_target or current_size >= target_size:
                break

        if current_size >= target_size:
            break

    novel_taxa = set(selected_taxa)
    seen_taxa = set(df_overlap[taxonomy_rank]) - novel_taxa

    # -----------------------------
    # 5. Build novelty_holdout (overlap only)
    # -----------------------------
    novelty_holdout = df_overlap[df_overlap[taxonomy_rank].isin(novel_taxa)].copy()

    # -----------------------------
    # 6. Seen-only dataframe (overlap only)
    # -----------------------------
    seen_df = df_overlap[df_overlap[taxonomy_rank].isin(seen_taxa)].copy()

    # -----------------------------
    # 7. Train pool (seen taxa only)
    # -----------------------------
    train_pool = seen_df

    # -----------------------------
    # 8. Split train pool into:
    #    true_train, calibration, novelty_fit_seen
    # -----------------------------
    remaining_fraction = (
        true_train_fraction + calibration_fraction + novelty_fit_fraction
    )

    rel_true = true_train_fraction / remaining_fraction
    rel_calib = calibration_fraction / remaining_fraction
    rel_novel_fit = novelty_fit_fraction / remaining_fraction

    genomes = train_pool[ID_COL].unique()

    true_train_genomes, temp = sk_train_test_split(
        genomes,
        test_size=(1 - rel_true),
        random_state=random_seed
    )

    calib_genomes, novelty_fit_seen_genomes = sk_train_test_split(
        temp,
        test_size=rel_novel_fit / (rel_calib + rel_novel_fit),
        random_state=random_seed
    )

    true_train = train_pool[train_pool[ID_COL].isin(true_train_genomes)]
    calibration = train_pool[train_pool[ID_COL].isin(calib_genomes)]
    novelty_fit_seen = train_pool[
        train_pool[ID_COL].isin(novelty_fit_seen_genomes)
    ]

    # -----------------------------
    # 9. Build novelty_fit (50% seen / 50% unseen)
    # -----------------------------
    n_seen = len(novelty_fit_seen)

    novelty_fit_unseen = novelty_holdout.sample(
        n=n_seen,
        random_state=random_seed,
        replace=len(novelty_holdout) < n_seen
    )

    novelty_fit = pd.concat(
        [novelty_fit_seen, novelty_fit_unseen],
        ignore_index=True
    )

    # -----------------------------
    # 10. Determine known taxa for downstream usage
    # -----------------------------
    known_taxa = set(true_train[taxonomy_rank]) \
                 | set(calibration[taxonomy_rank]) \
                 | set(novelty_fit_seen[taxonomy_rank])
    known_taxa_df = pd.DataFrame({"taxon": list(known_taxa)})

    # -----------------------------
    # 11. Build random/novelty tests from new genomes
    # -----------------------------
    novelty_mask = ~df_new[taxonomy_rank].isin(known_taxa)
    novelty_test = df_new[novelty_mask].copy()
    random_test = df_new[~novelty_mask].copy()

    # -----------------------------
    # 12. Print summary
    # -----------------------------
    def frac(x):
        return f"{len(x)} ({len(x)/total_genomes:.2%})"

    print("\n--- FINAL SPLIT ---")
    print("Novelty test:", frac(novelty_test))
    print("Random test:", frac(random_test))
    print("True train:", frac(true_train))
    print("Calibration:", frac(calibration))
    print("Novelty fit:", frac(novelty_fit))

    # -----------------------------
    # 13. Output
    # -----------------------------
    return (
        novelty_test,
        random_test,
        true_train,
        calibration,
        novelty_fit,
        known_taxa_df
    )
