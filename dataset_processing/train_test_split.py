import pickle
import random

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset_split(
    train_df,
    test_df,
    taxonomy_rank,
    calibration_fraction,
    novelty_fit_fraction,
    random_seed=42,
):
    """
    Load pre-generated dataset split and subdivide train into:

        - true_train
        - calibration
        - novelty_fit

    novelty_fit contains:
        - seen taxa (present in training)
        - unseen taxa (withheld from training)

    Returns:
        test_df
        calibration_df
        novelty_fit_df
        known_taxa_df
        true_train_df
    """

    random.seed(random_seed)

    ID_COL = "Accession"

    # ---------------------------------------------------
    # Filter unknown taxa
    # ---------------------------------------------------
    train_df = train_df[train_df[taxonomy_rank].notna()]
    train_df = train_df[train_df[taxonomy_rank] != "Unknown"]

    # ---------------------------------------------------
    # Select unseen novelty-fit taxa
    # ---------------------------------------------------
    taxa = train_df[taxonomy_rank].unique().tolist()

    n_unseen_taxa = max(
        1,
        int(len(taxa) * novelty_fit_fraction * 0.5)
    )

    unseen_novelty_taxa = set(
        random.sample(taxa, n_unseen_taxa)
    )

    # ---------------------------------------------------
    # Build unseen novelty-fit subset
    # ---------------------------------------------------
    novelty_fit_unseen = train_df[
        train_df[taxonomy_rank].isin(unseen_novelty_taxa)
    ].copy()

    remaining_train = train_df[
        ~train_df[taxonomy_rank].isin(unseen_novelty_taxa)
    ].copy()

    # ---------------------------------------------------
    # Split remaining genomes into:
    #   true_train / calibration / novelty_fit_seen
    # ---------------------------------------------------
    genomes = remaining_train[ID_COL].unique()

    temp_fraction = calibration_fraction + novelty_fit_fraction

    true_train_genomes, temp_genomes = train_test_split(
        genomes,
        test_size=temp_fraction,
        random_state=random_seed,
    )

    relative_novelty = (
        novelty_fit_fraction / temp_fraction
    )

    calibration_genomes, novelty_fit_seen_genomes = train_test_split(
        temp_genomes,
        test_size=relative_novelty,
        random_state=random_seed,
    )

    # ---------------------------------------------------
    # Build final dataframes
    # ---------------------------------------------------
    true_train_df = remaining_train[
        remaining_train[ID_COL].isin(true_train_genomes)
    ].copy()

    calibration_df = remaining_train[
        remaining_train[ID_COL].isin(calibration_genomes)
    ].copy()

    novelty_fit_seen = remaining_train[
        remaining_train[ID_COL].isin(novelty_fit_seen_genomes)
    ].copy()

    # ---------------------------------------------------
    # Balance seen/unseen novelty-fit
    # ---------------------------------------------------
    n_unseen = len(novelty_fit_unseen)

    novelty_fit_seen = novelty_fit_seen.sample(
        n=n_unseen,
        replace=len(novelty_fit_seen) < n_unseen,
        random_state=random_seed,
    )

    novelty_fit_df = pd.concat(
        [novelty_fit_seen, novelty_fit_unseen],
        ignore_index=True,
    )

    # ---------------------------------------------------
    # Determine known taxa for downstream usage
    # ---------------------------------------------------
    known_taxa = (
        set(true_train_df[taxonomy_rank])
        | set(calibration_df[taxonomy_rank])
        | set(novelty_fit_seen[taxonomy_rank])
    )

    known_taxa_df = pd.DataFrame({
        "taxon": list(known_taxa)
    })

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------
    unseen_taxa = set(novelty_fit_unseen[taxonomy_rank])

    print()
    print("=== Dataset Load Split ===")
    print()

    print(f"True-train genomes: {len(true_train_df)}")
    print(f"Calibration genomes: {len(calibration_df)}")
    print(f"Novelty-fit genomes: {len(novelty_fit_df)}")
    print(f"External test genomes: {len(test_df)}")
    print()

    print(f"Known taxa: {len(known_taxa)}")
    print(f"Novelty-fit unseen taxa: {len(unseen_taxa)}")
    print()

    return (
        test_df,
        true_train_df,
        calibration_df,
        novelty_fit_df,
        known_taxa_df,
    )