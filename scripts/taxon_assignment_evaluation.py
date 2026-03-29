import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

from dataset_processing import DATASET_PICKLE_PATH, get_evaluation_output_dir, get_dataset_filepath, \
    get_intermediate_output_path
from dataset_processing.hmm_data_processing import build_name_to_hmm_set_dict, compute_features
from dataset_processing.util import load_filtered_dataframe


def evaluate_taxon_assignment(df_agg, df_train, taxonomical_rank):
    """
    Evaluate aggregated taxon assignment performance.

    Expected columns in df_agg:
        - genome_test
        - taxon_test   (true taxon of test genome)
        - taxon_train  (candidate predicted taxon)
        - probability  (aggregated mean probability)

    df_train is used to determine train family sizes.
    """

    seen_rows = []
    unseen_rows = []

    # how many train genomes each taxon has
    train_taxon_sizes = df_train[taxonomical_rank].value_counts().to_dict()

    # taxa that exist in training candidates
    train_taxa = set(df_agg["taxon_train"].unique())

    for genome_test, group in df_agg.groupby("genome_test"):
        group = group.sort_values("probability", ascending=False).reset_index(drop=True)

        true_taxon = group["taxon_test"].iloc[0]
        top_taxon = group["taxon_train"].iloc[0]
        top_prob = group["probability"].iloc[0]

        if true_taxon in train_taxa:
            matches = group.index[group["taxon_train"] == true_taxon].tolist()

            if len(matches) == 0:
                true_rank = np.nan
                true_prob = np.nan
            else:
                true_rank = matches[0] + 1
                true_prob = group.loc[matches[0], "probability"]

            true_taxon_size = train_taxon_sizes.get(true_taxon, 0)

            seen_rows.append({
                "genome_test": genome_test,
                "true_taxon": true_taxon,
                "true_taxon_train_size": true_taxon_size,
                "top_taxon": top_taxon,
                "top_prob": top_prob,
                "true_rank": true_rank,
                "true_prob": true_prob,
                "top1_correct": true_rank == 1 if pd.notna(true_rank) else False,
                "top2_correct": true_rank <= 2 if pd.notna(true_rank) else False,
                "top3_correct": true_rank <= 3 if pd.notna(true_rank) else False,
            })

        else:
            unseen_rows.append({
                "genome_test": genome_test,
                "true_taxon": true_taxon,
                "top_taxon": top_taxon,
                "top_prob": top_prob,
            })

    df_seen = pd.DataFrame(seen_rows)
    df_unseen = pd.DataFrame(unseen_rows)

    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    # -------------------------
    # Seen taxa report
    # -------------------------
    print("\n[1] TEST GENOMES WITH TRUE TAXON PRESENT IN TRAIN")
    print("-" * 60)
    print(f"Number of genomes: {len(df_seen)}")

    if len(df_seen) > 0:
        print(f"Top-1 accuracy: {df_seen['top1_correct'].mean():.4f}")
        print(f"Top-2 accuracy: {df_seen['top2_correct'].mean():.4f}")
        print(f"Top-3 accuracy: {df_seen['top3_correct'].mean():.4f}")
        print(f"Mean rank of true taxon: {df_seen['true_rank'].mean():.4f}")
        print(f"Median rank of true taxon: {df_seen['true_rank'].median():.4f}")
        print(f"Mean probability of true taxon: {df_seen['true_prob'].mean():.4f}")
        print(f"Median probability of true taxon: {df_seen['true_prob'].median():.4f}")
        print(f"Mean probability of top predicted taxon: {df_seen['top_prob'].mean():.4f}")
    else:
        print("No seen-taxon genomes found.")

    # -------------------------
    # Unseen taxa report
    # -------------------------
    print("\n[2] TEST GENOMES WITH TRUE TAXON ABSENT FROM TRAIN")
    print("-" * 60)
    print(f"Number of genomes: {len(df_unseen)}")

    if len(df_unseen) > 0:
        print(f"Mean probability of top predicted taxon: {df_unseen['top_prob'].mean():.4f}")
        print(f"Median probability of top predicted taxon: {df_unseen['top_prob'].median():.4f}")
        print(f"Max probability of top predicted taxon: {df_unseen['top_prob'].max():.4f}")
    else:
        print("No unseen-taxon genomes found.")

    # -------------------------
    # Small-family analysis
    # -------------------------
    if len(df_seen) > 0:
        print("\n[3] PERFORMANCE BY TRUE TAXON TRAIN SIZE")
        print("-" * 60)

        def size_bucket(n):
            if n == 1:
                return "1"
            elif n == 2:
                return "2"
            elif n == 3:
                return "3"
            elif n <= 5:
                return "4-5"
            elif n <= 10:
                return "6-10"
            elif n <= 20:
                return "11-20"
            else:
                return "21+"

        df_seen["size_bucket"] = df_seen["true_taxon_train_size"].apply(size_bucket)

        summary = (
            df_seen.groupby("size_bucket")
            .agg(
                n_genomes=("genome_test", "count"),
                n_families=("true_taxon", "nunique"),
                top1=("top1_correct", "mean"),
                top2=("top2_correct", "mean"),
                top3=("top3_correct", "mean"),
                mean_rank=("true_rank", "mean"),
                median_rank=("true_rank", "median"),
                mean_true_prob=("true_prob", "mean"),
            )
            .reset_index()
        )

        print(summary.to_string(index=False))

        # direct "small family" slice
        print("\n[4] VERY SMALL FAMILY SUMMARY (<= 6 TRAIN GENOMES)")
        print("-" * 60)

        df_small = df_seen[df_seen["true_taxon_train_size"] <= 6]

        if len(df_small) > 0:
            print(f"Number of genomes: {len(df_small)}")
            print(f"Number of families: {df_small['true_taxon'].nunique()}")
            print(f"Top-1 accuracy: {df_small['top1_correct'].mean():.4f}")
            print(f"Top-2 accuracy: {df_small['top2_correct'].mean():.4f}")
            print(f"Top-3 accuracy: {df_small['top3_correct'].mean():.4f}")
            print(f"Mean rank: {df_small['true_rank'].mean():.4f}")
            print(f"Median rank: {df_small['true_rank'].median():.4f}")
            print(f"Mean true probability: {df_small['true_prob'].mean():.4f}")
        else:
            print("No very small-family genomes found.")

    return df_seen, df_unseen

def main():
    TAXONOMICAL_RANK = 'Family'
    MODEL_TYPE = "xgboost"

    hmm_dataset_path = DATASET_PICKLE_PATH
    train_genome_set_path = get_dataset_filepath(TAXONOMICAL_RANK, 'train')
    test_genome_set_path = get_dataset_filepath(TAXONOMICAL_RANK, 'random')
    model_path = get_intermediate_output_path(TAXONOMICAL_RANK, 'hmm', MODEL_TYPE)
    output_path = get_evaluation_output_dir(TAXONOMICAL_RANK, 'hmm')

    print("Loading and filtering data...")
    df_train = load_filtered_dataframe(
        hmm_dataset_path, train_genome_set_path
    )
    df_test = load_filtered_dataframe(
        hmm_dataset_path, test_genome_set_path
    )
    # all combinations
    df_pairs = df_train[['Accession']].merge(df_test[['Accession']], how='cross')
    df_pairs = df_pairs.rename(columns={
        "Accession_x": "genome1",
        "Accession_y": "genome2"
    })
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    print("Building genome dictionary...")
    genome_dict = build_name_to_hmm_set_dict(df_all)

    print("Computing features...")
    features, _, genome_list_train, genome_list_test = compute_features(df_pairs, genome_dict, process_target_var=False)
    print(f"Computed features for {len(features)} pairs.")

    # -----------------------------
    # Train model
    # -----------------------------
    print("Loading model...")
    model = joblib.load(model_path)

    # -----------------------------
    # Evaluation
    # -----------------------------
    print("Predicting...")

    # 1. Predict all at once
    probabilities = model.predict_proba(features)[:, 1]

    print("Building genome to taxon dictionary")
    # 2. Build fast accession -> taxon lookup dicts
    train_taxon_lookup = df_train.set_index("Accession")[TAXONOMICAL_RANK].to_dict()
    test_taxon_lookup = df_test.set_index("Accession")[TAXONOMICAL_RANK].to_dict()

    # 3. Map taxons in bulk
    taxon_train_list = [train_taxon_lookup[g] for g in genome_list_train]
    taxon_test_list = [test_taxon_lookup[g] for g in genome_list_test]

    # 4. Build dataframe directly
    df_probabilities = pd.DataFrame({
        "genome_test": genome_list_test,
        "genome_train": genome_list_train,
        "taxon_train": taxon_train_list,
        "taxon_test": taxon_test_list,
        "probability": probabilities
    })
    print("Aggregating probabilities by taxon...")
    df_agg = (
        df_probabilities
        .groupby(["genome_test", "taxon_test", "taxon_train"])["probability"]
        .agg("mean")
        .reset_index()
    )
    df_seen, df_unseen = evaluate_taxon_assignment(df_agg, df_train, TAXONOMICAL_RANK)


if __name__ == "__main__":
    main()
