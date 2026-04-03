import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from dataset_processing.hmm_data_processing import compute_features


def _aggregate_scores(df_probabilities, method="mean", topk=None):
    print("    Aggregating taxon scores...")
    start = time.time()

    grouped = []

    grouped_iter = df_probabilities.groupby(
        ["genome_test", "true_taxon", "taxon_train"]
    )

    for (genome_test, true_taxon, taxon_train), group in tqdm(
        grouped_iter,
        desc="    Aggregating groups"
    ):
        probs = group["probability"].sort_values(ascending=False).to_numpy()

        if topk is not None:
            probs = probs[:topk]

        if len(probs) == 0:
            continue

        if method == "mean":
            score = np.mean(probs)
        elif method == "median":
            score = np.median(probs)
        elif method == "max":
            score = np.max(probs)
        elif method == "logit_sum":
            eps = 1e-6
            probs = np.clip(probs, eps, 1 - eps)
            score = np.sum(np.log(probs / (1 - probs)))
        elif method == "topk_mean":
            score = np.mean(probs)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        grouped.append({
            "genome_test": genome_test,
            "true_taxon": true_taxon,
            "candidate_taxon": taxon_train,
            "score": score
        })

    df_scores = pd.DataFrame(grouped)

    print(f"    Aggregation done in {time.time() - start:.2f}s")
    print(f"    Aggregated score rows: {len(df_scores):,}")

    return df_scores


def _summarize_per_genome(df_scores, known_taxon=True):
    print("    Summarizing per genome...")
    start = time.time()

    rows = []

    for genome_test, group in tqdm(df_scores.groupby("genome_test"), desc="    Summarizing genomes"):
        group_sorted = group.sort_values("score", ascending=False).reset_index(drop=True)

        true_taxon = group_sorted["true_taxon"].iloc[0]
        predicted_taxon = group_sorted["candidate_taxon"].iloc[0]
        top_score = group_sorted["score"].iloc[0]

        candidate_taxa = group_sorted["candidate_taxon"].tolist()
        candidate_scores = group_sorted["score"].tolist()

        row = {
            "genome_test": genome_test,
            "true_taxon": true_taxon,
            "predicted_taxon": predicted_taxon,
            "top_score": top_score,
            "known_taxon": known_taxon,
        }

        if known_taxon:
            if true_taxon in candidate_taxa:
                true_rank = candidate_taxa.index(true_taxon) + 1
                true_score = candidate_scores[true_rank - 1]
            else:
                true_rank = None
                true_score = None

            row.update({
                "true_rank": true_rank,
                "true_score": true_score,
                "top1_correct": int(true_rank == 1) if true_rank is not None else 0,
                "top2_correct": int(true_rank is not None and true_rank <= 2),
                "top3_correct": int(true_rank is not None and true_rank <= 3),
            })
        else:
            row.update({
                "true_rank": None,
                "true_score": None,
                "top1_correct": None,
                "top2_correct": None,
                "top3_correct": None,
            })

        rows.append(row)

    df_summary = pd.DataFrame(rows)

    print(f"    Genome summarization done in {time.time() - start:.2f}s")
    print(f"    Genome summaries: {len(df_summary):,}")

    return df_summary


def assign_taxa(
    df_train,
    df_test,
    model,
    taxon_rank,
    feature_config,
    aggregation_config,
    known_taxon=True
):
    total_start = time.time()

    print("    Building all test × train pairs...")
    start = time.time()

    # -----------------------------
    # 1. Build all test x train pairs
    # -----------------------------
    df_pairs = df_train[["Accession"]].merge(df_test[["Accession"]], how="cross")
    df_pairs = df_pairs.rename(columns={
        "Accession_x": "genome1",
        "Accession_y": "genome2"
    })

    print(f"    Pair dataframe shape: {df_pairs.shape}")
    print(f"    Pair generation done in {time.time() - start:.2f}s")

    # -----------------------------
    # 2. Build genome lookup
    # -----------------------------
    print("    Building genome lookup...")
    start = time.time()

    df_all = pd.concat([df_train, df_test], ignore_index=True)

    genome_dict = {
        row["Accession"]: set(row["hmms_hits"])
        for _, row in df_all.iterrows()
    }

    print(f"    Genome lookup built for {len(genome_dict):,} genomes in {time.time() - start:.2f}s")

    # -----------------------------
    # 3. Compute pairwise features
    # -----------------------------
    print("    Computing pairwise features...")
    start = time.time()

    X, _, genome_list_train, genome_list_test = compute_features(
        pairs_df=df_pairs,
        genome_dict=genome_dict,
        feature_config=feature_config,
        process_target_var=False
    )

    print(f"    Feature matrix shape: {X.shape}")
    print(f"    Feature computation done in {time.time() - start:.2f}s")

    # -----------------------------
    # 4. Predict pairwise probabilities
    # -----------------------------
    print("    Predicting pairwise probabilities...")
    start = time.time()

    probabilities = model.predict_proba(X)[:, 1]

    print(f"    Probability prediction done in {time.time() - start:.2f}s")

    print("    Building probability dataframe...")
    start = time.time()

    train_taxon_lookup = df_train.set_index("Accession")[taxon_rank].to_dict()
    test_taxon_lookup = df_test.set_index("Accession")[taxon_rank].to_dict()

    taxon_train_list = [train_taxon_lookup[g] for g in genome_list_train]
    taxon_test_list = [test_taxon_lookup[g] for g in genome_list_test]

    df_probabilities = pd.DataFrame({
        "genome_test": genome_list_test,
        "genome_train": genome_list_train,
        "taxon_train": taxon_train_list,
        "true_taxon": taxon_test_list,
        "probability": probabilities
    })

    print(f"    Probability dataframe shape: {df_probabilities.shape}")
    print(f"    Probability dataframe built in {time.time() - start:.2f}s")

    # -----------------------------
    # 5. Aggregate family/taxon scores
    # -----------------------------
    agg_method = aggregation_config["method"]
    topk_values = aggregation_config["topk_values"]

    all_results = []

    for topk in topk_values:
        k_label = "all" if topk is None else f"top{topk}"
        print(f"\n    Running aggregation: method={agg_method}, k={k_label}")

        df_scores = _aggregate_scores(
            df_probabilities,
            method=agg_method,
            topk=topk
        )

        df_summary = _summarize_per_genome(
            df_scores,
            known_taxon=known_taxon
        )

        df_summary["aggregation_method"] = agg_method
        df_summary["aggregation_k"] = k_label

        all_results.append(df_summary)

    df_final = pd.concat(all_results, ignore_index=True)

    print(f"\n    Assignment finished in {time.time() - total_start:.2f}s")
    print(f"    Final result rows: {len(df_final):,}")

    return df_final