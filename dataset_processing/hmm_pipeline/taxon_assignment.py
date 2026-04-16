import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from scipy.special import softmax


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

def _get_novelty_score(df_probabilities, coeffs):
    print("Determining novelty scores...")
    a = coeffs["a"]
    b = coeffs["b"]

    rows = []

    for genome_test, group in tqdm(
            df_probabilities.groupby("genome_test"),
            desc="    Determining novelty scores"):
        probs = group["probability"].to_numpy()
        true_taxon = group["true_taxon"].iloc[0]

        eps = 1e-6
        probs = np.clip(probs, eps, 1 - eps)

        novelty_evidence = np.sum(np.log(probs / (1 - probs)))
        score = a + b * novelty_evidence

        rows.append({
            "genome_test": genome_test,
            "true_taxon": true_taxon,
            "candidate_taxon": "NOVEL",
            "score": score
        })

    return pd.DataFrame(rows)

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

        # -----------------------------------
        # Convert aggregated scores into
        # normalized taxon probabilities
        # -----------------------------------
        scores_array = np.array(candidate_scores, dtype=float)
        # softmax
        candidate_probs = np.asarray(softmax(scores_array), dtype=float)
        top_prob = float(candidate_probs[0])

        # find NOVEL prob if present
        if "NOVEL" in candidate_taxa:
            novel_idx = candidate_taxa.index("NOVEL")
            novel_prob = float(candidate_probs[novel_idx])
        else:
            novel_prob = 0.0

        row = {
            "genome_test": genome_test,
            "true_taxon": true_taxon,
            "predicted_taxon": predicted_taxon,
            "top_score": top_score,     # raw aggregated score
            "top_prob": top_prob,       # normalized probability
            "known_taxon": known_taxon,
            "novel_prob": novel_prob
        }

        # determine what the "true label" is
        if known_taxon:
            target_taxon = true_taxon
        else:
            target_taxon = "NOVEL"

        if target_taxon in candidate_taxa:
            target_rank = candidate_taxa.index(target_taxon) + 1
            target_score = candidate_scores[target_rank - 1]
            target_prob = float(candidate_probs[target_rank - 1])
        else:
            target_rank = None
            target_score = None
            target_prob = None

        row.update({
            "target_rank": target_rank,
            "target_score": target_score,
            "target_prob": target_prob,

            "top1_correct": int(target_rank == 1) if target_rank is not None else 0,
            "top2_correct": int(target_rank is not None and target_rank <= 2),
            "top3_correct": int(target_rank is not None and target_rank <= 3),
        })

        rows.append(row)

    df_summary = pd.DataFrame(rows)

    print(f"    Genome summarization done in {time.time() - start:.2f}s")
    print(f"    Genome summaries: {len(df_summary):,}")

    return df_summary

def aggregate_and_summarize(
    df_probabilities,
    novelty_coeffs,
    aggregation_config,
    known_taxon=True
):
    total_start = time.time()

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

        df_novelty = _get_novelty_score(
            df_probabilities,
            novelty_coeffs
        )

        df_scores = pd.concat([df_scores, df_novelty], ignore_index=True)

        df_summary = _summarize_per_genome(
            df_scores,
            known_taxon=known_taxon
        )

        df_summary["aggregation_method"] = agg_method
        df_summary["aggregation_k"] = k_label

        all_results.append(df_summary)

    df_final = pd.concat(all_results, ignore_index=True)

    print(f"\n    Assignment/aggregation finished in {time.time() - total_start:.2f}s")
    print(f"    Final result rows: {len(df_final):,}")

    return df_final
