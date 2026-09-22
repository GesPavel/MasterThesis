import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def _assignment_quality(seen_group):
    """
    Top-1 assignment quality over the genomes whose taxon is actually known.

    Micro and macro answer different questions. Micro pools every genome into
    one tally, so big taxa dominate it. Macro scores each taxon separately and
    averages, so a taxon holding two genomes counts as much as one holding two
    hundred -- which is the honest view when taxon sizes are as skewed as they
    are here.

    Note that micro precision, recall and F1 are all equal to plain accuracy in
    this setting: every genome gets exactly one predicted taxon, so a wrong
    prediction is simultaneously one false positive for the taxon guessed and
    one false negative for the taxon missed. They are reported because readers
    expect the set, not because they carry three separate pieces of information.
    """
    y_true = seen_group["true_taxon"].to_numpy()
    y_pred = seen_group["predicted_taxon"].to_numpy()

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "assignment_accuracy_micro": float(micro_p),
        "assignment_precision_micro": float(micro_p),
        "assignment_recall_micro": float(micro_r),
        "assignment_f1_micro": float(micro_f1),

        # Mean per-class recall, which is what "accuracy averaged over classes"
        # means and is the same number as macro recall by definition.
        "assignment_accuracy_macro": float(macro_r),
        "assignment_precision_macro": float(macro_p),
        "assignment_recall_macro": float(macro_r),
        "assignment_f1_macro": float(macro_f1),

        # Macro averages over every taxon appearing as a true or a predicted
        # label, so predicting into a taxon that holds no test genome still
        # costs precision.
        "n_classes_evaluated": int(len(set(y_true.tolist()) | set(y_pred.tolist()))),
        "n_true_classes": int(len(set(y_true.tolist()))),
    }

    # ---- rank-based scores ----
    rank = seen_group["target_rank"].to_numpy(dtype=float)
    found = ~np.isnan(rank)

    # A genome whose true taxon never appeared as a candidate scores 0, the
    # usual convention: there is no rank to take a reciprocal of.
    reciprocal = np.zeros(len(rank), dtype=float)
    reciprocal[found] = 1.0 / rank[found]
    metrics["mrr"] = float(reciprocal.mean())

    # Rank as a fraction of the candidate list: 0 is first, 1 is last. Absolute
    # ranks are not comparable across ranks or subsets, since the number of
    # candidate taxa differs; this is.
    n_candidates = seen_group["n_candidates"].to_numpy(dtype=float)
    spread = np.maximum(n_candidates - 1.0, 1.0)
    normalized = (rank - 1.0) / spread

    # Averaged over the genomes whose taxon was found at all, so it is not
    # silently mixing in a "worst possible" value for the ones that were not.
    metrics["n_target_found"] = int(found.sum())
    metrics["target_found_fraction"] = float(found.mean()) if len(rank) else 0.0
    metrics["mean_normalized_rank"] = float(np.nanmean(normalized[found])) if found.any() else 0.0
    metrics["median_normalized_rank"] = float(np.nanmedian(normalized[found])) if found.any() else 0.0

    return metrics


def _empty_assignment_quality():
    """Placeholders for a subset with no genomes of a known taxon."""
    keys = [
        "assignment_accuracy_micro", "assignment_precision_micro",
        "assignment_recall_micro", "assignment_f1_micro",
        "assignment_accuracy_macro", "assignment_precision_macro",
        "assignment_recall_macro", "assignment_f1_macro",
        "mrr", "target_found_fraction",
        "mean_normalized_rank", "median_normalized_rank",
    ]
    metrics = {key: 0.0 for key in keys}
    metrics.update(n_classes_evaluated=0, n_true_classes=0, n_target_found=0)
    return metrics


def evaluate_assignment_results(df_results, novelty_threshold, use_normalized_probs):
    metrics: dict[str, dict[str, float | int]] = {}

    for agg_k, group in df_results.groupby("aggregation_k"):
        is_novel = group["is_actually_novel"].astype(bool)
        seen_group = group.loc[~is_novel]

        result: dict[str, float | int] = {
            "n_genomes": int(len(group)),
            "n_seen_genomes": int(len(seen_group)),
            "n_novel_genomes": int(is_novel.sum()),
        }

        # ---- Novelty classification metrics (always computed) ----
        y_true = is_novel.astype(int)

        # recompute prediction from threshold
        y_pred = (group["top_prob" if use_normalized_probs else "top_score"] < novelty_threshold).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        accuracy = (tp + tn) / len(group) if len(group) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        result.update({
            "novelty_accuracy": float(accuracy),
            "novelty_precision": float(precision),
            "novelty_recall": float(recall),
            "novelty_f1": float(f1),
            "novel_genome_count": int(y_true.sum()),
        })

        # ---- Rank-based assignment metrics (seen genomes only) ----
        if len(seen_group) > 0:
            result.update(_assignment_quality(seen_group))
            result.update({
                "top1": float(seen_group["top1_correct"].mean()),
                "top2": float(seen_group["top2_correct"].mean()),
                "top3": float(seen_group["top3_correct"].mean()),
                "mean_rank": float(seen_group["target_rank"].mean()),
                "median_rank": float(seen_group["target_rank"].median()),
                "mean_true_prob": float(seen_group["target_prob"].mean()),
                "mean_true_score": float(seen_group["target_score"].mean()),
            })
        else:
            result.update(_empty_assignment_quality())
            result.update({
                "top1": 0.0,
                "top2": 0.0,
                "top3": 0.0,
                "mean_rank": 0.0,
                "median_rank": 0.0,
                "mean_true_prob": 0.0,
                "mean_true_score": 0.0,
            })

        # Confidence summaries over the full mixed dataset.
        result.update({
            "mean_top_prob": float(group["top_prob"].mean()),
            "median_top_prob": float(group["top_prob"].median()),
            "max_top_prob": float(group["top_prob"].max()),
            "mean_top_score": float(group["top_score"].mean()),
            "median_top_score": float(group["top_score"].median()),
            "max_top_score": float(group["top_score"].max()),
        })

        # Optional seen-only confidence summaries for debugging mixed datasets.
        if len(seen_group) > 0:
            result.update({
                "seen_mean_top_prob": float(seen_group["top_prob"].mean()),
                "seen_median_top_prob": float(seen_group["top_prob"].median()),
                "seen_max_top_prob": float(seen_group["top_prob"].max()),
                "seen_mean_top_score": float(seen_group["top_score"].mean()),
                "seen_median_top_score": float(seen_group["top_score"].median()),
                "seen_max_top_score": float(seen_group["top_score"].max()),
            })
        else:
            result.update({
                "seen_mean_top_prob": 0.0,
                "seen_median_top_prob": 0.0,
                "seen_max_top_prob": 0.0,
                "seen_mean_top_score": 0.0,
                "seen_median_top_score": 0.0,
                "seen_max_top_score": 0.0,
            })

        metrics[agg_k] = result

    return metrics