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