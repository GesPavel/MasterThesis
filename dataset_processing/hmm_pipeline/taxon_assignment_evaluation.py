def evaluate_assignment_results(df_results, novelty_threshold, use_normalized_probs, subset_name):
    metrics = {}

    for agg_k, group in df_results.groupby("aggregation_k"):
        result = {
            "n_genomes": int(len(group)),
        }

        # ---- Novelty classification metrics (always computed) ----
        y_true = group["is_actually_novel"].astype(int)

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
            "novel_genome_count": float(y_true.sum()),
        })

        # ---- Existing metrics ----
        if subset_name == "random":
            result.update({
                "top1": float(group["top1_correct"].mean()),
                "top2": float(group["top2_correct"].mean()),
                "top3": float(group["top3_correct"].mean()),
                "mean_rank": float(group["target_rank"].mean()),
                "median_rank": float(group["target_rank"].median()),
                "mean_true_prob": float(group["target_prob"].mean()),
                "mean_top_prob": float(group["top_prob"].mean()),
            })

        elif subset_name == "novelty":
            result.update({
                "mean_top_prob": float(group["top_prob"].mean()),
                "median_top_prob": float(group["top_prob"].median()),
                "max_top_prob": float(group["top_prob"].max()),
            })

        else:
            raise ValueError(f"Unsupported subset_name: {subset_name}")

        metrics[agg_k] = result

    return metrics