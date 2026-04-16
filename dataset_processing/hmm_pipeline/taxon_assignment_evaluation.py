def evaluate_assignment_results(df_results, subset_name):
    metrics = {}

    for agg_k, group in df_results.groupby("aggregation_k"):
        result = {
            "n_genomes": int(len(group)),
        }

        # -------------------------
        # TOP-K (works for both datasets now)
        # -------------------------
        result.update({
            "top1": float(group["top1_correct"].mean()),
            "top2": float(group["top2_correct"].mean()),
            "top3": float(group["top3_correct"].mean()),
            "mean_rank": float(group["target_rank"].mean()),
            "median_rank": float(group["target_rank"].median()),
        })

        # -------------------------
        # PROBABILITIES
        # -------------------------
        result.update({
            "mean_novel_prob": float(group["novel_prob"].mean()),
            "mean_top_prob": float(group["top_prob"].mean()),
        })

        # -------------------------
        # FALSE NOVEL (only meaningful for random)
        # -------------------------
        if subset_name == "random":
            is_pred_novel = group["predicted_taxon"] == "NOVEL"

            result.update({
                "false_novel_count": int(is_pred_novel.sum()),
                "false_novel_rate": float(is_pred_novel.mean()),
            })

        # -------------------------
        # NOVEL RECALL (only meaningful for novelty set)
        # -------------------------
        elif subset_name in ["novelty", "model_fit_val"]:
            is_pred_novel = group["predicted_taxon"] == "NOVEL"

            result.update({
                "novel_recall": float(is_pred_novel.mean()),
            })

        else:
            raise ValueError(f"Unsupported subset_name: {subset_name}")

        metrics[agg_k] = result

    return metrics