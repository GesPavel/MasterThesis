def evaluate_assignment_results(df_results, subset_name):
    metrics = {}

    for agg_k, group in df_results.groupby("aggregation_k"):
        result = {
            "n_genomes": int(len(group)),
        }

        if subset_name == "random":
            result.update({
                "top1": float(group["top1_correct"].mean()),
                "top2": float(group["top2_correct"].mean()),
                "top3": float(group["top3_correct"].mean()),
                "mean_rank": float(group["true_rank"].mean()),
                "median_rank": float(group["true_rank"].median()),
                "mean_true_score": float(group["true_score"].mean()),
                "mean_top_score": float(group["top_score"].mean()),
            })

        elif subset_name == "novelty":
            result.update({
                "mean_top_score": float(group["top_score"].mean()),
                "median_top_score": float(group["top_score"].median()),
                "max_top_score": float(group["top_score"].max()),
            })

        else:
            raise ValueError(f"Unsupported subset_name: {subset_name}")

        metrics[agg_k] = result

    return metrics