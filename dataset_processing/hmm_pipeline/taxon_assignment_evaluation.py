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

                # use probabilities instead of raw scores
                "mean_true_prob": float(group["true_prob"].mean()),
                "mean_top_prob": float(group["top_prob"].mean()),
            })

        elif subset_name == "novelty":
            result.update({
                # use probabilities instead of raw scores
                "mean_top_prob": float(group["top_prob"].mean()),
                "median_top_prob": float(group["top_prob"].median()),
                "max_top_prob": float(group["top_prob"].max()),
            })

        else:
            raise ValueError(f"Unsupported subset_name: {subset_name}")

        metrics[agg_k] = result

    return metrics