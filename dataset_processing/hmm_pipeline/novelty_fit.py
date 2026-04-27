import numpy as np

def get_novelty_threshold(
        df_assigned_taxa,
        metric,
        num_thresholds=200
):
    """
    Finds optimal novelty threshold based on chosen metric.

    metric: "f1", "accuracy", "precision", "recall"
    """

    y_true = df_assigned_taxa["is_actually_novel"].astype(int).values
    probs = df_assigned_taxa["top_prob"].values

    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    best_t = 0.0
    best_score = -1.0

    for t in thresholds:
        y_pred = (probs < t).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

        if metric == "f1":
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "accuracy":
            score = accuracy
        elif metric == "precision":
            score = precision
        elif metric == "recall":
            score = recall
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score