from dataclasses import dataclass, field

import numpy as np

import matplotlib
matplotlib.use("Agg")  # figures are only ever written to disk, never shown
import matplotlib.pyplot as plt


# Escalating ceilings on the false positive rate. The threshold is picked under
# the first budget that admits a non-trivial operating point.
DEFAULT_FPR_BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0)

# numpy >= 2 renamed trapz to trapezoid.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class NoveltyFitResult:
    """Everything the novelty fit produces: the curves, their areas, the chosen
    operating point and the two plots."""

    threshold: float
    auroc: float
    auprc: float
    score_column: str

    # Curve points, all aligned index-wise with `thresholds`.
    thresholds: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    precision: np.ndarray
    recall: np.ndarray

    # Index into the curve arrays of the selected threshold.
    chosen_index: int
    fpr_budget: float

    n_novel: int
    n_known: int

    roc_figure: object = field(default=None, repr=False)
    pr_figure: object = field(default=None, repr=False)

    def save_figures(self, roc_path, pr_path, dpi=150):
        """Write both plots and release them -- figures held open leak memory."""
        self.roc_figure.savefig(roc_path, dpi=dpi)
        self.pr_figure.savefig(pr_path, dpi=dpi)
        plt.close(self.roc_figure)
        plt.close(self.pr_figure)

    def summary(self) -> dict:
        """JSON-serialisable summary (no curve arrays, no figures)."""
        i = self.chosen_index
        return {
            "novelty_threshold": float(self.threshold),
            "auroc": float(self.auroc),
            "auprc": float(self.auprc),
            "score_column": self.score_column,
            "selection": {
                "rule": "max TPR subject to FPR <= budget, smallest budget first",
                "fpr_budget": float(self.fpr_budget),
                "tpr": float(self.tpr[i]),
                "fpr": float(self.fpr[i]),
                "precision": float(self.precision[i]),
                "recall": float(self.recall[i]),
            },
            "n_novel": int(self.n_novel),
            "n_known": int(self.n_known),
            "n_thresholds_evaluated": int(len(self.thresholds)),
        }


def compute_novelty_curves(df_assigned_taxa, use_normalized_probs):
    """
    ROC and precision-recall curves for the novelty decision rule
    `predicted_novel = score < threshold`.

    Every threshold that produces a distinct prediction is evaluated: the unique
    scores (each of which flips its own group of genomes to "known") plus one
    threshold above the maximum score (everything predicted novel).

    Returns a dict with the curve arrays and their areas.
    """
    score_column = "top_prob" if use_normalized_probs else "top_score"

    y_true = df_assigned_taxa["is_actually_novel"].astype(int).values
    scores = df_assigned_taxa[score_column].values.astype(float)

    positives = np.sort(scores[y_true == 1])
    negatives = np.sort(scores[y_true == 0])
    n_pos, n_neg = len(positives), len(negatives)

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "Novelty fit needs both novel and known genomes, got "
            f"{n_pos} novel and {n_neg} known."
        )

    unique_scores = np.unique(scores)
    # At t = min(score) nothing is predicted novel (FPR = TPR = 0); the extra
    # threshold just above the maximum predicts everything novel (FPR = TPR = 1),
    # so the curves span the full [0, 1] range.
    thresholds = np.concatenate(
        [unique_scores, [np.nextafter(unique_scores[-1], np.inf)]]
    )

    # searchsorted(..., "left") counts the entries strictly below the threshold,
    # which is exactly the `score < threshold` rule.
    tp = np.searchsorted(positives, thresholds, side="left")
    fp = np.searchsorted(negatives, thresholds, side="left")

    tpr = tp / n_pos
    fpr = fp / n_neg

    predicted_positive = tp + fp
    precision = np.divide(
        tp,
        predicted_positive,
        out=np.ones_like(tpr),  # nothing predicted novel -> precision defined as 1
        where=predicted_positive > 0,
    )
    recall = tpr

    # FPR is non-decreasing over ascending thresholds, so the trapezoid is taken
    # over the ROC curve traversed left to right.
    auroc = float(_trapezoid(tpr, fpr))

    # Average precision (step-wise), the same convention as sklearn's
    # average_precision_score -- trapezoidal PR area is optimistic.
    auprc = float(np.sum(np.diff(recall, prepend=0.0) * precision))

    return {
        "score_column": score_column,
        "thresholds": thresholds,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "auprc": auprc,
        "n_novel": n_pos,
        "n_known": n_neg,
    }


def select_threshold_by_fpr_budget(curves, fpr_budgets=DEFAULT_FPR_BUDGETS):
    """
    Highest-TPR operating point whose FPR fits inside a budget, trying the
    budgets in the given (ascending) order.

    A budget is only accepted if it admits a point with TPR > 0 -- the trivial
    "predict nothing novel" point always has FPR = 0, so without that guard the
    first budget would always win with a useless threshold. Ties on TPR are
    broken by the lower FPR, then by the lower threshold.

    Returns (chosen_index, budget_used).
    """
    tpr = curves["tpr"]
    fpr = curves["fpr"]

    for budget in fpr_budgets:
        candidates = np.flatnonzero(fpr <= budget)
        if len(candidates) == 0:
            continue

        best_tpr = tpr[candidates].max()
        if best_tpr <= 0.0:
            # No usable point at this budget -- relax to the next one.
            continue

        candidates = candidates[tpr[candidates] == best_tpr]
        candidates = candidates[fpr[candidates] == fpr[candidates].min()]

        return int(candidates[0]), float(budget)

    # Every budget was degenerate (only reachable if no threshold gives TPR > 0);
    # fall back to the most permissive point.
    return int(len(tpr) - 1), float(fpr_budgets[-1])


def plot_roc_curve(curves, chosen_index):
    fig, ax = plt.subplots(figsize=(6, 5.5))

    ax.plot(
        curves["fpr"],
        curves["tpr"],
        color="#1f77b4",
        lw=2,
        label=f"ROC (AUROC = {curves['auroc']:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="Chance")

    ax.scatter(
        [curves["fpr"][chosen_index]],
        [curves["tpr"][chosen_index]],
        color="#d62728",
        zorder=5,
        label=(
            f"Chosen threshold = {curves['thresholds'][chosen_index]:.4g}\n"
            f"TPR = {curves['tpr'][chosen_index]:.3f}, "
            f"FPR = {curves['fpr'][chosen_index]:.3f}"
        ),
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Novelty detection - ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig


def plot_precision_recall_curve(curves, chosen_index):
    fig, ax = plt.subplots(figsize=(6, 5.5))

    baseline = curves["n_novel"] / (curves["n_novel"] + curves["n_known"])

    ax.plot(
        curves["recall"],
        curves["precision"],
        color="#2ca02c",
        lw=2,
        label=f"PRC (AUPRC = {curves['auprc']:.4f})",
    )
    ax.axhline(
        baseline,
        color="grey",
        lw=1,
        ls="--",
        label=f"Novel prevalence = {baseline:.3f}",
    )

    ax.scatter(
        [curves["recall"][chosen_index]],
        [curves["precision"][chosen_index]],
        color="#d62728",
        zorder=5,
        label=(
            f"Chosen threshold = {curves['thresholds'][chosen_index]:.4g}\n"
            f"P = {curves['precision'][chosen_index]:.3f}, "
            f"R = {curves['recall'][chosen_index]:.3f}"
        ),
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Novelty detection - precision-recall curve")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig


def fit_novelty_threshold(
    df_assigned_taxa,
    use_normalized_probs,
    fpr_budgets=DEFAULT_FPR_BUDGETS,
):
    """
    Full novelty fit: ROC + PRC over every threshold, their areas, the selected
    threshold and the two plots.

    The caller owns the returned figures and is responsible for closing them.
    """
    curves = compute_novelty_curves(df_assigned_taxa, use_normalized_probs)
    chosen_index, budget = select_threshold_by_fpr_budget(curves, fpr_budgets)

    return NoveltyFitResult(
        threshold=float(curves["thresholds"][chosen_index]),
        auroc=curves["auroc"],
        auprc=curves["auprc"],
        score_column=curves["score_column"],
        thresholds=curves["thresholds"],
        fpr=curves["fpr"],
        tpr=curves["tpr"],
        precision=curves["precision"],
        recall=curves["recall"],
        chosen_index=chosen_index,
        fpr_budget=budget,
        n_novel=curves["n_novel"],
        n_known=curves["n_known"],
        roc_figure=plot_roc_curve(curves, chosen_index),
        pr_figure=plot_precision_recall_curve(curves, chosen_index),
    )