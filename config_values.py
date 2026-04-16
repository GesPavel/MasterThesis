REPRESENTATIONS = ["hmm", "pc", "hybrid"]

TAXON_RANKS = [
    "Family",
    "Genus",
    "Order",
    "Class",
    "Phylum",
    "Kingdom",
    "Realm",
]

SIMILARITY_METRICS = [
    "shared_count",
    "jaccard",
]

MODEL_TYPES = [
    "logreg",
    "xgboost",
]

NOVELTY_FIT_MODEL_TYPES = [
    "logreg",
    "xgboost",
]

LOGREG_SOLVERS = [
    "lbfgs",
    "liblinear",
    "saga",
]

XGBOOST_EVAL_METRICS = [
    "logloss",
    "auc",
    "error",
]

AGGREGATION_METHODS = [
    "mean",
    "median",
    "max",
    "logit_sum",
    "topk_mean",
]

CALIBRATION_METHODS = [
    "sigmoid",
    "isotonic",
]