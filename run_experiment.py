import yaml
from pathlib import Path
from datetime import datetime
import json
import shutil
import argparse

from dataset_processing.hmm_pipeline.ground_truth_extraction import build_pairs_dataset
from dataset_processing.hmm_pipeline.novelty_fit import get_novelty_threshold
from dataset_processing.hmm_pipeline.probability_model_calibration import calibrate_model
from dataset_processing.hmm_pipeline.probability_model_training import train_probability_model
from dataset_processing.hmm_pipeline.probability_prediction import predict_probabilities
from dataset_processing.hmm_pipeline.taxon_assignment import aggregate_and_summarize
from dataset_processing.hmm_pipeline.taxon_assignment_evaluation import evaluate_assignment_results
from dataset_processing.train_test_split import load_dataset_split
from dataset_processing.util import filter_unpickled_dataframe, load_pickle_given_config
from dataset_processing.paths import (
    get_split_paths,
    get_train_pairs_path,
    get_val_pairs_path,
    get_model_path,
    get_calibrated_model_path,
    get_metrics_path,
    get_assignment_results_path,
    get_probabilities_path,
    get_novelty_threshold_path, get_known_taxa_path
)
import pandas as pd
import joblib


# =========================
# Config loading
# =========================

def read_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =========================
# Experiment folder
# =========================

def create_experiment_dir(config: dict) -> Path:
    exp_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = Path("experiments") / f"{timestamp}_{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    return exp_dir


def find_parent_experiment(parent_name: str) -> Path:
    root = Path("experiments")
    matches = [p for p in root.iterdir() if p.is_dir() and p.name.endswith(f"_{parent_name}")]
    if not matches:
        raise ValueError(f"Parent experiment '{parent_name}' not found")
    return sorted(matches)[-1]


# =========================
# Stage definitions
# =========================

STAGES = ["split", "pairs", "train", "calibrate", "probability_prediction", "assign", "novelty_fit", "eval"]

STAGE_DEPENDENCIES = {
    "split": [],
    "pairs": ["split"],
    "train": ["pairs"],
    "calibrate": ["train"],
    "probability_prediction": ["calibrate"],
    "assign": ["probability_prediction"],
    "novelty_fit": ["assign"],
    "eval": ["assign", "novelty_fit"],
}


def _subset_equal(current: dict, parent: dict, keys: list) -> bool:
    return all(current.get(k) == parent.get(k) for k in keys)


def compare_configs_for_stage(stage, current, parent):
    if stage == "split":
        return _subset_equal(current, parent, ["data", "split"])

    if stage == "pairs":
        return _subset_equal(current, parent, ["data", "pair_generation"])

    if stage == "train":
        return _subset_equal(current, parent, ["data", "features", "model"])

    if stage == "calibrate":
        return _subset_equal(current, parent, ["data", "features", "model", "calibration"])

    if stage == "probability_prediction":
        return _subset_equal(current, parent, ["data", "features", "model", "calibration"])

    if stage == "assign":
        return _subset_equal(current, parent, ["data", "features", "aggregation", "calibration"])

    if stage == "novelty_fit":
        return _subset_equal(current, parent, ["data", "features", "model", "calibration", "novelty_fit"])

    if stage == "eval":
        return _subset_equal(current, parent,
                             ["data", "features", "aggregation", "calibration"])

    return True


def validate_stage_plan(stages_to_run, parent_dir):
    stages_to_run = set(stages_to_run)

    for stage in stages_to_run:
        for dep in STAGE_DEPENDENCIES[stage]:
            if dep not in stages_to_run and parent_dir is None:
                raise ValueError(
                    f"Stage '{stage}' requires '{dep}', but '{dep}' is not rerun "
                    f"and no parent experiment is provided."
                )


def copy_stage_outputs(stage, parent_dir, current_dir, config):
    print(f"Copying outputs for stage: {stage} from parent")

    rank = config["data"]["taxon_rank"]

    if stage == "split":
        parent_paths = get_split_paths(parent_dir, rank)
        current_paths = get_split_paths(current_dir, rank)

        for key in parent_paths:
            shutil.copy(parent_paths[key], current_paths[key])

        shutil.copy(
            get_known_taxa_path(parent_dir),
            get_known_taxa_path(current_dir)
        )

    elif stage == "pairs":
        shutil.copy(
            get_train_pairs_path(parent_dir, rank),
            get_train_pairs_path(current_dir, rank)
        )
        shutil.copy(
            get_val_pairs_path(parent_dir, rank),
            get_val_pairs_path(current_dir, rank)
        )

    elif stage == "train":
        shutil.copy(
            get_model_path(parent_dir),
            get_model_path(current_dir)
        )

    elif stage == "calibrate":
        shutil.copy(
            get_calibrated_model_path(parent_dir),
            get_calibrated_model_path(current_dir)
        )
        shutil.copy(
            parent_dir / "calibration_report.txt",
            current_dir / "calibration_report.txt"
        )

    elif stage == "probability_prediction":
        for subset_name in ["novelty_fit", "test"]:
            shutil.copy(
                get_probabilities_path(parent_dir, subset_name),
                get_probabilities_path(current_dir, subset_name)
            )

    elif stage == "assign":
        for subset_name in ["novelty_fit", "test"]:
            shutil.copy(
                get_assignment_results_path(parent_dir, subset_name),
                get_assignment_results_path(current_dir, subset_name)
            )

    elif stage == "novelty_fit":
        shutil.copy(
            get_novelty_threshold_path(parent_dir),
            get_novelty_threshold_path(current_dir)
        )

    elif stage == "eval":
        shutil.copy(
            get_metrics_path(parent_dir),
            get_metrics_path(current_dir)
        )


# =========================
# Stages
# =========================

def run_split_stage(config: dict, exp_dir: Path):
    print("=== [1/8] SPLITTING DATASET ===")
    taxonomy_rank = config["data"]["taxon_rank"]

    train_df = load_pickle_given_config(config, "train")
    test_df = load_pickle_given_config(config, "test")
    split_config = config["split"]

    test, true_train, calibration, novelty_fit, known_taxa = load_dataset_split(
            train_df=train_df,
            test_df=test_df,
            taxonomy_rank=taxonomy_rank,
            calibration_fraction=split_config["calibration_fraction"],
            novelty_fit_fraction=split_config["novelty_fit_fraction"],
            random_seed=config["experiment"]["random_seed"],
    )


    paths = get_split_paths(exp_dir, taxonomy_rank)

    test.to_csv(paths["test"], index=False)
    true_train.to_csv(paths["true_train"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    novelty_fit.to_csv(paths["novelty_fit"], index=False)

    known_taxa_path = get_known_taxa_path(exp_dir)
    known_taxa.to_csv(known_taxa_path, index=False)


def run_pair_generation_stage(config: dict, exp_dir: Path):
    print("=== [2/8] BUILDING GROUND TRUTH DATASET FOR PROBABILISTIC MODEL ===")

    taxon_rank = config["data"]["taxon_rank"]
    split_paths = get_split_paths(exp_dir, taxon_rank)
    train_df = load_pickle_given_config(config, "train")

    df_true_train = filter_unpickled_dataframe(
        train_df,
        split_paths["true_train"]
    )

    df_calibration = filter_unpickled_dataframe(
        train_df,
        split_paths["calibration"]
    )

    print("--- Generating pairs for true_train ---")
    train_pairs_df = build_pairs_dataset(
        df=df_true_train,
        rank=taxon_rank,
        k_neighbors=config["pair_generation"]["k_neighbors"],
        k_random=config["pair_generation"]["k_random"],
        random_seed=config["experiment"]["random_seed"],
    )
    train_pairs_df.to_csv(get_train_pairs_path(exp_dir, taxon_rank), index=False)

    print("--- Generating pairs for calibration ---")
    val_pairs_df = build_pairs_dataset(
        df=df_calibration,
        rank=taxon_rank,
        k_neighbors=config["pair_generation"]["k_neighbors"],
        k_random=config["pair_generation"]["k_random"],
        random_seed=config["experiment"]["random_seed"],
    )
    val_pairs_df.to_csv(get_val_pairs_path(exp_dir, taxon_rank), index=False)


def run_training_stage(config: dict, exp_dir: Path):
    print("=== [3/8] TRAINING MODEL ===")

    taxon_rank = config["data"]["taxon_rank"]
    train_df =  load_pickle_given_config(config, "train")

    true_train_df = filter_unpickled_dataframe(
        train_df,
        get_split_paths(exp_dir, taxon_rank)["true_train"]
    )

    pairs_df = pd.read_csv(get_train_pairs_path(exp_dir, taxon_rank))

    report_str, model = train_probability_model(
        df=true_train_df,
        pairs_df=pairs_df,
        taxon_rank=taxon_rank,
        model_type=config["model"]["type"],
        model_config=config["model"],
        feature_config=config["features"],
        random_seed=config["experiment"]["random_seed"],
    )

    model_path = get_model_path(exp_dir)
    report_path = exp_dir / "pairwise_training_report.txt"

    joblib.dump(model, model_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    return model


def run_calibration_stage(config: dict, exp_dir: Path):
    print("=== [4/8] CALIBRATING MODEL ===")

    taxon_rank = config["data"]["taxon_rank"]

    train_df = load_pickle_given_config(config, "train")

    df_cal_val = filter_unpickled_dataframe(
        train_df,
        get_split_paths(exp_dir, taxon_rank)["calibration"]
    )

    model = joblib.load(get_model_path(exp_dir))
    val_pairs = pd.read_csv(get_val_pairs_path(exp_dir, taxon_rank))

    calibration_report, calibrated_model = calibrate_model(
        base_model=model,
        df=df_cal_val,
        val_pairs_df=val_pairs,
        feature_config=config["features"],
        calibration_config=config["calibration"]
    )

    calibrated_model_path = get_calibrated_model_path(exp_dir)
    joblib.dump(calibrated_model, calibrated_model_path)

    report_path = exp_dir / "calibration_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(calibration_report)

    return calibrated_model


def run_probability_prediction_stage(config: dict, exp_dir: Path, model):
    print("=== [5/8] PROBABILITY PREDICTION ===")

    taxon_rank = config["data"]["taxon_rank"]

    train_df = load_pickle_given_config(config, "train")
    true_train_df = filter_unpickled_dataframe(
        train_df,
        get_split_paths(exp_dir, taxon_rank)["true_train"]
    )

    novelty_fit_df = filter_unpickled_dataframe(
        train_df,
        get_split_paths(exp_dir, taxon_rank)["novelty_fit"]
    )

    test_df = load_pickle_given_config(config, "test")

    for subset_name, subset_to_predict in [("novelty_fit", novelty_fit_df),
                                           ("test", test_df),]:
        print(f"--- Predicting probabilities for subset: {subset_name} ---")

        df_probabilities = predict_probabilities(
            df_train=true_train_df,
            df_test=subset_to_predict,
            model=model,
            taxon_rank=taxon_rank,
            feature_config=config["features"]
        )

        prob_path = get_probabilities_path(exp_dir, subset_name)
        df_probabilities.to_pickle(prob_path)


def run_assignment_stage(config: dict, exp_dir: Path):
    print("=== [6/8] TAXON ASSIGNMENT ===")

    for subset_name in ["novelty_fit", "test"]:
        print(f"--- Running assignment for subset: {subset_name} ---")

        prob_path = get_probabilities_path(exp_dir, subset_name)
        df_probabilities = pd.read_pickle(prob_path)

        known_taxa_path = get_known_taxa_path(exp_dir)
        known_taxa = pd.read_csv(known_taxa_path)

        df_results = aggregate_and_summarize(
            df_probabilities=df_probabilities,
            aggregation_config=config["aggregation"],
            known_taxa=known_taxa,
        )

        results_path = get_assignment_results_path(exp_dir, subset_name)
        df_results.to_csv(results_path, index=False)


def run_novelty_fit_stage(config: dict, exp_dir: Path):
    print("=== [7/8] NOVELTY FIT ===")

    results_path = get_assignment_results_path(exp_dir, "novelty_fit")
    df_results = pd.read_csv(results_path)

    optimization_metric = config["novelty"]['optimization_metric']
    use_normalized_probs = config["novelty"]['use_normalized_probabilities']
    threshold, _ = get_novelty_threshold(df_results, metric=optimization_metric,
                                         use_normalized_probs=use_normalized_probs)

    threshold_path = get_novelty_threshold_path(exp_dir)
    with open(threshold_path, "w") as f:
        json.dump(threshold, f, indent=2)

    print(f"    Novelty fit complete. Optimal threshold: {threshold}")


def run_evaluation_stage(config: dict, exp_dir: Path):
    print("=== [8/8] EVALUATION ===")

    threshold_path = get_novelty_threshold_path(exp_dir)
    with open(threshold_path, "r") as f:
        threshold = json.load(f)

    use_normalized_probs = config["novelty"]['use_normalized_probabilities']


    results_path = get_assignment_results_path(exp_dir, "test")
    df_results = pd.read_csv(results_path)

    metrics = evaluate_assignment_results(df_results, threshold, use_normalized_probs)

    with open(get_metrics_path(exp_dir), "w") as f:
        json.dump({"assignment": metrics}, f, indent=2)


# =========================
# Main
# =========================

def run_experiment(config_path: Path, stages_to_run: list):
    config = read_config(config_path)
    exp_dir = create_experiment_dir(config)

    parent_name = config["experiment"].get("parent_experiment_name")

    if parent_name is None:
        print("No parent experiment, rerunning all stages")
        stages_to_run = STAGES
        parent_dir = None
    else:
        parent_dir = find_parent_experiment(parent_name)
        parent_config = read_config(parent_dir / "config_used.yaml")

        # pre-check config compatibility for skipped stages
        for stage in STAGES:
            if stage not in stages_to_run:
                if not compare_configs_for_stage(stage, config, parent_config):
                    raise ValueError(
                        f"Config mismatch for skipped stage '{stage}' with parent experiment."
                    )

    # pre-check dependency validity
    validate_stage_plan(stages_to_run, parent_dir)

    model = None

    for stage in STAGES:
        if stage in stages_to_run:
            print(f"Running stage: {stage}")

            if stage == "split":
                run_split_stage(config, exp_dir)

            elif stage == "pairs":
                run_pair_generation_stage(config, exp_dir)

            elif stage == "train":
                model = run_training_stage(config, exp_dir)

            elif stage == "calibrate":
                model = run_calibration_stage(config, exp_dir)

            elif stage == "probability_prediction":
                if model is None:
                    model_path = get_calibrated_model_path(exp_dir)
                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Model file not found at {model_path}. "
                            f"Run train and calibrate stages or provide a valid parent experiment."
                        )
                    model = joblib.load(model_path)

                run_probability_prediction_stage(config, exp_dir, model)

            elif stage == "assign":
                run_assignment_stage(config, exp_dir)

            elif stage == "novelty_fit":
                run_novelty_fit_stage(config, exp_dir)

            elif stage == "eval":
                run_evaluation_stage(config, exp_dir)

        else:
            if parent_dir is not None:
                copy_stage_outputs(stage, parent_dir, exp_dir, config)

    print("Experiment finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--stages",
        type=str,
        required=True,
        help="Comma separated stages: split,pairs,train,calibrate,probability_prediction,assign,novelty_fit,eval"
    )

    args = parser.parse_args()

    stages_to_run = [s.strip() for s in args.stages.split(",")]

    invalid_stages = [s for s in stages_to_run if s not in STAGES]
    if invalid_stages:
        raise ValueError(f"Invalid stages requested: {invalid_stages}. Valid stages: {STAGES}")

    run_experiment(Path(args.config), stages_to_run)
