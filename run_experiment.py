import yaml
from pathlib import Path
from datetime import datetime
import json
import shutil
import argparse

from dataset_processing.hmm_pipeline.ground_truth_extraction import build_pairs_dataset
from dataset_processing.hmm_pipeline.probability_model_calibration import calibrate_model
from dataset_processing.hmm_pipeline.probability_model_training import train_probability_model
from dataset_processing.hmm_pipeline.taxon_assignment import assign_taxa
from dataset_processing.hmm_pipeline.taxon_assignment_evaluation import evaluate_assignment_results
from dataset_processing.train_test_split import split_dataset
from dataset_processing.util import load_filtered_dataframe
from dataset_processing.paths import (
    get_split_paths,
    get_pairs_path,
    get_val_pairs_path,
    get_model_path,
    get_calibrated_model_path,
    get_metrics_path,
    get_assignment_results_path,
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

STAGES = ["split", "pairs", "train", "calibrate", "assign"]

STAGE_DEPENDENCIES = {
    "split": [],
    "pairs": ["split"],
    "train": ["pairs"],
    "calibrate": ["train"],
    "assign": ["calibrate"],
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

    if stage == "assign":
        return _subset_equal(current, parent, ["data", "features", "aggregation", "calibration"])

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

    elif stage == "pairs":
        shutil.copy(
            get_pairs_path(parent_dir, rank),
            get_pairs_path(current_dir, rank)
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

    elif stage == "assign":
        shutil.copy(
            get_assignment_results_path(parent_dir, "random"),
            get_assignment_results_path(current_dir, "random")
        )
        shutil.copy(
            get_assignment_results_path(parent_dir, "novelty"),
            get_assignment_results_path(current_dir, "novelty")
        )
        shutil.copy(
            get_metrics_path(parent_dir),
            get_metrics_path(current_dir)
        )


# =========================
# Stages
# =========================

def run_split_stage(config: dict, exp_dir: Path):
    print("=== [1/5] SPLITTING DATASET ===")

    df = pd.read_pickle(config["data"]["dataset_pickle"])
    split_config = config["split"]

    novelty_test, random_test, true_train, calibration_val, model_fit_val = split_dataset(
        df=df,
        taxonomy_rank=config["data"]["taxon_rank"],
        novelty_fraction=split_config["novelty_fraction"],
        random_test_fraction=split_config["random_test_fraction"],
        true_train_fraction=split_config["train_fraction"],
        calibration_val_fraction=split_config["val_fraction"],
        novelty_fit_val_fraction=split_config["test_fraction"],
        random_seed=config["experiment"]["random_seed"]
    )

    paths = get_split_paths(exp_dir, config["data"]["taxon_rank"])

    novelty_test.to_csv(paths["novelty"], index=False)
    random_test.to_csv(paths["random"], index=False)
    true_train.to_csv(paths["true_train"], index=False)
    calibration_val.to_csv(paths["calibration_val"], index=False)
    model_fit_val.to_csv(paths["model_fit_val"], index=False)


def run_pair_generation_stage(config: dict, exp_dir: Path):
    print("=== [2/5] BUILDING GROUND TRUTH DATASET FOR PROBABILISTIC MODEL ===")

    taxon_rank = config["data"]["taxon_rank"]
    split_paths = get_split_paths(exp_dir, taxon_rank)

    df_true_train = load_filtered_dataframe(
        config["data"]["dataset_pickle"],
        split_paths["true_train"]
    )
    
    df_calibration_val = load_filtered_dataframe(
        config["data"]["dataset_pickle"],
        split_paths["calibration_val"]
    )

    print("--- Generating pairs for true_train ---")
    train_pairs_df = build_pairs_dataset(
        df=df_true_train,
        rank=taxon_rank,
        k_neighbors=config["pair_generation"]["k_neighbors"],
        k_random=config["pair_generation"]["k_random"],
        random_seed=config["experiment"]["random_seed"],
    )
    train_pairs_df.to_csv(get_pairs_path(exp_dir, taxon_rank), index=False)

    print("--- Generating pairs for calibration_val ---")
    val_pairs_df = build_pairs_dataset(
        df=df_calibration_val,
        rank=taxon_rank,
        k_neighbors=config["pair_generation"]["k_neighbors"],
        k_random=config["pair_generation"]["k_random"],
        random_seed=config["experiment"]["random_seed"],
    )
    val_pairs_df.to_csv(get_val_pairs_path(exp_dir, taxon_rank), index=False)


def run_training_stage(config: dict, exp_dir: Path):
    print("=== [3/5] TRAINING MODEL ===")

    taxon_rank = config["data"]["taxon_rank"]

    df_train = load_filtered_dataframe(
        config["data"]["dataset_pickle"],
        get_split_paths(exp_dir, taxon_rank)["true_train"]
    )

    pairs_df = pd.read_csv(get_pairs_path(exp_dir, taxon_rank))

    report_str, model = train_probability_model(
        df=df_train,
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
    print("=== [4/5] CALIBRATING MODEL ===")

    taxon_rank = config["data"]["taxon_rank"]

    df_cal_val = load_filtered_dataframe(
        config["data"]["dataset_pickle"],
        get_split_paths(exp_dir, taxon_rank)["calibration_val"]
    )

    model = joblib.load(get_model_path(exp_dir))
    val_pairs = pd.read_csv(get_val_pairs_path(exp_dir, taxon_rank))

    calibration_report, calibrated_model  = calibrate_model(
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


def run_assignment_stage(config: dict, exp_dir: Path, model):
    print("=== [5/5] TAXON ASSIGNMENT ===")

    taxon_rank = config["data"]["taxon_rank"]

    df_train = load_filtered_dataframe(
        config["data"]["dataset_pickle"],
        get_split_paths(exp_dir, taxon_rank)["true_train"]
    )

    all_metrics = {}

    for subset_name in ["random", "novelty", "model_fit_val"]:
        print(f"--- Running assignment for subset: {subset_name} ---")

        df_test = load_filtered_dataframe(
            config["data"]["dataset_pickle"],
            get_split_paths(exp_dir, taxon_rank)[subset_name]
        )

        known_taxon = (subset_name == "random")

        df_results = assign_taxa(
            df_train=df_train,
            df_test=df_test,
            model=model,
            taxon_rank=taxon_rank,
            feature_config=config["features"],
            aggregation_config=config["aggregation"],
            known_taxon=known_taxon
        )

        results_path = get_assignment_results_path(exp_dir, subset_name)
        df_results.to_csv(results_path, index=False)

        metrics = evaluate_assignment_results(df_results, subset_name)
        all_metrics[subset_name] = metrics

    with open(get_metrics_path(exp_dir), "w") as f:
        json.dump({"assignment": all_metrics}, f, indent=2)


# =========================
# Main
# =========================

def run_experiment(config_path: Path, stages_to_run: list):
    config = read_config(config_path)
    exp_dir = create_experiment_dir(config)

    parent_name = config["experiment"].get("parent_experiment_name")

    if parent_name is None:
        print("No parent experiment, rerunning all stages")
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

            elif stage == "assign":
                if model is None:
                    model_path = get_calibrated_model_path(exp_dir)
                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Model file not found at {model_path}. "
                            f"Run train and calibrate stages or provide a valid parent experiment."
                        )
                    model = joblib.load(model_path)

                run_assignment_stage(config, exp_dir, model)

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
        help="Comma separated stages: split,pairs,train,calibrate,assign"
    )

    args = parser.parse_args()

    stages_to_run = [s.strip() for s in args.stages.split(",")]

    invalid_stages = [s for s in stages_to_run if s not in STAGES]
    if invalid_stages:
        raise ValueError(f"Invalid stages requested: {invalid_stages}. Valid stages: {STAGES}")

    run_experiment(Path(args.config), stages_to_run)
