import yaml
from pathlib import Path
from datetime import datetime
import json
import shutil
import argparse

from dataset_processing.hmm_pipeline.ground_truth_extraction import build_pairs_dataset
from dataset_processing.hmm_pipeline.probability_model_training import train_probability_model
from dataset_processing.hmm_pipeline.taxon_assignment import assign_taxa
from dataset_processing.hmm_pipeline.taxon_assignment_evaluation import evaluate_assignment_results
from dataset_processing.train_test_split import split_dataset
from dataset_processing.util import load_filtered_dataframe
from dataset_processing.paths import (
    get_split_paths,
    get_pairs_path,
    get_model_path,
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

STAGES = ["split", "pairs", "train", "assign"]

STAGE_DEPENDENCIES = {
    "split": [],
    "pairs": ["split"],
    "train": ["pairs"],
    "assign": ["train"],
}


def _subset_equal(current: dict, parent: dict, keys: list) -> bool:
    return all(current.get(k) == parent.get(k) for k in keys)


def compare_configs_for_stage(stage, current, parent):
    if stage == "split":
        return _subset_equal(current, parent, ["data", "split"])

    if stage == "pairs":
        return _subset_equal(current, parent, ["data", "pair_generation"])

    if stage == "train":
        return _subset_equal(current, parent, ["data", "features", "model", "training"])

    if stage == "assign":
        return _subset_equal(current, parent, ["data", "features", "aggregation"])

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

    elif stage == "train":
        shutil.copy(
            get_model_path(parent_dir),
            get_model_path(current_dir)
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
    print("\n=== [1/4] SPLITTING DATASET ===")

    df = pd.read_pickle(config["data"]["dataset_pickle"])

    novelty_test, random_test, train_set = split_dataset(
        df=df,
        taxonomy_rank=config["data"]["taxon_rank"],
        novelty_fraction=config["split"]["novelty_fraction"],
        random_test_fraction=config["split"]["random_test_fraction"],
        random_seed=config["experiment"]["random_seed"]
    )

    paths = get_split_paths(exp_dir, config["data"]["taxon_rank"])

    novelty_test.to_csv(paths["novelty"], index=False)
    random_test.to_csv(paths["random"], index=False)
    train_set.to_csv(paths["train"], index=False)


def run_pair_generation_stage(config: dict, exp_dir: Path):
    print("\n=== [2/4] BUILDING GROUND TRUTH DATASET FOR PROBABILISTIC MODEL ===")

    taxon_rank = config["data"]["taxon_rank"]

    df_train = load_filtered_dataframe(
        config["data"]["dataset_pickle"],
        get_split_paths(exp_dir, taxon_rank)["train"]
    )

    pairs_df = build_pairs_dataset(
        df=df_train,
        rank=taxon_rank,
        k_neighbors=config["pair_generation"]["k_neighbors"],
        k_random=config["pair_generation"]["k_random"],
        random_seed=config["experiment"]["random_seed"],
    )

    pairs_df.to_csv(get_pairs_path(exp_dir, taxon_rank), index=False)


def run_training_stage(config: dict, exp_dir: Path):
    print("\n=== [3/4] TRAINING MODEL ===")

    taxon_rank = config["data"]["taxon_rank"]

    df_train = load_filtered_dataframe(
        config["data"]["dataset_pickle"],
        get_split_paths(exp_dir, taxon_rank)["train"]
    )

    pairs_df = pd.read_csv(get_pairs_path(exp_dir, taxon_rank))

    report_str, model = train_probability_model(
        df=df_train,
        pairs_df=pairs_df,
        taxon_rank=taxon_rank,
        model_type=config["model"]["type"],
        model_config=config["model"],
        training_config=config["training"],
        feature_config=config["features"],
        random_seed=config["experiment"]["random_seed"],
    )

    model_path = get_model_path(exp_dir)
    report_path = exp_dir / "pairwise_training_report.txt"

    joblib.dump(model, model_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    return model


def run_assignment_stage(config: dict, exp_dir: Path, model):
    print("\n=== [4/4] TAXON ASSIGNMENT ===")

    taxon_rank = config["data"]["taxon_rank"]

    df_train = load_filtered_dataframe(
        config["data"]["dataset_pickle"],
        get_split_paths(exp_dir, taxon_rank)["train"]
    )

    all_metrics = {}

    for subset_name in ["random", "novelty"]:
        print(f"\n--- Running assignment for subset: {subset_name} ---")

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
            print(f"\nRunning stage: {stage}")

            if stage == "split":
                run_split_stage(config, exp_dir)

            elif stage == "pairs":
                run_pair_generation_stage(config, exp_dir)

            elif stage == "train":
                model = run_training_stage(config, exp_dir)

            elif stage == "assign":
                if model is None:
                    model_path = get_model_path(exp_dir)
                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Model file not found at {model_path}. "
                            f"Run train stage or provide a valid parent experiment."
                        )
                    model = joblib.load(model_path)

                run_assignment_stage(config, exp_dir, model)

        else:
            if parent_dir is not None:
                copy_stage_outputs(stage, parent_dir, exp_dir, config)

    print("\nExperiment finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--stages",
        type=str,
        required=True,
        help="Comma separated stages: split,pairs,train,assign"
    )

    args = parser.parse_args()

    stages_to_run = [s.strip() for s in args.stages.split(",")]

    invalid_stages = [s for s in stages_to_run if s not in STAGES]
    if invalid_stages:
        raise ValueError(f"Invalid stages requested: {invalid_stages}. Valid stages: {STAGES}")

    run_experiment(Path(args.config), stages_to_run)