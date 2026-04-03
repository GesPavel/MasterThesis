from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_split_dir(exp_dir: Path) -> Path:
    return ensure_dir(exp_dir / "split")


def get_split_paths(exp_dir: Path, rank: str) -> dict:
    split_dir = ensure_dir(get_split_dir(exp_dir) / rank)

    return {
        "train": split_dir / "train.csv",
        "random": split_dir / "random.csv",
        "novelty": split_dir / "novelty.csv",
    }


def get_pairs_path(exp_dir: Path, rank: str) -> Path:
    output_dir = ensure_dir(exp_dir / "pairs")
    return output_dir / f"{rank}_pairs.csv"


def get_model_path(exp_dir: Path) -> Path:
    return exp_dir / "model.joblib"


def get_metrics_path(exp_dir: Path) -> Path:
    return exp_dir / "metrics.json"


def get_config_path(exp_dir: Path) -> Path:
    return exp_dir / "config_used.yaml"

def get_assignment_results_path(exp_dir: Path, subset_name: str) -> Path:
    return exp_dir / f"{subset_name}_assignment_results.csv"