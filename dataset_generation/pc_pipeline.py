import logging
from pathlib import Path
from typing import Dict, Sequence

from dataset_generation.constants import canonical_rank
from dataset_generation.io_utils import ensure_dir
from dataset_generation.pc_features import augment_test_with_pc, augment_train_with_pc

logger = logging.getLogger(__name__)


def augment_scenario_with_pc(scenario_dir: Path, mmseqs_bin: str) -> Dict[str, Path]:
    faa_dir = scenario_dir / "faa"
    pkl_dir = scenario_dir / "pkl"
    train_faa = faa_dir / "train.faa"
    train_pkl = pkl_dir / "train.pkl"

    if not train_faa.exists() or not train_pkl.exists():
        logger.warning(
            "Skipping %s: missing train.faa or train.pkl (faa=%s, pkl=%s)",
            scenario_dir, train_faa.exists(), train_pkl.exists(),
        )
        return {}

    logger.info("=== Augmenting scenario %s with PC features ===", scenario_dir)
    pkl_ext_dir = ensure_dir(scenario_dir / "pkl_ext")
    work_dir = ensure_dir(scenario_dir / "mmseqs")

    written: Dict[str, Path] = {}

    protein_to_pc = augment_train_with_pc(
        train_faa=train_faa,
        train_pkl=train_pkl,
        output_pkl=pkl_ext_dir / "train.pkl",
        work_dir=work_dir,
        mmseqs_bin=mmseqs_bin,
    )
    written["train"] = pkl_ext_dir / "train.pkl"

    for test_faa in sorted(faa_dir.glob("*.faa")):
        if test_faa.name == "train.faa":
            continue

        stem = test_faa.stem
        test_pkl = pkl_dir / f"{stem}.pkl"
        if not test_pkl.exists():
            logger.warning("Skipping %s: no matching pickle %s", test_faa, test_pkl)
            continue

        output_pkl = augment_test_with_pc(
            test_faa=test_faa,
            test_pkl=test_pkl,
            train_faa=train_faa,
            protein_to_pc=protein_to_pc,
            output_pkl=pkl_ext_dir / f"{stem}.pkl",
            work_dir=work_dir,
            mmseqs_bin=mmseqs_bin,
            result_label=stem,
        )
        written[stem] = output_pkl

    logger.info("=== Finished scenario %s: wrote %d pickle(s) to %s ===", scenario_dir, len(written), pkl_ext_dir)
    return written


def build_pc_features(
    ranks: Sequence[str],
    scenarios: Sequence[str],
    output_root: Path,
    mmseqs_bin: str,
) -> Dict[str, Dict[str, Dict[str, Path]]]:
    results: Dict[str, Dict[str, Dict[str, Path]]] = {}

    logger.info("Building PC features for rank(s)=%s, scenario(s)=%s", list(ranks), list(scenarios))

    for rank in ranks:
        rank_key = canonical_rank(rank)
        results[rank_key] = {}

        for scenario in scenarios:
            scenario_dir = output_root / rank_key / f"scenario{scenario}"
            if not scenario_dir.exists():
                logger.warning("Skipping rank='%s' scenario=%s: %s does not exist", rank_key, scenario, scenario_dir)
                continue

            results[rank_key][scenario] = augment_scenario_with_pc(scenario_dir, mmseqs_bin)

    logger.info("All requested rank(s)/scenario(s) complete.")
    return results
