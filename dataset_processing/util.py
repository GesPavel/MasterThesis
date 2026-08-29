from pathlib import Path

import pandas as pd


def filter_unpickled_dataframe(df, csv_path):
    """
    Load full dataset and keep only genomes present in train_set.csv.

    Matching is done via Accession.
    """
    genomes_to_filter_df = pd.read_csv(csv_path)

    filter_ids = set(genomes_to_filter_df["Accession"])

    df = df[df["Accession"].isin(filter_ids)].copy()

    return df


def get_scenario_data_dir(config, scenario):
    """
    Resolve the directory holding the extended (PC-augmented) pickles for a
    given scenario and the configured taxon rank.

    Layout: <data_root>/<Rank>/scenario<N>/pkl_ext/
    The rank is capitalized to match the on-disk directories (Family/Genus/Order),
    and we always read the extended pickles (they carry both hmms_hits and pc_hits,
    so they serve the hmm/pc/hybrid representations).
    """
    return (
        Path(config["data"]["data_root"])
        / config["data"]["taxon_rank"].capitalize()
        / f"scenario{scenario}"
        / "pkl_ext"
    )


def load_scenario_pickle(config, scenario, stem):
    """Load a single pickle (e.g. 'train', 'test', 'test_highm') for a scenario."""
    data_dir = get_scenario_data_dir(config, scenario)
    return pd.read_pickle(data_dir / f"{stem}.pkl")


def discover_test_variants(config, scenario):
    """
    Return the ordered list of available test pickle stems for a scenario:
    the main 'test' first (if present -- scenario 1 has none), followed by the
    dark-matter variants (test_*.pkl) sorted by name.
    """
    data_dir = get_scenario_data_dir(config, scenario)

    variants = []
    if (data_dir / "test.pkl").exists():
        variants.append("test")

    variants.extend(sorted(p.stem for p in data_dir.glob("test_*.pkl")))

    return variants
