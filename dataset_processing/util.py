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


def load_pickle_given_config(config, type):
    if type not in ["train", "test"]:
        raise Exception("Trying to load invalid pickle type")
    if config["data"]["representation"] == "hybrid":
        raise NotImplementedError("Hybrid representation is not implemented for loading")

    path_to_data_dir = Path(config["data"]["data_root"]) / config["data"][
        "taxon_rank"].lower() / f"scenario{config["data"]["dataset_number"]}"
    df = pd.read_pickle(path_to_data_dir / f"{type}_{config["data"]["representation"]}.pkl")
    return df
