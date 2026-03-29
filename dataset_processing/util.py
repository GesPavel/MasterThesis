import pandas as pd


def load_filtered_dataframe(pickle_path, csv_path):
    """
    Load full dataset and keep only genomes present in train_set.csv.

    Matching is done via Accession.
    """
    df = pd.read_pickle(pickle_path)
    genomes_to_filter_df = pd.read_csv(csv_path)

    filter_ids = set(genomes_to_filter_df["Accession"])

    df = df[df["Accession"].isin(filter_ids)].copy()

    return df
