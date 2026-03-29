import numpy as np
from tqdm import tqdm


def build_name_to_hmm_set_dict(df):
    genome_dict = {}

    for _, row in df.iterrows():
        acc = row["Accession"]
        hmms = row["hmms_hits"]

        hmms_set = set(hmms) if isinstance(hmms, list) else set()

        genome_dict[acc] = {
            "hmms": hmms_set,
            "len": len(hmms_set)
        }

    return genome_dict



def compute_features(pairs_df, genome_dict, process_target_var=True):
    X = []
    y = []

    genome1_arr = pairs_df["genome1"].to_numpy()
    genome2_arr = pairs_df["genome2"].to_numpy()

    if process_target_var:
        y_arr = pairs_df["y"].to_numpy()

    for i in tqdm(range(len(pairs_df)), total=len(pairs_df)):
        g1 = genome1_arr[i]
        g2 = genome2_arr[i]

        d1 = genome_dict.get(g1)
        d2 = genome_dict.get(g2)

        if d1 is None or d2 is None:
            continue

        h1 = d1["hmms"]
        h2 = d2["hmms"]

        len1 = d1["len"]
        len2 = d2["len"]

        inter = len(h1 & h2)
        union = len1 + len2 - inter
        jaccard = inter / union if union > 0 else 0.0

        X.append([
            inter,
            jaccard,
            len1,
            len2,
            abs(len1 - len2)
        ])


    return (
        np.array(X),
        y_arr if process_target_var else np.array([]),
        genome1_arr,
        genome2_arr
    )