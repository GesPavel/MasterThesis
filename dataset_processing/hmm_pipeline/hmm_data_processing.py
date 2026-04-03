import numpy as np


def build_name_to_hmm_set_dict(df):
    accessions = df["Accession"].to_numpy()
    hmm_hits = df["hmms_hits"].to_numpy()
    return {accession: set(hits) for accession, hits in zip(accessions, hmm_hits)}


def compute_features(pairs_df, genome_dict, feature_config, process_target_var=True):
    g1_all = pairs_df["genome1"].to_numpy()
    g2_all = pairs_df["genome2"].to_numpy()

    genome_list_1 = g1_all.tolist()
    genome_list_2 = g2_all.tolist()

    set1_list = [genome_dict[g] for g in g1_all]
    set2_list = [genome_dict[g] for g in g2_all]

    len1 = np.fromiter((len(s) for s in set1_list), dtype=np.int32, count=len(set1_list))
    len2 = np.fromiter((len(s) for s in set2_list), dtype=np.int32, count=len(set2_list))

    inter = None
    union = None

    if feature_config.get("use_intersection", False) or feature_config.get("use_jaccard", False):
        inter = np.fromiter(
            (len(s1 & s2) for s1, s2 in zip(set1_list, set2_list)),
            dtype=np.int32,
            count=len(set1_list),
        )

    if feature_config.get("use_jaccard", False):
        union = np.fromiter(
            (len(s1 | s2) for s1, s2 in zip(set1_list, set2_list)),
            dtype=np.int32,
            count=len(set1_list),
        )

    columns = []

    if feature_config.get("use_intersection", False):
        columns.append(inter)

    if feature_config.get("use_jaccard", False):
        jaccard = np.divide(
            inter,
            union,
            out=np.zeros_like(inter, dtype=float),
            where=union > 0,
        )
        columns.append(jaccard)

    if feature_config.get("use_len1", False):
        columns.append(len1)

    if feature_config.get("use_len2", False):
        columns.append(len2)

    if feature_config.get("use_abs_len_diff", False):
        columns.append(np.abs(len1 - len2))

    if columns:
        X = np.column_stack(columns)
    else:
        X = np.empty((len(genome_list_1), 0))

    y = None
    if process_target_var:
        y = pairs_df["same_taxon"].to_numpy()

    return X, y, genome_list_1, genome_list_2