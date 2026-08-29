import numpy as np

from dataset_processing.timing import timed, log


def build_name_to_hmm_set_dict(df):
    accessions = df["Accession"].to_numpy()
    hmm_hits = df["hmms_hits"].to_numpy()
    return {accession: set(hits) for accession, hits in zip(accessions, hmm_hits)}


def build_name_to_feature_dict(df, representation):
    accessions = df["Accession"].to_numpy()

    if representation == "hmm":
        hmm_hits = df["hmms_hits"].to_numpy()
        return {accession: set(hits) for accession, hits in zip(accessions, hmm_hits)}

    if representation == "pc":
        pc_hits = df["pc_hits"].to_numpy()
        return {accession: set(hits) for accession, hits in zip(accessions, pc_hits)}

    if representation == "hybrid":
        hmm_hits = df["hmms_hits"].to_numpy()
        pc_hits = df["pc_hits"].to_numpy()
        return {
            accession: {"hmm": set(hmm), "pc": set(pc)}
            for accession, hmm, pc in zip(accessions, hmm_hits, pc_hits)
        }

    raise ValueError(f"Unsupported representation: {representation}")


def _compute_feature_block(set1_list, set2_list, feature_config):
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
        return np.column_stack(columns)

    return np.empty((len(set1_list), 0))


def compute_features(pairs_df, genome_dict, feature_config, representation="hmm", process_target_var=True):
    log(f"compute_features: {len(pairs_df):,} pairs, representation={representation}")

    with timed("compute_features: total"):
        with timed("compute_features: pair column extraction"):
            g1_all = pairs_df["genome1"].to_numpy()
            g2_all = pairs_df["genome2"].to_numpy()

            genome_list_1 = g1_all.tolist()
            genome_list_2 = g2_all.tolist()

        if representation in {"hmm", "pc"}:
            with timed("compute_features: set lookup"):
                set1_list = [genome_dict[g] for g in g1_all]
                set2_list = [genome_dict[g] for g in g2_all]

            with timed("compute_features: feature block"):
                X = _compute_feature_block(set1_list, set2_list, feature_config)

        elif representation == "hybrid":
            with timed("compute_features: set lookup"):
                set1_hmm = [genome_dict[g]["hmm"] for g in g1_all]
                set2_hmm = [genome_dict[g]["hmm"] for g in g2_all]
                set1_pc = [genome_dict[g]["pc"] for g in g1_all]
                set2_pc = [genome_dict[g]["pc"] for g in g2_all]

            with timed("compute_features: feature block (hmm)"):
                X_hmm = _compute_feature_block(set1_hmm, set2_hmm, feature_config)

            with timed("compute_features: feature block (pc)"):
                X_pc = _compute_feature_block(set1_pc, set2_pc, feature_config)

            with timed("compute_features: hstack"):
                X = np.hstack([X_hmm, X_pc])

        else:
            raise ValueError(f"Unsupported representation: {representation}")

        y = None
        if process_target_var:
            y = pairs_df["same_taxon"].to_numpy()

    log(f"compute_features: X shape {X.shape}")

    return X, y, genome_list_1, genome_list_2

