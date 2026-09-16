import numpy as np
from scipy import sparse

from dataset_processing.timing import timed, log


# Slice size for the pairwise intersection loop, in matrix nonzeros. Each slice
# materialises two hit-matrix subsets plus their product, so the peak stays a few
# hundred MB regardless of how many hits a genome carries.
NNZ_PER_SLICE = 20_000_000


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


def _build_hit_matrix(genome_dict, key=None):
    """One 0/1 row per genome, one column per distinct hit.

    There are far fewer genomes than pairs, so building this once lets the
    per-pair set operations become sparse row products that scipy runs in C.
    """
    genomes = list(genome_dict)
    row_of_genome = {genome: i for i, genome in enumerate(genomes)}

    column_of_hit = {}
    indices = []
    indptr = [0]
    for genome in genomes:
        hits = genome_dict[genome] if key is None else genome_dict[genome][key]
        for hit in hits:
            indices.append(column_of_hit.setdefault(hit, len(column_of_hit)))
        indptr.append(len(indices))

    matrix = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.int32),
            np.array(indices, dtype=np.int32),
            np.array(indptr, dtype=np.int64),
        ),
        shape=(len(genomes), len(column_of_hit)),
    )

    return matrix, row_of_genome


def _row_indices(matrix_lookup, genomes):
    return np.fromiter(
        (matrix_lookup[genome] for genome in genomes),
        dtype=np.int32,
        count=len(genomes),
    )


def _intersection_sizes(matrix, rows1, rows2):
    """|hits(a) & hits(b)| for every pair."""
    unique1, inverse1 = np.unique(rows1, return_inverse=True)
    unique2, inverse2 = np.unique(rows2, return_inverse=True)

    # Prediction pairs are a full test x train cross product, so the grid of all
    # unique-genome combinations is exactly the pair list and one sparse matrix
    # product answers every pair at once. Sampled pair lists (training,
    # calibration) cover only a sliver of that grid, and the condition below
    # sends them to the slice loop instead of computing the whole rectangle.
    if len(unique1) * len(unique2) <= len(rows1):
        grid = (matrix[unique1] @ matrix[unique2].T).toarray()
        return grid[inverse1, inverse2]

    # Slice the pairs so the two matrix subsets and their product stay small.
    mean_nnz = max(1.0, matrix.nnz / max(1, matrix.shape[0]))
    slice_size = max(1, int(NNZ_PER_SLICE / mean_nnz))

    intersections = np.empty(len(rows1), dtype=np.int32)
    for start in range(0, len(rows1), slice_size):
        stop = start + slice_size
        product = matrix[rows1[start:stop]].multiply(matrix[rows2[start:stop]])
        intersections[start:stop] = np.asarray(product.sum(axis=1)).ravel()

    return intersections


def _compute_feature_block(matrix, rows1, rows2, feature_config):
    hits_per_genome = matrix.getnnz(axis=1)
    len1 = hits_per_genome[rows1]
    len2 = hits_per_genome[rows2]

    inter = None
    union = None

    if feature_config.get("use_intersection", False) or feature_config.get("use_jaccard", False):
        inter = _intersection_sizes(matrix, rows1, rows2)

    if feature_config.get("use_jaccard", False):
        # |a | b| == |a| + |b| - |a & b|, so the union needs no second pass.
        union = len1 + len2 - inter

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

    return np.empty((len(rows1), 0))


def compute_features(pairs_df, genome_dict, feature_config, representation="hmm", process_target_var=True):
    log(f"compute_features: {len(pairs_df):,} pairs, representation={representation}")

    with timed("compute_features: total"):
        with timed("compute_features: pair column extraction"):
            g1_all = pairs_df["genome1"].to_numpy()
            g2_all = pairs_df["genome2"].to_numpy()

            genome_list_1 = g1_all.tolist()
            genome_list_2 = g2_all.tolist()

        if representation in {"hmm", "pc"}:
            with timed("compute_features: hit matrix"):
                matrix, row_of_genome = _build_hit_matrix(genome_dict)

            with timed("compute_features: row lookup"):
                rows1 = _row_indices(row_of_genome, g1_all)
                rows2 = _row_indices(row_of_genome, g2_all)

            with timed("compute_features: feature block"):
                X = _compute_feature_block(matrix, rows1, rows2, feature_config)

        elif representation == "hybrid":
            with timed("compute_features: hit matrix"):
                matrix_hmm, row_of_genome = _build_hit_matrix(genome_dict, key="hmm")
                matrix_pc, _ = _build_hit_matrix(genome_dict, key="pc")

            with timed("compute_features: row lookup"):
                rows1 = _row_indices(row_of_genome, g1_all)
                rows2 = _row_indices(row_of_genome, g2_all)

            with timed("compute_features: feature block (hmm)"):
                X_hmm = _compute_feature_block(matrix_hmm, rows1, rows2, feature_config)

            with timed("compute_features: feature block (pc)"):
                X_pc = _compute_feature_block(matrix_pc, rows1, rows2, feature_config)

            with timed("compute_features: hstack"):
                X = np.hstack([X_hmm, X_pc])

        else:
            raise ValueError(f"Unsupported representation: {representation}")

        y = None
        if process_target_var:
            y = pairs_df["same_taxon"].to_numpy()

    log(f"compute_features: X shape {X.shape}")

    return X, y, genome_list_1, genome_list_2
