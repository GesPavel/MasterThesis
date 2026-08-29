import gc
import random

import pandas as pd
from tqdm import tqdm

from dataset_processing.timing import timed, log


def _add_feature_sets(df):
    df = df.copy()

    if "hmms_hits" in df.columns:
        df["hmm_set"] = df["hmms_hits"].apply(set)

    if "pc_hits" in df.columns:
        df["pc_set"] = df["pc_hits"].apply(set)

    return df


def _fast_similarity(features_i, features_j, representation):
    if representation == "hmm":
        return len(features_i & features_j)

    elif representation == "pc":
        return len(features_i & features_j)

    elif representation == "hybrid":
        hmm_sim = len(features_i["hmm"] & features_j["hmm"])
        pc_sim = len(features_i["pc"] & features_j["pc"])

        return hmm_sim + pc_sim

    else:
        raise ValueError(f"Unsupported representation: {representation}")


def _build_genome_lookup(df, representation):
    if representation == "hmm":
        return dict(zip(df["Accession"], df["hmm_set"]))

    elif representation == "pc":
        return dict(zip(df["Accession"], df["pc_set"]))

    elif representation == "hybrid":
        return {
            acc: {
                "hmm": hmm_set,
                "pc": pc_set,
            }
            for acc, hmm_set, pc_set in zip(
                df["Accession"],
                df["hmm_set"],
                df["pc_set"]
            )
        }

    else:
        raise ValueError(f"Unsupported representation: {representation}")


def _generate_pairs_for_genome(
    genome_i,
    same_taxon_candidates,
    candidate_genomes,
    genome_dict,
    representation,
    k_pos
):
    """
    Partners for one query genome, as (accession, same_taxon) tuples:

      * k_pos same-taxon partners, capped by how many actually exist,
      * ceil(k_pos / 2) hard negatives -- the most similar wrong-taxon genomes,
      * floor(k_pos / 2) random negatives drawn from the wrong-taxon rest.

    The label is known from the bucket a partner was drawn from, so no taxon
    lookup is needed downstream. A genome with no same-taxon partner returns
    nothing: it would otherwise contribute negatives only.
    """
    positive_pool = [g for g in same_taxon_candidates if g != genome_i]
    k_pos = min(k_pos, len(positive_pool))

    if k_pos == 0:
        return []

    n_hard = -(-k_pos // 2)  # ceil
    n_random = k_pos // 2

    features_i = genome_dict[genome_i]
    same_taxon = set(same_taxon_candidates)

    # genome_i itself is in same_taxon, so this also drops the self-pair.
    sims = [
        (genome_j, _fast_similarity(features_i, genome_dict[genome_j], representation))
        for genome_j in candidate_genomes
        if genome_j not in same_taxon
    ]
    sims.sort(key=lambda x: x[1], reverse=True)

    hard_negatives = [g for g, _ in sims[:n_hard]]

    remaining = [g for g, _ in sims[n_hard:]]
    random_negatives = random.sample(remaining, min(n_random, len(remaining)))

    return (
        [(g, 1) for g in random.sample(positive_pool, k_pos)]
        + [(g, 0) for g in hard_negatives + random_negatives]
    )


def build_pairs_dataset(
    df,
    rank,
    representation,
    partner_df=None,
    k_pos=10,
    random_seed=42
):
    """
    Build labelled genome pairs: for every genome of `df`, up to k_pos same-taxon
    partners and an equal number of wrong-taxon ones, half of them the most
    similar (hard) negatives and half random. See _generate_pairs_for_genome.

    With `partner_df` left as None the partners are drawn from `df` itself
    (self-pairing). Passing a `partner_df` draws them from that frame instead --
    e.g. calibration genomes paired against the training set, so calibration sees
    the same cross-set setting as prediction rather than a self-paired one.

    Column order follows prediction time, where genome1 is the train genome and
    genome2 the queried one (see probability_prediction.predict_probabilities):
    in cross-set mode the partner goes to genome1 and the `df` genome to genome2,
    so len1/len2 keep the same meaning at every stage of the pipeline.
    """
    random.seed(random_seed)

    cross_set = partner_df is not None

    with timed("build_pairs_dataset: feature sets + lookups"):
        df = _add_feature_sets(df)
        partner_df = _add_feature_sets(partner_df) if cross_set else df

        genome_dict = _build_genome_lookup(df, representation)

        if cross_set:
            genome_dict.update(_build_genome_lookup(partner_df, representation))

        # The only taxonomy structure kept: candidates grouped by taxon. It gives
        # the positive pool directly and, by exclusion, the negative one.
        taxon_to_candidates = partner_df.groupby(rank)["Accession"].apply(list).to_dict()

        query_genomes = list(zip(df["Accession"], df[rank]))
        candidate_genomes = partner_df["Accession"].tolist()

    log(
        f"build_pairs_dataset: {len(query_genomes):,} query x {len(candidate_genomes):,} candidate genomes "
        f"-> {len(query_genomes) * len(candidate_genomes):,} similarity comparisons"
    )

    rows = []
    skipped = 0

    with timed("build_pairs_dataset: all-vs-all similarity + partner selection"):
        for genome_i, taxon_i in tqdm(query_genomes, desc="Building genome pairs"):
            partners = _generate_pairs_for_genome(
                genome_i=genome_i,
                same_taxon_candidates=taxon_to_candidates.get(taxon_i, ()),
                candidate_genomes=candidate_genomes,
                genome_dict=genome_dict,
                representation=representation,
                k_pos=k_pos,
            )

            if not partners:
                skipped += 1
                continue

            for genome_j, same_taxon in partners:
                genome1, genome2 = (genome_j, genome_i) if cross_set else (genome_i, genome_j)

                rows.append({
                    "genome1": genome1,
                    "genome2": genome2,
                    "same_taxon": same_taxon,
                })

    # Feature sets and lookups are dead from here on; the frame built below is
    # the memory peak, so drop them before allocating it.
    with timed("build_pairs_dataset: free lookups"):
        del df, partner_df, genome_dict, taxon_to_candidates, query_genomes, candidate_genomes
        gc.collect()

    with timed("build_pairs_dataset: dataframe construction"):
        pairs_df = pd.DataFrame(rows)

    print(f"Built {len(pairs_df)} total pairs")
    print("Positive pairs:", pairs_df["same_taxon"].sum())
    print("Negative pairs:", len(pairs_df) - pairs_df["same_taxon"].sum())
    print(f"Genomes skipped (no same-taxon partner available): {skipped}")

    return pairs_df