import random
import pandas as pd
from tqdm import tqdm


def _get_hmm_set(row):
    return set(row["hmms_hits"])


def _fast_similarity(set_i, set_j):
    return len(set_i & set_j)


def _add_hmm_sets(df):
    df = df.copy()
    df["hmm_set"] = df.apply(_get_hmm_set, axis=1)
    return df


def _build_genome_lookup(df):
    return dict(zip(df["Accession"], df["hmm_set"]))


def _build_taxon_lookup(df, rank):
    return dict(zip(df["Accession"], df[rank]))


def _generate_pairs_for_genome(genome_i, all_genomes, genome_dict, k_neighbors, k_random):
    set_i = genome_dict[genome_i]

    sims = []
    for genome_j in all_genomes:
        if genome_i == genome_j:
            continue
        sim = _fast_similarity(set_i, genome_dict[genome_j])
        sims.append((genome_j, sim))

    sims.sort(key=lambda x: x[1], reverse=True)

    neighbors = [g for g, _ in sims[:k_neighbors]]

    remaining = [g for g in all_genomes if g != genome_i and g not in neighbors]
    random_partners = random.sample(remaining, min(k_random, len(remaining)))

    return neighbors + random_partners


def build_pairs_dataset(df, rank, k_neighbors=10, k_random=10, random_seed=42):
    random.seed(random_seed)

    df = _add_hmm_sets(df)

    genome_dict = _build_genome_lookup(df)
    taxon_dict = _build_taxon_lookup(df, rank)
    all_genomes = df["Accession"].tolist()

    rows = []

    for genome_i in tqdm(all_genomes, desc="Building genome pairs"):
        partners = _generate_pairs_for_genome(
            genome_i=genome_i,
            all_genomes=all_genomes,
            genome_dict=genome_dict,
            k_neighbors=k_neighbors,
            k_random=k_random,
        )

        for genome_j in partners:
            rows.append({
                "genome1": genome_i,
                "genome2": genome_j,
                "same_taxon": int(taxon_dict[genome_i] == taxon_dict[genome_j]),
            })

    pairs_df = pd.DataFrame(rows)

    print(f"Built {len(pairs_df)} total pairs")
    print("Positive pairs:", pairs_df["same_taxon"].sum())
    print("Negative pairs:", len(pairs_df) - pairs_df["same_taxon"].sum())

    return pairs_df