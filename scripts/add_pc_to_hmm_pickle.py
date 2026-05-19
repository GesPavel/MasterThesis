from pathlib import Path

import pandas as pd
from tqdm import tqdm


def extract_genome_accession(protein_id: str) -> str:
    """
    Example:
        HQ641347.1_20 -> HQ641347.1
    """
    return protein_id.rsplit("_", 1)[0]


def build_pc_pickle(
    mmseqs_tsv_path: str,
    hmm_pickle_path: str,
    output_pickle_path: str,
):
    Path(output_pickle_path).parent.mkdir(parents=True, exist_ok=True)
    print("Loading HMM dataframe...")
    df = pd.read_pickle(hmm_pickle_path)

    print("Loading MMseqs2 cluster TSV...")
    clusters_df = pd.read_csv(
        mmseqs_tsv_path,
        sep="\t",
        header=None,
        names=["pc_id", "protein_id"]
    )

    print(f"Loaded {len(clusters_df):,} protein-cluster assignments")

    # ---------------------------------------------------------
    # Build genome -> PCs mapping
    # ---------------------------------------------------------
    print("Building genome -> PC mapping...")

    genome_to_pcs = {}

    for row in tqdm(clusters_df.itertuples(index=False), total=len(clusters_df)):
        pc_id = row.pc_id
        protein_id = row.protein_id

        genome_id = extract_genome_accession(protein_id)

        if genome_id not in genome_to_pcs:
            genome_to_pcs[genome_id] = set()

        genome_to_pcs[genome_id].add(pc_id)

    print(f"Built mappings for {len(genome_to_pcs):,} genomes")

    # ---------------------------------------------------------
    # Attach PC profiles to dataframe
    # ---------------------------------------------------------
    print("Attaching PC profiles to dataframe...")

    accessions = df["Accession"].to_numpy()

    pc_hits = []
    missing_genomes = 0

    for accession in tqdm(accessions):
        pcs = genome_to_pcs.get(accession)

        if pcs is None:
            pcs = set()
            missing_genomes += 1

        pc_hits.append(list(pcs))

    # Keep ALL original columns from HMM dataframe
    # and simply add pc_hits as a new column.
    df["pc_hits"] = pc_hits

    print(f"Genomes missing PCs: {missing_genomes:,}")

    # ---------------------------------------------------------
    # Some useful diagnostics
    # ---------------------------------------------------------
    pc_counts = [len(x) for x in pc_hits]

    print("\nPC statistics:")
    print(f"Mean PCs/genome: {sum(pc_counts) / len(pc_counts):.2f}")
    print(f"Median PCs/genome: {pd.Series(pc_counts).median():.2f}")
    print(f"Max PCs/genome: {max(pc_counts)}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    print("\nSaving pickle...")
    df.to_pickle(output_pickle_path)

    print(f"Done. Saved to: {output_pickle_path}")


if __name__ == "__main__":
    build_pc_pickle(
        mmseqs_tsv_path="raw_data/default_cluster_MSL40.tsv",
        hmm_pickle_path="raw_data/MSL40_complete.pkl",
        output_pickle_path="raw_data/joined/MSL40_hmm_pc.pkl",
    )