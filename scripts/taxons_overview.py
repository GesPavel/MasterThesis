import os

import pandas as pd

from dataset_processing import DATASET_PICKLE_PATH, DATASET_OVERVIEW_FOLDER

def main():
    TAXONOMIC_RANK = 'Genus'  # Change this to 'Genus', 'Order', etc. if desired
    output_file = f'{TAXONOMIC_RANK}_distribution.csv'
    output_path = os.path.join(DATASET_OVERVIEW_FOLDER, output_file)
    # -----------------------------
    # 1. Load data
    # -----------------------------
    df = pd.read_pickle(DATASET_PICKLE_PATH)

    total_genomes = len(df)
    print(f"Dataset loaded from pickle at {DATASET_PICKLE_PATH}, total genomes: {total_genomes}")
    # -----------------------------
    # 2. Group by taxon
    # -----------------------------
    print(f"Grouping genomes by {TAXONOMIC_RANK} rank... ")
    taxon_counts = df.groupby(TAXONOMIC_RANK).size().reset_index(name='genome_count')


    # -----------------------------
    # 3. Compute relative size
    # -----------------------------
    taxon_counts['fraction'] = taxon_counts['genome_count'] / total_genomes
    taxon_counts['percentage'] = taxon_counts['fraction'] * 100

    # -----------------------------
    # 4. Sort by size (largest first)
    # -----------------------------
    taxon_counts = taxon_counts.sort_values(by='genome_count', ascending=False)

    # -----------------------------
    # 5. Save to CSV
    # -----------------------------
    taxon_counts.to_csv(output_path, index=False)

    # -----------------------------
    # 6. Print quick summary
    # -----------------------------
    print(f"Saved to {output_path}")
    print(f"Total genomes: {total_genomes}")
    print(f"Total families: {len(taxon_counts)}")

    print("\nTop 10 largest families:")
    print(taxon_counts.head(10))


if __name__ == '__main__':
    main()