import logging
import random
from typing import List, Tuple

import pandas as pd

from dataset_generation.constants import ID_COL
from dataset_generation.io_utils import normalize_accession

logger = logging.getLogger(__name__)


def bucketed_taxon_holdout(
    df: pd.DataFrame,
    rank: str,
    test_fraction: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, set[str], set[str]]:
    """Hold out a subset of taxa (at ``rank``) for testing.

    Held-out taxa are sampled across large/medium/small size buckets (by
    cumulative genome-count share) so the held-out set isn't dominated by a
    handful of oversized taxa.
    """
    random.seed(random_seed)
    total_genomes = len(df)
    if total_genomes == 0:
        raise ValueError("Cannot build a taxon holdout from an empty dataframe.")

    target_size = max(1, int(total_genomes * test_fraction))
    logger.info(
        "Taxon holdout for rank='%s': targeting ~%d/%d genomes (%.0f%%)",
        rank, target_size, total_genomes, test_fraction * 100,
    )

    taxon_sizes = df.groupby(rank).size().sort_values(ascending=False)
    total = taxon_sizes.sum()
    cum_frac = taxon_sizes.cumsum() / total

    large = taxon_sizes[cum_frac <= 0.2]
    medium = taxon_sizes[(cum_frac > 0.2) & (cum_frac <= 0.5)]
    small = taxon_sizes[cum_frac > 0.5]
    logger.info(
        "Taxon size buckets for '%s': large=%d, medium=%d, small=%d",
        rank, len(large), len(medium), len(small),
    )

    selected_taxa: List[str] = []
    current_size = 0
    bucket_plan = [(large, 0.3), (medium, 0.3), (small, 0.4)]

    for bucket, fraction in bucket_plan:
        taxa_list = bucket.index.tolist()
        random.shuffle(taxa_list)

        bucket_target = max(1, int(target_size * fraction))
        bucket_size = 0

        for taxon in taxa_list:
            selected_taxa.append(taxon)
            size = int(taxon_sizes[taxon])
            current_size += size
            bucket_size += size
            if bucket_size >= bucket_target or current_size >= target_size:
                break

        if current_size >= target_size:
            break

    test_taxa = set(selected_taxa)
    train_taxa = set(df[rank].unique()) - test_taxa

    train_df = df[df[rank].isin(train_taxa)].copy()
    test_df = df[df[rank].isin(test_taxa)].copy()
    train_ids = set(train_df[ID_COL].map(normalize_accession).unique())
    test_ids = set(test_df[ID_COL].map(normalize_accession).unique())
    logger.info(
        "Taxon holdout complete for '%s': %d test taxa / %d train taxa, %d test genomes / %d train genomes",
        rank, len(test_taxa), len(train_taxa), len(test_ids), len(train_ids),
    )
    return train_df, test_df, train_ids, test_ids
