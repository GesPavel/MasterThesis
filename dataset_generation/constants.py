ID_COL = "Accession"
VALID_RANKS = ("Family", "Genus", "Order")
VALID_SCENARIOS = ("1", "2", "3", "4")
DEFAULT_SCENARIO2_TEST_FRACTION = 0.30
DEFAULT_SCENARIO3_TEST_FRACTION = 0.20
UNKNOWN_LABELS = {"Unknown", None}


def canonical_rank(rank: str) -> str:
    normalized = rank.strip()
    if normalized not in VALID_RANKS:
        raise ValueError(f"Invalid rank '{rank}'. Choose one of: {list(VALID_RANKS) + ['all']}")
    return normalized
