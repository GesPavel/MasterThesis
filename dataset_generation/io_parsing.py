from typing import List

from dataset_generation.constants import VALID_RANKS, VALID_SCENARIOS, canonical_rank


def parse_rank_selection(rank_arg: str) -> List[str]:
    normalized = rank_arg.strip().lower()
    if normalized == "all":
        return list(VALID_RANKS)
    return [canonical_rank(normalized)]


def parse_scenarios_selection(scenarios_arg: str) -> List[str]:
    normalized = scenarios_arg.strip().lower()
    if normalized == "all":
        return list(VALID_SCENARIOS)

    scenarios = [item.strip() for item in normalized.split(",") if item.strip()]
    invalid = [s for s in scenarios if s not in VALID_SCENARIOS]
    if invalid:
        raise ValueError(f"Invalid scenario(s): {invalid}. Valid values are 1, 2, 3, 4, or all.")
    if not scenarios:
        raise ValueError("No scenarios selected.")
    return scenarios
