"""
DeepFold - Feature Engineering Module
Converts Hero's 7 cards (2 hole cards + 5 community cards) and opponent count N
into a fixed-size, explainable numerical feature vector for model training and inference.
"""

from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd

from evaluator import parse_card, evaluate_7card_hand


# Canonical list and ordering of feature columns
FEATURE_NAMES = [
    "hand_category",
    "has_pair",
    "has_two_pair",
    "has_trips",
    "has_straight",
    "has_flush",
    "has_full_house",
    "has_quads",
    "has_straight_flush",
    "high_card_rank",
    "second_high_rank",
    "primary_rank",
    "secondary_rank",
    "max_suit_count",
    "n_opponents",
    "is_pocket_pair",
    "is_suited_hole",
    "hole_high_rank",
    "hole_low_rank"
]


def extract_features_from_parsed(
    hero_cards: List[Tuple[int, str]],
    comm_cards: List[Tuple[int, str]],
    n_opponents: int
) -> Dict[str, Union[int, float]]:
    """
    Extracts numerical features given parsed (rank, suit) tuples for Hero's 2 hole cards,
    5 community cards, and integer number of opponents N.
    """
    all_7 = hero_cards + comm_cards
    
    # 1. Evaluate best 5-card hand
    (cat, tiebreaker), _ = evaluate_7card_hand(all_7)
    
    # 2. Binary category flags
    has_pair = 1 if cat == 1 else 0
    has_two_pair = 1 if cat == 2 else 0
    has_trips = 1 if cat == 3 else 0
    has_straight = 1 if cat == 4 else 0
    has_flush = 1 if cat == 5 else 0
    has_full_house = 1 if cat == 6 else 0
    has_quads = 1 if cat == 7 else 0
    has_straight_flush = 1 if cat >= 8 else 0
    
    # 3. Ranks of all 7 cards
    all_ranks = sorted([c[0] for c in all_7], reverse=True)
    distinct_ranks = sorted(set(all_ranks), reverse=True)
    
    high_card_rank = distinct_ranks[0]
    second_high_rank = distinct_ranks[1] if len(distinct_ranks) > 1 else distinct_ranks[0]
    
    # 4. Tiebreaker primary and secondary ranks from best hand
    primary_rank = tiebreaker[0] if len(tiebreaker) > 0 else high_card_rank
    secondary_rank = tiebreaker[1] if len(tiebreaker) > 1 else 0
    
    # 5. Suit distribution among the 7 cards
    suit_counts = {}
    for c in all_7:
        suit_counts[c[1]] = suit_counts.get(c[1], 0) + 1
    max_suit_count = max(suit_counts.values()) if suit_counts else 0
    
    # 6. Hero hole card specific signals
    hole_r1, hole_s1 = hero_cards[0]
    hole_r2, hole_s2 = hero_cards[1]
    is_pocket_pair = 1 if hole_r1 == hole_r2 else 0
    is_suited_hole = 1 if hole_s1 == hole_s2 else 0
    hole_high_rank = max(hole_r1, hole_r2)
    hole_low_rank = min(hole_r1, hole_r2)
    
    return {
        "hand_category": cat,
        "has_pair": has_pair,
        "has_two_pair": has_two_pair,
        "has_trips": has_trips,
        "has_straight": has_straight,
        "has_flush": has_flush,
        "has_full_house": has_full_house,
        "has_quads": has_quads,
        "has_straight_flush": has_straight_flush,
        "high_card_rank": high_card_rank,
        "second_high_rank": second_high_rank,
        "primary_rank": primary_rank,
        "secondary_rank": secondary_rank,
        "max_suit_count": max_suit_count,
        "n_opponents": int(n_opponents),
        "is_pocket_pair": is_pocket_pair,
        "is_suited_hole": is_suited_hole,
        "hole_high_rank": hole_high_rank,
        "hole_low_rank": hole_low_rank
    }


def extract_features_single(
    hero_card_strs: List[str],
    comm_card_strs: List[str],
    n_opponents: int
) -> pd.DataFrame:
    """
    Extracts features for single inference call from card strings.
    Returns pandas DataFrame of shape (1, n_features) matching FEATURE_NAMES.
    """
    hero_cards = [parse_card(c) for c in hero_card_strs]
    comm_cards = [parse_card(c) for c in comm_card_strs]
    feat_dict = extract_features_from_parsed(hero_cards, comm_cards, n_opponents)
    return pd.DataFrame([feat_dict])[FEATURE_NAMES]


def extract_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw simulation DataFrame containing:
    ['hero_c1', 'hero_c2', 'comm_c1', 'comm_c2', 'comm_c3', 'comm_c4', 'comm_c5', 'n_opponents', ...]
    Extracts feature matrix DataFrame for all rows.
    """
    features_list = []
    
    # Process row by row
    for row in df.itertuples(index=False):
        hero_cards = [parse_card(row.hero_c1), parse_card(row.hero_c2)]
        comm_cards = [
            parse_card(row.comm_c1),
            parse_card(row.comm_c2),
            parse_card(row.comm_c3),
            parse_card(row.comm_c4),
            parse_card(row.comm_c5)
        ]
        feats = extract_features_from_parsed(hero_cards, comm_cards, row.n_opponents)
        features_list.append(feats)
        
    feat_df = pd.DataFrame(features_list)[FEATURE_NAMES]
    return feat_df
