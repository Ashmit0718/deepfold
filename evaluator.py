"""
DeepFold - Poker Hand Evaluator
Implements card parsing, 5-card hand evaluation, tie-breaking logic,
and 7-card best-hand selection for Texas Hold'em.
"""

from typing import List, Tuple, Union
import itertools

# Category mappings
CATEGORY_NAMES = {
    0: "High Card",
    1: "One Pair",
    2: "Two Pair",
    3: "Three of a Kind",
    4: "Straight",
    5: "Flush",
    6: "Full House",
    7: "Four of a Kind",
    8: "Straight Flush",
    9: "Royal Flush"
}

RANK_MAP = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

REV_RANK_MAP = {
    2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
}

SUITS = ['h', 'd', 'c', 's']
SUIT_SYMBOLS = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}


def parse_card(card_str: str) -> Tuple[int, str]:
    """
    Parses a card string such as 'Ah', '10s', 'Ts', '2c' into (rank: int, suit: str).
    Rank is 2-14 (14 is Ace). Suit is 'h', 'd', 'c', or 's'.
    """
    card_str = card_str.strip()
    if len(card_str) < 2 or len(card_str) > 3:
        raise ValueError(f"Invalid card string: '{card_str}'")
    
    suit = card_str[-1].lower()
    if suit not in SUITS:
        raise ValueError(f"Invalid suit in card string: '{card_str}'")
    
    rank_str = card_str[:-1].upper()
    if rank_str not in RANK_MAP:
        raise ValueError(f"Invalid rank in card string: '{card_str}'")
        
    return (RANK_MAP[rank_str], suit)


def format_card(card: Tuple[int, str]) -> str:
    """Formats a (rank, suit) tuple into a standard string like 'Ah' or '10s'."""
    return f"{REV_RANK_MAP[card[0]]}{card[1]}"


def create_deck() -> List[Tuple[int, str]]:
    """Returns a standard fresh 52-card deck."""
    return [(rank, suit) for rank in range(2, 15) for suit in SUITS]


def evaluate_5card_hand(cards: List[Tuple[int, str]]) -> Tuple[int, Tuple]:
    """
    Evaluates exactly 5 cards and returns a comparable score tuple:
    (category: int, tiebreaker: tuple)
    
    Categories (0 to 9):
      0: High Card        - tiebreaker: (r1, r2, r3, r4, r5)
      1: One Pair         - tiebreaker: (pair_rank, k1, k2, k3)
      2: Two Pair         - tiebreaker: (high_pair, low_pair, k1)
      3: Three of a Kind  - tiebreaker: (trips_rank, k1, k2)
      4: Straight         - tiebreaker: (high_rank,)  [Wheel A-2-3-4-5 has high_rank = 5]
      5: Flush            - tiebreaker: (r1, r2, r3, r4, r5)
      6: Full House       - tiebreaker: (trips_rank, pair_rank)
      7: Four of a Kind   - tiebreaker: (quad_rank, kicker)
      8: Straight Flush   - tiebreaker: (high_rank,)
      9: Royal Flush      - tiebreaker: (14,)
    """
    if len(cards) != 5:
        raise ValueError("evaluate_5card_hand requires exactly 5 cards.")
    
    ranks = sorted([c[0] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    
    is_flush = (suits[0] == suits[1] == suits[2] == suits[3] == suits[4])
    
    # Check straight
    is_straight = False
    straight_high = 0
    unique_ranks = sorted(set(ranks), reverse=True)
    if len(unique_ranks) == 5:
        if ranks[0] - ranks[4] == 4:
            is_straight = True
            straight_high = ranks[0]
        elif ranks == [14, 5, 4, 3, 2]:  # Wheel (Ace-to-5)
            is_straight = True
            straight_high = 5

    # Check Straight Flush / Royal Flush
    if is_flush and is_straight:
        if straight_high == 14:
            return (9, (14,))
        return (8, (straight_high,))
        
    # Count rank frequencies
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    freq = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    
    # Four of a Kind: 4 of same rank + 1 kicker
    if freq[0][1] == 4:
        return (7, (freq[0][0], freq[1][0]))
        
    # Full House: 3 of same rank + 2 of another rank
    if freq[0][1] == 3 and freq[1][1] == 2:
        return (6, (freq[0][0], freq[1][0]))
        
    # Flush: 5 cards same suit, not straight
    if is_flush:
        return (5, tuple(ranks))
        
    # Straight: 5 sequential ranks, not flush
    if is_straight:
        return (4, (straight_high,))
        
    # Three of a Kind: 3 of same rank + 2 distinct kickers
    if freq[0][1] == 3:
        return (3, (freq[0][0], freq[1][0], freq[2][0]))
        
    # Two Pair: 2 of one rank + 2 of second rank + 1 kicker
    if freq[0][1] == 2 and freq[1][1] == 2:
        return (2, (freq[0][0], freq[1][0], freq[2][0]))
        
    # One Pair: 2 of one rank + 3 distinct kickers
    if freq[0][1] == 2:
        return (1, (freq[0][0], freq[1][0], freq[2][0], freq[3][0]))
        
    # High Card: all distinct, not straight, not flush
    return (0, tuple(ranks))


def evaluate_7card_hand(cards: List[Tuple[int, str]]) -> Tuple[Tuple[int, Tuple], List[Tuple[int, str]]]:
    """
    Given 7 cards (e.g. 2 hole cards + 5 community cards), returns:
    (best_score, best_5_cards)
    where best_score is comparable with other scores via > and ==.
    """
    if len(cards) != 7:
        raise ValueError(f"evaluate_7card_hand requires exactly 7 cards, got {len(cards)}.")
    
    best_score = None
    best_5 = None
    
    for combo in itertools.combinations(cards, 5):
        score = evaluate_5card_hand(list(combo))
        if best_score is None or score > best_score:
            best_score = score
            best_5 = list(combo)
            
    return best_score, best_5


def get_hand_description(score: Tuple[int, Tuple]) -> str:
    """Returns a readable description of the hand score."""
    cat, tb = score
    cat_name = CATEGORY_NAMES.get(cat, "Unknown")
    
    if cat == 9:
        return "Royal Flush"
    elif cat == 8:
        return f"Straight Flush, {REV_RANK_MAP[tb[0]]}-high"
    elif cat == 7:
        return f"Four of a Kind, {REV_RANK_MAP[tb[0]]}s (kicker {REV_RANK_MAP[tb[1]]})"
    elif cat == 6:
        return f"Full House, {REV_RANK_MAP[tb[0]]}s full of {REV_RANK_MAP[tb[1]]}s"
    elif cat == 5:
        ranks_str = ", ".join(REV_RANK_MAP[r] for r in tb)
        return f"Flush ({ranks_str})"
    elif cat == 4:
        return f"Straight, {REV_RANK_MAP[tb[0]]}-high"
    elif cat == 3:
        return f"Three of a Kind, {REV_RANK_MAP[tb[0]]}s (kickers {REV_RANK_MAP[tb[1]]}, {REV_RANK_MAP[tb[2]]})"
    elif cat == 2:
        return f"Two Pair, {REV_RANK_MAP[tb[0]]}s and {REV_RANK_MAP[tb[1]]}s (kicker {REV_RANK_MAP[tb[2]]})"
    elif cat == 1:
        return f"One Pair, {REV_RANK_MAP[tb[0]]}s"
    else:
        return f"High Card, {REV_RANK_MAP[tb[0]]}"
