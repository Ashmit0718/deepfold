"""
Comprehensive unit tests for DeepFold poker hand evaluator.
Tests all 10 hand categories and detailed kicker tie-breaking rules.
"""

import pytest
from evaluator import (
    parse_card,
    format_card,
    create_deck,
    evaluate_5card_hand,
    evaluate_7card_hand,
    get_hand_description,
    CATEGORY_NAMES
)


def cards_from_str(s: str):
    return [parse_card(c) for c in s.split()]


def test_deck_and_parsing():
    deck = create_deck()
    assert len(deck) == 52
    assert len(set(deck)) == 52
    
    # Test card parsing
    assert parse_card("Ah") == (14, "h")
    assert parse_card("10s") == (10, "s")
    assert parse_card("Ts") == (10, "s")
    assert parse_card("2c") == (2, "c")
    assert parse_card("Kd") == (13, "d")
    
    # Format roundtrip
    assert format_card((14, "h")) == "Ah"
    assert format_card((10, "s")) == "10s"


def test_royal_flush_beats_straight_flush():
    royal = evaluate_5card_hand(cards_from_str("Ah Kh Qh Jh 10h"))
    straight_flush = evaluate_5card_hand(cards_from_str("Kh Qh Jh 10h 9h"))
    
    assert royal[0] == 9
    assert straight_flush[0] == 8
    assert royal > straight_flush


def test_straight_flush_rankings():
    sf_king = evaluate_5card_hand(cards_from_str("Kh Qh Jh 10h 9h"))
    sf_nine = evaluate_5card_hand(cards_from_str("9d 8d 7d 6d 5d"))
    wheel_sf = evaluate_5card_hand(cards_from_str("5c 4c 3c 2c Ac"))  # 5-high straight flush
    six_sf = evaluate_5card_hand(cards_from_str("6s 5s 4s 3s 2s"))    # 6-high straight flush
    
    assert sf_king > sf_nine
    assert six_sf > wheel_sf  # 6-high beats 5-high (wheel)
    assert wheel_sf[0] == 8
    assert wheel_sf[1] == (5,)


def test_straight_flush_beats_four_of_a_kind():
    sf = evaluate_5card_hand(cards_from_str("5s 4s 3s 2s As"))
    quads = evaluate_5card_hand(cards_from_str("Ah As Ac Ad Kd"))
    assert sf[0] == 8
    assert quads[0] == 7
    assert sf > quads


def test_four_of_a_kind_tiebreakers():
    quad_aces = evaluate_5card_hand(cards_from_str("Ah As Ac Ad 2d"))
    quad_kings = evaluate_5card_hand(cards_from_str("Kh Ks Kc Kd Ad"))
    assert quad_aces > quad_kings
    
    # Same quads, kicker tiebreak
    quad_tens_k = evaluate_5card_hand(cards_from_str("10h 10s 10c 10d Kd"))
    quad_tens_q = evaluate_5card_hand(cards_from_str("10h 10s 10c 10d Qd"))
    assert quad_tens_k > quad_tens_q


def test_four_of_a_kind_beats_full_house():
    quads = evaluate_5card_hand(cards_from_str("2h 2s 2c 2d 3d"))
    fh = evaluate_5card_hand(cards_from_str("Ah As Ac Kd Kh"))
    assert quads[0] == 7
    assert fh[0] == 6
    assert quads > fh


def test_full_house_tiebreakers():
    # Higher trips wins regardless of pair
    fh_7s_full_of_2s = evaluate_5card_hand(cards_from_str("7h 7s 7c 2d 2h"))
    fh_6s_full_of_aces = evaluate_5card_hand(cards_from_str("6h 6s 6c Ad Ah"))
    assert fh_7s_full_of_2s > fh_6s_full_of_aces
    
    # Same trips, higher pair wins
    fh_aces_kings = evaluate_5card_hand(cards_from_str("Ah As Ac Kd Kh"))
    fh_aces_queens = evaluate_5card_hand(cards_from_str("Ah As Ac Qd Qh"))
    assert fh_aces_kings > fh_aces_queens


def test_full_house_beats_flush():
    fh = evaluate_5card_hand(cards_from_str("2h 2s 2c 3d 3h"))
    flush = evaluate_5card_hand(cards_from_str("Ah Kh Qh Jh 9h"))
    assert fh[0] == 6
    assert flush[0] == 5
    assert fh > flush


def test_flush_tiebreakers():
    flush_high_9 = evaluate_5card_hand(cards_from_str("Ah Kh 9h 7h 2h"))
    flush_high_8 = evaluate_5card_hand(cards_from_str("As Ks 8s 7s 6s"))
    assert flush_high_9 > flush_high_8
    
    # 5th card kicker tiebreak
    flush_a = evaluate_5card_hand(cards_from_str("Kd Qd Jd 8d 4d"))
    flush_b = evaluate_5card_hand(cards_from_str("Kc Qc Jc 8c 3c"))
    assert flush_a > flush_b


def test_flush_beats_straight():
    flush = evaluate_5card_hand(cards_from_str("2d 4d 6d 8d 10d"))
    straight = evaluate_5card_hand(cards_from_str("10h Jh Qh Kh As"))
    assert flush[0] == 5
    assert straight[0] == 4
    assert flush > straight


def test_straight_tiebreakers():
    broadway = evaluate_5card_hand(cards_from_str("10h Jh Qh Kh As"))
    king_high = evaluate_5card_hand(cards_from_str("9c 10s Jd Qh Kc"))
    six_high = evaluate_5card_hand(cards_from_str("2h 3s 4c 5d 6h"))
    wheel = evaluate_5card_hand(cards_from_str("Ah 2s 3c 4d 5h"))
    
    assert broadway > king_high
    assert king_high > six_high
    assert six_high > wheel  # 6-high beats 5-high (wheel A-2-3-4-5)
    assert wheel[1] == (5,)


def test_straight_beats_three_of_a_kind():
    straight = evaluate_5card_hand(cards_from_str("Ah 2s 3c 4d 5h"))
    trips = evaluate_5card_hand(cards_from_str("Ah As Ac Kd Qh"))
    assert straight[0] == 4
    assert trips[0] == 3
    assert straight > trips


def test_three_of_a_kind_tiebreakers():
    trips_8 = evaluate_5card_hand(cards_from_str("8h 8s 8c 3d 2h"))
    trips_7 = evaluate_5card_hand(cards_from_str("7h 7s 7c Ad Kh"))
    assert trips_8 > trips_7
    
    # Same trips, kicker tiebreak
    trips_8_ak = evaluate_5card_hand(cards_from_str("8h 8s 8c Ad Kh"))
    trips_8_aq = evaluate_5card_hand(cards_from_str("8h 8s 8c As Qh"))
    assert trips_8_ak > trips_8_aq


def test_three_of_a_kind_beats_two_pair():
    trips = evaluate_5card_hand(cards_from_str("2h 2s 2c 4d 5h"))
    two_pair = evaluate_5card_hand(cards_from_str("Ah As Kd Kh Qc"))
    assert trips[0] == 3
    assert two_pair[0] == 2
    assert trips > two_pair


def test_two_pair_tiebreakers():
    # Top pair decides
    tp_aces_twos = evaluate_5card_hand(cards_from_str("Ah As 2c 2d 3h"))
    tp_kings_queens = evaluate_5card_hand(cards_from_str("Kh Ks Qd Qh Jh"))
    assert tp_aces_twos > tp_kings_queens
    
    # Same top pair, second pair decides
    tp_aces_kings = evaluate_5card_hand(cards_from_str("Ah As Kh Ks 3h"))
    tp_aces_queens = evaluate_5card_hand(cards_from_str("Ah As Qh Qs Jh"))
    assert tp_aces_kings > tp_aces_queens
    
    # Both pairs same, kicker decides
    tp_kicker_jack = evaluate_5card_hand(cards_from_str("Ah As Kh Ks Jh"))
    tp_kicker_ten = evaluate_5card_hand(cards_from_str("Ah As Kh Ks 10h"))
    assert tp_kicker_jack > tp_kicker_ten


def test_two_pair_beats_one_pair():
    two_pair = evaluate_5card_hand(cards_from_str("2h 2s 3c 3d 4h"))
    one_pair = evaluate_5card_hand(cards_from_str("Ah As Kd Qh Jc"))
    assert two_pair[0] == 2
    assert one_pair[0] == 1
    assert two_pair > one_pair


def test_one_pair_tiebreakers():
    pair_aces = evaluate_5card_hand(cards_from_str("Ah As 2c 3d 4h"))
    pair_kings = evaluate_5card_hand(cards_from_str("Kh Ks Ad Qh Jc"))
    assert pair_aces > pair_kings
    
    # Same pair, kicker comparisons
    pair_aces_kqj = evaluate_5card_hand(cards_from_str("Ah As Kd Qh Jc"))
    pair_aces_kq10 = evaluate_5card_hand(cards_from_str("Ah As Kd Qh 10c"))
    assert pair_aces_kqj > pair_aces_kq10


def test_one_pair_beats_high_card():
    pair = evaluate_5card_hand(cards_from_str("2h 2s 3c 4d 5h"))
    high_card = evaluate_5card_hand(cards_from_str("Ah Kd Qc Jh 9s"))
    assert pair[0] == 1
    assert high_card[0] == 0
    assert pair > high_card


def test_high_card_tiebreakers():
    hc_1 = evaluate_5card_hand(cards_from_str("Ah Kd Qc Jh 9s"))
    hc_2 = evaluate_5card_hand(cards_from_str("Ah Kd Qc Jh 8s"))
    assert hc_1 > hc_2


def test_identical_hands_tie():
    h1 = evaluate_5card_hand(cards_from_str("Ah Kd Qc Jh 9s"))
    h2 = evaluate_5card_hand(cards_from_str("Ac Kh Qd Js 9c"))
    assert h1 == h2


def test_evaluate_7card_hand():
    # Hero has hole cards (Ah, Kh) with board (Qh, Jh, 10h, 2c, 3s) -> Royal Flush
    cards_7 = cards_from_str("Ah Kh Qh Jh 10h 2c 3s")
    score, best_5 = evaluate_7card_hand(cards_7)
    assert score[0] == 9
    assert len(best_5) == 5
    
    # 7 cards with three pairs: A-A, K-K, Q-Q, 2d -> Best is Two Pair Aces and Kings with Q kicker
    cards_3pair = cards_from_str("Ah As Kh Ks Qh Qs 2d")
    score_3p, best_5_3p = evaluate_7card_hand(cards_3pair)
    assert score_3p[0] == 2
    assert score_3p[1] == (14, 13, 12)
    
    # 7 cards with Full House vs Flush: should select Full House
    cards_fh_flush = cards_from_str("Ah As Ac 2h 3h 4h 2s")
    score_fh, _ = evaluate_7card_hand(cards_fh_flush)
    assert score_fh[0] == 6  # Full house AAA22 beats flush A432h


def test_hand_descriptions():
    score_rf = evaluate_5card_hand(cards_from_str("Ah Kh Qh Jh 10h"))
    assert get_hand_description(score_rf) == "Royal Flush"
    
    score_fh = evaluate_5card_hand(cards_from_str("Ah As Ac Kd Kh"))
    assert "Full House, As full of Ks" in get_hand_description(score_fh)
