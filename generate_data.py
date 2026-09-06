"""
DeepFold - Synthetic Data Generation
Simulates Texas Hold'em showdowns against 2 to 5 random opponents using Monte Carlo sampling.
Generates 100,000+ raw showdown rows with strict no-card-reuse rules.
Opponents' hole cards are discarded after evaluating the showdown outcome.
"""

import os
import random
import argparse
import time
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from evaluator import create_deck, evaluate_7card_hand, format_card


def simulate_single_showdown(deck_template: List[Tuple[int, str]], rng: random.Random) -> dict:
    """
    Simulates a single showdown:
    - Deals 2 hole cards to Hero
    - Deals 5 community cards
    - Selects N opponents randomly in [2, 5]
    - Deals N opponents 2 cards each without replacement
    - Evaluates Hero's best 5-card hand vs each opponent's best 5-card hand
    - Determines outcome: 'Win', 'Loss', or 'Tie'
    - Opponent cards are discarded; only Hero's 7 cards, N, and outcome are returned.
    """
    # Create a fresh shuffled copy of deck
    deck = deck_template.copy()
    rng.shuffle(deck)
    
    # 1. Deal hero 2 hole cards
    hero_cards = [deck[0], deck[1]]
    
    # 2. Deal 5 community cards
    comm_cards = [deck[2], deck[3], deck[4], deck[5], deck[6]]
    
    # 3. Pick random number of opponents N between 2 and 5
    n_opponents = rng.randint(2, 5)
    
    # 4. Deal 2 cards to each of the N opponents from the remaining deck
    # Total cards needed for opponents: 2 * n_opponents
    # Deck indices: 7 to 7 + 2*N
    opponents_cards = []
    card_idx = 7
    for _ in range(n_opponents):
        opp_hole = [deck[card_idx], deck[card_idx + 1]]
        opponents_cards.append(opp_hole)
        card_idx += 2
        
    # 5. Evaluate hero's best 5-card hand from 7 cards
    hero_7 = hero_cards + comm_cards
    hero_score, _ = evaluate_7card_hand(hero_7)
    
    # 6. Evaluate each opponent's best 5-card hand (their 2 hole cards + same 5 community cards)
    opp_scores = []
    for opp_hole in opponents_cards:
        opp_7 = opp_hole + comm_cards
        score, _ = evaluate_7card_hand(opp_7)
        opp_scores.append(score)
        
    # 7. Compare hero against all opponents:
    # - Win: hero strictly beats every opponent
    # - Loss: at least one opponent strictly beats hero
    # - Tie: hero ties with at least one opponent for the best hand, and no one beats hero
    hero_beats_all = True
    hero_is_beaten = False
    hero_ties_with_best = False
    
    max_opp_score = max(opp_scores)
    
    if hero_score > max_opp_score:
        outcome = "Win"
    elif hero_score < max_opp_score:
        outcome = "Loss"
    else:
        # hero_score == max_opp_score
        outcome = "Tie"
        
    # Discard opponents' cards completely.
    # Return hero cards, community cards, N, and outcome.
    return {
        "hero_c1": format_card(hero_cards[0]),
        "hero_c2": format_card(hero_cards[1]),
        "comm_c1": format_card(comm_cards[0]),
        "comm_c2": format_card(comm_cards[1]),
        "comm_c3": format_card(comm_cards[2]),
        "comm_c4": format_card(comm_cards[3]),
        "comm_c5": format_card(comm_cards[4]),
        "n_opponents": n_opponents,
        "outcome": outcome
    }


def _worker_simulate_batch(args: Tuple[int, int]) -> List[dict]:
    """Worker task to simulate a batch of rows with a distinct seed."""
    batch_size, seed = args
    rng = random.Random(seed)
    deck_template = create_deck()
    batch = []
    for _ in range(batch_size):
        batch.append(simulate_single_showdown(deck_template, rng))
    return batch


def generate_dataset(n_rows: int = 100000, output_path: str = "data/poker_simulation_100k.csv", n_jobs: int = None) -> pd.DataFrame:
    """
    Generates n_rows poker showdown rows in parallel and saves to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
        
    print(f"[*] Starting Monte Carlo simulation for {n_rows:,} rows across {n_jobs} workers...")
    
    batch_size = max(1000, n_rows // (n_jobs * 8))
    chunks = []
    remaining = n_rows
    current_seed = int(time.time())
    
    tasks = []
    while remaining > 0:
        sz = min(batch_size, remaining)
        current_seed += 137
        tasks.append((sz, current_seed))
        remaining -= sz
        
    rows = []
    with mp.Pool(processes=n_jobs) as pool:
        for batch in tqdm(pool.imap_unordered(_worker_simulate_batch, tasks), total=len(tasks), desc="Simulating Showdowns"):
            rows.extend(batch)
            
    df = pd.DataFrame(rows)
    
    # Save raw CSV
    df.to_csv(output_path, index=False)
    print(f"\n[+] Successfully saved {len(df):,} rows to '{output_path}'")
    
    # Print class balance statistics
    counts = df["outcome"].value_counts()
    percentages = df["outcome"].value_counts(normalize=True) * 100
    
    print("\n" + "=" * 50)
    print("SHOWDOWN CLASS DISTRIBUTION:")
    print("=" * 50)
    for cls in ["Win", "Loss", "Tie"]:
        c = counts.get(cls, 0)
        p = percentages.get(cls, 0.0)
        print(f"  {cls:<6}: {c:>8,} rows ({p:6.2f}%)")
    print("=" * 50)
    print("Note on Class Imbalance:")
    print("  Loss dominates heavily (~70-80%) because Hero must beat between 2 and 5 independent")
    print("  opponents simultaneously. In an (N+1)-player game with random cards, Hero's expected")
    print("  fair win rate is 1/(N+1) (e.g. 16.7% at N=5, 25% at N=3).")
    print("  This class imbalance is a fundamental property of multi-opponent poker showdowns,")
    print("  and is handled at training time using class_weight='balanced'.")
    print("=" * 50 + "\n")
    
    # Also print breakdown by opponent count N
    print("Outcome breakdown by Number of Opponents (N):")
    cross_tab = pd.crosstab(df["n_opponents"], df["outcome"], normalize="index") * 100
    print(cross_tab.round(2))
    print()
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFold Synthetic Data Generator")
    parser.add_argument("--rows", type=int, default=100000, help="Number of showdown rows to simulate (default: 100,000)")
    parser.add_argument("--output", type=str, default="data/poker_simulation_100k.csv", help="Output CSV path")
    parser.add_argument("--jobs", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()
    
    generate_dataset(n_rows=args.rows, output_path=args.output, n_jobs=args.jobs)
