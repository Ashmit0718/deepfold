# DeepFold: ML-Focused Poker Outcome Predictor

**DeepFold** is a machine learning project that predicts a poker player's showdown outcome (**Win**, **Loss**, or **Tie**) in Texas Hold'em, given only:
- Hero's **2 hole cards**
- The **5 community cards** (7 known cards total)
- The number of unknown opponents at the table (**2 to 5 opponents**)

The project focuses on the end-to-end ML pipeline: a rigorously tested poker hand evaluator, synthetic Monte Carlo data generation, domain-informed feature engineering, and class-balanced classification. A minimal Flask backend and single-page card picker demonstrate the trained model in real time.

---

## 1. Problem Formulation

- **Input at inference time**: Hero's 2 hole cards + 5 board cards (7 cards total) + number of opponents $N \in [2, 5]$.
- **Output**: 3-class prediction (**Win**, **Loss**, **Tie**) with class probabilities.
- **Strict Information Barrier**: Opponents' hole cards only exist transiently inside the simulation engine to determine the showdown outcome. They are discarded immediately and are **never** stored in datasets or accessible to the model at training or inference time.
- **Framing**: Multi-class classification (3 classes), not regression.

---

## 2. Hand Evaluator (`evaluator.py` & `test_evaluator.py`)

A Texas Hold'em hand evaluator maps 5-card combinations into standard hand categories (0–9) and computes hierarchical kicker tuples for exact tie-breaking:

| Category ID | Hand Category | Tie-breaking Tuple |
|:-----------:|:--------------|:-------------------|
| **9** | Royal Flush | `(14,)` |
| **8** | Straight Flush | `(high_rank,)` (Wheel A-2-3-4-5 has high rank 5) |
| **7** | Four of a Kind | `(quad_rank, kicker)` |
| **6** | Full House | `(trips_rank, pair_rank)` |
| **5** | Flush | `(r1, r2, r3, r4, r5)` (ranks descending) |
| **4** | Straight | `(high_rank,)` (Wheel A-2-3-4-5 has high rank 5) |
| **3** | Three of a Kind | `(trips_rank, kicker1, kicker2)` |
| **2** | Two Pair | `(high_pair, low_pair, kicker)` |
| **1** | One Pair | `(pair_rank, kicker1, kicker2, kicker3)` |
| **0** | High Card | `(r1, r2, r3, r4, r5)` |

The 7-card evaluator chooses $\arg\max$ over all $\binom{7}{5} = 21$ combinations.

### Unit Tests
The evaluator was verified with 22 unit tests in `test_evaluator.py`:
- Royal flush beats straight flush
- Straight flush rankings (King-high vs 9-high vs 5-high wheel)
- Quads vs Full House, and quads kicker tie-breakers
- Full house tie-breakers (higher trips wins regardless of pair)
- Flush kicker comparisons (5th card kicker tie-breaks)
- Straight wheel (A-2-3-4-5) vs 6-high straight (2-3-4-5-6)
- Two pair rankings (higher top pair vs higher second pair vs kicker)
- Three pairs among 7 cards (correctly picking top two pairs)
- Identical hands tie (split pot)

All 22 unit tests pass in 0.09s (`python -m pytest test_evaluator.py`).

---

## 3. Synthetic Data Generation (`generate_data.py`)

100,000 Texas Hold'em showdowns were simulated using parallel Monte Carlo sampling across 11 CPU workers in ~8 seconds:

1. Shuffle a fresh 52-card deck.
2. Deal Hero 2 random hole cards.
3. Deal 5 random community cards.
4. Pick opponent count $N \in \{2, 3, 4, 5\}$ uniformly at random.
5. Deal $N$ opponents 2 cards each without replacement (strict no-card-reuse).
6. Evaluate Hero's best 5-card hand vs each opponent's best 5-card hand.
7. Label outcome:
   - **Win**: Hero beats all $N$ opponents.
   - **Tie**: Hero ties with at least one opponent for the winning hand, and no opponent beats Hero.
   - **Loss**: At least one opponent strictly beats Hero.
8. Opponents' cards are discarded; only Hero's 7 cards, $N$, and outcome are saved to `data/poker_simulation_100k.csv`.

### Class Distribution (100,000 rows)
```
==================================================
SHOWDOWN CLASS DISTRIBUTION:
==================================================
  Win   :   22,600 rows ( 22.60%)
  Loss  :   74,403 rows ( 74.40%)
  Tie   :    2,997 rows (  3.00%)
==================================================
```

### Breakdown by Opponent Count ($N$)
```
n_opponents      Loss (%)    Tie (%)     Win (%)
2                 64.61       3.38       32.02
3                 73.55       3.01       23.43
4                 78.23       2.69       19.08
5                 81.36       2.90       15.73
```

> **Why Loss Dominates:**
> In an $(N+1)$-player showdown with random cards, Hero's theoretical fair share of wins is $1 / (N+1)$ (e.g. 16.7% with 5 opponents, 25.0% with 3 opponents). Because Hero must beat all independent opponents simultaneously, Loss is naturally the majority class. This imbalance is handled at training time using `class_weight='balanced'`, rather than artificially distorting the data generation.

---

## 4. Feature Engineering (`features.py`)

A 19-dimensional explainable numerical feature vector is derived from Hero's 7 cards + $N$:

1. `hand_category`: Integer 0 to 9.
2. Binary flags: `has_pair`, `has_two_pair`, `has_trips`, `has_straight`, `has_flush`, `has_full_house`, `has_quads`, `has_straight_flush`.
3. `high_card_rank`: Highest card rank among all 7 cards (2–14).
4. `second_high_rank`: Second highest distinct rank (kicker strength).
5. `primary_rank`: Rank of the primary combination (e.g. pair rank, quad rank, or straight high card).
6. `secondary_rank`: Rank of the secondary combination (e.g. low pair in two pair, pair in full house).
7. `max_suit_count`: Maximum count of any single suit among the 7 cards (flush strength/potential).
8. `n_opponents`: Raw number of opponents (2 to 5).
9. `is_pocket_pair`: 1 if Hero holds a pair in hole cards, else 0.
10. `is_suited_hole`: 1 if Hero's hole cards share the same suit, else 0.
11. `hole_high_rank`: Max rank between Hero's 2 hole cards.
12. `hole_low_rank`: Min rank between Hero's 2 hole cards.

---

## 5. Model Training & Evaluation (`train_model.py`)

- **Dataset**: 100,000 showdown rows.
- **Split**: 80% Train (80,000) / 20% Test (20,000), stratified by class.
- **Class Balancing**: Balanced class/sample weighting applied to prevent the model from collapsing to predicting only "Loss".

### Model Comparison on Test Set (20,000 rows)

| Metric | Multinomial Logistic Regression | Small MLP (64x32) |
|:-------|:-------------------------------:|:-----------------:|
| **Accuracy** | 69.20% | 68.57% |
| **Macro F1** | 0.5038 | **0.5164** |
| **Win F1** | 0.5070 | **0.5717** |
| **Loss F1** | **0.8248** | 0.8136 |
| **Tie Recall** | 54.59% | **59.60%** |

### Selected Model: MLP Classifier (64 $\rightarrow$ 32)
The MLP captures non-linear interactions between hand category, kicker ranks, and opponent count $N$, delivering higher Macro F1 and significantly stronger Win identification (F1: 0.5717 vs 0.5070).

```
MLP Classification Report:
              precision    recall  f1-score   support
        Loss     0.9420    0.7160    0.8136     14,881
         Tie     0.0950    0.5960    0.1639        599
         Win     0.5477    0.5978    0.5717      4,520

    accuracy                         0.6857     20,000
   macro avg     0.5283    0.6366    0.5164     20,000

MLP Confusion Matrix:
Actual \ Pred        Loss       Tie       Win
---------------------------------------------
Loss               10,655     2,125     2,101
Tie                   112       357       130
Win                   544     1,274     2,702
```

The trained model pipeline (`StandardScaler` + `MLPClassifier`) is serialized to `models/deepfold_model.joblib`.

---

## 6. Backend API (`app.py`)

A minimal Flask app exposes a single prediction endpoint:

- **Endpoint**: `POST /api/predict`
- **Request**:
  ```json
  {
    "hero_cards": ["Ah", "Kh"],
    "community_cards": ["Qh", "Jh", "10h", "2c", "7s"],
    "n_opponents": 3
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "prediction": "Win",
    "n_opponents": 3,
    "hero_best_hand": {
      "category": "Royal Flush",
      "category_id": 9,
      "description": "Royal Flush"
    },
    "probabilities": {
      "win": 0.9821,
      "loss": 0.0105,
      "tie": 0.0074
    }
  }
  ```
- **Validation**: Rejects invalid card strings, duplicate cards, and opponent counts outside 2–5.

---

## 7. Frontend (`static/index.html`)

A single-page HTML/JS card picker and probability visualizer:
- Card selectors for Hero's 2 hole cards and 5 board cards.
- Quick test presets: *Royal Flush*, *Four Aces*, *Full House*, *Nut Flush*, *One Pair*, *Weak 7-2*.
- Duplicate card detection preventing illegal board configurations.
- Dynamic probability bars for Win, Loss, and Tie.

---

## 8. Quickstart & Verification

### 1. Run Unit & Integration Tests
```bash
python -m pytest test_evaluator.py test_api.py -v
```

### 2. (Optional) Re-generate Dataset
```bash
python generate_data.py --rows 100000
```

### 3. (Optional) Re-train Model
```bash
python train_model.py
```

### 4. Start the Application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.
