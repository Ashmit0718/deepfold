# DeepFold: ML-Focused Poker Outcome Predictor — Walkthrough

**DeepFold** is a complete machine learning system that predicts Texas Hold'em showdown outcomes (**Win / Loss / Tie**) given Hero's 2 hole cards, 5 community cards, and 2 to 5 unknown opponents.

---

## 1. Deliverable Checklist Status

- [x] **`evaluator.py` & `test_evaluator.py`**: Standard 52-card deck, 5-card ranking (0–9), kicker tie-breaking, and 7-card best-hand evaluator with 22 unit tests.
- [x] **`generate_data.py` & Dataset**: 100,000 simulated showdowns with random opponents (2–5) saved to `data/poker_simulation_100k.csv` with class distribution printed.
- [x] **`features.py`**: 19-dimensional explainable domain feature vector extracted for batch training and single-instance inference.
- [x] **`train_model.py`**: Stratified 80/20 train/test split, balanced weighting, evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix), and comparison of Logistic Regression vs MLP.
- [x] **Trained Model**: Serialized pipeline saved to `models/deepfold_model.joblib` with metadata in `models/model_metadata.json`.
- [x] **`app.py`**: Minimal Flask backend with single endpoint `POST /api/predict` and input validation.
- [x] **Frontend (`static/index.html`)**: Interactive single-page card picker with presets, duplicate checking, and dynamic probability bars.
- [x] **`README.md`**: Project documentation explaining methodology, mathematical rationale, and setup.

---

## 2. Hand Evaluator Verification

The hand evaluator in [evaluator.py](file:///c:/Users/desai/Desktop/deepfold/evaluator.py) was validated with 22 tests in [test_evaluator.py](file:///c:/Users/desai/Desktop/deepfold/test_evaluator.py) using `pytest`.

```
============================= test session starts =============================
test_evaluator.py::test_deck_and_parsing PASSED                          [  4%]
test_evaluator.py::test_royal_flush_beats_straight_flush PASSED          [  9%]
test_evaluator.py::test_straight_flush_rankings PASSED                   [ 13%]
test_evaluator.py::test_straight_flush_beats_four_of_a_kind PASSED       [ 18%]
test_evaluator.py::test_four_of_a_kind_tiebreakers PASSED                [ 22%]
test_evaluator.py::test_four_of_a_kind_beats_full_house PASSED           [ 27%]
test_evaluator.py::test_full_house_tiebreakers PASSED                    [ 31%]
test_evaluator.py::test_full_house_beats_flush PASSED                    [ 36%]
test_evaluator.py::test_flush_tiebreakers PASSED                         [ 40%]
test_evaluator.py::test_flush_beats_straight PASSED                      [ 45%]
test_evaluator.py::test_straight_tiebreakers PASSED                      [ 50%]
test_evaluator.py::test_straight_beats_three_of_a_kind PASSED            [ 54%]
test_evaluator.py::test_three_of_a_kind_tiebreakers PASSED               [ 59%]
test_evaluator.py::test_three_of_a_kind_beats_two_pair PASSED            [ 63%]
test_evaluator.py::test_two_pair_tiebreakers PASSED                      [ 68%]
test_evaluator.py::test_two_pair_beats_one_pair PASSED                   [ 72%]
test_evaluator.py::test_one_pair_tiebreakers PASSED                      [ 77%]
test_evaluator.py::test_one_pair_beats_high_card PASSED                  [ 81%]
test_evaluator.py::test_high_card_tiebreakers PASSED                     [ 86%]
test_evaluator.py::test_identical_hands_tie PASSED                       [ 90%]
test_evaluator.py::test_evaluate_7card_hand PASSED                       [ 95%]
test_evaluator.py::test_hand_descriptions PASSED                         [100%]
============================= 22 passed in 0.09s ==============================
```

---

## 3. Synthetic Data Generation

Generated 100,000 showdown rows via parallel Monte Carlo simulation in `generate_data.py`:

```
==================================================
SHOWDOWN CLASS DISTRIBUTION (100,000 rows):
==================================================
  Win   :   22,600 rows ( 22.60%)
  Loss  :   74,403 rows ( 74.40%)
  Tie   :    2,997 rows (  3.00%)
==================================================

Outcome breakdown by Number of Opponents (N):
n_opponents      Loss (%)    Tie (%)     Win (%)
2                 64.61       3.38       32.02
3                 73.55       3.01       23.43
4                 78.23       2.69       19.08
5                 81.36       2.90       15.73
```

> **Class Imbalance Note**:
> Loss naturally dominates because Hero must beat all $N$ independent opponents simultaneously. Expected win rate is approximately $1 / (N+1)$ in an fair showdown. This reflects real poker physics and is handled via balanced loss weighting at train time.

---

## 4. Model Training & Evaluation

Trained on 80,000 samples, evaluated on 20,000 held-out test samples (stratified).

### Model Performance Comparison

| Metric | Multinomial Logistic Regression | Small MLP Classifier (64x32) |
|:-------|:-------------------------------:|:----------------------------:|
| **Test Accuracy** | 69.20% | 68.57% |
| **Macro F1** | 0.5038 | **0.5164** |
| **Win F1-Score** | 0.5070 | **0.5717** |
| **Loss F1-Score** | **0.8248** | 0.8136 |
| **Tie Recall** | 54.59% | **59.60%** |

### Confusion Matrix (Selected MLP Model)
```
Actual \ Pred        Loss       Tie       Win
---------------------------------------------
Loss               10,655     2,125     2,101
Tie                   112       357       130
Win                   544     1,274     2,702
```

The model pipeline is saved to `models/deepfold_model.joblib`.

---

## 5. Automated API & System Tests

Ran test suite `test_api.py` covering positive scenarios (Royal Flush, weak hands), edge cases, and validation rejections (duplicate cards, out-of-range opponents).

```
============================= test session starts =============================
test_api.py::test_predict_royal_flush PASSED                             [ 20%]
test_api.py::test_predict_weak_hand PASSED                               [ 40%]
test_api.py::test_predict_duplicate_cards_rejected PASSED                [ 60%]
test_api.py::test_predict_invalid_opponents PASSED                       [ 80%]
test_api.py::test_index_page PASSED                                      [100%]
============================== 5 passed in 1.99s ==============================
```

---

## 6. Live Frontend Verification

The UI was verified directly in the browser via automated browser testing.

### Initial Page State
![Initial Page Load](C:\Users\desai\.gemini\antigravity-ide\brain\10f07982-8b10-4d10-978a-20adcf471765\deepfold_initial_load_1788688278141.png)

### Four Aces Prediction Output
![Prediction Results](C:\Users\desai\.gemini\antigravity-ide\brain\10f07982-8b10-4d10-978a-20adcf471765\deepfold_four_aces_results_1788688307818.png)

**Observed Values for 'Four Aces' Preset (`Ah As` on `Ac Ad Kd 2c 7s` with 3 opponents):**
- **Hero Best 5-Card Hand**: `Four of a Kind, As (kicker K)`
- **Predicted Class**: `WIN`
- **Win Probability**: `74.1%`
- **Tie Probability**: `25.5%`
- **Loss Probability**: `0.4%`
