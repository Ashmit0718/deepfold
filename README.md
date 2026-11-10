# DeepFold – Poker Outcome Prediction Using Machine Learning

DeepFold is a machine learning project I put together to predict the outcome of a Texas Hold'em showdown. The concept was to take a player's two hole cards, the five community cards and the number of opponents at the table and then work out whether the player is going to Win, Lose or Tie.

I wanted to work on something that wasn't just about training a model on some existing dataset. So I built the entire pipeline from scratch. Started with a poker hand evaluator and synthetic data generator, then did the feature engineering, trained the model, evaluated it and then put it all together in a Flask app.

The project also has a simple web interface whereby you can select cards and see the predictions coming in in real time.

## How It Works

DeepFold takes three inputs:

- Hero's two hole cards.
- 5 community cards.
- Number of opponents (2 to 5).

These inputs are processed and converted into numerical features which are then fed into the trained classification model.

The model returns one of three possible outcomes:

- **Win**
- **Loss**
- **Tie**

One thing we very specifically wanted to maintain throughout the project was an information barrier. The opponents' hole cards are only ever used for the purpose of determining the actual showdown result during a simulation. They are discarded immediately afterwards and are never part of the dataset or ever seen by the model.

This means the model only learns from the information that would in fact be available to a player at showdown.

## Project Architecture

I broke the project down into discrete components so that the hand evaluation, data generation, feature extraction, model training and the prediction logic could all be developed and tested in isolation.

### 1. Poker Hand Evaluator (`evaluator.py`)

I started off with a poker hand evaluator as the whole dataset hinges on correctly determining the best hand.

The evaluator assigns a category to each hand from 0 to 9.

| ID | Hand Category |
|---|---|
| 9 | Royal Flush |
| 8 | Straight Flush |
| 7 | Four of a Kind |
| 6 | Full House |
| 5 | Flush |
| 4 | Straight |
| 3 | Three of a Kind |
| 2 | Two Pair |
| 1 | One Pair |
| 0 | High Card |

Other than working out the hand category we also implemented kicker comparisons to deal with the case where two players have the same hand category.

For example if two players have a pair of Aces then the remaining cards are used to determine who has the best hand.

Since Texas Hold'em allows players to use any five cards out of their seven available cards, the evaluator checks all possible five-card combinations:

\[
\binom{7}{5}=21
\]

It then picks the best combination.

**Testing**

I've written 22 unit tests in test_evaluator.py to test out the various hand rankings and edge cases.

Some of the cases I tested include:

- Royal Flush to Straight Flush.
- Four of a Kind v Full House.
- Kicker comparisons.
- Full House tie-breaking.
- Wheel straight (A-2-3-4-5).
- Three of the seven cards were pairs.
- Same hands resulting in a split pot.

All 22 tests passed during testing.

### 2. Synthetic Data Generation (`generate_data.py`)

One of the key elements of the project was generating sufficient data to train the classification models.

Rather than use an existing poker dataset we just implemented a Monte Carlo simulation which will generate random Texas Hold'em showdowns.

I generated 100,000 showdown records using multiprocessing with 11 CPU workers.

The simulation follows this process:

1. A fresh 52 card deck is made up and shuffled.
2. Two hole cards are dealt to Hero.
3. Five community cards are dealt.
4. A random number of opponents (between 2 and 5) is selected.
5. Two cards are dealt to each opponent (without replacement).
6. The evaluator works out the best 5 card hand Hero can make and the best 5 card hand each of the opponents can make.
7. The results are then compared to establish the final outcome.
8. Only the Hero's cards, the opponent count and the outcome are saved.

The opponents' cards are discarded after each simulation. That was important as I didn't want the model to have access to any information that wouldn't be available at the time of actual prediction.

The generated dataset is stored in:

`data/poker_simulation_100k.csv`

#### Dataset Distribution

After generating 100,000 records, the class distribution was:

| Outcome | Records | Percentage |
|---|---:|---:|
| Win | 22,600 | 22.60% |
| Loss | 74,403 | 74.40% |
| Tie | 2,997 | 3.00% |
| **Total** | **100,000** | **100%** |

Loss is the dominant class as Hero has to beat every opponent in order to win the showdown.

I also looked at how the number of opponents affects the outcome distribution.

| Opponents | Loss (%) | Tie (%) | Win (%) |
|---|---:|---:|---:|
| 2 | 64.61 | 3.38 | 32.02 |
| 3 | 73.55 | 3.01 | 23.43 |
| 4 | 78.23 | 2.69 | 19.08 |
| 5 | 81.36 | 2.90 | 15.73 |

As we'd expect the chance of winning goes down as more opponents are added.

### 3. Feature Engineering (`features.py`)

Once we'd got the dataset generated we needed to convert the card data into something the ML models could work with.

For this I've implemented a feature extraction module which takes the seven cards in Hero's hand and the number of opponents and produces a 19 dimensional numerical feature vector.

The features include:

- Hand category: the strongest hand, from High Card through to Royal Flush.
- Hand flags: binary values indicating the presence of pairs, two pairs, trips, straights, flushes, full houses, quads and straight flushes.
- Card ranks: Highest rank, second highest distinct rank and primary/secondary combination ranks.
- Suit information: Maximum number of cards of the same suit.
- Number of opponents in the showdown.
- Hole card information: pocket pairs, suited hole cards and the high and low of the hole cards.

I kept the features fully explainable so that I knew exactly what information was going in to the model rather than treating the card combinations as some sort of black box.

### 4. Model Training (`train_model.py`)

For the classification part, I experimented with two different models:

1. Multinomial Logistic Regression.
2. Multi-layer perceptron (MLP) with two hidden layers (64 and 32 neurons).

I used an 80:20 train/test split with stratification to preserve the class distribution.

| Dataset | Samples |
|---|---:|
| Training | 80,000 |
| Testing | 20,000 |
| Total | 100,000 |

Given the substantial class imbalance in the dataset I used balanced class weighting during training.

#### Model Comparison

I've evaluated both models on the same test set.

| Metric | Logistic Regression | MLP (64 → 32) |
|---|---:|---:|
| Accuracy | 69.20% | 68.57% |
| Macro F1 | 0.5038 | 0.5164 |
| Win F1 | 0.5070 | 0.5717 |
| Loss F1 | 0.8248 | 0.8136 |
| Tie Recall | 54.59% | 59.60% |

I went with the MLP as the final model.

Although Logistic Regression actually achieved a slightly better accuracy, I felt the MLP was the more appropriate model for this project as it did better on the Macro F1 and was better at picking out the Win and Tie cases.

The final model uses:

- StandardScaler for feature scaling.
- MLPClassifier for classification.
- Two hidden layers (64 and 32 neurons respectively).

The trained pipeline is saved using Joblib:

`models/deepfold_model.joblib`

#### MLP Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Loss | 94.20% | 71.60% | 81.36% |
| Tie | 9.50% | 59.60% | 16.39% |
| Win | 54.77% | 59.78% | 57.17% |

One thing that emerged during the evaluation was that tie prediction remains a very weak area. It does a reasonable number of actual ties, but the precision is very low so there are a lot of predicted ties that are wrong.

I kept this result in the evaluation rather than just looking at accuracy as that alone doesn't tell the whole story when you've got an imbalanced dataset.

### 5. Flask Backend (`app.py`)

Having trained the model I've now embedded it into a Flask app so that it can make predictions via an API.

The backend loads the saved model, takes the card details, extracts the features and returns the prediction along with the class probabilities.

**API Endpoint**

`POST /api/predict`

Example request:

```json
{
  "hero_cards": ["Ah", "Kh"],
  "community_cards": ["Qh", "Jh", "10h", "2c", "7s"],
  "n_opponents": 3
}
```

Example response:

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

I've also added in input validation to stop invalid predictions.

The API rejects:

- Invalid card values.
- Duplicate cards.
- Incorrect card counts.
- Opponent count outside the 2 to 5 supported range.
