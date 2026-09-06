"""
DeepFold - Minimal Flask Backend
Serves a single prediction endpoint POST /api/predict
Accepts Hero's 2 hole cards, 5 community cards, and opponent count N (2-5).
Applies features.py and returns model Win/Loss/Tie class probabilities.
Also serves the minimal demo frontend at GET /.
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from evaluator import parse_card, evaluate_7card_hand, get_hand_description, CATEGORY_NAMES
from features import extract_features_single

app = Flask(__name__, static_folder="static")
CORS(app)

# Load trained model pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "deepfold_model.joblib")
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please run train_model.py first.")
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/")
def index():
    """Serves the minimal single-page frontend."""
    return send_from_directory("static", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Request Body:
    {
        "hero_cards": ["Ah", "Kd"],
        "community_cards": ["Qh", "Jh", "10h", "2c", "7s"],
        "n_opponents": 3
    }
    """
    try:
        data = request.get_json(force=True, silent=False)
        if not data:
            return jsonify({"error": "Missing JSON request body"}), 400

        hero_cards = data.get("hero_cards")
        comm_cards = data.get("community_cards")
        n_opponents = data.get("n_opponents")

        # 1. Validate card arrays
        if not isinstance(hero_cards, list) or len(hero_cards) != 2:
            return jsonify({"error": "hero_cards must be a list of exactly 2 card strings (e.g. ['Ah', 'Kd'])"}), 400

        if not isinstance(comm_cards, list) or len(comm_cards) != 5:
            return jsonify({"error": "community_cards must be a list of exactly 5 card strings (e.g. ['Qh', 'Jh', '10h', '2c', '7s'])"}), 400

        # 2. Validate opponent count
        try:
            n_opponents = int(n_opponents)
            if n_opponents < 2 or n_opponents > 5:
                return jsonify({"error": "n_opponents must be an integer between 2 and 5"}), 400
        except (TypeError, ValueError):
            return jsonify({"error": "n_opponents must be an integer between 2 and 5"}), 400

        # 3. Validate and parse cards
        all_card_strs = hero_cards + comm_cards
        parsed_cards = []
        for c in all_card_strs:
            try:
                parsed_cards.append(parse_card(c))
            except Exception as e:
                return jsonify({"error": f"Invalid card '{c}': {str(e)}"}), 400

        # 4. Check for duplicate cards across table
        if len(set(parsed_cards)) != 7:
            return jsonify({"error": "Duplicate cards detected. All 7 cards (2 hole + 5 community) must be distinct."}), 400

        # 5. Evaluate hero's best 5-card hand description
        hero_score, best_5 = evaluate_7card_hand(parsed_cards)
        category_name = CATEGORY_NAMES.get(hero_score[0], "Unknown")
        description = get_hand_description(hero_score)

        # 6. Extract features and predict using loaded model
        clf_pipeline = get_model()
        features = extract_features_single(hero_cards, comm_cards, n_opponents)
        
        # Predict class probabilities
        probs = clf_pipeline.predict_proba(features)[0]
        classes = list(clf_pipeline.classes_)  # e.g. ['Loss', 'Tie', 'Win']
        
        prob_dict = {}
        for cls_name, p in zip(classes, probs):
            prob_dict[cls_name.lower()] = float(round(p, 4))
            
        pred_label = str(clf_pipeline.predict(features)[0])

        return jsonify({
            "status": "success",
            "prediction": pred_label,
            "probabilities": {
                "win": prob_dict.get("win", 0.0),
                "loss": prob_dict.get("loss", 0.0),
                "tie": prob_dict.get("tie", 0.0)
            },
            "hero_best_hand": {
                "category": category_name,
                "category_id": int(hero_score[0]),
                "description": description
            },
            "n_opponents": n_opponents
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    print("Starting DeepFold Flask Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
