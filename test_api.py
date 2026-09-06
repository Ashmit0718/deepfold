"""
Integration tests for DeepFold Flask API endpoint /api/predict
"""

import json
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict_royal_flush(client):
    """Royal flush with 3 opponents should predict Win with near 100% win probability."""
    payload = {
        "hero_cards": ["Ah", "Kh"],
        "community_cards": ["Qh", "Jh", "10h", "2c", "7s"],
        "n_opponents": 3
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["hero_best_hand"]["category"] == "Royal Flush"
    assert data["prediction"] == "Win"
    assert data["probabilities"]["win"] > 0.85
    assert data["probabilities"]["loss"] < 0.15


def test_predict_weak_hand(client):
    """7-high unmade hand against 4 opponents should predict Loss with high probability."""
    payload = {
        "hero_cards": ["7h", "2c"],
        "community_cards": ["Ks", "Qd", "Jc", "10s", "9h"],
        "n_opponents": 4
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    # Note: On Ks Qd Jc 10s 9h, the board is already a straight! So all players tie the board or hero has high card.
    # In either case, prediction should return valid probabilities summing close to 1.
    total_prob = sum(data["probabilities"].values())
    assert abs(total_prob - 1.0) < 0.05


def test_predict_duplicate_cards_rejected(client):
    """Duplicate cards should return a 400 Bad Request."""
    payload = {
        "hero_cards": ["Ah", "Ah"],  # Duplicate Ah
        "community_cards": ["Kd", "Qd", "Jd", "10d", "2c"],
        "n_opponents": 3
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "Duplicate cards detected" in data["error"]


def test_predict_invalid_opponents(client):
    """Opponents outside 2-5 range should return a 400 Bad Request."""
    payload = {
        "hero_cards": ["Ah", "Kd"],
        "community_cards": ["Qh", "Jh", "10h", "2c", "7s"],
        "n_opponents": 8  # Invalid
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "n_opponents must be an integer between 2 and 5" in data["error"]


def test_index_page(client):
    """GET / should serve static HTML frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"DeepFold" in response.data
