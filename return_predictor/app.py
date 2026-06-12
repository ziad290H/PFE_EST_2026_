from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
MODEL_PATH = "return_predictor/best_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

if model is None:
    print("Model not loaded")
    exit()

DECISION_THRESHOLD = 0.31


# ─────────────────────────────────────────────
# Static knowledge (from training)
# ─────────────────────────────────────────────
CATEGORY_RETURN_RATES = {
    "Intimates": 0.15, "Jeans": 0.12, "Tops & Tees": 0.11,
    "Swim": 0.13, "Outerwear & Coats": 0.10, "Dresses": 0.14,
    "Accessories": 0.08, "Socks": 0.06, "Shorts": 0.09,
    "Sleep & Lounge": 0.11, "Active": 0.10,
    "Other": 0.10
}

BRAND_RETURN_RATES = {
    "default": 0.10
}

DEPARTMENT_ENC = {"Women": 0, "Men": 1, "Kids": 2, "Other": 3}
GENDER_ENC = {"F": 0, "M": 1, "Other": 2}
COUNTRY_ENC = {
    "United States": 1, "United Kingdom": 7, "France": 4,
    "Germany": 5, "Spain": 6, "Other": 10
}
PRICE_SEG_ENC = {"Low": 0, "Medium": 1, "High": 2}


# ─────────────────────────────────────────────
# Feature builder (MATCHES TRAINING EXACTLY)
# ─────────────────────────────────────────────
def build_features(data: dict) -> pd.DataFrame:

    sale_price   = float(data.get("sale_price", 50))
    retail_price = float(data.get("retail_price", 80))
    num_of_item  = int(data.get("num_of_item", 1))
    age          = int(data.get("age", 30))

    order_month     = int(data.get("order_month", 6))
    order_dayofweek = int(data.get("order_dayofweek", 2))
    order_hour      = int(data.get("order_hour", 12))
    is_weekend      = int(data.get("is_weekend", 0))
    order_quarter   = (order_month - 1) // 3 + 1

    # engineered numeric features
    price_ratio  = sale_price / retail_price if retail_price else 1
    discount_pct = ((retail_price - sale_price) / retail_price * 100) if retail_price else 0
    delai_traitement_jours = float(data.get("delai_traitement_jours", 3))

    a_ete_expedie = int(data.get("a_ete_expedie", 1))
    anciennete_compte_jours = int(data.get("anciennete_compte_jours", 100))

    # user behavior
    user_total_orders  = int(data.get("user_total_orders", 1))
    user_total_returns = int(data.get("user_total_returns", 0))
    user_return_rate_hist = (
        user_total_returns / user_total_orders if user_total_orders > 0 else 0
    )

    # categorical raw inputs
    category = data.get("category", "Other")
    brand = data.get("brand", "default")
    department = data.get("department", "Other")
    gender = data.get("gender", "Other")
    country = data.get("country", "Other")
    state = data.get("state", "Other")
    city = data.get("city", "Other")
    traffic = data.get("traffic_source", "Other")

    # encoded features (MUST MATCH TRAINING)
    features = {
        "sale_price": sale_price,
        "retail_price": retail_price,
        "num_of_item": num_of_item,
        "age": age,

        "price_ratio": price_ratio,
        "discount_pct": discount_pct,
        "delai_traitement_jours": delai_traitement_jours,

        "a_ete_expedie": a_ete_expedie,
        "anciennete_compte_jours": anciennete_compte_jours,

        "user_return_rate_hist": user_return_rate_hist,
        "user_total_returns": user_total_returns,
        "user_total_orders": user_total_orders,

        "cat_return_rate": CATEGORY_RETURN_RATES.get(category, 0.10),
        "brand_return_rate": BRAND_RETURN_RATES.get(brand, 0.10),

        "order_month": order_month,
        "order_dayofweek": order_dayofweek,
        "order_hour": order_hour,
        "is_weekend": is_weekend,
        "order_quarter": order_quarter,

        "category_enc": hash(category) % 100,
        "brand_enc": hash(brand) % 100,
        "department_enc": DEPARTMENT_ENC.get(department, 3),
        "user_gender_enc": GENDER_ENC.get(gender, 2),
        "country_enc": COUNTRY_ENC.get(country, 10),

        "state_enc": hash(state) % 100,
        "city_enc": hash(city) % 100,

        "user_traffic_source_enc": hash(traffic) % 100,
        "price_segment_enc": 0 if sale_price < 30 else (1 if sale_price < 100 else 2),
    }

    return pd.DataFrame([features])


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    X = build_features(data)

    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= DECISION_THRESHOLD)

    return jsonify({
        "prediction": pred,
        "probability": round(proba * 100, 2),
        "label": "Retourné" if pred else "Non retourné",
        "risk_level": "Élevé" if proba > 0.6 else "Moyen" if proba > 0.31 else "Faible"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)