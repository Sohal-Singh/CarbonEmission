"""
Carbon Footprint Calculator — Flask backend
Loads the trained Logistic Regression (classifier) + Linear Regression
(continuous CO2 estimate) models along with the OneHotEncoder and the two
MultiLabelBinarizers used for the checkbox groups, reproduces the exact
preprocessing pipeline, and serves predictions to the single-page frontend.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from dotenv import load_dotenv
from google import genai

app = Flask(__name__)

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ---------------------------------------------------------------------
# LOAD MODELS + ENCODERS (once, at startup)
# ---------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "Models")

logistic_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
linear_model = joblib.load(os.path.join(MODELS_DIR, "linear_regression_model.pkl"))
onehot_encoder = joblib.load(os.path.join(MODELS_DIR, "onehot_encoder.joblib"))
cooking_encoder = joblib.load(os.path.join(MODELS_DIR, "cooking_encoder.joblib"))
recycling_encoder = joblib.load(os.path.join(MODELS_DIR, "recycling_encoder.joblib"))

# ---------------------------------------------------------------------
# COLUMN ORDER — pulled straight from the encoders themselves, so this
# stays correct even if you retrain and swap the .joblib/.pkl files.
# ---------------------------------------------------------------------
NUMERIC_COLS = [
    "Monthly Grocery Bill",
    "Vehicle Monthly Distance Km",
    "Waste Bag Weekly Count",
    "How Long TV PC Daily Hour",
    "How Many New Clothes Monthly",
    "How Long Internet Daily Hour",
]

CATEGORICAL_COLS = list(onehot_encoder.feature_names_in_)

# Final column order the models were trained on.
MODEL_FEATURE_ORDER = list(logistic_model.feature_names_in_)

# Confirmed by testing a clearly-low-impact profile (class 0) and a
# clearly-high-impact profile (class 2) against the linear model's
# continuous estimate — classes are ordinal LOW < MEDIUM < HIGH.
CLASS_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


# ---------------------------------------------------------------------
# FEATURE ENGINEERING — mirrors training-time preprocessing exactly
# ---------------------------------------------------------------------
def build_feature_frame(form):
    """Turn the raw submitted form into the 56-column DataFrame the
    models were trained on."""

    # 1. Numeric columns, cast to float, single row.
    numeric_values = [[float(form.get(col, 0) or 0) for col in NUMERIC_COLS]]
    numeric_df = pd.DataFrame(numeric_values, columns=NUMERIC_COLS)

    # 2. Cooking methods — multi-select checkboxes -> MultiLabelBinarizer.
    cooking_selected = form.getlist("Cooking_With")
    cooking_arr = cooking_encoder.transform([cooking_selected])
    cooking_df = pd.DataFrame(
        cooking_arr, columns=[f"Cooking_{c}" for c in cooking_encoder.classes_]
    )

    # 3. Recycling habits — multi-select checkboxes -> MultiLabelBinarizer.
    recycling_selected = form.getlist("Recycling")
    recycling_arr = recycling_encoder.transform([recycling_selected])
    recycling_df = pd.DataFrame(
        recycling_arr, columns=[f"Recycle_{c}" for c in recycling_encoder.classes_]
    )

    # 4. Categorical dropdowns -> OneHotEncoder (order matters here, must
    #    match onehot_encoder.feature_names_in_ exactly).
    categorical_values = [[form.get(col, "") for col in CATEGORICAL_COLS]]
    categorical_df = pd.DataFrame(categorical_values, columns=CATEGORICAL_COLS)
    onehot_arr = onehot_encoder.transform(categorical_df)
    onehot_df = pd.DataFrame(onehot_arr, columns=onehot_encoder.get_feature_names_out())

    # 5. Combine, then reindex to the exact column order the models
    #    expect — this is the safety net against ordering mistakes above.
    final_df = pd.concat([numeric_df, cooking_df, recycling_df, onehot_df], axis=1)
    final_df = final_df.reindex(columns=MODEL_FEATURE_ORDER, fill_value=0)

    return final_df


# ---------------------------------------------------------------------
# AI SUGGESTIONS — simple rule-based tips derived from the submitted
# lifestyle inputs and predicted category (swap in an LLM call here later
# if you want genuinely generative suggestions).
# ---------------------------------------------------------------------
def generate_suggestions(form, category):
    tips = []

    if form.get("Transport") == "private":
        tips.append("Try public transport or carpooling a few days a week to cut vehicle emissions.")
    if form.get("Frequency of Traveling by Air") in ("frequently", "very frequently"):
        tips.append("Reduce flight frequency where possible, or offset unavoidable trips.")
    if form.get("Energy efficiency") == "No":
        tips.append("Switch to energy-efficient appliances and LED lighting at home.")
    if form.get("Diet") == "omnivore":
        tips.append("Swapping a few meals a week for plant-based options meaningfully lowers your footprint.")
    if form.get("Heating Energy Source") in ("coal", "wood"):
        tips.append("Consider transitioning your heating source to electricity or natural gas.")
    try:
        if float(form.get("Vehicle Monthly Distance Km", 0) or 0) > 1000:
            tips.append("Look into combining errands or remote work days to reduce monthly driving distance.")
    except ValueError:
        pass
    if not form.getlist("Recycling"):
        tips.append("Start recycling paper, plastic, glass, or metal — even one category helps.")

    if not tips:
        tips.append("You're already following great sustainable habits — keep it up!")

    # Cap so the UI stays tidy regardless of how many rules fire.
    return tips[:5]

def generate_gemini_suggestions(form, category, estimated_co2):
    prompt = f"""
You are an environmental sustainability expert.

A user has the following lifestyle:

Body Type: {form.get("Body Type")}
Sex: {form.get("Sex")}
Diet: {form.get("Diet")}
Transport: {form.get("Transport")}
Vehicle Type: {form.get("Vehicle Type")}
Heating Source: {form.get("Heating Energy Source")}
Energy Efficiency: {form.get("Energy efficiency")}

Monthly Grocery Bill: {form.get("Monthly Grocery Bill")}
Vehicle Monthly Distance: {form.get("Vehicle Monthly Distance Km")} km
TV/PC Usage: {form.get("How Long TV PC Daily Hour")} hours/day
Internet Usage: {form.get("How Long Internet Daily Hour")} hours/day
New Clothes Bought Monthly: {form.get("How Many New Clothes Monthly")}

Cooking Methods:
{", ".join(form.getlist("Cooking_With"))}

Recycling:
{", ".join(form.getlist("Recycling"))}

Predicted Carbon Emission:
{estimated_co2:.2f} kg/year

Emission Category:
{category}

Give exactly 5 personalized eco-friendly suggestions.

Rules:
- Keep each suggestion to one sentence.
- Return ONLY the 5 suggestions.
- Do not write any introduction.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    suggestions = [
        line.replace("*", "").replace("-", "").strip()
        for line in response.text.split("\n")
        if line.strip()
    ]

    return suggestions[:5]


# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    try:
        features = build_feature_frame(form)
    except Exception as exc:  # malformed/missing fields
        return jsonify({"error": f"Could not process inputs: {exc}"}), 400

    predicted_class = int(logistic_model.predict(features)[0])
    category = CLASS_LABELS.get(predicted_class, "MEDIUM")

    # Linear Regression can extrapolate below 0 for very low-impact
    # profiles (a known limitation of unregularized linear models near
    # the edge of the training distribution) — clamp for display.
    estimated_co2 = max(0.0, float(linear_model.predict(features)[0]))

    try:
        suggestions = generate_gemini_suggestions(
                form,
                category,
                estimated_co2
            )

    
    except Exception as e:
        print("Gemini Error:", e)
        
        suggestions = generate_suggestions(
                form,
                category
            )

    

    return jsonify({
        "category": category,
        "estimated_co2_per_month": round(estimated_co2, 1),
        "suggestions": suggestions,
    })


if __name__ == "__main__":
    app.run(debug=True)