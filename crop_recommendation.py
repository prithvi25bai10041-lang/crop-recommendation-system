"""
  CROP RECOMMENDATION SYSTEM
  Fundamentals of AI and ML — BYOP Capstone Project

  Uses: Nitrogen (N), Phosphorus (P), Potassium (K),
        Temperature, Humidity, Soil pH, Rainfall
  Models: Random Forest, Naive Bayes, SVM (best auto-selected)

"""

import os 
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

#  CONSTANTS-

DATA_FILE   = "crop_data.csv"
MODEL_FILE  = "best_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "encoder.pkl"

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL   = "label"

CROP_INFO = {
    "rice":        "🌾 Grows best in waterlogged/flooded fields. Staple food crop.",
    "maize":       "🌽 Versatile cereal crop. Used for food, feed and industry.",
    "chickpea":    "🫘 Legume that fixes nitrogen. Drought-tolerant.",
    "kidneybeans": "🫘 High-protein legume. Needs well-drained, fertile soil.",
    "pigeonpeas":  "🫘 Drought-hardy legume. Popular in tropical regions.",
    "mothbeans":   "🫘 Highly drought-resistant legume grown in arid zones.",
    "mungbean":    "🫘 Fast-growing legume. Good for crop rotation.",
    "blackgram":   "🫘 Protein-rich pulse. Thrives in warm humid climates.",
    "lentil":      "🫘 Cool-season legume. Rich in protein and fiber.",
    "pomegranate": "🍎 Fruit tree. Thrives in semi-arid tropical climates.",
    "banana":      "🍌 Tropical fruit. Needs high humidity and rainfall.",
    "mango":       "🥭 King of fruits. Needs dry spell before flowering.",
    "grapes":      "🍇 Fruit vine. Prefers well-drained sandy/loamy soil.",
    "watermelon":  "🍉 High-water-content fruit. Needs warm, sunny climate.",
    "muskmelon":   "🍈 Sweet melon. Needs dry, warm weather to ripen.",
    "apple":       "🍎 Cool-climate fruit. Needs winter chill for flowering.",
    "orange":      "🍊 Citrus fruit. Thrives in subtropical climates.",
    "papaya":      "🍈 Tropical fruit. Fast-growing, sensitive to frost.",
    "coconut":     "🥥 Tropical palm. Grows near coasts and high humidity.",
    "cotton":      "🌿 Fibre crop. Needs long warm season and moderate rain.",
    "jute":        "🌿 Natural fibre crop. Needs warm humid climate.",
    "coffee":      "☕ Cash crop. Thrives in tropical highland climates.",
}

#  DISPLAY HELPERS

def banner():
    print("\n" + "=" * 60)
    print("        🌱  CROP RECOMMENDATION SYSTEM  🌱")
    print("     Fundamentals of AI & ML — BYOP Project")
    print("=" * 60 + "\n")

def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def success(msg): print(f"  ✅ {msg}")
def info(msg):    print(f"  ℹ️  {msg}")
def warn(msg):    print(f"  ⚠️  {msg}")
def error(msg):   print(f"  ❌ {msg}")

#  DATA LOADING

def load_data(path=DATA_FILE):
    if not os.path.exists(path):
        error(f"Dataset '{path}' not found. Run generate_dataset.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    success(f"Loaded dataset: {len(df)} samples | {df[TARGET_COL].nunique()} crops")
    return df

#  MODEL TRAINING & SELECTION

def train_models(df):
    section("TRAINING MODELS")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Naive Bayes":   GaussianNB(),
        "SVM":           SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    }

    results = {}
    print()
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cv     = cross_val_score(model, X_scaled, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
        results[name] = {
            "model":    model,
            "accuracy": acc,
            "cv_mean":  cv.mean(),
            "cv_std":   cv.std(),
            "y_test":   y_test,
            "y_pred":   y_pred,
        }
        print(f"  📊 {name:<18} | Test Acc: {acc:.4f} | CV: {cv.mean():.4f} ± {cv.std():.4f}")

    # Pick best by CV mean
    best_name = max(results, key=lambda n: results[n]["cv_mean"])
    best      = results[best_name]

    section(f"BEST MODEL → {best_name}")
    success(f"Test Accuracy : {best['accuracy']*100:.2f}%")
    success(f"CV Accuracy   : {best['cv_mean']*100:.2f}% ± {best['cv_std']*100:.2f}%")

    print("\n  Classification Report (Best Model):\n")
    crop_names = [le.classes_[i] for i in sorted(set(best["y_test"]))]
    print(classification_report(
        best["y_test"], best["y_pred"],
        target_names=le.classes_,
        zero_division=0
    ))

    # Save artefacts
    with open(MODEL_FILE,  "wb") as f: pickle.dump(best["model"], f)
    with open(SCALER_FILE, "wb") as f: pickle.dump(scaler, f)
    with open(ENCODER_FILE,"wb") as f: pickle.dump(le, f)

    success(f"Model saved → {MODEL_FILE}")
    return best["model"], scaler, le, best_name, results


#  MODEL LOADING

def load_trained():
    if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, ENCODER_FILE]):
        return None, None, None
    with open(MODEL_FILE,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_FILE, "rb") as f: scaler = pickle.load(f)
    with open(ENCODER_FILE,"rb") as f: le     = pickle.load(f)
    return model, scaler, le


#  INPUT VALIDATION

PARAM_LIMITS = {
    "N":           (0,   200,  "kg/ha",  "Nitrogen content in soil"),
    "P":           (0,   200,  "kg/ha",  "Phosphorus content in soil"),
    "K":           (0,   200,  "kg/ha",  "Potassium content in soil"),
    "temperature": (-10, 55,   "°C",     "Average temperature"),
    "humidity":    (0,   100,  "%",      "Relative humidity"),
    "ph":          (0,   14,   "",       "Soil pH level"),
    "rainfall":    (0,   400,  "mm",     "Annual rainfall"),
}

def get_float_input(prompt, lo, hi, unit):
    while True:
        try:
            val = float(input(f"    {prompt} [{lo}–{hi} {unit}]: ").strip())
            if lo <= val <= hi:
                return val
            warn(f"Value must be between {lo} and {hi}. Try again.")
        except ValueError:
            warn("Please enter a valid number.")

def collect_inputs():
    section("ENTER SOIL & CLIMATE PARAMETERS")
    print()
    vals = {}
    for key, (lo, hi, unit, desc) in PARAM_LIMITS.items():
        vals[key] = get_float_input(f"{desc} ({key})", lo, hi, unit)
    return vals


#  PREDICTION

def predict(inputs, model, scaler, le, top_n=3):
    x = np.array([[inputs[f] for f in FEATURE_COLS]])
    x_scaled = scaler.transform(x)

    if hasattr(model, "predict_proba"):
        probs    = model.predict_proba(x_scaled)[0]
        top_idx  = np.argsort(probs)[::-1][:top_n]
        top_crops = [(le.classes_[i], probs[i]) for i in top_idx]
    else:
        pred = model.predict(x_scaled)[0]
        top_crops = [(le.classes_[pred], 1.0)]

    return top_crops

def display_results(top_crops, inputs):
    section("RECOMMENDATION RESULTS")
    print()
    print(f"  {'Rank':<6} {'Crop':<15} {'Confidence':<12} {'Description'}")
    print(f"  {'─'*4}   {'─'*13}   {'─'*10}   {'─'*30}")
    for rank, (crop, conf) in enumerate(top_crops, 1):
        stars = "★" * min(5, round(conf * 5))
        desc  = CROP_INFO.get(crop, "")
        print(f"  #{rank:<5} {crop.capitalize():<15} {conf*100:>6.1f}%  {stars}   {desc}")

    print()
    best_crop = top_crops[0][0]
    section(f"TOP RECOMMENDATION: {best_crop.upper()}")
    success(f"{CROP_INFO.get(best_crop, 'A suitable crop for your conditions.')}")
    print()
    print("  Input Summary:")
    for key, (_, _, unit, desc) in PARAM_LIMITS.items():
        print(f"    • {desc:<30}: {inputs[key]:.2f} {unit}")


#  FEATURE IMPORTANCE (Random Forest only)

def show_feature_importance(model, model_name):
    if model_name != "Random Forest":
        return
    section("FEATURE IMPORTANCE (Random Forest)")
    importances = model.feature_importances_
    pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    max_imp = pairs[0][1]
    print()
    for feat, imp in pairs:
        bar_len = int((imp / max_imp) * 30)
        bar     = "█" * bar_len
        print(f"  {feat:<14} {bar:<30} {imp:.4f}")


#  MAIN MENU

def main():
    banner()

    print("  OPTIONS:")
    print("  [1] Train models on dataset & predict")
    print("  [2] Load existing trained model & predict")
    print("  [3] Train models only (no prediction)")
    print("  [0] Exit")
    print()

    choice = input("  Enter choice: ").strip()

    if choice == "0":
        info("Goodbye! 🌱")
        return

    model, scaler, le, model_name = None, None, None, None

    if choice in ("1", "3"):
        df = load_data()
        model, scaler, le, model_name, all_results = train_models(df)
        show_feature_importance(model, model_name)

    elif choice == "2":
        model, scaler, le = load_trained()
        if model is None:
            warn("No saved model found. Training a new one...")
            df = load_data()
            model, scaler, le, model_name, _ = train_models(df)
        else:
            success("Loaded saved model successfully.")
            model_name = type(model).__name__

    else:
        error("Invalid choice. Exiting.")
        return

    if choice == "3":
        info("Training complete. Model saved. Run again with option [2] to predict.")
        return

    # Prediction loop
    while True:
        inputs    = collect_inputs()
        top_crops = predict(inputs, model, scaler, le, top_n=3)
        display_results(top_crops, inputs)

        print()
        again = input("  🔄 Recommend for another set of conditions? (y/n): ").strip().lower()
        if again != "y":
            break

    info("Thank you for using the Crop Recommendation System! 🌾")


if __name__ == "__main__":
    main()
