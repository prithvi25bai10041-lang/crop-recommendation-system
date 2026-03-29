# 🌱 Crop Recommendation System

A machine learning-based command-line application that recommends the most suitable crops to grow based on soil nutrients and climate conditions.

---

## 📌 Project Overview

This project is a **Bring Your Own Project (BYOP)** submission for the course *Fundamentals of AI and ML*. It applies core ML concepts — classification, model comparison, feature engineering, cross-validation — to a practical agricultural problem.

Given seven environmental and soil parameters, the system trains three machine learning models, automatically selects the best-performing one, and recommends the top 3 crops most suited to those conditions.

---

## 🧠 Machine Learning Models Used

| Model | Type | Notes |
|---|---|---|
| Random Forest | Ensemble (Bagging) | Best performer; also provides feature importance |
| Gaussian Naive Bayes | Probabilistic | Fast, interpretable baseline |
| SVM (RBF Kernel) | Kernel Method | Strong on high-dimensional boundaries |

The best model is selected by **5-fold cross-validation accuracy** and saved to disk for reuse.

---

## 🌾 Crops Covered (22 crops)

Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

## 📥 Input Parameters

| Parameter | Unit | Range | Description |
|---|---|---|---|
| N | kg/ha | 0–200 | Nitrogen content in soil |
| P | kg/ha | 0–200 | Phosphorus content in soil |
| K | kg/ha | 0–200 | Potassium content in soil |
| Temperature | °C | -10–55 | Average ambient temperature |
| Humidity | % | 0–100 | Relative humidity |
| pH | — | 0–14 | Soil pH level |
| Rainfall | mm | 0–400 | Annual rainfall |

---

## 📁 Project Structure

```
crop_recommendation/
├── crop_recommendation.py   # Main application (train + predict)
├── generate_dataset.py      # Generates synthetic crop dataset
├── crop_data.csv            # Generated dataset (2200 samples, 22 crops)
├── best_model.pkl           # Saved trained model (auto-generated)
├── scaler.pkl               # Feature scaler (auto-generated)
├── encoder.pkl              # Label encoder (auto-generated)
└── README.md                # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/crop-recommendation-system.git
cd crop-recommendation-system
```

### Step 2 — Install Dependencies

```bash
pip install scikit-learn pandas numpy
```

### Step 3 — Generate the Dataset

```bash
python generate_dataset.py
```

This creates `crop_data.csv` with 2,200 samples across 22 crop types.

---

## 🚀 Running the Application

```bash
python crop_recommendation.py
```

You will see a menu:

```
============================================================
        🌱  CROP RECOMMENDATION SYSTEM  🌱
     Fundamentals of AI & ML — BYOP Project
============================================================

  OPTIONS:
  [1] Train models on dataset & predict
  [2] Load existing trained model & predict
  [3] Train models only (no prediction)
  [0] Exit
```

### First Run — Use Option 1

Option `[1]` trains all three models, displays a comparison, selects the best model, and then prompts you for inputs to make a prediction.

### Subsequent Runs — Use Option 2

Option `[2]` loads the saved model instantly and skips retraining. Ideal for repeated use.

---

## 📊 Sample Output

```
  📊 Random Forest      | Test Acc: 0.9000 | CV: 0.9205 ± 0.0032
  📊 Naive Bayes        | Test Acc: 0.8841 | CV: 0.9018 ± 0.0068
  📊 SVM                | Test Acc: 0.8727 | CV: 0.8845 ± 0.0079

  BEST MODEL → Random Forest
  ✅ Test Accuracy : 90.00%
  ✅ CV Accuracy   : 92.05% ± 0.32%

  RECOMMENDATION RESULTS
  ─────────────────────────────────────────────────────────
  Rank   Crop            Confidence   Description
  ─────────────────────────────────────────────────────────
  #1     Rice              72.0%  ████   🌾 Grows best in waterlogged/flooded fields.
  #2     Jute              18.0%  ██     🌿 Natural fibre crop. Needs warm humid climate.
  #3     Banana             8.0%  █      🍌 Tropical fruit. Needs high humidity.
```

---

## 📈 Feature Importance

The Random Forest model reveals which factors matter most:

```
  humidity       ██████████████████████████████ 0.2150
  rainfall       █████████████████████████      0.1794
  N              ██████████████████████         0.1604
  P              ████████████████████           0.1499
  K              ████████████████████           0.1498
  temperature    ████████████                   0.0915
  ph             ███████                        0.0540
```

---

## 🧪 Dataset Details

- **Source:** Synthetically generated based on published agronomic data for each crop
- **Samples:** 2,200 (100 per crop × 22 crops)
- **Features:** 7 numerical features
- **Target:** Crop label (22 classes)
- **Split:** 80% training / 20% testing

---

## 📚 Dependencies

| Library | Version | Purpose |
|---|---|---|
| scikit-learn | ≥1.0 | ML models, metrics, preprocessing |
| pandas | ≥1.3 | Data loading and manipulation |
| numpy | ≥1.21 | Numerical operations |

---

## 👤 Author

**[Your Name]**  
Course: Fundamentals of AI and ML  
Submission: BYOP Capstone Project  
Deadline: March 31, 2026

---

## 📄 License

This project is submitted as academic coursework. Feel free to reference or build upon it with attribution.
