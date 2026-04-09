"""
BOQ Text Classification Pipeline
- TF-IDF vectorization with bigrams
- Logistic Regression with class weight balancing
- Full evaluation with classification report
- Confidence-based prediction
"""

import os
import re
import warnings
import numpy as np
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from config import MODEL_PATH, TRAINING_DATA_PATH, CATEGORIES

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. TEXT PREPROCESSING
# ──────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Clean and normalize BOQ description text."""
    text = str(text).lower().strip()
    # remove special chars but keep spaces and alphanumeric
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# 2. LOAD TRAINING DATA
# ──────────────────────────────────────────────

def load_training_data() -> tuple[list[str], list[str]]:
    """Load and preprocess training data from CSV."""
    df = pd.read_csv(TRAINING_DATA_PATH)
    df = df.dropna(subset=["description", "category"])
    df["description"] = df["description"].apply(preprocess_text)
    df = df[df["description"].str.len() > 3]  # drop empty/tiny rows

    texts = df["description"].tolist()
    labels = df["category"].tolist()

    print(f"[OK] Loaded {len(texts)} training samples")
    print(f"     Classes: {dict(df['category'].value_counts())}")
    return texts, labels


# ──────────────────────────────────────────────
# 3. BUILD PIPELINE
# ──────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    TF-IDF (unigrams + bigrams + trigrams) → Calibrated LinearSVC.

    Why this combo:
    - TF-IDF with trigrams captures multi-word BOQ terms like 'fire hose reel'
    - LinearSVC is more decisive on small datasets than LogisticRegression
    - CalibratedClassifierCV wraps SVC to provide probability estimates
    - class_weight='balanced' handles imbalanced categories
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),       # unigrams + bigrams + trigrams
            max_features=8000,        # larger vocab for trigrams
            sublinear_tf=True,        # apply log normalization to TF
            min_df=1,                 # keep rare terms (small dataset)
            strip_accents="unicode",
            analyzer="word",
        )),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                max_iter=2000,
                C=1.0,
            ),
            cv=3,
        )),
    ])


# ──────────────────────────────────────────────
# 4. TRAIN + EVALUATE
# ──────────────────────────────────────────────

def train_model() -> Pipeline:
    """Train model, evaluate with cross-validation, save to disk."""
    texts, labels = load_training_data()

    pipeline = build_pipeline()

    # Cross-validation (stratified to preserve class ratios)
    n_splits = min(5, min(pd.Series(labels).value_counts()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="f1_macro")
        print(f"\n[EVAL] {n_splits}-Fold Cross-Validation:")
        print(f"       F1 Macro: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Train on full data
    pipeline.fit(texts, labels)

    # Full training set evaluation (to check fit)
    preds = pipeline.predict(texts)
    print(f"\n[EVAL] Training Set Accuracy: {accuracy_score(labels, preds):.4f}")
    print("\n" + classification_report(labels, preds, zero_division=0))

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[OK] Model saved -> {MODEL_PATH}")

    return pipeline


# ──────────────────────────────────────────────
# 5. LOAD OR TRAIN
# ──────────────────────────────────────────────

def load_or_train() -> Pipeline:
    """Load saved model or train a new one."""
    if os.path.exists(MODEL_PATH):
        print("[OK] Loaded existing model")
        return joblib.load(MODEL_PATH)
    return train_model()


# ──────────────────────────────────────────────
# 6. PREDICT WITH CONFIDENCE
# ──────────────────────────────────────────────

def predict_single(pipeline, text: str) -> tuple[str, float]:
    """Predict category and confidence for a single description."""
    cleaned = preprocess_text(text)
    proba = pipeline.predict_proba([cleaned])[0]
    best_idx = np.argmax(proba)
    return pipeline.classes_[best_idx], round(float(proba[best_idx]), 4)


# ──────────────────────────────────────────────
# 7. CLASSIFY EXTRACTED BOQ DATA
# ──────────────────────────────────────────────

def classify(pipeline, extracted: list[tuple[str, list[dict]]]) -> tuple[list[dict], dict]:
    """Classify extracted BOQ items and return records + grouped output."""
    records = []
    grouped_output = {}

    for filename, payload in extracted:
        # Handle both formats
        items = payload.get("items", []) if isinstance(payload, dict) else payload

        if not items:
            print(f"[!] No items to classify from {filename}")
            continue

        texts = [preprocess_text(item.get("description", "")) for item in items]
        probs = pipeline.predict_proba(texts)
        labels = pipeline.classes_

        for i, item in enumerate(items):
            best_idx = np.argmax(probs[i])
            category = labels[best_idx]
            confidence = round(float(probs[i][best_idx]), 4)

            item["category"] = category
            item["confidence"] = confidence

            section = item.get("section", "General")
            section_code = item.get("section_code", "")
            key = f"{filename} > {section}"

            if key not in grouped_output:
                grouped_output[key] = []
            grouped_output[key].append(item)

            records.append({
                "file": filename,
                "section": section,
                "section_code": section_code,
                "id": item.get("id"),
                "description": item.get("description", ""),
                "unit": item.get("unit", ""),
                "quantity": item.get("quantity", 0),
                "supply_rate": item.get("supply_rate", ""),
                "itc_rate": item.get("itc_rate", ""),
                "supply_amount": item.get("supply_amount", ""),
                "itc_amount": item.get("itc_amount", ""),
                "total_amount": item.get("total_amount", ""),
                "remarks": item.get("remarks", ""),
                "category": category,
                "confidence": confidence,
            })

        print(f"[OK] Classified {len(items)} items from {filename}")

    # Sort inside each section
    for section in grouped_output:
        grouped_output[section] = sorted(
            grouped_output[section],
            key=lambda x: str(x.get("id", "")),
        )

    return records, grouped_output
