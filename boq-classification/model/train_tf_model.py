import pandas as pd
import numpy as np
import re
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info logs

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
# 1. TEXT CLEANING
# ──────────────────────────────────────────────

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────
# 2. LOAD & PREPARE DATA
# ──────────────────────────────────────────────

print("=" * 50)
print("BOQ CLASSIFICATION - TensorFlow Model Training")
print("=" * 50)

df = pd.read_csv("data/training_data.csv")
df["description"] = df["description"].apply(clean_text)
df = df[df["description"].str.len() > 3].dropna().drop_duplicates()

print(f"\n📊 Total samples: {len(df)}")
print(f"📊 Categories: {df['category'].nunique()}")
print(f"\nCategory Distribution:")
print(df["category"].value_counts().to_string())

texts = df["description"].tolist()
labels = df["category"].tolist()


# ──────────────────────────────────────────────
# 3. VECTORIZE + ENCODE
# ──────────────────────────────────────────────

le = LabelEncoder()
y_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

# Train/test split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# TextVectorization (fit on ALL texts)
from tensorflow.keras.layers import TextVectorization
vectorizer = TextVectorization(max_tokens=4000, output_sequence_length=30)
vectorizer.adapt(texts)

X_train = vectorizer(X_train_text)
X_test = vectorizer(X_test_text)

print(f"\n🔹 Train: {len(X_train)} | Test: {len(X_test)}")
print(f"🔹 Classes: {list(le.classes_)}")


# ──────────────────────────────────────────────
# 4. BUILD KERAS MODEL
# ──────────────────────────────────────────────

model = Sequential([
    Embedding(input_dim=4000, output_dim=64),
    GlobalAveragePooling1D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"\n{'=' * 50}")
print("TRAINING STARTED")
print(f"{'=' * 50}\n")


# ──────────────────────────────────────────────
# 5. TRAIN (Epoch 1/10 ... Epoch 10/10)
# ──────────────────────────────────────────────

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1,
)


# ──────────────────────────────────────────────
# 6. RESULTS
# ──────────────────────────────────────────────

final_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print(f"\nTrain Accuracy: {final_acc}")
print(f"Validation Accuracy: {val_acc}")


# ──────────────────────────────────────────────
# 7. SAVE MODEL + VECTORIZER + ENCODER
# ──────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
model.save("models/boq_tf_model.keras")
vectorizer_model = tf.keras.Sequential([vectorizer])
vectorizer_model.save("models/vectorizer_model.keras")
joblib.dump(le, "models/label_encoder.pkl")

print(f"\n💾 Model saved  -> models/boq_tf_model.keras")
print(f"💾 Vectorizer saved -> models/vectorizer_model.keras")
print(f"💾 Encoder saved -> models/label_encoder.pkl")


# ──────────────────────────────────────────────
# 8. TEST PREDICTIONS
# ──────────────────────────────────────────────

test_texts = [
    "Supplying, fixing and testing PVC pipes for water supply system",
    "LED panel light 24W installation with wiring",
    "Fire sprinkler head pendent type installation",
    "RCC concrete M25 for column casting",
    "Sewage pump 3HP submersible for drainage",
    "CCTV camera dome type with DVR installation",
]

print(f"\n{'=' * 60}")
print("SAMPLE PREDICTIONS")
print(f"{'=' * 60}")

for text in test_texts:
    cleaned = clean_text(text)
    vec = vectorizer(tf.constant([cleaned]))
    proba = model.predict(vec, verbose=0)[0]
    idx = np.argmax(proba)
    cat = le.classes_[idx]
    conf = proba[idx]
    print(f"  {cat:12s} ({conf:.2%}) | {text[:50]}")

print(f"\n{'=' * 60}")
print("✅ DONE!")
print(f"{'=' * 60}")
