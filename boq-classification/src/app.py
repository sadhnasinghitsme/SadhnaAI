from model import BOQModel

# ──────────────────────────────────────────────
# 1. TRAIN
# ──────────────────────────────────────────────

print("=" * 50)
print("BOQ CLASSIFICATION - Train & Predict")
print("=" * 50)

model = BOQModel()
model.train("data/training_data.csv")

# ──────────────────────────────────────────────
# 2. PREDICT
# ──────────────────────────────────────────────

test_texts = [
    "pvc pipe water supply system",
    "LED panel light 24W installation with wiring",
    "Fire sprinkler head pendent type installation",
    "RCC concrete M25 for column casting",
    "Sewage pump 3HP submersible for drainage",
    "CCTV camera dome type with DVR installation",
]

print(f"\n{'=' * 50}")
print("PREDICTIONS")
print(f"{'=' * 50}")

for text in test_texts:
    category, confidence = model.predict(text)
    print(f"  {text[:45]:45s} → {category} ({confidence:.2f})")

# ──────────────────────────────────────────────
# 3. SAVE
# ──────────────────────────────────────────────

model.save()
