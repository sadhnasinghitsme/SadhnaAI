from src.model import BOQModel

model = BOQModel()

# 👉 Train model (so accuracy prints)
model.train("data/clean_training_data.csv")

# 👉 Predict after training
text = "pvc pipe water supply system"

category, confidence = model.predict(text)

print("\nCategory:", category)
print("Confidence:", round(confidence, 2))