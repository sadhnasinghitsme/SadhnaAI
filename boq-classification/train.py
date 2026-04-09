from src.model import BOQModel

model = BOQModel()

model.train("data/clean_training_data.csv")
model.save()