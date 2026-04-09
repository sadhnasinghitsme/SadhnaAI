import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR      = os.path.join(BASE_DIR, "data", "processed")
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODEL_PATH         = os.path.join(BASE_DIR, "models", "boq_model.pkl")

OUT_JSON_DIR   = os.path.join(BASE_DIR, "outputs", "json")
OUT_CSV_DIR    = os.path.join(BASE_DIR, "outputs", "csv")
OUT_PLOT_DIR   = os.path.join(BASE_DIR, "outputs", "plots")
OUT_LOG_DIR    = os.path.join(BASE_DIR, "outputs", "logs")

CATEGORIES = ["Plumbing", "Electrical", "Fire", "Civil", "PHE", "Other"]
