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

# Google Cloud Storage config
# Set these via environment variables or update directly
GCS_BUCKET_NAME    = os.environ.get("GCS_BUCKET_NAME", "your-boq-bucket")
GCS_DATA_PREFIX    = os.environ.get("GCS_DATA_PREFIX", "boq-data/")  # folder path in bucket
GCS_CREDENTIALS    = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
