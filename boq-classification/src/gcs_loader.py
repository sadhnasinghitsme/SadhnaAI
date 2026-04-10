"""
Google Cloud Storage loader for BOQ files and training data.

Usage:
    1. Set env vars: GCS_BUCKET_NAME, GOOGLE_APPLICATION_CREDENTIALS
    2. Upload your BOQ Excel files / CSVs to the bucket under GCS_DATA_PREFIX
    3. Call download functions to pull data locally or load directly into pandas
"""

import os
import io
import pandas as pd
from google.cloud import storage
from config import GCS_BUCKET_NAME, GCS_DATA_PREFIX, DATA_RAW_DIR, TRAINING_DATA_PATH


def get_gcs_client():
    """Create and return a GCS client (uses GOOGLE_APPLICATION_CREDENTIALS env var)."""
    return storage.Client()


def list_gcs_files(extension=None):
    """List all files in the GCS bucket under the configured prefix."""
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=GCS_DATA_PREFIX)

    files = []
    for blob in blobs:
        if extension and not blob.name.endswith(extension):
            continue
        files.append(blob.name)
    return files


def download_boq_files_from_gcs(local_dir=None):
    """Download all Excel BOQ files from GCS to local data/raw directory."""
    local_dir = local_dir or DATA_RAW_DIR
    os.makedirs(local_dir, exist_ok=True)

    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=GCS_DATA_PREFIX)

    downloaded = []
    for blob in blobs:
        if blob.name.endswith((".xlsx", ".xls")):
            filename = os.path.basename(blob.name)
            local_path = os.path.join(local_dir, filename)
            blob.download_to_filename(local_path)
            downloaded.append(local_path)
            print(f"[GCS] Downloaded: {filename}")

    print(f"[GCS] Total files downloaded: {len(downloaded)}")
    return downloaded


def load_training_csv_from_gcs(gcs_path=None):
    """Load a training CSV directly from GCS into a DataFrame.

    Args:
        gcs_path: Full blob path like 'boq-data/training_data.csv'.
                  Defaults to GCS_DATA_PREFIX + 'training_data.csv'.
    """
    gcs_path = gcs_path or f"{GCS_DATA_PREFIX}training_data.csv"

    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(gcs_path)

    content = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(content))
    print(f"[GCS] Loaded training data: {len(df)} rows from gs://{GCS_BUCKET_NAME}/{gcs_path}")
    return df


def merge_local_and_gcs_training_data(gcs_path=None):
    """Merge local training_data.csv with GCS training data (deduplicates)."""
    # Load local
    local_df = pd.read_csv(TRAINING_DATA_PATH) if os.path.exists(TRAINING_DATA_PATH) else pd.DataFrame()

    # Load from GCS
    try:
        gcs_df = load_training_csv_from_gcs(gcs_path)
    except Exception as e:
        print(f"[GCS] Could not load from GCS: {e}")
        return local_df

    # Merge and deduplicate
    merged = pd.concat([local_df, gcs_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["description"], keep="last")
    print(f"[GCS] Merged dataset: {len(local_df)} local + {len(gcs_df)} GCS = {len(merged)} unique rows")
    return merged


def upload_training_csv_to_gcs(df=None, gcs_path=None):
    """Upload local training data to GCS (for sharing with team)."""
    gcs_path = gcs_path or f"{GCS_DATA_PREFIX}training_data.csv"

    if df is None:
        df = pd.read_csv(TRAINING_DATA_PATH)

    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(gcs_path)

    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
    print(f"[GCS] Uploaded {len(df)} rows to gs://{GCS_BUCKET_NAME}/{gcs_path}")
