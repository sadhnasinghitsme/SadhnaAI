import os
import pandas as pd


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def detect_description_column(df: pd.DataFrame):
    for col in df.columns:
        if any(k in col.lower() for k in ["desc", "item", "work", "particular"]):
            return col
    str_cols = df.select_dtypes(include="object").columns
    return str_cols[0] if len(str_cols) else None


def detect_unit_column(df: pd.DataFrame):
    for col in df.columns:
        if "unit" in col.lower():
            return col
    return None


def detect_id_column(df: pd.DataFrame):
    for col in df.columns:
        if "sl" in col.lower() or col.lower() == "id":
            return col
    return None


def detect_qty_column(df: pd.DataFrame):
    for col in df.columns:
        if "qty" in col.lower() or "quantity" in col.lower():
            return col
    return None


def is_numeric(val) -> bool:
    try:
        f = float(val)
        return f == f  # False for NaN
    except:
        return False
