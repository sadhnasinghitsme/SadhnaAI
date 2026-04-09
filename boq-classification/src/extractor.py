import re
import pandas as pd
from utils import (
    detect_description_column,
    detect_unit_column,
    detect_id_column,
    detect_qty_column,
)


# ---------- HELPERS ----------

def normalize_column_name(col):
    text = str(col).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def clean_value(val):
    if pd.isna(val):
        return ""
    if isinstance(val, str):
        return val.strip()
    return val


def find_col(df, keyword):
    keyword = keyword.lower()
    for col in df.columns:
        if keyword in normalize_column_name(col):
            return col
    return None


# ---------- MAIN EXTRACTION ----------

def extract_line_items(
    df, id_col, desc_col, unit_col, qty_col,
    supply_rate_col, itc_rate_col,
    supply_amt_col, itc_amt_col,
    total_amt_col, remarks_col
):
    items = []

    current_section = "General"
    current_section_code = ""

    for _, row in df.iterrows():

        raw_desc = clean_value(row[desc_col]).replace("\n", " ") if desc_col else ""

        # ---------- SECTION DETECTION ----------
        if raw_desc.isupper() and len(raw_desc) > 5:
            current_section = raw_desc
            continue

        raw_id   = clean_value(row[id_col]) if id_col else ""
        raw_unit = clean_value(row[unit_col]) if unit_col else ""
        raw_qty  = row[qty_col] if qty_col and pd.notna(row[qty_col]) else 0

        # ---------- AMOUNT FIELDS ----------
        supply_rate = clean_value(row.get(supply_rate_col, "")) if supply_rate_col else ""
        itc_rate    = clean_value(row.get(itc_rate_col, "")) if itc_rate_col else ""
        supply_amt  = clean_value(row.get(supply_amt_col, "")) if supply_amt_col else ""
        itc_amt     = clean_value(row.get(itc_amt_col, "")) if itc_amt_col else ""
        total_amt   = clean_value(row.get(total_amt_col, "")) if total_amt_col else ""
        remarks     = clean_value(row.get(remarks_col, "")) if remarks_col else ""

        # ---------- SKIP INVALID ----------
        if not raw_desc or raw_desc.lower() == "nan":
            continue

        # ---------- FINAL MERGED OBJECT ----------
        items.append({
            "section": current_section,
            "section_code": current_section_code,
            "id": raw_id,
            "description": raw_desc,
            "unit": raw_unit,
            "quantity": raw_qty,

            # amounts (NOW INCLUDED)
            "supply_rate": supply_rate,
            "itc_rate": itc_rate,
            "supply_amount": supply_amt,
            "itc_amount": itc_amt,
            "total_amount": total_amt,
            "remarks": remarks
        })

    return items


# ---------- FILE PROCESSING ----------

def extract_from_files(boq_files):
    results = []

    for filename, df in boq_files:

        # Clean columns
        df.columns = [normalize_column_name(c) for c in df.columns]
        df = df.ffill()

        # Detect columns
        supply_rate_col = find_col(df, "supply rate")
        itc_rate_col    = find_col(df, "itc rate")
        supply_amt_col  = find_col(df, "supply amount")
        itc_amt_col     = find_col(df, "itc amount")
        total_amt_col   = find_col(df, "total")
        remarks_col     = find_col(df, "remark")

        desc_col = detect_description_column(df)
        unit_col = detect_unit_column(df)
        id_col   = detect_id_column(df)
        qty_col  = detect_qty_column(df)

        print("\n🔍 DEBUG COLUMNS:", df.columns)
        print("✔ Detected:", {
            "desc": desc_col,
            "unit": unit_col,
            "qty": qty_col,
            "supply_rate": supply_rate_col,
            "total": total_amt_col
        })

        if desc_col is None:
            print(f"[!] No description column in {filename}, skipping.")
            continue

        items = extract_line_items(
            df,
            id_col,
            desc_col,
            unit_col,
            qty_col,
            supply_rate_col,
            itc_rate_col,
            supply_amt_col,
            itc_amt_col,
            total_amt_col,
            remarks_col,
        )

        if items:
            results.append((filename, items))
        else:
            print(f"[!] No line items extracted from {filename}.")

    return results