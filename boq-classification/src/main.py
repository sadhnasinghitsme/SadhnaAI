import json
import os
import pandas as pd
from db.mongo import save_to_mongo

from config import OUT_JSON_DIR, OUT_CSV_DIR, OUT_LOG_DIR, OUT_PLOT_DIR
from utils import ensure_dirs
from data_loader import load_boq_files
from extractor import extract_from_files
from classifier import load_or_train, classify
from visualizer import plot_2d_category_map
from db.mongo import save_to_mongo

def main():
    ensure_dirs(OUT_JSON_DIR, OUT_CSV_DIR, OUT_LOG_DIR, OUT_PLOT_DIR)

    pipeline  = load_or_train()
    boq_files = load_boq_files()

    extracted = extract_from_files(boq_files)
    records, _ = classify(pipeline, extracted)

    # -- Build JSON output in {items, amounts} shape ----------
    json_output = {"items": [], "amounts": []}
    for _, payload in extracted:
        if isinstance(payload, dict):
            json_output["items"].extend(payload.get("items", []))
            json_output["amounts"].extend(payload.get("amounts", []))
        else:
            json_output["items"].extend(payload)

    json_output = {
        "items": [
            {
                "id": item.get("id"),
                "description": item.get("description", ""),
                "unit": item.get("unit", ""),
                "quantity": item.get("quantity", 0),
            }
            for item in json_output["items"]
        ],
        "amounts": [
            {
                "id": amount.get("id"),
                "supply_rate": amount.get("supply_rate", ""),
                "itc_rate": amount.get("itc_rate", ""),
                "supply_amount": amount.get("supply_amount", ""),
                "itc_amount": amount.get("itc_amount", ""),
                "total_amount": amount.get("total_amount", ""),
            }
            for amount in json_output["amounts"]
        ],
    }

    json_path = os.path.join(OUT_JSON_DIR, "output.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, allow_nan=False)
    print(f"[OK] JSON saved -> {json_path}")
    save_to_mongo(json_output)

    # -- Save CSV --------------------------------------─
    csv_path = os.path.join(OUT_CSV_DIR, "output.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"[OK] CSV  saved -> {csv_path}")

    # -- Plot ------------------------------------------─
    plot_2d_category_map(records)

    # -- Summary ----------------------------------------
    total = len(records)
    low   = sum(1 for r in records if r["confidence"] < 0.6)
    print(f"\n[i] Total items: {total} | Low confidence (<60%): {low}")


if __name__ == "__main__":
    main()
