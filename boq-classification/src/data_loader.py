import glob
import os
import pandas as pd
from config import DATA_RAW_DIR


def find_header_row(df_raw: pd.DataFrame, max_rows: int = 15) -> int:
    for i, row in df_raw.iterrows():
        if i >= max_rows:
            break
        vals = [str(v).upper() for v in row.tolist()]
        if any("DESCRIPTION" in v or "SL.NO" in v for v in vals):
            return i
    return 0


def load_boq_files() -> list[tuple[str, pd.DataFrame]]:
    files = glob.glob(os.path.join(DATA_RAW_DIR, "*.xlsx")) + \
            glob.glob(os.path.join(DATA_RAW_DIR, "*.xls"))
    results = []
    for f in files:
        xl = pd.ExcelFile(f)
        for sheet in xl.sheet_names:
            raw = pd.read_excel(xl, sheet_name=sheet, header=None, nrows=15)
            header_row = find_header_row(raw)
            df = pd.read_excel(xl, sheet_name=sheet, header=header_row)
            label = f"{os.path.basename(f)} [{sheet}]"
            results.append((label, df))
            print(f"[OK] Loaded: {label} (header row={header_row}, {len(df)} rows)")
    return results
