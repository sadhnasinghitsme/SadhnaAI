import glob
import os
import pandas as pd
from data_loader import load_boq_files
from extractor import find_col
from utils import detect_description_column, detect_unit_column, detect_id_column, detect_qty_column
from config import DATA_RAW_DIR

files = glob.glob(os.path.join(DATA_RAW_DIR, '*.xlsx')) + glob.glob(os.path.join(DATA_RAW_DIR, '*.xls'))
print('files:', files)
for f in files:
    xl = pd.ExcelFile(f)
    print('file:', os.path.basename(f))
    for sheet in xl.sheet_names:
        print('sheet:', sheet)
        raw = pd.read_excel(xl, sheet_name=sheet, header=None, nrows=15)
        header_row = -1
        for i, row in raw.iterrows():
            vals = [str(v).upper() for v in row.tolist()]
            if any('DESCRIPTION' in v or 'SL.NO' in v for v in vals):
                header_row = i
                break
        print('header_row:', header_row)
        df = pd.read_excel(xl, sheet_name=sheet, header=header_row if header_row >= 0 else 0)
        print('raw cols:', list(df.columns))
        norm_cols = [str(c).lower().replace('\n', ' ').strip() for c in df.columns]
        print('normalized cols:', norm_cols)
        print('desc_col:', detect_description_column(df))
        print('unit_col:', detect_unit_column(df))
        print('id_col:', detect_id_column(df))
        print('qty_col:', detect_qty_column(df))
        print('supply_rate:', find_col(df, 'supply rate'))
        print('itc_rate:', find_col(df, 'itc rate'))
        print('supply_amount:', find_col(df, 'supply amount'))
        print('itc_amount:', find_col(df, 'itc amount'))
        print('total:', find_col(df, 'total'))
        print('-'*80)
