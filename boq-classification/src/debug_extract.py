import glob
import os
import pandas as pd
from src.data_loader import find_header_row
from src.extractor import find_col
from src.utils import detect_description_column, detect_unit_column, detect_id_column, detect_qty_column
from config import DATA_RAW_DIR

files = glob.glob(os.path.join(DATA_RAW_DIR, '*.xlsx')) + glob.glob(os.path.join(DATA_RAW_DIR, '*.xls'))
print('files', files)
for f in files[:1]:
    xl = pd.ExcelFile(f)
    for sheet in xl.sheet_names[:2]:
        raw = pd.read_excel(xl, sheet_name=sheet, header=None, nrows=20)
        print('file', os.path.basename(f), 'sheet', sheet)
        for i, row in raw.iterrows():
            print(i, row.tolist())
        header_row = find_header_row(raw)
        print('header_row', header_row)
        df = pd.read_excel(xl, sheet_name=sheet, header=header_row)
        cols = [str(c).lower().replace('\n', ' ').strip() for c in df.columns]
        print('columns', list(df.columns))
        print('normalized cols', cols)
        print('detect desc', detect_description_column(df))
        print('detect unit', detect_unit_column(df))
        print('detect id', detect_id_column(df))
        print('detect qty', detect_qty_column(df))
        print('supply_rate', find_col(df, 'supply rate'))
        print('itc_rate', find_col(df, 'itc rate'))
        print('supply_amount', find_col(df, 'supply amount'))
        print('itc_amount', find_col(df, 'itc amount'))
        print('total', find_col(df, 'total'))
        print('---')
