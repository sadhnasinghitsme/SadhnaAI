# BOQ Classification

Automatically classifies BOQ (Bill of Quantity) items into **Plumbing, Electrical, Fire, PHE, Other** using a TensorFlow text classification model.

## Run

```bash
pip install -r requirements.txt
python src/main.py
```

## Input

Drop `.xlsx` / `.xls` files into the `data/` folder. The script auto-detects the description column.  
If no files are found, it runs on built-in sample data.

## Output

- Trained model saved to `model/boq_model.keras`
- Predictions printed with confidence scores
