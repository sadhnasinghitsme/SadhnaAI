"""
Data augmentation for BOQ training data.
Expands the dataset by generating variations of existing descriptions.
No external APIs or cloud services needed.
"""

import random
import pandas as pd
from config import TRAINING_DATA_PATH

# Synonym replacements common in BOQ descriptions
SYNONYMS = {
    "supplying": ["providing", "supply of", "procurement of"],
    "providing": ["supplying", "supply of", "furnishing"],
    "fixing": ["installing", "fitting", "mounting", "erecting"],
    "installing": ["fixing", "fitting", "mounting", "erecting"],
    "installation": ["fixing", "fitting", "erection", "mounting"],
    "laying": ["placing", "installing", "fixing"],
    "pipe": ["pipeline", "piping"],
    "pipeline": ["pipe", "piping"],
    "work": ["works", "job"],
    "works": ["work", "job"],
    "system": ["setup", "arrangement"],
    "testing": ["commissioning", "checking"],
    "construction": ["building", "erection"],
    "providing and fixing": ["supply and installation of", "supplying and fitting"],
    "supplying and fixing": ["providing and installing", "supply and fixing of"],
    "providing and laying": ["supplying and placing", "supply and laying of"],
}

# Unit/size variations
SIZE_VARIANTS = {
    "15mm": ["20mm", "25mm"],
    "20mm": ["15mm", "25mm"],
    "25mm": ["20mm", "32mm"],
    "32mm": ["25mm", "40mm"],
    "50mm": ["40mm", "65mm"],
    "100mm": ["80mm", "110mm", "150mm"],
    "150mm": ["100mm", "200mm"],
    "200mm": ["150mm", "250mm"],
    "2.5 sqmm": ["1.5 sqmm", "4 sqmm"],
    "4 sqmm": ["2.5 sqmm", "6 sqmm"],
    "1.5 sqmm": ["1 sqmm", "2.5 sqmm"],
    "600x600": ["300x300", "800x800"],
    "800x800": ["600x600", "1000x1000"],
    "12mm": ["10mm", "15mm"],
    "18W": ["12W", "24W"],
    "12W": ["9W", "18W"],
}


def synonym_replace(text):
    """Replace random words/phrases with synonyms."""
    result = text.lower()
    # Try phrase-level replacements first
    for phrase, replacements in SYNONYMS.items():
        if phrase in result and random.random() > 0.4:
            result = result.replace(phrase, random.choice(replacements), 1)
    return result.strip()


def size_swap(text):
    """Swap dimensions/sizes with similar values."""
    result = text
    for size, variants in SIZE_VARIANTS.items():
        if size in result and random.random() > 0.5:
            result = result.replace(size, random.choice(variants), 1)
    return result


def word_shuffle(text):
    """Lightly shuffle word order while keeping it readable."""
    words = text.split()
    if len(words) < 5:
        return text
    # Swap two adjacent words randomly
    idx = random.randint(0, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return " ".join(words)


def add_prefix_suffix(text):
    """Add common BOQ prefixes or suffixes."""
    prefixes = [
        "Rate for ", "Cost of ", "Item: ", "BOQ item - ",
        "As per specification ", "As per drawing ",
    ]
    suffixes = [
        " as per specification", " as per drawing", " including all accessories",
        " complete in all respects", " as directed by engineer",
        " including labour and material", " with all fittings",
    ]
    if random.random() > 0.5:
        text = random.choice(prefixes) + text
    else:
        text = text + random.choice(suffixes)
    return text


def augment_row(description, num_variants=3):
    """Generate multiple augmented variants of a single description."""
    variants = []
    transforms = [synonym_replace, size_swap, word_shuffle, add_prefix_suffix]

    for _ in range(num_variants):
        text = description
        # Apply 1-2 random transforms
        chosen = random.sample(transforms, k=random.randint(1, 2))
        for fn in chosen:
            text = fn(text)
        if text != description:
            variants.append(text)

    return variants


def augment_dataset(input_path=None, output_path=None, variants_per_row=3):
    """Augment the full training dataset and save."""
    input_path = input_path or TRAINING_DATA_PATH
    output_path = output_path or TRAINING_DATA_PATH

    df = pd.read_csv(input_path)
    original_count = len(df)

    new_rows = []
    for _, row in df.iterrows():
        variants = augment_row(row["description"], num_variants=variants_per_row)
        for v in variants:
            new_rows.append({"description": v, "category": row["category"]})

    augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    augmented_df = augmented_df.drop_duplicates(subset=["description"], keep="first")

    augmented_df.to_csv(output_path, index=False)
    print(f"[Augment] {original_count} -> {len(augmented_df)} rows (saved to {output_path})")
    return augmented_df


if __name__ == "__main__":
    augment_dataset(variants_per_row=5)
