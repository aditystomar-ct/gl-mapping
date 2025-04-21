import os
import glob
import pandas as pd
import numpy as np
import re
import unicodedata

# Best normalize and clean functions
def normalize_text(text):
    """Normalize unicode, lowercase, and strip extra spaces."""
    if pd.isnull(text):
        return ''
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii')
    return text.lower().strip()

def clean_text(text):
    """Remove unwanted characters but keep basic meaning."""
    if pd.isnull(text):
        return ''
    text = re.sub(r'[^a-zA-Z0-9.%/,-]', '', text)  # Keep ., %, -, /
    return text

def is_numeric(s):
    return s.apply(lambda x: re.fullmatch(r'\d+', x) is not None).all()

def is_decimal(s):
    return s.apply(lambda x: re.fullmatch(r'\d+\.\d+', x) is not None).all()

def has_special_characters(s):
    return s.apply(lambda x: any(c in x for c in ['/', '-', ',', '.', '%'])).all()

def is_alphanumeric(s):
    return s.apply(lambda x: re.fullmatch(r'[A-Za-z0-9]+', x) is not None).all()

def is_mostly_letters(s, threshold=1.0):
    letter_counts = s.apply(lambda x: sum(c.isalpha() for c in x))
    total_counts = s.apply(lambda x: len(x))
    ratios = letter_counts / total_counts
    return (ratios == threshold).mean() >= 1.0

def fixed_length(s):
    cleaned = s.apply(clean_text)
    lengths = cleaned.apply(len).unique()
    return len(lengths) == 1

def load_input_file(input_directory):
    files = glob.glob(os.path.join(input_directory, "*.xlsx"))
    if not files:
        raise FileNotFoundError("No XLSX file found.")
    file_path = files[0]
    df = pd.read_excel(file_path, skiprows=0)
    df_for_mapping = df.head(10)
    print(f"Loaded input file '{file_path}' with {len(df)} rows. Using top 10 rows for mapping.")
    return df, df_for_mapping

def load_seed_files(seed_directory):
    seed_files = glob.glob(os.path.join(seed_directory, "*.csv"))
    seeds = {}
    for file in seed_files:
        df = pd.read_csv(file, skipinitialspace=True, dtype=str)
        df.columns = df.columns.str.strip()
        filename = os.path.splitext(os.path.basename(file))[0]
        seeds[filename] = df
        print(f"Loaded seed file: '{filename}' with {len(df)} rows.")
    return seeds

def find_primary_keys(df, tolerance=0):
    primary_keys = []
    for col in df.columns:
        s = df[col].dropna().astype(str).apply(normalize_text).apply(clean_text)
        if s.empty:
            continue
        total_count = len(s)
        unique_count = len(s.unique())
        
        if total_count - unique_count <= tolerance:
            primary_keys.append(col)
            print(f"Primary key candidate: '{col}' (Total: {total_count}, Unique: {unique_count})")
        else:
            # 🆕 Debug print if duplicates found
            print(f"\n⚠️ Column '{col}' not picked as primary key (Total: {total_count}, Unique: {unique_count}).")
            print(f"Finding duplicates for '{col}'...\n")
            duplicated_values = s[s.duplicated(keep=False)]
            if not duplicated_values.empty:
                print(df.loc[duplicated_values.index, [col]])
            print("-" * 50)

    return primary_keys


def analyze_column_properties(s):
    cleaned_s = s.dropna().astype(str).apply(normalize_text).apply(clean_text)
    properties = {
        'is_numeric': is_numeric(cleaned_s),
        'is_decimal': is_decimal(cleaned_s),
        'has_special_characters': has_special_characters(cleaned_s),
        'is_alphanumeric': is_alphanumeric(cleaned_s),
        'is_mostly_letters': is_mostly_letters(cleaned_s),
        'is_fixed_length': fixed_length(cleaned_s),
        'length': cleaned_s.apply(len).unique()[0] if fixed_length(cleaned_s) else None
    }
    return properties

def map_columns(input_sample, seed_col_name, seed_col_values, seed_props):
    matches = []
    for col in input_sample.columns:
        s_input = input_sample[col].dropna().astype(str).apply(normalize_text).apply(clean_text)
        if s_input.empty:
            continue
        input_props = {
            'is_numeric': is_numeric(s_input),
            'is_decimal': is_decimal(s_input),
            'has_special_characters': has_special_characters(s_input),
            'is_alphanumeric': is_alphanumeric(s_input),
            'is_mostly_letters': is_mostly_letters(s_input),
            'is_fixed_length': fixed_length(s_input),
            'length': s_input.apply(len).unique()[0] if fixed_length(s_input) else None
        }
        conditions = all(seed_props[key] == input_props.get(key) for key in seed_props)
        if conditions:
            print(f"Matched input column '{col}' for seed column '{seed_col_name}'")
            matches.append(col)
    if not matches:
        print(f"No match found for seed column '{seed_col_name}'")
    return matches

# === Step 8: Full Processing Flow ===
input_directory = "input"
seed_directory = "seeds"

full_input_df, input_sample = load_input_file(input_directory)
seeds = load_seed_files(seed_directory)

all_mappings = {}

for seed_name, seed_df in seeds.items():
    print(f"\n=== Processing seed file '{seed_name}' ===")
    primary_keys = find_primary_keys(seed_df, tolerance=0)

    mappings = []
    for pk_col in primary_keys:
        if pk_col not in seed_df.columns:
            continue
        seed_col_values = seed_df[pk_col]
        seed_props = analyze_column_properties(seed_col_values)

        print(f"\nProperties for seed column '{pk_col}': {seed_props}")
        
        matched_cols = map_columns(input_sample, pk_col, seed_col_values, seed_props)

        if matched_cols:
            mappings.append({
                'seed_column': pk_col,
                'matched_input_columns': matched_cols,
                'seed_properties': seed_props
            })

    all_mappings[seed_name] = mappings
