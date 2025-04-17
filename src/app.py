import os
import glob
import pandas as pd
import numpy as np


def load_and_split_input(input_directory, n_chunks=10):
    files = glob.glob(os.path.join(input_directory, "*.xlsx"))
    if not files:
        raise FileNotFoundError("No input XLSX files found in the directory.")
    file_path = files[0]
    df = pd.read_excel(file_path, skiprows=2)
    df_for_mapping = df.head(10)
    chunks = np.array_split(df_for_mapping, n_chunks)
    print(f"Loaded input file '{file_path}' with {len(df)} rows (skipped first 2 rows), and using top 10 rows for mapping split into {n_chunks} chunks.")
    return df, chunks


def load_seed_files(directory_path):
    seed_files = glob.glob(os.path.join(directory_path, "*.csv"))
    seeds = {}
    for file in seed_files:
        df = pd.read_csv(file, skipinitialspace=True, dtype=str)
        filename = os.path.splitext(os.path.basename(file))[0]
        seeds[filename] = df
        print(f"Loaded seed file: '{filename}' with {len(df)} rows.")
    return seeds


def find_primary_keys_fixed_length(df):
    primary_keys = []
    df_full = df
    for col in df_full.columns:
        s = df_full[col].dropna().astype(str)
        if s.empty:
            continue
        lengths = s.apply(len).unique()
        if len(lengths) == 1 and len(s) == len(s.unique()):
            primary_keys.append((col, lengths[0]))
            print(f"Primary key candidate: '{col}' with fixed length {lengths[0]}")
    return primary_keys


def map_fixed_length_columns(input_sample, seed_column, seed_length):
    matches = []
    for col in input_sample.columns:
        s = input_sample[col].dropna().astype(str)
        if s.empty:
            continue
        unique_lengths = s.apply(len).unique()
        if len(unique_lengths) == 1 and unique_lengths[0] == seed_length:
            matches.append(col)
    return matches


def process_seed_file_mapping(input_sample, seed_df, seed_name, fixed_length_keys):
    print(f"STEP 4: Processing mapping for seed file '{seed_name}'")
    mappings = []
    if not fixed_length_keys:
        print(f"No fixed-length primary key candidates found in seed file '{seed_name}'.")
        return mappings

    for pk_col, pk_length in fixed_length_keys:
        current_matches = map_fixed_length_columns(input_sample, pk_col, pk_length)
        if current_matches:
            mappings.append({
                'seed_file': seed_name,
                'seed_column': pk_col,
                'fixed_length': pk_length,
                'input_columns': current_matches
            })
            print(f"Candidate '{pk_col}' maps to input column(s): {current_matches}")
        else:
            print(f"No matching input column found for seed candidate '{pk_col}' with fixed length {pk_length}")
    return mappings


def run_gl_mapping(input_dir="input", seed_dir="seeds"):
    full_input_df, _ = load_and_split_input(input_dir)
    input_sample = full_input_df.head(10)

    seeds = load_seed_files(seed_dir)
    fixed_length_primary_keys = {}
    for seed_name, seed_df in seeds.items():
        print(f"\nSeed File: '{seed_name}'")
        pk_candidates = find_primary_keys_fixed_length(seed_df)
        fixed_length_primary_keys[seed_name] = pk_candidates
        print("Primary Key Candidates:", pk_candidates)

    all_mappings = {}
    for seed_name, seed_df in seeds.items():
        print(f"\n=== Processing seed file: '{seed_name}' ===")
        mapping_for_seed = process_seed_file_mapping(
            input_sample, seed_df, seed_name,
            fixed_length_primary_keys.get(seed_name, [])
        )
        all_mappings[seed_name] = mapping_for_seed

    print("\n=== Final Mapping Results ===")
    for seed_name, mapping_list in all_mappings.items():
        for mapping in mapping_list:
            input_cols_str = ", ".join(mapping["input_columns"])
            print(f"Seed File '{seed_name}': '{mapping['seed_column']}' -> {input_cols_str} (Fixed Length: {mapping['fixed_length']})")


run_gl_mapping()
