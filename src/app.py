import os
import argparse
import pandas as pd
import re
import unicodedata

# === Utility Functions ===

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9.%/,-]', '', text)

def normalize_text(text):
    """Remove hidden Unicode control characters like \u202d"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def is_numeric(s):
    return s.apply(lambda x: re.fullmatch(r'\d+', x) is not None).all()

def is_decimal(s):
    return s.apply(lambda x: re.fullmatch(r'\d+\.\d+', x) is not None).all()

def has_special_characters(s):
    return s.apply(lambda x: any(c in x for c in ['/', '-', ',', '.', '%'])).any()

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

def find_primary_keys(df, tolerance=0): # From testing.ipynb
    primary_keys = []
    for col in df.columns:
        s = df[col].dropna().astype(str).apply(normalize_text).apply(clean_text)
        if s.empty:
            continue
        total_count = len(s)
        unique_count = len(s.unique())
        
        if total_count - unique_count <= tolerance:
            primary_keys.append(col)
            # print(f"Primary key candidate: '{col}' (Total: {total_count}, Unique: {unique_count})")
        else:
            # Debug print if duplicates found
            # print(f"\nColumn '{col}' not picked as primary key (Total: {total_count}, Unique: {unique_count}).")
            # print(f"Finding duplicates for '{col}'...\n")
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
    
    if not matches: # Added from notebook
        print(f"No match found for seed column '{seed_col_name}'")
    return matches

# === Main Execution Logic ===

def infer_keys(transactions_path, master_paths):
    # Load input file (transaction file)
    full_input_df = pd.read_excel(transactions_path, skiprows=2, dtype=str)
    input_sample = full_input_df.head(10)
    print(f"Loaded input file '{transactions_path}' with {len(full_input_df)} rows. Using top 10 rows for mapping.")

    # Load seed files (master files)
    seeds = {}
    for path in master_paths:
        df = pd.read_csv(path, skipinitialspace=True, dtype=str)
        df.columns = df.columns.str.strip()
        seed_name = os.path.splitext(os.path.basename(path))[0]
        seeds[seed_name] = df
        print(f"Loaded seed file: '{seed_name}' with {len(df)} rows.")

    all_mappings = {}
    final_verified_mappings = {}

    for seed_name, seed_df_loop_var in seeds.items(): # Renamed seed_df to avoid conflict with outer scope if any
        print(f"\n=== Processing seed file '{seed_name}' ===")
        primary_keys = find_primary_keys(seed_df_loop_var)

        mappings_list_inner = [] # Renamed to avoid conflict
        for pk_col in primary_keys:
            if pk_col not in seed_df_loop_var.columns:
                continue

            seed_col_values = seed_df_loop_var[pk_col]
            seed_props = analyze_column_properties(seed_col_values)

            print(f"\nProperties for seed column '{pk_col}': {seed_props}")

            matched_cols = map_columns(input_sample, pk_col, seed_col_values, seed_props)

            if matched_cols:
                mappings_list_inner.append({
                    'seed_column': pk_col,
                    'matched_input_columns': matched_cols,
                    'seed_properties': seed_props
                })
        all_mappings[seed_name] = mappings_list_inner

    print("\n=== Final Mapping Results ===") # From notebook
    for seed_name_map_results, mappings_list_results in all_mappings.items():
        for mapping_item_results in mappings_list_results:
            input_cols_results = ", ".join(mapping_item_results['matched_input_columns'])
            print(f"Seed File '{seed_name_map_results}': '{mapping_item_results['seed_column']}' -> {input_cols_results}")

    # Verification loop from notebook
    for seed_name_verify, mappings_list_verify in all_mappings.items():
        final_verified_mappings[seed_name_verify] = []
        for mapping_item_verify in mappings_list_verify:
            seed_column_name_verify = mapping_item_verify['seed_column']
            
            current_seed_df = seeds[seed_name_verify] # Get the correct seed_df for this iteration
            full_seed_col_values = current_seed_df[seed_column_name_verify].dropna().astype(str).apply(lambda x: x.strip())

            for input_col_name_verify in mapping_item_verify['matched_input_columns']:
                if input_col_name_verify not in full_input_df.columns:
                    continue

                full_input_col_values = full_input_df[input_col_name_verify].dropna().astype(str).apply(lambda x: normalize_text(x.strip()))
                input_unique = set(full_input_col_values)
                seed_unique = set(full_seed_col_values)

                if not input_unique: # if input_unique is empty
                    continue

                common_elements = input_unique & seed_unique
                # Handle division by zero if input_unique is empty (though checked above)
                ratio = len(common_elements) / len(input_unique) if len(input_unique) > 0 else 0.0


                print(f"\nMapping Attempt: '{seed_name_verify}:{seed_column_name_verify}' -> '{input_col_name_verify}'")
                # Optional detailed prints from notebook (can be very verbose)
                # print(f"Input Unique Values ({input_col_name_verify}): {sorted(list(input_unique))}")
                # print(f"Seed Unique Values ({seed_column_name_verify}): {sorted(list(seed_unique))}")
                # print(f"Common Values: {sorted(list(common_elements))}")
                print(f"Input Unique Count: {len(input_unique)}, Common Count: {len(common_elements)}, Ratio: {ratio:.2f}")

                if ratio >= 0.8:
                    print(" Mapping Accepted based on Common Elements Ratio.\n")
                    final_verified_mappings[seed_name_verify].append({
                        'seed_column': seed_column_name_verify,
                        'input_column': input_col_name_verify,
                        'ratio': ratio
                    })
                else:
                    print("Mapping Rejected based on Common Elements Ratio.\n")

    print("\n=== Final Verified Mappings ===")
    for seed_name_final, verified_list_final in final_verified_mappings.items():
        for verified_item_final in verified_list_final:
            print(f"Seed File '{seed_name_final}': Seed Column '{verified_item_final['seed_column']}' -> Input Column '{verified_item_final['input_column']}' (Ratio: {verified_item_final['ratio']:.2f})")

# === CLI Parser ===

def parse_args():
    parser = argparse.ArgumentParser(
        description='Infer foreign keys in transactions file and identify associated master file columns.'
    )
    parser.add_argument('--transactions', type=str, required=True, help='Path to the transactions XLSX file.')
    parser.add_argument('--masters', type=str, nargs='+', required=True, help='Paths to one or more master CSV files.')
    return parser.parse_args()

# === Entry Point ===

if __name__ == '__main__':
    args = parse_args()
    infer_keys(args.transactions, args.masters)
