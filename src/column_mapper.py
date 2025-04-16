import pandas as pd
import re
class ColumnMapper:
    def __init__(self):
        self.mapping_cache = {}

    def _get_alphanumeric_length(self, value):
        """
        Calculate the length of a string considering only alphanumeric characters.
        
        Args:
            value (str): The string to process.
        
        Returns:
            int: The length of the string with only alphanumeric characters.
        """
        return len(re.sub(r'[^a-zA-Z0-9]', '', value))
    
    def find_fixed_length_column(self, df, check_unique=False, sample_size=None):
        """
        Find a column in the dataframe where all non-null values are of fixed length.
        Counts all characters (letters, digits, spaces, etc.), not just digits.
        Optionally check if the column has unique values (potential primary key).
        
        Args:
            df (pd.DataFrame): The dataframe to scan.
            check_unique (bool): If True, check if the column has unique values.
            sample_size (int): Optional, number of rows to sample. If None, check all rows.
            
        Returns:
            tuple: (column_name, length) if found, otherwise (None, None).
        """
        if sample_size is not None and len(df) > sample_size:
            df_sample = df.head(sample_size)
        else:
            df_sample = df
            
        for col in df.columns:
            s = df_sample[col].dropna().astype(str)
            
            if s.empty:
                continue
            
            lengths = s.apply(self._get_alphanumeric_length).unique()
            if len(lengths) == 1:
                if check_unique:
                    if len(s) == len(s.unique()):
                        return col, lengths[0]
                else:
                    return col, lengths[0]
            
        return None, None

    def find_all_fixed_length_columns(self, df, check_unique=False, sample_size=None):
        """
        Find all columns in the dataframe where all non-null values have the same length.
        
        Args:
            df (pd.DataFrame): The dataframe to scan.
            check_unique (bool): If True, only return columns with unique values.
            sample_size (int): Optional, number of rows to sample. If None, check all rows.
            
        Returns:
            list: List of tuples (column_name, length) for all fixed-length columns.
        """
        if sample_size is not None and len(df) > sample_size:
            df_sample = df.head(sample_size)
        else:
            df_sample = df
            
        fixed_length_columns = []
        
        for col in df.columns:
            s = df_sample[col].dropna().astype(str)
            if s.empty:
                continue
                
            lengths = s.apply(self._get_alphanumeric_length).unique()
            if len(lengths) == 1:
                fixed_length = lengths[0]
                if check_unique:
                    if len(s) == len(s.unique()):
                        fixed_length_columns.append((col, fixed_length))
                else:
                    fixed_length_columns.append((col, fixed_length))
                    
        return fixed_length_columns

    def map_columns(self, input_df, seed_df):
        """
        Map columns from input dataframe to seed dataframe based on fixed length identification.
        
        Args:
            input_df (pd.DataFrame): The input dataframe.
            seed_df (pd.DataFrame): The seed dataframe.
        
        Returns:
            tuple: (result_df, mapping_info)
        """
        input_df = self._preprocess_dataframe(input_df)
        seed_df = self._preprocess_dataframe(seed_df)
        
        mapping_info = {'matched_columns': []}

        primary_keys_with_fixed_length = []
        
        for col in seed_df.columns:
            col_values = seed_df[col].dropna()
            is_unique = len(col_values.unique()) == len(col_values)
            
            if is_unique:
                s = col_values.astype(str)
                lengths = s.apply(self._get_alphanumeric_length).unique()
                
                if len(lengths) == 1:
                    fixed_length = lengths[0]
                    primary_keys_with_fixed_length.append((col, fixed_length))
                    print(f"Primary Key Found: Column '{col}' | Fixed Length: {fixed_length} | Unique: Yes")
        
        for seed_col, seed_length in primary_keys_with_fixed_length:
            for input_col in input_df.columns:
                fk_col, fk_length = self.find_fixed_length_column(input_df[[input_col]], sample_size=10)
                if fk_col and fk_length == seed_length:
                    is_pk = len(seed_df[seed_col].dropna().unique()) == len(seed_df[seed_col].dropna())
                    
                    mapping_info['matched_columns'].append({
                        'seed': seed_col,
                        'input': fk_col,
                        'length': seed_length,
                        'is_primary_key': is_pk
                    })
        print("\nMapping Information:")
        for match in mapping_info['matched_columns']:
            print(f"Seed Column: {match['seed']} | Input Column: {match['input']} | Length: {match['length']} | Is Primary Key: {'Yes' if match['is_primary_key'] else 'No'}")    
        return input_df, mapping_info

    def _preprocess_dataframe(self, df):
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        return df
