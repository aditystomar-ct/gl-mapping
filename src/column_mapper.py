#!/usr/bin/env python3
"""
Column Mapper module for GL Mapping project.
This module provides functionality to map columns between different data sources
based on seed files that define the mapping relationships.
"""

import pandas as pd
import numpy as np
import re


class ColumnMapper:
    """
    A class to handle column mapping between different data sources.
    """
    
    def __init__(self):
        """
        Initialize the ColumnMapper.
        """
        self.mapping_cache = {}
        # Common variations of GSTIN column names
        self.gstin_column_patterns = [
            r'gstin',
            r'gst.*(id|number|no)',
            r'tax.*(id|number|no)'
        ]
    
    def map_columns(self, input_df, seed_df):
        """
        Map columns from input dataframe based on the seed dataframe.
        
        Args:
            input_df (pd.DataFrame): The input dataframe to map columns from
            seed_df (pd.DataFrame): The seed dataframe containing mapping information
            
        Returns:
            tuple: (pd.DataFrame, dict) A new dataframe with mapped columns and a dictionary of mapping info
        """
        # Preprocess dataframes to standardize column names and data formats
        input_df = self._preprocess_dataframe(input_df)
        seed_df = self._preprocess_dataframe(seed_df)
        
        # Identify potential key columns for joining
        potential_keys = self._identify_potential_keys(input_df, seed_df)
        
        # Create a dictionary to store mapping information
        mapping_info = {
            'join_key': None,
            'input_columns': list(input_df.columns),
            'seed_columns': list(seed_df.columns),
            'matched_columns': []
        }
        
        if not potential_keys:
            print("Warning: No potential key columns identified for mapping")
            return input_df.copy(), mapping_info
        
        # Use the first identified key for joining
        join_key = potential_keys[0]
        print(f"Using '{join_key['input_col']}' from input and '{join_key['seed_col']}' from seed as the join key")
        
        # Store the join key in mapping info
        mapping_info['join_key'] = {
            'input': join_key['input_col'],
            'seed': join_key['seed_col'],
            'match_type': join_key['match_type'],
            'confidence': join_key['confidence']
        }
        
        # Find column matches based on names
        for input_col in input_df.columns:
            for seed_col in seed_df.columns:
                # Skip the join keys, we already know about those
                if input_col == join_key['input_col'] and seed_col == join_key['seed_col']:
                    continue
                    
                # Check for exact or similar column names
                if input_col == seed_col or (input_col in seed_col or seed_col in input_col):
                    mapping_info['matched_columns'].append({
                        'input': input_col,
                        'seed': seed_col,
                        'match_type': 'exact' if input_col == seed_col else 'similar',
                        'confidence': 1.0 if input_col == seed_col else 0.7
                    })
        
        # Perform the join
        result_df = pd.merge(
            input_df,
            seed_df,
            left_on=join_key['input_col'],
            right_on=join_key['seed_col'],
            how='left'
        )
        
        return result_df, mapping_info
    
    def _identify_potential_keys(self, input_df, seed_df):
        """
        Identify potential key columns that can be used for joining.
        
        Args:
            input_df (pd.DataFrame): The input dataframe
            seed_df (pd.DataFrame): The seed dataframe
            
        Returns:
            list: List of dictionaries containing potential key mappings
        """
        potential_keys = []
        
        # First, prioritize GSTIN-related columns
        input_gstin_cols = self._find_gstin_columns(input_df)
        seed_gstin_cols = self._find_gstin_columns(seed_df)
        
        if input_gstin_cols and seed_gstin_cols:
            # If GSTIN columns are found in both dataframes, prioritize them
            for input_col in input_gstin_cols:
                for seed_col in seed_gstin_cols:
                    potential_keys.append({
                        'input_col': input_col,
                        'seed_col': seed_col,
                        'match_type': 'gstin',
                        'confidence': 0.95
                    })
        
        # Check for exact column name matches
        common_cols = set(input_df.columns).intersection(set(seed_df.columns))
        for col in common_cols:
            # Check if the column has unique values in the seed file (potential primary key)
            if seed_df[col].nunique() == len(seed_df):
                potential_keys.append({
                    'input_col': col,
                    'seed_col': col,
                    'match_type': 'exact_name',
                    'confidence': 0.9
                })
        
        # Check for similar column names
        if not potential_keys:
            for input_col in input_df.columns:
                for seed_col in seed_df.columns:
                    # Simple similarity check based on substring
                    if (input_col in seed_col or seed_col in input_col) and input_col != seed_col:
                        # Check data type compatibility
                        if input_df[input_col].dtype == seed_df[seed_col].dtype:
                            potential_keys.append({
                                'input_col': input_col,
                                'seed_col': seed_col,
                                'match_type': 'similar_name',
                                'confidence': 0.7
                            })
        
        # Check for value overlaps if still no keys found
        if not potential_keys:
            for input_col in input_df.columns:
                for seed_col in seed_df.columns:
                    # Skip if data types are incompatible
                    if input_df[input_col].dtype != seed_df[seed_col].dtype:
                        continue
                    
                    # Check for value overlap
                    input_values = set(input_df[input_col].dropna().unique())
                    seed_values = set(seed_df[seed_col].dropna().unique())
                    
                    if input_values and seed_values:  # Ensure non-empty sets
                        overlap = len(input_values.intersection(seed_values))
                        overlap_ratio = overlap / len(seed_values)
                        
                        if overlap_ratio > 0.5:  # More than 50% overlap
                            potential_keys.append({
                                'input_col': input_col,
                                'seed_col': seed_col,
                                'match_type': 'value_overlap',
                                'confidence': overlap_ratio
                            })
        
        # Sort by confidence
        potential_keys.sort(key=lambda x: x['confidence'], reverse=True)
        
        return potential_keys
    
    def _find_gstin_columns(self, df):
        """
        Find columns that likely contain GSTIN numbers.
        
        Args:
            df (pd.DataFrame): The dataframe to search
            
        Returns:
            list: List of column names that likely contain GSTIN numbers
        """
        gstin_columns = []
        
        # Check column names for GSTIN patterns
        for col in df.columns:
            col_lower = col.lower()
            for pattern in self.gstin_column_patterns:
                if re.search(pattern, col_lower):
                    gstin_columns.append(col)
                    break
        
        # If no columns found by name, try to identify by content pattern
        if not gstin_columns:
            for col in df.columns:
                # Skip non-string columns
                if df[col].dtype != 'object':
                    continue
                
                # Check if column contains values that look like GSTIN
                # GSTIN format: 2 digits, 10 chars, 1 digit, 1 char, 1 digit, 1 char
                sample = df[col].dropna().astype(str).head(10)
                gstin_pattern = r'\d{2}[A-Z0-9]{10}\d[A-Z]\d[A-Z]'
                
                matches = sample.str.contains(gstin_pattern, regex=True).sum()
                if matches > 0:
                    gstin_columns.append(col)
        
        return gstin_columns
    
    def _preprocess_dataframe(self, df):
        """
        Preprocess dataframe to standardize column names and data formats.
        
        Args:
            df (pd.DataFrame): The dataframe to preprocess
            
        Returns:
            pd.DataFrame: The preprocessed dataframe
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Strip whitespace from column names
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        
        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        
        return df