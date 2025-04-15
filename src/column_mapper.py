#!/usr/bin/env python3
"""
Column Mapper module for GL Mapping project.
This module provides functionality to map columns between different data sources
based on seed files that define the mapping relationships.
"""

import pandas as pd
import numpy as np
import re
import itertools


class ColumnMapper:
    """
    A class to handle column mapping between different data sources.
    Identifies primary-foreign key relationships between datasets.
    """
    
    def __init__(self):
        """
        Initialize the ColumnMapper.
        """
        self.mapping_cache = {}
        
        # Common variations of business identifier column names
        self.gstin_column_patterns = [
            r'gstin',
            r'gst.*(id|number|no)',
            r'tax.*(id|number|no)'
        ]
        
        # Common business key patterns for mapping
        self.business_key_patterns = {
            'company_id': [r'company.*(id|code|no)', r'org.*(id|code|no)', r'entity.*(id|code|no)'],
            'party_id': [r'(party|customer|vendor|supplier).*(id|code|no)'],
            'invoice_id': [r'invoice.*(id|no|number|code)', r'bill.*(id|no|number)'],
            'location_id': [r'location.*(id|code)', r'branch.*(id|code)', r'place.*(id|code)', r'site.*(id|code)']
        }
        
        # Common primary key column names
        self.primary_key_patterns = [
            r'id$', r'^code$', r'key$', r'number$',
            r'^pk_', r'primary.*key'
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
    
    def analyze_column_semantic_similarity(self, input_df, seed_df, seed_name):
        """
        Analyze semantic similarity between columns in input_df and seed_df.
        
        Args:
            input_df (pd.DataFrame): The input dataframe
            seed_df (pd.DataFrame): The seed dataframe
            seed_name (str): Name of the seed file
            
        Returns:
            list: List of dictionaries containing column mappings with semantic similarity
        """
        # Preprocess dataframes
        input_df = self._preprocess_dataframe(input_df)
        seed_df = self._preprocess_dataframe(seed_df)
        
        semantic_mappings = []
        
        # Get common values between columns for better similarity detection
        for input_col in input_df.columns:
            input_col_lower = input_col.lower()
            
            for seed_col in seed_df.columns:
                seed_col_lower = seed_col.lower()
                match_reason = None
                confidence = 0
                
                # Case 1: Exact name match
                if input_col_lower == seed_col_lower:
                    match_reason = "exact_name_match"
                    confidence = 1.0
                
                # Case 2: One is contained in the other
                elif input_col_lower in seed_col_lower or seed_col_lower in input_col_lower:
                    match_reason = "substring_match"
                    # Calculate similarity based on the length of shared characters
                    common_length = min(len(input_col_lower), len(seed_col_lower))
                    total_length = max(len(input_col_lower), len(seed_col_lower))
                    confidence = 0.6 + (0.3 * (common_length / total_length))
                
                # Only check content if both are string columns and no high confidence match yet
                elif input_df[input_col].dtype == 'object' and seed_df[seed_col].dtype == 'object':
                    # Get sample values
                    try:
                        input_values = set(input_df[input_col].dropna().astype(str).sample(min(10, len(input_df))).values)
                        seed_values = set(seed_df[seed_col].dropna().astype(str).sample(min(10, len(seed_df))).values)
                        
                        if input_values and seed_values:
                            # Check for value overlap
                            common_values = input_values.intersection(seed_values)
                            if common_values:
                                match_reason = "content_overlap"
                                confidence = 0.5 + (0.4 * (len(common_values) / min(len(input_values), len(seed_values))))
                    except:
                        # Skip if there's an error during sampling
                        pass
                
                # Add to mappings if confidence is high enough
                if match_reason and confidence > 0.3:
                    semantic_mappings.append({
                        'input_col': input_col,
                        'seed_col': seed_col,
                        'seed_name': seed_name,
                        'match_reason': match_reason,
                        'confidence': confidence
                    })
        
        # Sort by confidence
        semantic_mappings.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates (keep highest confidence for each input column)
        unique_mappings = {}
        for mapping in semantic_mappings:
            input_col = mapping['input_col']
            if input_col not in unique_mappings or mapping['confidence'] > unique_mappings[input_col]['confidence']:
                unique_mappings[input_col] = mapping
        
        return list(unique_mappings.values())
    
    def analyze_seed_files_together(self, input_df, seed_dfs_dict):
        """
        Analyze multiple seed files together to find the best mapping strategy.
        
        Args:
            input_df (pd.DataFrame): The input dataframe
            seed_dfs_dict (dict): Dictionary of {seed_name: seed_df} pairs
            
        Returns:
            dict: Information about cross-seed relationships and best mapping strategy
        """
        # Preprocess all dataframes
        input_df = self._preprocess_dataframe(input_df)
        preprocessed_seeds = {name: self._preprocess_dataframe(df) for name, df in seed_dfs_dict.items()}
        
        # Find primary/foreign key candidates in each seed file
        pk_candidates_by_seed = {}
        for seed_name, seed_df in preprocessed_seeds.items():
            pk_candidates_by_seed[seed_name] = self._find_primary_key_candidates(seed_df)
        
        # Find common columns between seed files
        common_columns = {}
        for seed1, seed2 in itertools.combinations(preprocessed_seeds.keys(), 2):
            seed1_df = preprocessed_seeds[seed1]
            seed2_df = preprocessed_seeds[seed2]
            
            seed1_cols = set(seed1_df.columns)
            seed2_cols = set(seed2_df.columns)
            
            # Check for exact name matches
            exact_matches = seed1_cols.intersection(seed2_cols)
            
            # Check for content overlaps in columns
            content_matches = []
            for col1 in seed1_df.columns:
                for col2 in seed2_df.columns:
                    # Skip if already an exact match
                    if col1 == col2:
                        continue
                    
                    # Check for content overlap
                    vals1 = set(seed1_df[col1].dropna().astype(str))
                    vals2 = set(seed2_df[col2].dropna().astype(str))
                    
                    if vals1 and vals2:
                        overlap = len(vals1.intersection(vals2))
                        if overlap > 0:
                            overlap_ratio = overlap / min(len(vals1), len(vals2))
                            if overlap_ratio > 0.5:  # More than 50% overlap
                                content_matches.append({
                                    'seed1_col': col1,
                                    'seed2_col': col2,
                                    'overlap_ratio': overlap_ratio
                                })
            
            common_columns[f"{seed1}_{seed2}"] = {
                'exact_matches': list(exact_matches),
                'content_matches': content_matches
            }
        
        # Find the best mapping strategy between input and all seed files
        input_to_seed_mappings = {}
        for seed_name, seed_df in preprocessed_seeds.items():
            # 1. Check for direct primary/foreign key relationships
            pk_fk_relationships = self.identify_primary_foreign_keys(input_df, seed_df, seed_name)
            
            # 2. Get column semantic mappings
            semantic_mappings = self.analyze_column_semantic_similarity(input_df, seed_df, seed_name)
            
            input_to_seed_mappings[seed_name] = {
                'pk_fk_relationships': pk_fk_relationships,
                'semantic_mappings': semantic_mappings
            }
        
        # Identify which seed file provides the best mapping to the input file
        best_seed = None
        best_score = -1
        best_key_pair = None
        
        for seed_name, mappings in input_to_seed_mappings.items():
            if mappings['pk_fk_relationships']:
                best_rel = mappings['pk_fk_relationships'][0]  # Best relationship for this seed
                score = best_rel['confidence'] * best_rel['overlap_ratio']
                
                if score > best_score:
                    best_score = score
                    best_seed = seed_name
                    best_key_pair = {
                        'primary_key': f"{seed_name}.{best_rel['primary_key']}",
                        'foreign_key': best_rel['foreign_key'],
                        'confidence': best_rel['confidence']
                    }
        
        # Generate recommendations
        recommendations = {
            'best_seed_file': best_seed,
            'best_key_pair': best_key_pair,
            'common_columns_between_seeds': common_columns,
            'all_mappings': input_to_seed_mappings
        }
        
        return recommendations
        
    def identify_primary_foreign_keys(self, input_df, seed_df, seed_name):
        """
        Identify potential primary key in seed file and corresponding foreign key in input file.
        
        Args:
            input_df (pd.DataFrame): The input dataframe
            seed_df (pd.DataFrame): The seed dataframe (master file)
            seed_name (str): Name of the seed file
            
        Returns:
            dict: Dictionary containing primary-foreign key relationship info
        """
        # Preprocess dataframes
        input_df = self._preprocess_dataframe(input_df)
        seed_df = self._preprocess_dataframe(seed_df)
        
        # Step 1: Identify primary key candidates in seed file
        pk_candidates = self._find_primary_key_candidates(seed_df)
        
        # Step 2: For each primary key candidate, find potential foreign key matches in input file
        relationships = []
        
        for pk_info in pk_candidates:
            pk_col = pk_info['column']
            fk_matches = self._find_foreign_key_matches(pk_col, seed_df, input_df)
            
            for fk_match in fk_matches:
                relationships.append({
                    'seed_name': seed_name,
                    'primary_key': pk_col,
                    'foreign_key': fk_match['column'],
                    'match_type': fk_match['match_type'],
                    'overlap_ratio': fk_match['overlap_ratio'],
                    'confidence': fk_match['confidence']
                })
        
        # Step 3: Sort relationships by confidence
        relationships.sort(key=lambda x: x['confidence'], reverse=True)
        
        return relationships
    
    def _find_primary_key_candidates(self, df):
        """
        Find columns that could serve as primary keys in the dataframe.
        
        Args:
            df (pd.DataFrame): The dataframe to analyze
            
        Returns:
            list: List of dictionaries containing primary key candidate info
        """
        pk_candidates = []
        
        # Check each column for primary key properties
        for col in df.columns:
            # Skip columns with too many missing values
            missing_ratio = df[col].isna().mean()
            if missing_ratio > 0.1:  # More than 10% missing values
                continue
            
            # Calculate uniqueness ratio
            uniqueness_ratio = df[col].nunique() / len(df)
            
            # Check if column name matches primary key patterns
            name_match = False
            col_lower = str(col).lower()
            for pattern in self.primary_key_patterns:
                if re.search(pattern, col_lower):
                    name_match = True
                    break
            
            # Calculate confidence based on uniqueness and name
            base_confidence = uniqueness_ratio
            if name_match:
                base_confidence += 0.2  # Boost confidence if name suggests key
            
            # Only consider as PK if reasonably unique
            if uniqueness_ratio > 0.8:  # At least 80% unique values
                pk_candidates.append({
                    'column': col,
                    'uniqueness_ratio': uniqueness_ratio,
                    'name_match': name_match,
                    'confidence': min(base_confidence, 1.0)  # Cap at 1.0
                })
            
        # Sort by confidence
        pk_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return pk_candidates
    
    def _find_foreign_key_matches(self, pk_col, seed_df, input_df):
        """
        Find columns in input_df that could be foreign keys to seed_df[pk_col].
        
        Args:
            pk_col (str): Primary key column in seed_df
            seed_df (pd.DataFrame): The seed dataframe
            input_df (pd.DataFrame): The input dataframe
            
        Returns:
            list: List of dictionaries containing foreign key match info
        """
        fk_matches = []
        
        # Get primary key values
        pk_values = set(seed_df[pk_col].dropna().astype(str))
        
        # Check each column in input_df for overlap with pk_values
        for col in input_df.columns:
            # Get column values
            col_values = set(input_df[col].dropna().astype(str))
            
            if not col_values:  # Skip empty columns
                continue
            
            # Calculate overlap
            overlap = len(pk_values.intersection(col_values))
            
            if overlap > 0:
                # Calculate ratio of input values that exist in the primary key
                overlap_ratio = overlap / len(col_values)
                
                # Check for name similarity
                name_similarity = self._calculate_name_similarity(pk_col, col)
                
                # Calculate confidence based on overlap and name similarity
                confidence = (0.7 * overlap_ratio) + (0.3 * name_similarity)
                
                # Determine match type
                match_type = "content_match"
                if name_similarity > 0.8:
                    match_type = "name_and_content_match"
                
                # Only include if reasonable overlap
                if overlap_ratio > 0.1:  # At least 10% overlap
                    fk_matches.append({
                        'column': col,
                        'overlap': overlap,
                        'overlap_ratio': overlap_ratio,
                        'name_similarity': name_similarity,
                        'match_type': match_type,
                        'confidence': confidence
                    })
        
        # Sort by confidence
        fk_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return fk_matches
    
    def _calculate_name_similarity(self, col1, col2):
        """
        Calculate similarity between column names.
        
        Args:
            col1 (str): First column name
            col2 (str): Second column name
            
        Returns:
            float: Similarity score between 0 and 1
        """
        col1_lower = str(col1).lower()
        col2_lower = str(col2).lower()
        
        # Exact match
        if col1_lower == col2_lower:
            return 1.0
        
        # One is substring of the other
        if col1_lower in col2_lower or col2_lower in col1_lower:
            # Calculate ratio based on length
            longer = max(len(col1_lower), len(col2_lower))
            shorter = min(len(col1_lower), len(col2_lower))
            return 0.5 + (0.4 * (shorter / longer))
        
        # Check for common business key patterns
        for key_type, patterns in self.business_key_patterns.items():
            col1_match = any(re.search(pattern, col1_lower) for pattern in patterns)
            col2_match = any(re.search(pattern, col2_lower) for pattern in patterns)
            if col1_match and col2_match:
                return 0.8  # Both match the same business key pattern
        
        # Default similarity for different names
        return 0.1
    
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
        
        # Handle unnamed columns
        new_column_names = []
        for i, col_name in enumerate(df.columns):
            col_str = str(col_name)
            
            # Check if this is an unnamed column
            if 'Unnamed:' in col_str:
                # Try to infer a better name from the first few non-empty values
                non_empty_values = df[col_name].dropna().head(5)
                
                if len(non_empty_values) > 0:
                    # Try to get a good column name from the data
                    potential_headers = [str(v) for v in non_empty_values
                                      if not re.match(r'^\d+(\.\d+)?$', str(v))  # Not just a number
                                      and len(str(v)) < 30                      # Not too long
                                      and len(str(v)) > 2]                      # Not too short
                    
                    if potential_headers:
                        # Use the first good potential header
                        new_name = potential_headers[0]
                    else:
                        # If no good header found, use a descriptive name with column position
                        col_idx = int(re.search(r'\d+', col_str).group()) if re.search(r'\d+', col_str) else i
                        new_name = f"Data_Column_{col_idx}"
                else:
                    # If column is empty, name it accordingly
                    col_idx = int(re.search(r'\d+', col_str).group()) if re.search(r'\d+', col_str) else i
                    new_name = f"Empty_Column_{col_idx}"
            else:
                # Not an unnamed column, keep the original name
                new_name = col_str
            
            # Ensure unique column names
            if new_name in new_column_names:
                suffix = 1
                while f"{new_name}_{suffix}" in new_column_names:
                    suffix += 1
                new_name = f"{new_name}_{suffix}"
                
            new_column_names.append(new_name)
        
        # Set the new column names
        df.columns = new_column_names
        
        # Strip whitespace from column names
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        
        # Convert all object dtype columns to strings and handle mixed types
        # This ensures proper Arrow serialization
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Convert to string and handle special cases
                df[col] = df[col].astype(str)
                # Replace 'nan' strings and empty strings with NaN
                df[col] = df[col].replace(['nan', 'None', ''], np.nan)
                # Strip whitespace
                df[col] = df[col].str.strip()
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {str(e)}")
                # Fallback: force convert problematic values
                df[col] = df[col].apply(lambda x: str(x) if x is not None else np.nan)
        
        return df