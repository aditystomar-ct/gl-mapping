#!/usr/bin/env python3
"""
Main application file for GL Mapping project.
This script handles the main workflow for mapping columns from input files
using seed files as reference.

The application provides both a command-line interface and a Streamlit web interface.
"""

import os
import pandas as pd
import glob
import sys
import tempfile
from datetime import datetime
from column_mapper import ColumnMapper

# Import streamlit for the web interface
import streamlit as st

def process_mapping(input_df, seed_df, seed_name, output_dir='output'):
    """
    Process the mapping between input dataframe and seed dataframe.
    
    Args:
        input_df (pd.DataFrame): The input dataframe
        seed_df (pd.DataFrame): The seed dataframe
        seed_name (str): Name of the seed file (for output filename)
        output_dir (str): Directory to save output files
        
    Returns:
        tuple: (result_dataframe, output_file_path, mapping_info, semantic_mappings)
    """
    # Initialize the column mapper
    mapper = ColumnMapper()
    
    # Map columns using the seed file
    result_df, mapping_info = mapper.map_columns(input_df, seed_df)
    
    # Analyze semantic similarity between columns
    semantic_mappings = mapper.analyze_column_semantic_similarity(input_df, seed_df, seed_name)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"mapped_{seed_name}_{timestamp}.csv")
    
    # Save the result
    result_df.to_csv(output_file, index=False)
    
    # Print the join key information
    print("\n" + "="*60)
    print(f"COLUMN MAPPING FOR {seed_name.upper()}")
    print("="*60)
    
    if mapping_info['join_key']:
        print(f"Join Key: {mapping_info['join_key']['input']} -> {seed_name}.{mapping_info['join_key']['seed']}")
    
    # Print the semantic column mappings
    print("\nColumn Mappings:")
    if semantic_mappings:
        # Sort by confidence
        for mapping in sorted(semantic_mappings, key=lambda x: x['confidence'], reverse=True):
            confidence_percent = int(mapping['confidence'] * 100)
            if confidence_percent >= 60:  # Only show reasonably confident mappings
                print(f"{mapping['input_col']} -> {seed_name}.{mapping['seed_col']} ({confidence_percent}% confidence)")
    else:
        print("No column mappings found")
    
    return result_df, output_file, mapping_info, semantic_mappings


def run_cli(output_dir='output'):
    """
    Run the GL mapping process from command line.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Check if input directory exists
    input_dir = 'input'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created input directory: {input_dir}")
        print("Please place Excel files in the input directory and run again")
        return
    
    # Load the main input file (Excel file)
    input_files = glob.glob(os.path.join(input_dir, '*.xlsx'))
    if not input_files:
        print("Error: No Excel input files found in the input directory")
        return
    
    input_file_path = input_files[0]  # Use the first Excel file found
    print(f"Using input file: {input_file_path}")
    
    input_df = pd.read_excel(input_file_path)
    print(f"Loaded input file with {len(input_df)} rows and {len(input_df.columns)} columns")
    
    # Load seed files
    seed_dir = 'seeds'
    
    # Check if seed directory exists
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)
        print(f"Created seed directory: {seed_dir}")
        print("Please place CSV seed files in the seeds directory and run again")
        return
    
    seed_files = [f for f in os.listdir(seed_dir) if f.endswith('.csv')]
    
    if not seed_files:
        print("Error: No seed files found in the seeds directory")
        return
    
    print(f"Found {len(seed_files)} seed files")
    
    # Ask user which seed files to use if there are multiple
    if len(seed_files) > 1:
        print("\nAvailable seed files:")
        for i, file in enumerate(seed_files):
            print(f"{i+1}. {file}")
        
        try:
            selection = input("\nEnter seed file numbers to use (comma separated, or 'all'): ")
            if selection.lower().strip() == 'all':
                selected_seed_files = seed_files
            else:
                indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                selected_seed_files = [seed_files[i] for i in indices if 0 <= i < len(seed_files)]
        except (ValueError, IndexError):
            print("Invalid selection, using all seed files")
            selected_seed_files = seed_files
    else:
        selected_seed_files = seed_files
    
    # Process each selected seed file
    for seed_file in selected_seed_files:
        print("\n" + "="*60)
        print(f"PROCESSING SEED FILE: {seed_file}")
        print("="*60)
        
        seed_path = os.path.join(seed_dir, seed_file)
        # Handle potential issues with CSV files (like commas in headers)
        seed_df = pd.read_csv(seed_path, skipinitialspace=True)
        print(f"Processing seed file: {seed_file} with {len(seed_df)} rows")
        
        # Process the mapping
        seed_name = os.path.splitext(os.path.basename(seed_file))[0]
        _, output_file, mapping_info, semantic_mappings = process_mapping(input_df, seed_df, seed_name, output_dir)
        print(f"Saved mapped data to {output_file}")
    
    print("GL mapping completed successfully")


def run_streamlit():
    """
    Run the GL mapping process with Streamlit interface.
    """
    # Set page configuration
    st.set_page_config(
        page_title="GL Mapping Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("GL Mapping Tool")
    st.markdown("""
    This tool helps you map columns between a main input file (such as a GSTR-1 report)
    and seed files (reference data) based on common identifiers like GSTIN numbers.
    """)
    
    # Create sidebar for file uploads and options
    with st.sidebar:
        st.header("Upload Files")
        
        # Input file upload
        st.subheader("Input File (Excel)")
        input_file = st.file_uploader("Upload your main input file", type=["xlsx", "xls"])
        
        # Seed files upload
        st.subheader("Seed Files (CSV)")
        seed_files = st.file_uploader("Upload your seed/reference files", type=["csv"], accept_multiple_files=True)
        
        # Option to use existing seed files
        st.subheader("Or Use Existing Seed Files")
        use_existing_seeds = st.checkbox("Use existing seed files in seeds/ directory")
        
        # Output directory
        st.subheader("Output Settings")
        output_dir = st.text_input("Output Directory", value="output")
        
        # Run button
        run_mapping = st.button("Run Mapping", type="primary")
    
    # Main content area
    if not input_file and not run_mapping:
        st.info("Please upload an input file and seed files, then click 'Run Mapping'.")
        
        # Show example data
        with st.expander("View Example Data Structure"):
            st.markdown("""
            ### Expected Input File Structure
            The input file should be an Excel file (.xlsx or .xls) with columns that include identifiers
            like GSTIN numbers that can be used to join with seed files.
            
            ### Expected Seed File Structure
            Seed files should be CSV files with columns that can be mapped to the input file.
            At least one column should contain values that match with a column in the input file.
            
            ### Example Mapping
            If your input file has a column named "GSTIN" and your seed file has a column
            named "GSTIN Number", the tool will identify these as potential join keys.
            """)
        return
    
    # Process files when the Run button is clicked
    if run_mapping:
        if not input_file:
            st.error("Please upload an input file.")
            return
        
        if not seed_files and not use_existing_seeds:
            st.error("Please upload seed files or select 'Use existing seed files'.")
            return
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            st.info(f"Created output directory: {output_dir}")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load the input file
            status_text.text("Loading input file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                tmp.write(input_file.getvalue())
                tmp_path = tmp.name
            
            input_df = pd.read_excel(tmp_path)
            os.unlink(tmp_path)  # Delete the temporary file
            
            st.write(f"Loaded input file with {len(input_df)} rows and {len(input_df.columns)} columns")
            progress_bar.progress(20)
            
            # Display input file preview
            with st.expander("Input File Preview"):
                st.dataframe(input_df.head(5))
            
            # Process seed files
            all_seed_files = []
            
            # Process uploaded seed files
            if seed_files:
                status_text.text("Processing uploaded seed files...")
                for i, seed_file in enumerate(seed_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                        tmp.write(seed_file.getvalue())
                        tmp_path = tmp.name
                    
                    seed_df = pd.read_csv(tmp_path, skipinitialspace=True)
                    os.unlink(tmp_path)  # Delete the temporary file
                    
                    all_seed_files.append({
                        'name': seed_file.name,
                        'df': seed_df
                    })
            
            # Process existing seed files if selected
            if use_existing_seeds:
                status_text.text("Processing existing seed files...")
                seed_dir = 'seeds'
                existing_seed_files = glob.glob(os.path.join(seed_dir, '*.csv'))
                
                if existing_seed_files:
                    # Allow user to select which seed files to use
                    file_names = [os.path.basename(path) for path in existing_seed_files]
                    
                    # Create a selection widget in the main content area
                    st.subheader("Select Seed Files to Use")
                    selected_files = st.multiselect(
                        "Choose which seed files to process",
                        file_names,
                        default=file_names  # Default to all files selected
                    )
                    
                    if not selected_files:
                        st.warning("No seed files selected. Please select at least one seed file.")
                        return
                    
                    # Update status
                    status_text.text(f"Processing {len(selected_files)} selected seed files...")
                    
                    # Filter to only selected files
                    existing_seed_files = [
                        path for path in existing_seed_files
                        if os.path.basename(path) in selected_files
                    ]
                
                # Process the selected seed files
                for seed_path in existing_seed_files:
                    seed_df = pd.read_csv(seed_path, skipinitialspace=True)
                    all_seed_files.append({
                        'name': os.path.basename(seed_path),
                        'df': seed_df
                    })
            
            progress_bar.progress(40)
            
            if not all_seed_files:
                st.error("No seed files found.")
                return
            
            st.write(f"Found {len(all_seed_files)} seed files")
            
            # Create tabs for each seed file result
            tabs = st.tabs([seed['name'] for seed in all_seed_files])
            
            # Process each seed file
            for i, (tab, seed) in enumerate(zip(tabs, all_seed_files)):
                progress_percent = 40 + (i / len(all_seed_files)) * 50
                status_text.text(f"Processing seed file: {seed['name']}")
                progress_bar.progress(int(progress_percent))
                
                with tab:
                    # Display seed file preview
                    st.subheader("Seed File Preview")
                    st.dataframe(seed['df'].head(5))
                    
                    # Process the mapping
                    seed_name = os.path.splitext(seed['name'])[0]
                    result_df, output_file, mapping_info, semantic_mappings = process_mapping(input_df, seed['df'], seed_name, output_dir)
                    
                    # Display mapping information
                    st.subheader("Column Mapping Information")
                    
                    # Display semantic mappings in the requested format
                    if semantic_mappings:
                        # Create formatted strings like "input_col -> seed_name.seed_col"
                        st.markdown("### Semantic Column Mappings")
                        
                        # Create two columns for better presentation
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### High Confidence Mappings (80%+)")
                            high_confidence = [m for m in semantic_mappings if m['confidence'] >= 0.8]
                            if high_confidence:
                                for mapping in sorted(high_confidence, key=lambda x: x['confidence'], reverse=True):
                                    confidence = int(mapping['confidence'] * 100)
                                    st.markdown(f"**{mapping['input_col']}** → **{seed_name}.{mapping['seed_col']}** ({confidence}%)")
                            else:
                                st.info("No high confidence mappings found")
                        
                        with col2:
                            st.markdown("#### Medium Confidence Mappings (60-80%)")
                            medium_confidence = [m for m in semantic_mappings if 0.6 <= m['confidence'] < 0.8]
                            if medium_confidence:
                                for mapping in sorted(medium_confidence, key=lambda x: x['confidence'], reverse=True):
                                    confidence = int(mapping['confidence'] * 100)
                                    st.markdown(f"**{mapping['input_col']}** → **{seed_name}.{mapping['seed_col']}** ({confidence}%)")
                            else:
                                st.info("No medium confidence mappings found")
                    else:
                        st.info("No column mappings found")
                    
                    # For technical users, still show the detailed mapping information
                    with st.expander("Technical Mapping Details"):
                        # Display join key information
                        if mapping_info['join_key']:
                            st.success(f"Join Key: Input column '**{mapping_info['join_key']['input']}**' is mapped to Seed column '**{mapping_info['join_key']['seed']}**' (Confidence: {mapping_info['join_key']['confidence']:.2f})")
                        
                        # Display additional column mappings
                        if mapping_info['matched_columns']:
                            st.write("Additional Column Mappings:")
                            mapping_data = []
                            for match in mapping_info['matched_columns']:
                                mapping_data.append({
                                    "Input Column": match['input'],
                                    "Seed Column": match['seed'],
                                    "Match Type": match['match_type'],
                                    "Confidence": f"{match['confidence']:.2f}"
                                })
                            st.table(mapping_data)
                        else:
                            st.info("No additional column mappings found beyond the join key")
                    
                    # Download button for the result
                    with open(output_file, 'rb') as f:
                        st.download_button(
                            label=f"Download Mapped Result for {seed['name']}",
                            data=f,
                            file_name=os.path.basename(output_file),
                            mime="text/csv"
                        )
            
            progress_bar.progress(100)
            status_text.text("Mapping completed successfully!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")


def main(output_dir='output'):
    """
    Main function that determines whether to run CLI or Streamlit interface.
    """
    # Check if running as Streamlit app
    if 'streamlit' in sys.modules:
        run_streamlit()
    else:
        run_cli(output_dir)

if __name__ == "__main__":
    try:
        # Check if running with streamlit
        if 'streamlit' in sys.modules:
            main()
        else:
            # Check if output directory is provided as command line argument
            output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'
            main(output_dir)
    except Exception as e:
        if 'streamlit' in sys.modules:
            st.error(f"Error: {str(e)}")
        else:
            print(f"Error: {str(e)}")
            sys.exit(1)