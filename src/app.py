import os
import pandas as pd
import tempfile
from datetime import datetime
from column_mapper import ColumnMapper
import streamlit as st
import sys

def process_mapping(input_df, seed_df, seed_name, output_dir='output'):
    """
    Process the mapping between input dataframe and seed dataframe.
    Maps primary keys and fixed-length columns.
    
    Args:
        input_df (pd.DataFrame): The input dataframe
        seed_df (pd.DataFrame): The seed dataframe
        seed_name (str): Name of the seed file (for output filename)
        output_dir (str): Directory to save output files
        
    Returns:
        tuple: (output_file_path, column_mappings, fixed_length_columns_info)
    """
    # Initialize the column mapper
    mapper = ColumnMapper()
    
    # Map columns using the seed file (checks fixed-length columns)
    _, mapping_info = mapper.map_columns(input_df, seed_df)
    
    # Extract fixed-length column information
    fixed_length_columns_info = mapping_info.get('matched_columns', [])
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"mapped_{seed_name}_{timestamp}.csv")
    
    # Generate a mapping report instead of saving a merged dataframe
    report_df = pd.DataFrame([
        {
            'Seed Column': rel['seed'],
            'Input Column': rel['input'],
            'Fixed Length': rel['length'],
            'Is Primary Key': "Yes" if rel.get('is_primary_key', False) else "No"
        }
        for rel in fixed_length_columns_info
    ])
    
    # Save the mapping report instead of a merged dataframe
    report_df.to_csv(output_file, index=False)
    
    # Collect column mappings (input -> seed)
    column_mappings = []
    
    # Collect matched columns (input -> seed)
    for match in fixed_length_columns_info:
        input_col = match['input']
        seed_col = match['seed']
        length = match['length']
        column_mappings.append(f"Input: {input_col} → Seed: {seed_name}.{seed_col} (Fixed character length: {length})")
    
    # Print column mappings for terminal output
    print("\nColumn Mappings:")
    for mapping in column_mappings:
        print(mapping)
    
    return output_file, column_mappings, fixed_length_columns_info

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
    
    # Sidebar for file uploads and options
    with st.sidebar:
        st.header("Upload Files")
        
        # Input file upload
        st.subheader("Input File (Excel)")
        input_file = st.file_uploader("Upload your main input file", type=["xlsx", "xls"])
        
        # Seed files upload
        st.subheader("Seed Files (CSV)")
        seed_files = st.file_uploader("Upload your seed/reference files", type=["csv"], accept_multiple_files=True)
        
        # Output directory
        st.subheader("Output Settings")
        output_dir = st.text_input("Output Directory", value="output")
        
        # Run button
        run_mapping = st.button("Run Mapping", type="primary")
    
    # Main content area
    if not input_file and not run_mapping:
        st.info("Please upload an input file and seed files, then click 'Run Mapping'.")
        return
    
    if run_mapping:
        if not input_file:
            st.error("Please upload an input file.")
            return
        
        if not seed_files:
            st.error("Please upload seed files.")
            return
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            st.info(f"Created output directory: {output_dir}")
        
        # Load input file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(input_file.getvalue())
            tmp_path = tmp.name
        
        input_df = pd.read_excel(tmp_path)
        os.unlink(tmp_path)  # Delete temporary file
        
        st.write(f"Loaded input file with {len(input_df)} rows and {len(input_df.columns)} columns")
        
        # Process seed files
        all_seed_files = []
        for seed_file in seed_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(seed_file.getvalue())
                tmp_path = tmp.name
            
            seed_df = pd.read_csv(tmp_path, skipinitialspace=True)
            os.unlink(tmp_path)  # Delete temporary file
            all_seed_files.append({
                'name': seed_file.name,
                'df': seed_df
            })
        
        # Process each seed file
        for seed in all_seed_files:
            seed_name = os.path.splitext(seed['name'])[0]
            seed_df = seed['df']
            
            # Process the mapping and get column mappings
            output_file, column_mappings, fixed_length_columns_info = process_mapping(input_df, seed_df, seed_name, output_dir)
            
            if not column_mappings:
                st.info(f"No mappings found for {seed_name}")
                continue
            
            st.subheader(f"Mapping Results for {seed_name}")
            
            # Display fixed-length column information
            if fixed_length_columns_info:
                # Create a simple DataFrame showing all the column details
                columns_df = pd.DataFrame([
                    {
                        'Seed Column': item['seed'],
                        'Input Column': item['input'],
                        'Fixed Length': item['length'],
                        'Is Primary Key': 'Yes' if item.get('is_primary_key', False) else 'No',
                        'Is Unique': 'Yes' if item.get('is_primary_key', False) else 'No'
                    }
                    for item in fixed_length_columns_info
                ])
                
                # Show info about mapped columns
                st.subheader("Column Mapping Information")
                st.dataframe(columns_df)
                
                # Extract primary keys
                primary_keys = [info for info in fixed_length_columns_info if info.get('is_primary_key', False)]
                
                # Display dedicated primary key information section
                if primary_keys:
                    st.subheader("📌 Primary Key Columns")
                    st.info("Primary key columns are unique identifiers with fixed character length")
                    
                    # Create a specific DataFrame just for primary keys
                    pk_df = pd.DataFrame([
                        {
                            'Column Name': pk['seed'],
                            'Character Length': pk['length'],
                            'Is Unique': 'Yes'
                        }
                        for pk in primary_keys
                    ])
                    
                    # Show primary key details in a prominent way
                    st.markdown("### Primary Key Details")
                    st.table(pk_df)
                    
                    # Visual chart of primary key lengths
                    st.markdown("### Primary Key Column Lengths")
                    chart_data = pd.DataFrame({
                        'Column': [pk['seed'] for pk in primary_keys],
                        'Length': [pk['length'] for pk in primary_keys]
                    })
                    st.bar_chart(chart_data.set_index('Column'))
            else:
                st.warning("No fixed-length columns identified in the seed file")
            
            # Display column mappings
            st.subheader("Column Mappings")
            st.table(pd.DataFrame(column_mappings, columns=["Column Mappings"]))

def main(output_dir='output'):
    if 'streamlit' in sys.modules:
        run_streamlit()

if __name__ == "__main__":
    try:
        if 'streamlit' in sys.modules:
            main()
        else:
            output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'
            main(output_dir)
    except Exception as e:
        if 'streamlit' in sys.modules:
            st.error(f"Error: {str(e)}")
        else:
            print(f"Error: {str(e)}")
            sys.exit(1)
