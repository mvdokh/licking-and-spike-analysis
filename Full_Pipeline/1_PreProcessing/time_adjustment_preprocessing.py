#!/usr/bin/env python3
"""
Time Adjustment Preprocessing Script

This script processes the semi_from_table_designer.csv file by dividing the first 3 columns
(pico_Interval Start, pico_Interval End, pico_Interval Duration) by 30,000 to adjust
the time scale. The processed data is saved as pico_time_adjust.csv.

Usage:
    python time_adjustment_preprocessing.py
"""

import pandas as pd
import os

def process_time_adjustment():
    """
    Process the semi_from_table_designer.csv file by dividing the first 3 columns by 30,000.
    Save the result as pico_time_adjust.csv in the Data folder.
    """
    
    # Define file paths
    input_file = "../../Data/semi_from_table_designer.csv"
    output_file = "../../Data/pico_time_adjust.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Reading data from: {input_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns[:5])}...")  # Show first 5 column names
    
    # Get the first 3 columns
    first_three_cols = df.columns[:3]
    print(f"Processing columns: {list(first_three_cols)}")
    
    # Create a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Divide the first 3 columns by 30,000
    for col in first_three_cols:
        print(f"Processing column '{col}'...")
        print(f"  Original range: {df[col].min():.3f} - {df[col].max():.3f}")
        
        processed_df[col] = df[col] / 30000.0
        
        print(f"  Adjusted range: {processed_df[col].min():.6f} - {processed_df[col].max():.6f}")
    
    # Save the processed data
    print(f"Saving processed data to: {output_file}")
    processed_df.to_csv(output_file, index=False)
    
    print(f"Processing complete! Saved {len(processed_df)} rows to {output_file}")
    
    # Display summary statistics
    print("\nSummary of time adjustment:")
    print("=" * 50)
    for col in first_three_cols:
        print(f"{col}:")
        print(f"  Original: {df[col].mean():.3f} ± {df[col].std():.3f}")
        print(f"  Adjusted: {processed_df[col].mean():.6f} ± {processed_df[col].std():.6f}")
        print()

if __name__ == "__main__":
    try:
        process_time_adjustment()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)