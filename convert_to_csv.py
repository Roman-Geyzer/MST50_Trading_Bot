"""
Description: Converts all Parquet files in a directory to CSV format,
             filtering rows based on a specific date in the 'time' column.
             
Usage:
    - Ensure the source and destination directories are correctly set.
    - Set the FILTER_DATE to the desired cutoff date (format: DD.MM.YYYY).
    - Run the script to perform the conversion and filtering.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# Global parameter for the filter date (DD.MM.YYYY)
FILTER_DATE_STR = '01.01.2024'  # Modify this date as needed

def convert_parquet_to_csv(source_dir, destination_dir, filter_date_str):
    """
    Converts all Parquet files in the source_dir to CSV format,
    filtering rows where the 'time' column is on or after filter_date,
    and saves them in the destination_dir with the same base filenames.
    
    Parameters:
    - source_dir (str): Path to the source directory containing Parquet files.
    - destination_dir (str): Path to the destination directory for CSV files.
    - filter_date_str (str): The cutoff date in 'DD.MM.YYYY' format.
    """
    
    # Convert to Path objects for easier handling
    source_path = Path(source_dir)
    destination_path = Path(destination_dir)
    
    # Check if source directory exists
    if not source_path.exists():
        print(f"Source directory '{source_dir}' does not exist.")
        return
    
    # Create destination directory if it doesn't exist
    destination_path.mkdir(parents=True, exist_ok=True)
    print(f"Destination directory is set to '{destination_path.resolve()}'")
    
    # Iterate over all Parquet files in the source directory
    parquet_files = list(source_path.glob('*.parquet'))
    
    if not parquet_files:
        print(f"No Parquet files found in '{source_dir}'.")
        return
    
    print(f"Found {len(parquet_files)} Parquet file(s) in '{source_dir}'. Starting conversion with date filtering...")
    
    # Convert FILTER_DATE_STR to datetime object
    try:
        filter_date = datetime.strptime(filter_date_str, '%d.%m.%Y')
    except ValueError as ve:
        print(f"Error parsing FILTER_DATE '{filter_date_str}': {ve}")
        return
    
    for parquet_file in parquet_files:
        try:
            # Read Parquet file into DataFrame
            df = pd.read_parquet(parquet_file)
            
            # Check if 'time' column exists
            if 'time' not in df.columns:
                print(f"File '{parquet_file.name}' does not contain a 'time' column. Skipping this file.")
                continue
            
            # Convert 'time' column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            
            # Drop rows with invalid 'time' values
            initial_row_count = len(df)
            df = df.dropna(subset=['time'])
            dropped_rows = initial_row_count - len(df)
            if dropped_rows > 0:
                print(f"Dropped {dropped_rows} rows with invalid 'time' values in '{parquet_file.name}'.")
            
            # Filter the DataFrame based on FILTER_DATE
            filtered_df = df[df['time'] >= filter_date]
            filtered_row_count = len(filtered_df)
            
            if filtered_row_count == 0:
                print(f"No rows after filtering in '{parquet_file.name}'. Skipping CSV creation.")
                continue
            
            # Define CSV file path
            csv_filename = parquet_file.stem + '.csv'  # Same base name with .csv extension
            csv_file_path = destination_path / csv_filename
            
            # Write filtered DataFrame to CSV
            filtered_df.to_csv(csv_file_path, index=False)
            
            print(f"Converted '{parquet_file.name}' to '{csv_filename}' with {filtered_row_count} row(s).")
        
        except Exception as e:
            print(f"Failed to convert '{parquet_file.name}': {e}")
    
    print("Conversion process with date filtering completed.")

if __name__ == "__main__":
    # Define your source and destination directories
    source_directory = '/Volumes/TM/historical_data'          # Replace with your source folder path
    destination_directory = '/Volumes/TM/historical_data_csvs'  # Replace with your destination folder path
    
    # Call the conversion function with the filter date
    convert_parquet_to_csv(source_directory, destination_directory, FILTER_DATE_STR)