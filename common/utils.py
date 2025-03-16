import os
import sys
import pandas as pd
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)

class Utils():
    def __init__(self):
        """
        Initialize the class
        """
        pass
    
    def merge_csv_files(self, input_files: list, output_file: str, chunksize=100000, is_print_messege = False):
        """
        Reads multiple large CSV files in chunks, merges them, and saves the result to a new file.
        
        :param input_files: list of paths to CSV files
        :param output_file: path to the output CSV file
        :param chunksize: number of rows to read at a time (default: 100000)
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
        
        with open(output_file, 'w', newline='') as outfile:
            header_written = False
            for file in input_files:
                try:
                    for chunk in pd.read_csv(file, chunksize=chunksize):
                        chunk.to_csv(outfile, index=False, header=not header_written, mode='a')
                        header_written = True  # Ensure header is written only once
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        
        print(f"File saved: {output_file}")
