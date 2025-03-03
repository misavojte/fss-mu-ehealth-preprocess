import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import load_ehealth_data, load_session_ids
from processes.process_fixation import process_multiple_gaze_sessions
import pandas as pd
import random
import numpy as np

def main():
    # Get all available session IDs
    print("\n=== Loading Session IDs ===")
    # session_ids = load_session_ids()
    session_ids = ["1731573239729-3024"]
    
    if not session_ids:
        print("No sessions found!")
        return
    
    # Load gaze data for all sessions
    print("\n=== Loading Gaze Data ===")
    gaze_data_dict = {}
    for session_id in session_ids:
        gaze_df = load_ehealth_data(session_id)
        if gaze_df is not None:
            gaze_data_dict[session_id] = gaze_df
    
    if not gaze_data_dict:
        print("No gaze data found!")
        return
    
    # Process all sessions
    print("\n=== Processing Gaze Data ===")
    
    # Set a random seed before processing to ensure deterministic results
    random.seed(42)
    np.random.seed(42)
    
    summary_df = process_multiple_gaze_sessions(session_ids, gaze_data_dict, False)
    
    # Display results
    if not summary_df.empty:
        print("\n=== Gaze Summary Table ===")
        print(f"\nShape: {summary_df.shape}")
        print("\nColumns:")
        for col in summary_df.columns:
            print(f"- {col}")
            
        print("\nFirst few rows:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(summary_df.head())
        
        # Basic statistics for fixation metrics
        print("\n=== Fixation Statistics ===")
        fixation_cols = [col for col in summary_df.columns if 'fix_' in col]
        print("\nMean values:")
        print(summary_df[fixation_cols].mean())
    else:
        print("No data to process!")

if __name__ == "__main__":
    main() 