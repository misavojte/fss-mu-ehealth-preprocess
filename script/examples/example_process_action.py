import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import load_action_data, load_session_ids
from processes.process_action import process_multiple_sessions
import pandas as pd

def main():
    # Get all available session IDs
    print("\n=== Loading Session IDs ===")
    session_ids = load_session_ids()
    
    if not session_ids:
        print("No sessions found!")
        return
    
    # Load action data for all sessions
    print("\n=== Loading Action Data ===")
    action_data_dict = {}
    for session_id in session_ids:
        action_df = load_action_data(session_id)
        if action_df is not None:
            action_data_dict[session_id] = action_df
    
    # Process all sessions
    print("\n=== Processing Sessions ===")
    summary_df = process_multiple_sessions(session_ids, action_data_dict)
    
    # Display results
    if not summary_df.empty:
        print("\n=== Summary Table ===")
        print(f"\nShape: {summary_df.shape}")
        print("\nColumns:")
        for col in summary_df.columns:
            print(f"- {col}")
            
        print("\nFirst few rows:")
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)        # Don't wrap wide tables
        print(summary_df.head())
        
        # Basic statistics for validation metrics
        print("\n=== Validation Statistics ===")
        validation_cols = [col for col in summary_df.columns if 'vali' in col]
        print("\nMean values:")
        print(summary_df[validation_cols].mean())
        
        print("\nStandard deviations:")
        print(summary_df[validation_cols].std())
        
        # Save to CSV
        output_path = "output/session_summaries.csv"
        os.makedirs("output", exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f"\nSaved summary table to: {output_path}")
    else:
        print("No data to process!")

if __name__ == "__main__":
    main() 