import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import load_action_data, load_session_ids
from processes.process_action import process_multiple_sessions
import pandas as pd

# Example Usage:
# -----------------------------------------------------------------------------
# This script processes action logs from the eHealth application, extracting:
#   1. Validation metrics (accuracy, precision, number of validation rounds)
#   2. L1 token events (start time, end time, response value)
#   3. L2 token events (start time, end time, response value)
#   4. L3 ordered tasks (sequence, time at each stop)
#
# The eHealth action data format includes columns like:
# - id: Action event identifier
# - timestamp: Time the action was recorded (ISO format)
# - sessionId: Session identifier (matching gaze data)
# - type: Type of action (e.g., 'gaze-validation', 'L1_start', 'L1_response')
# - value: Value associated with the action (often JSON or structured strings)
#
# NOTE: Unlike fixation processing which creates individual CSV files for each session,
# action processing only creates a single summary CSV file containing all sessions.
# This design choice is intentional as action data is typically analyzed at the summary level.
#
# Examples:
#
# 1. Process all available sessions:
#    python script/examples/example_process_action.py
#
# 2. Process only a specific session:
#    python script/examples/example_process_action.py --session-id 1731932124378-2311
#
# 3. Process without saving to CSV:
#    python script/examples/example_process_action.py --no-save
#
# 4. Export with a custom base name:
#    python script/examples/example_process_action.py --output-name my_custom_name
#
# Output:
# - ONLY a single action summary CSV file in the 'outputs' directory (no individual files)
# - Timestamp format is standardized to match the fixation data format
#   for easy alignment between action events and fixation data
#
# The action summary includes:
# - Validation metrics for each validation point (accuracy, precision)
# - Token events with timestamps (good for aligning with fixation data)
# - Navigation task metrics and timestamps
# -----------------------------------------------------------------------------

def parse_arguments():
    """Parse command line arguments for the eHealth action processing script."""
    parser = argparse.ArgumentParser(
        description='Process eHealth action logs and extract metrics.'
    )
    parser.add_argument(
        '--session-id', 
        type=str, 
        default=None,
        help='Process only this session ID (optional, otherwise processes all available)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to CSV'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default="action_summary",
        help='Base name for the output CSV file'
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Get all available session IDs
    print("\n=== Loading Session IDs ===")
    all_session_ids = load_session_ids()
    
    if not all_session_ids:
        print("No sessions found!")
        return
    
    # Filter to specific session if requested
    if args.session_id:
        if args.session_id in all_session_ids:
            session_ids = [args.session_id]
            print(f"Processing single session: {args.session_id}")
        else:
            print(f"Session ID {args.session_id} not found!")
            print(f"Available session IDs: {', '.join(all_session_ids[:5])}" + 
                  ("..." if len(all_session_ids) > 5 else ""))
            return
    else:
        session_ids = all_session_ids
        print(f"Processing {len(session_ids)} sessions")
    
    # Load action data for selected sessions
    print("\n=== Loading Action Data ===")
    action_data_dict = {}
    for session_id in session_ids:
        action_df = load_action_data(session_id)
        if action_df is not None:
            # Ensure timestamp column is in datetime format
            if 'timestamp' in action_df.columns:
                action_df['timestamp'] = pd.to_datetime(action_df['timestamp'])
                print(f"Loaded {len(action_df)} action samples for session {session_id}")
                
                # Show timestamp format (important for alignment with fixation data)
                if not action_df.empty:
                    print(f"  Timestamp example: {action_df['timestamp'].iloc[0]}")
                    
                    # Display some of the action types to give an overview
                    action_types = action_df['type'].unique()
                    print(f"  Action types: {', '.join(action_types[:5])}" + 
                          ("..." if len(action_types) > 5 else ""))
            else:
                print(f"Warning: No timestamp column found in action data for session {session_id}")
            
            # Add to dictionary
            action_data_dict[session_id] = action_df
    
    if not action_data_dict:
        print("No action data found!")
        return
    
    # Process all sessions
    print("\n=== Processing Sessions ===")
    summary_df = process_multiple_sessions(
        session_ids, 
        action_data_dict, 
        save_output=not args.no_save,
        base_name=args.output_name
    )
    
    # Display results
    if not summary_df.empty:
        print("\n=== Summary Table ===")
        print(f"\nShape: {summary_df.shape}")
        print("\nColumns:")
        col_categories = {
            "Session Info": [col for col in summary_df.columns if any(term in col for term in ['session', 'linking'])],
            "Validation": [col for col in summary_df.columns if 'vali' in col],
            "L1 Tokens": [col for col in summary_df.columns if 'L1__' in col],
            "L2 Tokens": [col for col in summary_df.columns if 'L2__' in col],
            "L3 Navigation": [col for col in summary_df.columns if 'L3__' in col],
            "Other": []
        }
        
        # Categorize columns for better display
        for col in summary_df.columns:
            if not any(col in category for category in col_categories.values()):
                col_categories["Other"].append(col)
        
        # Print categorized columns
        for category, cols in col_categories.items():
            if cols:
                print(f"\n{category} columns:")
                for col in cols:
                    print(f"- {col}")
            
        # Display information about timestamp columns in the summary
        time_cols = [col for col in summary_df.columns if 'Time' in col or 'time' in col]
        if time_cols:
            print("\nTimestamp columns found in summary:")
            for col in time_cols[:5]:  # Show first 5 timestamp columns
                non_null_vals = summary_df[col].dropna()
                if len(non_null_vals) > 0:
                    print(f"  {col}: {non_null_vals.iloc[0]} ({type(non_null_vals.iloc[0]).__name__})")
            print("  (These timestamps can be aligned with fixation timestamps)")
        
        print("\nFirst few rows:")
        pd.set_option('display.max_columns', 10)  # Limit columns for readability
        pd.set_option('display.width', None)      # Don't wrap wide tables
        print(summary_df.head())
        
        # Basic statistics for validation metrics
        print("\n=== Validation Statistics ===")
        validation_cols = [col for col in summary_df.columns if 'vali' in col]
        if validation_cols:
            print("\nMean values:")
            print(summary_df[validation_cols].mean())
            
            print("\nStandard deviations:")
            print(summary_df[validation_cols].std())
            
            # Count sessions with validation data
            has_validation = summary_df[~summary_df['vali_avg_accuracy'].isna()]
            print(f"\nSessions with validation data: {len(has_validation)}/{len(summary_df)}")
            if not has_validation.empty:
                print(f"Average validation accuracy: {has_validation['vali_avg_accuracy'].mean():.2f}")
                print(f"Average validation precision: {has_validation['vali_avg_precision'].mean():.2f}")
        
        # Save to CSV if not already saved during processing
        if args.no_save:
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"{args.output_name}_{timestamp}.csv")
            
            # Use the same date format as in process_fixation.py
            summary_df.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S.%f')
            print(f"\nSaved action summary table to: {output_path}")
        
        print("\n=== eHealth Action Processing Complete ===")
        if not args.no_save:
            print("The results have been saved to the 'outputs' directory.")
            print(f"Output filename uses the base: {args.output_name}")
        
        print("\n=== Timestamp Alignment Information ===")
        print("The timestamps in this output are standardized to match the format used in fixation data.")
        print("This makes it easy to align action events with fixation events by comparing timestamps.")
        print("Example usage:")
        print("1. Find a task start event in the action data (e.g., L1_start)")
        print("2. Use the timestamp to locate fixations that occurred during that task")
        print("3. Analyze fixation patterns during specific cognitive tasks")
        
        print("\n=== CSV Format Notes ===")
        print("1. The CSV output contains standardized datetime columns in format: 'YYYY-MM-DD HH:MM:SS.fff'")
        print("2. This format matches the output from the fixation processing pipeline")
        print("3. Unlike fixation processing which creates individual files per session,")
        print("   action processing only creates a single summary file for all sessions")
        print("4. Column naming follows these patterns:")
        print("   - L1__token__startTime: Start time for a specific L1 token")
        print("   - L2__token__endTime: End time for a specific L2 token")
        print("   - L3__Ox__duration: Duration spent at ordered stop x")
        print("5. When aligning with fixation data, compare the following:")
        print("   - Action timestamps: Look for specific events like L1/L2/L3 start times")
        print("   - Fixation timestamps: Use the 'start_timestamp' and 'end_timestamp' columns")
        print("   - The fixation's 'aoi' column: Can be correlated with actions for specific areas")
    else:
        print("No data to process!")

if __name__ == "__main__":
    main() 