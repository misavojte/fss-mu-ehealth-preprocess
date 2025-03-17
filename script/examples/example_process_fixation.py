import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import load_ehealth_data, load_session_ids
from processes.process_fixation import process_multiple_gaze_sessions
import pandas as pd

# Example Usage:
# -----------------------------------------------------------------------------
# This script provides two approaches for fixation detection:
#   1. Real timestamps approach: Uses actual time intervals from the data
#   2. Fixed frequency approach: Assumes evenly spaced samples at a set frequency
#
# Examples:
#
# 1. Run both approaches and compare them (default):
#    python script/examples/example_process_fixation.py
#
# 2. Use only real timestamps approach:
#    python script/examples/example_process_fixation.py --approach real
#
# 3. Use only fixed frequency approach:
#    python script/examples/example_process_fixation.py --approach fixed
#
# 4. Specify a custom timestamp column:
#    python script/examples/example_process_fixation.py --timestamp-column timestamp
#
# 5. Specify a different sampling frequency (e.g., 250 Hz):
#    python script/examples/example_process_fixation.py --frequency 250.0
#
# 6. Process only a specific session:
#    python script/examples/example_process_fixation.py --session-id 1731932124378-2311
#
# 7. Combine multiple options:
#    python script/examples/example_process_fixation.py --approach real --timestamp-column ISOtimestamp --session-id 1731932124378-2311
#
# Output:
# - Fixation data CSV files in outputs/fixations/
# - Summary statistics CSV files in outputs/
# - Approach comparison CSV file (when both approaches are used)
# -----------------------------------------------------------------------------

def parse_arguments():
    """Parse command line arguments for the example script."""
    parser = argparse.ArgumentParser(
        description='Process eye tracking data for fixation detection using different timestamp approaches.'
    )
    parser.add_argument(
        '--approach', 
        type=str, 
        choices=['real', 'fixed', 'both'],
        default='both',
        help='Timestamp approach: "real" for real timestamps, "fixed" for fixed frequency, or "both" to compare'
    )
    parser.add_argument(
        '--timestamp-column', 
        type=str, 
        default='ISOtimestamp',
        help='Column name containing timestamps (for real timestamps approach)'
    )
    parser.add_argument(
        '--frequency', 
        type=float, 
        default=150.0,
        help='Sampling frequency in Hz (for fixed frequency approach)'
    )
    parser.add_argument(
        '--session-id', 
        type=str, 
        default=None,
        help='Process only this session ID (optional, otherwise processes all available)'
    )
    
    return parser.parse_args()

def process_with_real_timestamps(session_ids, gaze_data_dict, timestamp_column='ISOtimestamp'):
    """Process using real timestamps approach."""
    print("\n=== Processing with Real Timestamps Approach ===")
    print(f"Using timestamp column: {timestamp_column}")
    
    # Check if timestamp column exists in the first dataset
    if session_ids and session_ids[0] in gaze_data_dict:
        first_df = gaze_data_dict[session_ids[0]]
        if timestamp_column not in first_df.columns:
            print(f"WARNING: Specified timestamp column '{timestamp_column}' not found in data.")
            print(f"Available columns: {', '.join(first_df.columns[:10])}")
            print("The code will attempt to fall back to available timestamp columns.")
    
    summary_df = process_multiple_gaze_sessions(
        session_ids=session_ids,
        gaze_data_dict=gaze_data_dict,
        use_real_timestamps=True,
        timestamp_column=timestamp_column,
        save_output=True
    )
    
    return summary_df

def process_with_fixed_frequency(session_ids, gaze_data_dict, frequency=150.0):
    """Process using fixed frequency approach."""
    print(f"\n=== Processing with Fixed Frequency Approach ({frequency} Hz) ===")
    print("This approach assumes evenly spaced samples at the specified frequency.")
    print("Data will first be sorted by timestamp (if available) and then assigned evenly spaced time points.")
    
    summary_df = process_multiple_gaze_sessions(
        session_ids=session_ids,
        gaze_data_dict=gaze_data_dict,
        use_real_timestamps=False,
        fixed_frequency=frequency,
        save_output=True
    )
    
    return summary_df

def display_summary(summary_df, title="Summary"):
    """Display summary dataframe with useful statistics."""
    if summary_df.empty:
        print("No data to display!")
        return
    
    # Check for processing errors
    if 'processing_error' in summary_df.columns:
        error_sessions = summary_df[summary_df['processing_error'].notna()]
        if not error_sessions.empty:
            print("\n=== Processing Errors ===")
            for _, row in error_sessions.iterrows():
                print(f"Error in session {row['session_id']}: {row['processing_error']}")
    
    print(f"\n=== {title} ===")
    print(f"Shape: {summary_df.shape}")
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
    if fixation_cols:
        print("\nMean values:")
        print(summary_df[fixation_cols].mean())
    else:
        print("No fixation metrics found in data.")

def compare_approaches(real_summary, fixed_summary):
    """Compare results from both timestamp approaches."""
    if real_summary.empty or fixed_summary.empty:
        print("Cannot compare: One or both approaches did not produce results.")
        return
    
    print("\n=== Comparing Timestamp Approaches ===")
    
    # Merge on session_id
    comparison = pd.merge(
        real_summary.add_suffix('_real'), 
        fixed_summary.add_suffix('_fixed'),
        left_on='session_id_real', 
        right_on='session_id_fixed',
        how='inner'
    )
    
    if comparison.empty:
        print("No common sessions to compare.")
        return
    
    # Check key metrics differences
    metrics_to_compare = [
        'fix_total_fixations',
        'fix_mean_fixation_duration',
        'fix_median_fixation_duration',
        'fix_fixation_rate'
    ]
    
    print("\nMetric Differences (Real - Fixed):")
    differences = {}
    
    for metric in metrics_to_compare:
        real_col = f"{metric}_real"
        fixed_col = f"{metric}_fixed"
        
        if real_col in comparison.columns and fixed_col in comparison.columns:
            # Calculate absolute and percentage differences
            abs_diff = comparison[real_col] - comparison[fixed_col]
            
            # Handle division by zero
            with pd.option_context('mode.use_inf_as_na', True):
                pct_diff = (abs_diff / comparison[fixed_col]) * 100
                pct_diff = pct_diff.fillna(0)
            
            differences[metric] = {
                'mean_abs_diff': abs_diff.mean(),
                'mean_pct_diff': pct_diff.mean(),
                'max_abs_diff': abs_diff.abs().max(),
                'max_pct_diff': pct_diff.abs().max()
            }
            
            print(f"\n{metric}:")
            print(f"  Mean absolute difference: {abs_diff.mean():.2f}")
            print(f"  Mean percentage difference: {pct_diff.mean():.2f}%")
            print(f"  Max absolute difference: {abs_diff.abs().max():.2f}")
            print(f"  Max percentage difference: {pct_diff.abs().max():.2f}%")
    
    # Save comparison to CSV
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"approach_comparison_{timestamp}.csv")
    
    comparison.to_csv(output_path, index=False)
    print(f"\nSaved comparison to: {output_path}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Get session IDs
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
    
    # Load gaze data for selected sessions
    print("\n=== Loading Gaze Data ===")
    gaze_data_dict = {}
    for session_id in session_ids:
        gaze_df = load_ehealth_data(session_id)
        if gaze_df is not None:
            gaze_data_dict[session_id] = gaze_df
            print(f"Loaded {len(gaze_df)} samples for session {session_id}")
            
            # Print available columns to help users identify timestamp columns
            print(f"Available columns: {', '.join(gaze_df.columns[:5])}...")
            
            # Check for specified timestamp column if using real timestamps
            if args.approach in ['real', 'both'] and args.timestamp_column not in gaze_df.columns:
                print(f"WARNING: Specified timestamp column '{args.timestamp_column}' not found in session {session_id}.")
                # Check for alternative timestamp columns
                timestamp_alternatives = ['timestamp', 'ISOtimestamp', 'time']
                available_timestamps = [col for col in timestamp_alternatives if col in gaze_df.columns]
                if available_timestamps:
                    print(f"Available timestamp columns: {', '.join(available_timestamps)}")
                else:
                    print("No standard timestamp columns found. The processing will use sample order.")
    
    if not gaze_data_dict:
        print("No gaze data found!")
        return
    
    # Process based on selected approach
    real_summary = None
    fixed_summary = None
    
    if args.approach in ['real', 'both']:
        real_summary = process_with_real_timestamps(
            session_ids, 
            gaze_data_dict, 
            timestamp_column=args.timestamp_column
        )
        display_summary(real_summary, "Real Timestamps Approach Summary")
    
    if args.approach in ['fixed', 'both']:
        fixed_summary = process_with_fixed_frequency(
            session_ids, 
            gaze_data_dict, 
            frequency=args.frequency
        )
        display_summary(fixed_summary, "Fixed Frequency Approach Summary")
    
    # Compare approaches if both were used
    if args.approach == 'both' and real_summary is not None and fixed_summary is not None:
        compare_approaches(real_summary, fixed_summary)
    
    print("\n=== Processing Complete ===")
    print("The results have been saved to the 'outputs' directory.")
    if args.approach in ['real', 'both']:
        print(f"Real timestamps approach used column: {args.timestamp_column}")
    if args.approach in ['fixed', 'both']:
        print(f"Fixed frequency approach used sampling rate: {args.frequency} Hz")

if __name__ == "__main__":
    main() 