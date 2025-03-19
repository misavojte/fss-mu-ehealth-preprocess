import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processes.process_insights import (
    process_fixation_insights,
    get_latest_session_summary,
    load_all_fixation_files,
    match_fixations_to_stimuli
)
import pandas as pd

# Example Usage:
# -----------------------------------------------------------------------------
# This script demonstrates how to use the process_insights module to:
#   1. Load the most recent session_summaries CSV file 
#   2. Load all fixation data files and add sessionId column
#   3. Match fixations to stimuli based on timestamp ranges
#   4. Export the combined and enriched fixations to a CSV file
#
# Examples:
#
# 1. Run with default settings (limited to 5 files and 2 sessions for testing):
#    python script/examples/example_process_insights.py
#
# 2. Process all fixation files and sessions:
#    python script/examples/example_process_insights.py --all
#
# 3. Process specific number of files and sessions:
#    python script/examples/example_process_insights.py --max-files 10 --max-sessions 3
#
# 4. Run with verbose output:
#    python script/examples/example_process_insights.py --verbose
#
# 5. Run without saving the output:
#    python script/examples/example_process_insights.py --no-save
#
# Output:
# - Combined and enriched fixation data CSV file in outputs/
# -----------------------------------------------------------------------------

def parse_arguments():
    """Parse command line arguments for the process insights example script."""
    parser = argparse.ArgumentParser(
        description='Process fixation data and match with stimuli from session summaries.'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='Do not save the output CSV'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all files and sessions (not just a sample for testing)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=5,
        help='Maximum number of fixation files to load (default: 5, use 0 for all)'
    )
    parser.add_argument(
        '--max-sessions',
        type=int,
        default=2,
        help='Maximum number of sessions to process (default: 2, use 0 for all)'
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine max_files and max_sessions
    max_files = None if args.all else args.max_files
    max_sessions = None if args.all else args.max_sessions
    
    if max_files == 0:
        max_files = None
    if max_sessions == 0:
        max_sessions = None
    
    # Process the fixation insights
    print("\n=== Processing Fixation Insights ===")
    print(f"Max files: {max_files or 'All'}, Max sessions: {max_sessions or 'All'}")
    
    enriched_fixations = process_fixation_insights(
        save_output=not args.no_save,
        max_files=max_files,
        max_sessions=max_sessions
    )
    
    # Display results
    if not enriched_fixations.empty:
        print("\n=== Enriched Fixation Data ===")
        print(f"\nShape: {enriched_fixations.shape}")
        print("\nColumns:")
        for col in enriched_fixations.columns:
            print(f"- {col}")
            
        print("\nFirst few rows:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(enriched_fixations.head())
        
        # Display more detailed information if verbose is enabled
        if args.verbose:
            print("\n=== Stimulus Distribution ===")
            stimulus_counts = enriched_fixations['stimulus'].value_counts()
            print(stimulus_counts)
            
            print("\n=== Session Distribution ===")
            session_counts = enriched_fixations['sessionId'].value_counts()
            print(session_counts)
            
            # Check for any unmatched fixations
            unmatched = enriched_fixations[enriched_fixations['stimulus'].isna()]
            if not unmatched.empty:
                print("\n=== Unmatched Fixations ===")
                print(f"Total unmatched: {len(unmatched)} ({len(unmatched)/len(enriched_fixations)*100:.2f}%)")
                
                # Sample of unmatched fixations
                print("\nSample of unmatched fixations:")
                print(unmatched.head())
    else:
        print("No enriched fixation data to display!")
    
    print("\n=== Processing Complete ===")

if __name__ == "__main__":
    main() 