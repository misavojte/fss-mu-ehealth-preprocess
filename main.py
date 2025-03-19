import os
import sys
import pandas as pd
import argparse
from datetime import datetime

"""
eHealth Data Processing Pipeline

This script processes eHealth gaze and action data, performing fixation detection
on the gaze data and extracting metrics from the action data. The script uses the
modules from the script/processes/ and script/loaders/ directories.

Example Usage:
-----------------------------------------------------------------------------
1. Process all data with default settings (fixed frequency at 150 Hz):
   python main.py

2. Process only a specific session:
   python main.py --session-id 1731932124378-2311

3. Use real timestamps instead of fixed frequency:
   python main.py --real-timestamps --timestamp-column ISOtimestamp

4. Process only gaze data (skip action data):
   python main.py --gaze-only

5. Process only action data (skip gaze data):
   python main.py --action-only

6. Change the sampling frequency for gaze processing:
   python main.py --frequency 120.0

7. Save outputs to a different directory:
   python main.py --output-dir my_results

8. Show more detailed processing information:
   python main.py --verbose

9. Combine multiple options:
   python main.py --session-id 1731932124378-2311 --real-timestamps --verbose

Output:
- Fixation data CSV files in <output-dir>/fixations/ (one per session)
- Gaze summary CSV file in <output-dir>/
- Action summary CSV file in <output-dir>/
- All timestamps are standardized to the format: 'YYYY-MM-DD HH:MM:SS.fff'

Notes:
- Unlike the fixation process which creates individual files per session,
  action processing only creates a single summary file for all sessions.
- When using real timestamps, ensure the timestamp column exists in your data.
- The default sampling frequency (150 Hz) is appropriate for most eHealth data.
-----------------------------------------------------------------------------
"""

# Import necessary modules from processes and loaders
from script.loaders import load_session_ids, load_ehealth_data, load_action_data
from script.processes.process_fixation import process_multiple_gaze_sessions
from script.processes.process_action import process_multiple_sessions

def parse_arguments():
    """Parse command-line arguments for the eHealth data processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Process eHealth gaze and action data in a single pipeline."
    )
    
    # Data selection options
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Process only this specific session ID (optional, defaults to all sessions)"
    )
    parser.add_argument(
        "--gaze-only",
        action="store_true",
        help="Process only gaze data (skip action data)"
    )
    parser.add_argument(
        "--action-only",
        action="store_true",
        help="Process only action data (skip gaze data)"
    )
    
    # Gaze processing options
    parser.add_argument(
        "--frequency",
        type=float,
        default=150.0,
        help="Sampling frequency in Hz for gaze data processing (defaults to 150.0 Hz)"
    )
    parser.add_argument(
        "--real-timestamps",
        action="store_true",
        help="Use real timestamps from the data instead of fixed frequency"
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default="ISOtimestamp",
        help="Column name for timestamps when using real timestamps (defaults to 'ISOtimestamp')"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output files (defaults to 'outputs')"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to disk"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information"
    )
    
    return parser.parse_args()

def ensure_output_directory(output_dir):
    """Create outputs directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fixations"), exist_ok=True)
    print(f"Output directories created at {os.path.abspath(output_dir)}")

def process_ehealth_data(session_ids=None, frequency=150.0, use_real_timestamps=False, 
                        timestamp_column="ISOtimestamp", save_output=True, output_dir="outputs", verbose=False):
    """
    Process all eHealth gaze data using fixed frequency approach.
    
    Args:
        session_ids: List of session IDs to process (if None, process all available)
        frequency: Sampling frequency to use for fixation detection (Hz)
        use_real_timestamps: Whether to use real timestamps from data
        timestamp_column: Column name for timestamps when using real timestamps
        save_output: Whether to save results to disk
        output_dir: Directory to save output files
        verbose: Whether to print detailed processing information
    
    Returns:
        pd.DataFrame: Summary of processed gaze data
    """
    print("\n" + "="*80)
    print("PROCESSING EHEALTH GAZE DATA")
    if use_real_timestamps:
        print(f"Using real timestamps from column: {timestamp_column}")
    else:
        print(f"Using fixed frequency: {frequency} Hz")
    print("="*80)
    
    if session_ids is None:
        session_ids = load_session_ids()
    elif isinstance(session_ids, str):
        session_ids = [session_ids]  # Convert single session ID to list
    
    if not session_ids:
        print("No session IDs found for processing.")
        return None
    
    print(f"Found {len(session_ids)} sessions to process.")
    
    # Load gaze data for all sessions
    gaze_data_dict = {}
    for session_id in session_ids:
        try:
            gaze_df = load_ehealth_data(session_id)
            if gaze_df is not None:
                gaze_data_dict[session_id] = gaze_df
                if verbose:
                    print(f"Loaded {len(gaze_df)} gaze samples for session {session_id}")
                    if use_real_timestamps and timestamp_column in gaze_df.columns:
                        print(f"  Timestamp column '{timestamp_column}' found with {gaze_df[timestamp_column].nunique()} unique values")
                    elif use_real_timestamps:
                        print(f"  Warning: Timestamp column '{timestamp_column}' not found in session {session_id}")
                        print(f"  Available columns: {', '.join(gaze_df.columns[:5])}...")
            else:
                print(f"No gaze data found for session {session_id}")
        except Exception as e:
            print(f"Error loading gaze data for session {session_id}: {e}")
    
    if not gaze_data_dict:
        print("No gaze data loaded for any session. Skipping gaze processing.")
        return None
    
    # Process gaze data
    try:
        if use_real_timestamps:
            print(f"Processing gaze data with real timestamps from column '{timestamp_column}'...")
        else:
            print(f"Processing gaze data with fixed frequency approach ({frequency} Hz)...")
        
        gaze_summary = process_multiple_gaze_sessions(
            session_ids=list(gaze_data_dict.keys()),
            gaze_data_dict=gaze_data_dict,
            use_real_timestamps=use_real_timestamps,
            timestamp_column=timestamp_column,
            fixed_frequency=frequency,
            save_output=save_output
        )
        print("Gaze data processing completed successfully.")
        return gaze_summary
    except Exception as e:
        print(f"Error processing gaze data: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None

def process_action_data(session_ids=None, save_output=True, output_dir="outputs", verbose=False):
    """
    Process all eHealth action data.
    
    Args:
        session_ids: List of session IDs to process (if None, process all available)
        save_output: Whether to save results to disk
        output_dir: Directory to save output files
        verbose: Whether to print detailed processing information
    
    Returns:
        pd.DataFrame: Summary of processed action data
    """
    print("\n" + "="*80)
    print("PROCESSING EHEALTH ACTION DATA")
    print("="*80)
    
    if session_ids is None:
        session_ids = load_session_ids()
    elif isinstance(session_ids, str):
        session_ids = [session_ids]  # Convert single session ID to list
    
    if not session_ids:
        print("No session IDs found for processing.")
        return None
    
    print(f"Found {len(session_ids)} sessions to process.")
    
    # Load action data for all sessions
    action_data_dict = {}
    for session_id in session_ids:
        try:
            action_df = load_action_data(session_id)
            if action_df is not None:
                if 'timestamp' in action_df.columns:
                    action_df['timestamp'] = pd.to_datetime(action_df['timestamp'])
                action_data_dict[session_id] = action_df
                if verbose:
                    print(f"Loaded {len(action_df)} action events for session {session_id}")
                    if not action_df.empty:
                        action_types = action_df['type'].unique()
                        print(f"  Action types: {', '.join(action_types[:5])}" + 
                              ("..." if len(action_types) > 5 else ""))
            else:
                print(f"No action data found for session {session_id}")
        except Exception as e:
            print(f"Error loading action data for session {session_id}: {e}")
    
    if not action_data_dict:
        print("No action data loaded for any session. Skipping action processing.")
        return None
    
    # Process action data
    try:
        print("Processing action data...")
        action_summary = process_multiple_sessions(
            session_ids=list(action_data_dict.keys()),
            action_data_dict=action_data_dict,
            save_output=save_output,
            base_name="ehealth_action_summary"
        )
        print("Action data processing completed successfully.")
        return action_summary
    except Exception as e:
        print(f"Error processing action data: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None

def main():
    """Main function to run the eHealth data processing pipeline."""
    start_time = datetime.now()
    
    # Parse command line arguments
    args = parse_arguments()
    
    if args.verbose:
        print(f"Starting eHealth data processing at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {args.output_dir}")
    
    # Ensure output directories exist
    ensure_output_directory(args.output_dir)
    
    try:
        # Get list of session IDs to process
        if args.session_id:
            session_ids = [args.session_id]
            if args.verbose:
                print(f"Processing single session: {args.session_id}")
        else:
            session_ids = load_session_ids()
            if args.verbose:
                print(f"Found {len(session_ids)} sessions to process")
                
        if len(session_ids) == 0:
            print("No session IDs found. Check if the input directory contains valid data.")
            return
            
        # Process data based on command line flags
        gaze_summary = None
        action_summary = None
        
        # Process gaze data if not action-only
        if not args.action_only:
            gaze_start = datetime.now()
            if args.verbose:
                print(f"\nStarting gaze data processing at {gaze_start.strftime('%Y-%m-%d %H:%M:%S')}")
                if args.real_timestamps:
                    print(f"Using real timestamps with column: {args.timestamp_column}")
                else:
                    print(f"Using fixed frequency approach at {args.frequency} Hz")
            
            try:
                gaze_summary = process_ehealth_data(
                    session_ids, 
                    frequency=args.frequency,
                    use_real_timestamps=args.real_timestamps,
                    timestamp_column=args.timestamp_column,
                    output_dir=args.output_dir,
                    save_output=not args.no_save,
                    verbose=args.verbose
                )
                
                gaze_end = datetime.now()
                gaze_duration = (gaze_end - gaze_start).total_seconds()
                
                if args.verbose:
                    if gaze_summary is not None:
                        print(f"Gaze processing completed in {gaze_duration:.2f} seconds")
                        print(f"Processed {len(gaze_summary)} sessions with fixation data")
                    else:
                        print("Gaze processing completed, but no summary data was generated")
            except Exception as e:
                print(f"ERROR processing gaze data: {str(e)}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        # Process action data if not gaze-only
        if not args.gaze_only:
            action_start = datetime.now()
            if args.verbose:
                print(f"\nStarting action data processing at {action_start.strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                action_summary = process_action_data(
                    session_ids,
                    output_dir=args.output_dir,
                    save_output=not args.no_save,
                    verbose=args.verbose
                )
                
                action_end = datetime.now()
                action_duration = (action_end - action_start).total_seconds()
                
                if args.verbose:
                    if action_summary is not None:
                        print(f"Action processing completed in {action_duration:.2f} seconds")
                        print(f"Processed {len(action_summary)} session action logs")
                    else:
                        print("Action processing completed, but no summary data was generated")
            except Exception as e:
                print(f"ERROR processing action data: {str(e)}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        # Print summary information
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        print("\nProcessing Summary:")
        print(f"Total processing time: {total_duration:.2f} seconds")
        
        if gaze_summary is not None:
            print(f"Gaze data: {len(gaze_summary)} sessions processed")
            if not args.no_save:
                print(f"Fixation files saved to: {os.path.join(args.output_dir, 'fixations')}")
                print(f"Gaze summary saved to: {os.path.join(args.output_dir, 'gaze_summary.csv')}")
        
        if action_summary is not None:
            print(f"Action data: {len(action_summary)} sessions processed")
            if not args.no_save:
                print(f"Action summary saved to: {os.path.join(args.output_dir, 'action_summary.csv')}")
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())




