import os
import pandas as pd
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import re
from datetime import datetime
import traceback
import sys
import time

def get_latest_session_summary() -> Optional[pd.DataFrame]:
    """
    Find and load the most recent session_summaries CSV file from the outputs directory.
    
    Returns:
        Optional[pd.DataFrame]: The loaded session summaries dataframe or None if no file found
    """
    # Find all session_summaries files
    output_dir = "outputs"
    pattern = os.path.join(output_dir, "*session_summaries*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Error: No session_summaries file found in {output_dir}")
        return None
    
    # Sort files by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]
    
    print(f"Loading latest session summary: {latest_file}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} session summaries")
        return df
    except Exception as e:
        print(f"Error loading session summary file: {str(e)}")
        traceback.print_exc()
        return None

def load_all_fixation_files(max_files: int = None) -> pd.DataFrame:
    """
    Load all fixation CSV files from the outputs/fixations directory and combine them.
    
    Args:
        max_files (int, optional): Maximum number of files to load for testing. Default is None (load all).
    
    Returns:
        pd.DataFrame: Combined fixation data from all files with added sessionId column
    """
    fixation_dir = os.path.join("outputs", "fixations")
    if not os.path.exists(fixation_dir):
        print(f"Error: Fixation directory not found: {fixation_dir}")
        return pd.DataFrame()
    
    # Find all CSV files in the fixations directory
    pattern = os.path.join(fixation_dir, "*.csv")
    fixation_files = glob.glob(pattern)
    
    if not fixation_files:
        print(f"Error: No fixation files found in {fixation_dir}")
        return pd.DataFrame()
    
    # Limit the number of files for testing if specified
    if max_files is not None and max_files > 0:
        fixation_files = fixation_files[:max_files]
        print(f"Limited to {max_files} files for testing")
    
    print(f"Found {len(fixation_files)} fixation files")
    
    all_fixations = []
    
    for file_path in fixation_files:
        try:
            # Extract session ID from filename (assumes format like 'fixations_1731573239729-3024_*.csv')
            file_name = os.path.basename(file_path)
            match = re.search(r'fixations_([^_]+)', file_name)
            
            if not match:
                print(f"Warning: Could not extract session ID from filename: {file_name}")
                continue
                
            session_id = match.group(1)
            
            # Load the fixation data
            fixation_df = pd.read_csv(file_path)
            
            # Add session ID column
            fixation_df['sessionId'] = session_id
            
            all_fixations.append(fixation_df)
            print(f"Loaded {len(fixation_df)} fixations from session {session_id}")
            
        except Exception as e:
            print(f"Error loading fixation file {file_path}: {str(e)}")
            continue
    
    if not all_fixations:
        print("No valid fixation data loaded")
        return pd.DataFrame()
    
    # Combine all fixation dataframes
    combined_fixations = pd.concat(all_fixations, ignore_index=True)
    print(f"Combined {len(combined_fixations)} fixations from all sessions")
    
    return combined_fixations

def find_matching_session(sessions_df: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """
    Find a matching session in the sessions DataFrame for the given session_id.
    
    Args:
        sessions_df (pd.DataFrame): DataFrame with session summaries
        session_id (str): Session ID to find
        
    Returns:
        pd.DataFrame: Matching session row or empty DataFrame if no match
    """
    # First, try direct match on session_id column
    session_row = sessions_df[sessions_df['session_id'] == session_id]
    
    # If no match, try with different case
    if session_row.empty:
        session_row = sessions_df[sessions_df['session_id'].str.lower() == session_id.lower()]
    
    # If still no match, try removing any trailing/leading whitespace
    if session_row.empty:
        session_row = sessions_df[sessions_df['session_id'].str.strip() == session_id.strip()]
    
    # If still no match, the session_id might be stored in a different format
    # Try matching on substring (e.g., the numeric part might match)
    if session_row.empty:
        # Extract numeric part of session_id (if any)
        match = re.search(r'(\d+)', session_id)
        if match:
            numeric_part = match.group(1)
            session_row = sessions_df[sessions_df['session_id'].str.contains(numeric_part, regex=True)]
            
            # If multiple matches, print warning and use the first one
            if len(session_row) > 1:
                print(f"Warning: Multiple session matches found for {session_id}, using first match: {session_row['session_id'].iloc[0]}")
                session_row = session_row.iloc[:1]
    
    return session_row

def prepare_stimulus_timeranges(session_row: pd.DataFrame) -> List[Tuple[datetime, datetime, str]]:
    """
    Extract and prepare stimulus timeranges from the session summary row.
    
    Args:
        session_row (pd.DataFrame): A single row from the session summaries dataframe
    
    Returns:
        List[Tuple[datetime, datetime, str]]: List of tuples with (start_time, end_time, stimulus_name)
    """
    timeranges = []
    
    # Get all columns that contain startTime and endTime for stimuli
    start_time_cols = [col for col in session_row.columns if '__startTime' in col]
    
    for start_col in start_time_cols:
        # Find corresponding end time column
        stimulus_id = start_col.split('__startTime')[0]
        end_col = stimulus_id + '__endTime'
        
        if end_col not in session_row.columns:
            continue
        
        try:
            # Extract stimulus name from column name (middle part between __ and __)
            parts = stimulus_id.split('__')
            if len(parts) < 2:
                continue
                
            stimulus_name = parts[1]
            
            # Get and parse start/end times, ensuring they're timezone-naive
            start_str = session_row[start_col].iloc[0]
            end_str = session_row[end_col].iloc[0]
            
            # Convert to datetime and remove timezone info
            start_time = pd.to_datetime(start_str).replace(tzinfo=None)
            end_time = pd.to_datetime(end_str).replace(tzinfo=None)
            
            timeranges.append((start_time, end_time, stimulus_name))
        except Exception as e:
            print(f"Error preparing timerange for stimulus {stimulus_id}: {str(e)}")
    
    # Sort by start time to optimize matching
    timeranges.sort(key=lambda x: x[0])
    
    return timeranges

def match_fixations_to_stimuli(fixations_df: pd.DataFrame, sessions_df: pd.DataFrame, max_sessions: int = None) -> pd.DataFrame:
    """
    Match fixations to stimuli based on timestamp ranges in the session summaries.
    Optimized for time-ordered fixations.
    
    Args:
        fixations_df (pd.DataFrame): Combined fixation data with sessionId column
        sessions_df (pd.DataFrame): Session summaries with stimuli timestamp ranges
        max_sessions (int, optional): Maximum number of sessions to process for testing. Default is None (process all).
    
    Returns:
        pd.DataFrame: Fixation data with added stimulus information
    """
    if fixations_df.empty or sessions_df.empty:
        print("Error: Empty fixation or session data")
        return pd.DataFrame()
    
    # Create a new column for stimulus and linking_id
    fixations_df['stimulus'] = None
    fixations_df['linking_id'] = None
    
    # Determine which timestamp column to use
    timestamp_candidates = ['timestamp', 'start_timestamp', 'end_timestamp']
    available_timestamp_cols = [col for col in timestamp_candidates if col in fixations_df.columns]
    
    if not available_timestamp_cols:
        print("Error: No timestamp column found in fixation data. Available columns:")
        print(fixations_df.columns.tolist())
        sys.exit("Aborting: No timestamp column found in fixation data")
    
    # Prioritize start_timestamp if available, otherwise use timestamp
    timestamp_col = 'start_timestamp' if 'start_timestamp' in available_timestamp_cols else available_timestamp_cols[0]
    print(f"Using timestamp column: {timestamp_col}")
    
    # Sample a few session IDs for testing if specified
    unique_sessions = fixations_df['sessionId'].unique()
    if max_sessions is not None and max_sessions > 0:
        if max_sessions < len(unique_sessions):
            unique_sessions = unique_sessions[:max_sessions]
            print(f"Limited to {max_sessions} sessions for testing")
    
    # Process each session
    for session_idx, session_id in enumerate(unique_sessions):
        start_time = time.time()
        print(f"Processing session {session_idx+1}/{len(unique_sessions)}: {session_id}")
        
        # Get session row from summaries
        session_row = find_matching_session(sessions_df, session_id)
        
        if session_row.empty:
            print(f"Error: Session ID {session_id} not found in session summaries")
            print("Available session IDs in session summaries:")
            print(sessions_df['session_id'].tolist()[:10])  # Show the first 10 for reference
            print("Skipping this session...")
            continue
        
        # Get all stimulus timeranges for this session
        timeranges = prepare_stimulus_timeranges(session_row)
        print(f"  Found {len(timeranges)} stimulus time ranges for this session")
        
        if not timeranges:
            print(f"  Warning: No valid timeranges found for session {session_id}")
            continue
        
        # Get linking_id for this session
        linking_id = None
        if 'linking_id' in session_row.columns:
            linking_id = session_row['linking_id'].iloc[0]
            print(f"  Found linking_id: {linking_id}")
        else:
            print("  Warning: No linking_id column found in session summary")
        
        # Filter fixations for this session
        session_fixations = fixations_df[fixations_df['sessionId'] == session_id].copy()
        print(f"  Processing {len(session_fixations)} fixations for this session")
        
        # Add linking_id to all fixations for this session
        if linking_id is not None:
            fixations_df.loc[fixations_df['sessionId'] == session_id, 'linking_id'] = linking_id
        
        # Ensure our timestamp column is a datetime
        if 'timestamp_dt' not in session_fixations.columns:
            session_fixations['timestamp_dt'] = session_fixations[timestamp_col].apply(
                lambda x: pd.to_datetime(x).replace(tzinfo=None) if isinstance(x, str) else 
                          (x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else x)
            )
        
        # Important: Sort fixations by timestamp for efficient processing
        session_fixations.sort_values('timestamp_dt', inplace=True)
        
        # Track the current timerange index to avoid checking all timeranges for each fixation
        current_tr_idx = 0
        match_count = 0
        
        # Process each fixation
        batch_size = 5000  # Process in larger batches for report only
        num_fixations = len(session_fixations)
        for i, (idx, fixation) in enumerate(session_fixations.iterrows()):
            # Progress reporting
            if i % batch_size == 0 or i == num_fixations - 1:
                print(f"  Processing fixation {i+1}/{num_fixations} ({((i+1)/num_fixations)*100:.1f}%)")
            
            fix_time = fixation['timestamp_dt']
            
            # Skip if timestamp couldn't be parsed
            if pd.isna(fix_time):
                continue
            
            # Find the appropriate timerange for this fixation
            matched = False
            
            # First, check if the current timerange still applies
            while current_tr_idx < len(timeranges):
                start_time_tr, end_time_tr, stimulus_name = timeranges[current_tr_idx]
                
                # If fixation is before current timerange, it doesn't match any
                if fix_time < start_time_tr:
                    break
                
                # If fixation is within current timerange, we found a match
                if start_time_tr <= fix_time <= end_time_tr:
                    fixations_df.at[idx, 'stimulus'] = stimulus_name
                    match_count += 1
                    matched = True
                    break
                
                # If fixation is after current timerange, move to next timerange
                if fix_time > end_time_tr:
                    current_tr_idx += 1
                    continue
            
            # If not matched with the optimized approach, fall back to checking all remaining timeranges
            if not matched:
                for tr_idx in range(current_tr_idx, len(timeranges)):
                    start_time_tr, end_time_tr, stimulus_name = timeranges[tr_idx]
                    if start_time_tr <= fix_time <= end_time_tr:
                        fixations_df.at[idx, 'stimulus'] = stimulus_name
                        match_count += 1
                        # Update current_tr_idx for next fixation
                        current_tr_idx = tr_idx
                        break
        
        elapsed = time.time() - start_time
        print(f"  Completed session {session_id} in {elapsed:.2f} seconds. Matched {match_count} fixations.")
    
    # Check if any fixations were not matched to a stimulus
    unmatched = fixations_df[fixations_df['stimulus'].isna()]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} fixations ({len(unmatched)/len(fixations_df)*100:.2f}%) could not be matched to a stimulus")
    else:
        print("All fixations were successfully matched to stimuli")
    
    # Check if any fixations were not matched to a linking ID
    unlinked = fixations_df[fixations_df['linking_id'].isna()]
    if not unlinked.empty:
        print(f"Warning: {len(unlinked)} fixations ({len(unlinked)/len(fixations_df)*100:.2f}%) could not be matched to a linking ID")
    
    return fixations_df

def process_fixation_insights(save_output: bool = True, max_files: int = 5, max_sessions: int = 2) -> pd.DataFrame:
    """
    Main function to process fixation insights.
    
    1. Load the latest session summaries CSV
    2. Load and combine all fixation data
    3. Match fixations to stimuli based on timestamp ranges
    4. Optionally save the enriched fixation data
    
    Args:
        save_output (bool): Whether to save the output to CSV
        max_files (int): Maximum number of fixation files to load for testing (None for all)
        max_sessions (int): Maximum number of sessions to process for testing (None for all)
    
    Returns:
        pd.DataFrame: Enriched fixation data with stimulus information
    """
    # Load the latest session summaries
    sessions_df = get_latest_session_summary()
    if sessions_df is None:
        print("Error: Failed to load session summaries")
        return pd.DataFrame()
    
    # Load and combine all fixation data
    fixations_df = load_all_fixation_files(max_files)
    if fixations_df.empty:
        print("Error: No fixation data found")
        return pd.DataFrame()
    
    # Match fixations to stimuli
    enriched_fixations = match_fixations_to_stimuli(fixations_df, sessions_df, max_sessions)
    
    # Save output if requested
    if save_output and not enriched_fixations.empty:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"fixation_insights_{timestamp}.csv")
        
        enriched_fixations.to_csv(output_file, index=False)
        print(f"Saved enriched fixation data to: {output_file}")
    
    return enriched_fixations 