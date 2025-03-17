import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Union, List, Tuple, Optional


def extract_validation_metrics(action_df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """
    Extract validation metrics from action log, finding the last chain of 5 validations.
    
    Args:
        action_df (pd.DataFrame): Raw action log data
        
    Returns:
        Tuple[int, pd.DataFrame]: (number of validation repetitions, validation points data)
    """
    # Get all validation events
    validation_events = action_df[action_df['type'] == 'gaze-validation'].copy()
    if validation_events.empty:
        return 0, pd.DataFrame()
    
    # Parse the validation values from JSON strings
    validation_events['metrics'] = validation_events['value'].apply(
        lambda x: json.loads(x) if x and x.strip() else {})
    validation_events['accuracy'] = validation_events['metrics'].apply(
        lambda x: x.get('accuracy', float('nan')))
    validation_events['precision'] = validation_events['metrics'].apply(
        lambda x: x.get('precision', float('nan')))
    validation_events['points'] = validation_events['metrics'].apply(
        lambda x: x.get('gazePointCount', float('nan')))
    
    # Count total validation attempts (divided by 5 points)
    total_validations = len(validation_events)
    validation_rounds = (total_validations // 5) if total_validations >= 5 else 0
    
    # Get the last 5 validation points
    last_validations = validation_events.tail(5)
    
    return validation_rounds, last_validations

def extract_l1_token_data(action_df: pd.DataFrame) -> Dict[str, Dict[str, Union[str, int, float]]]:
    """
    Extract L1 token data including start time, end time, and response value.
    
    L1 tokens follow pattern t_XX_y where XX is a number and y is gender (m/f).
    Each token has start event (L1_start) and response event (L1_response).
    
    Args:
        action_df (pd.DataFrame): Action log DataFrame
        
    Returns:
        Dict: Dictionary with token data in format:
            {
                "t_01_m": {
                    "startTime": timestamp,
                    "endTime": timestamp, 
                    "value": int
                },
                ...
            }
    """
    # Define all possible tokens
    tokens = [f"t_{i:02d}_{g}" for i in range(1, 33) for g in ['m', 'f']]
    
    # Filter to only tokens that appear in the data
    valid_tokens = []
    for token in tokens:
        # Check if this token appears in any value field
        if action_df['value'].str.contains(token, regex=False).any():
            valid_tokens.append(token)
    
    # Initialize token data dictionary
    token_data = {token: {"startTime": None, "endTime": None, "value": None} for token in valid_tokens}
    
    # Ensure timestamp is datetime
    if 'timestamp' in action_df.columns:
        action_df['timestamp'] = pd.to_datetime(action_df['timestamp'])
    
    # Process start events
    start_events = action_df[action_df['type'] == 'L1_start']
    for _, row in start_events.iterrows():
        token = row['value'].strip()
        if token in token_data:
            token_data[token]['startTime'] = row['timestamp']
    
    # Process response events
    response_events = action_df[action_df['type'] == 'L1_response']
    for _, row in response_events.iterrows():
        # Parse the response value format: "t_XX_y; value; time_ms"
        parts = row['value'].split(';')
        if len(parts) >= 2:
            token = parts[0].strip()
            if token in token_data:
                token_data[token]['endTime'] = row['timestamp']
                try:
                    token_data[token]['value'] = int(parts[1].strip())
                except ValueError:
                    # Handle case where value might not be a valid integer
                    token_data[token]['value'] = None
    
    return token_data

def extract_l2_token_data(action_df: pd.DataFrame) -> Dict[str, Dict[str, Union[str, int, float]]]:
    """
    Extract L2 token data including start time, end time, and response value.
    
    L2 tokens are simple numbers (1-16).
    Each token has start event (L2_start) and response event (L2_response).
    
    Args:
        action_df (pd.DataFrame): Action log DataFrame
        
    Returns:
        Dict: Dictionary with token data in format:
            {
                "1": {
                    "startTime": timestamp,
                    "endTime": timestamp, 
                    "value": int
                },
                ...
            }
    """
    # Define all possible tokens for L2 (1-16)
    possible_tokens = [str(i) for i in range(1, 17)]
    
    # Filter to only tokens that appear in the data
    valid_tokens = []
    for token in possible_tokens:
        # Get L2 events for this token
        l2_events = action_df[
            (action_df['type'].str.startswith('L2_')) & 
            (action_df['value'].str.startswith(token + ';') | (action_df['value'] == token))
        ]
        if not l2_events.empty:
            valid_tokens.append(token)
    
    # Initialize token data dictionary
    token_data = {token: {"startTime": None, "endTime": None, "value": None} for token in valid_tokens}
    
    # Ensure timestamp is datetime
    if 'timestamp' in action_df.columns:
        action_df['timestamp'] = pd.to_datetime(action_df['timestamp'])
    
    # Process start events
    start_events = action_df[action_df['type'] == 'L2_start']
    for _, row in start_events.iterrows():
        token = row['value'].strip()
        if token in token_data:
            token_data[token]['startTime'] = row['timestamp']
    
    # Process response events
    response_events = action_df[action_df['type'] == 'L2_response']
    for _, row in response_events.iterrows():
        # Parse the response value format: "token; value; time_ms"
        parts = row['value'].split(';')
        if len(parts) >= 2:
            token = parts[0].strip()
            if token in token_data:
                token_data[token]['endTime'] = row['timestamp']
                try:
                    token_data[token]['value'] = int(parts[1].strip())
                except ValueError:
                    # Handle case where value might not be a valid integer
                    token_data[token]['value'] = None
    
    return token_data

def extract_l3_token_data(action_df: pd.DataFrame) -> Dict[str, Dict[str, Union[str, int, float]]]:
    """
    Extract L3 token data including start time and end time for each of the 4 ordered stops.
    
    L3 has exactly 4 ordered stops (O1, O2, O3, O4) with start and end events.
    The order is important and defined in the L3_init event.
    
    Args:
        action_df (pd.DataFrame): Action log DataFrame
        
    Returns:
        Dict: Dictionary with ordered stops data in format:
            {
                "O1": {
                    "token": value, (e.g. "3")
                    "startTime": timestamp,
                    "endTime": timestamp
                },
                "O2": {...},
                "O3": {...},
                "O4": {...}
            }
    """
    # Initialize with the 4 ordered stops
    ordered_stops = {f"O{i}": {"token": None, "startTime": None, "endTime": None} for i in range(1, 5)}
    
    # Ensure timestamp is datetime
    if 'timestamp' in action_df.columns:
        action_df['timestamp'] = pd.to_datetime(action_df['timestamp'])
    
    # Extract the sequence from L3_init
    init_event = action_df[action_df['type'] == 'L3_init']
    if not init_event.empty:
        # Get the tokens in the correct order
        tokens = init_event.iloc[0]['value'].split(';')
        # Store tokens in order (O1, O2, O3, O4)
        for i, token in enumerate(tokens[:4], 1):
            ordered_stops[f"O{i}"]["token"] = token.strip()
    
    # Extract start and end times
    for i in range(1, 5):
        stop_key = f"O{i}"
        token = ordered_stops[stop_key]["token"]
        
        if token:
            # Find start event for this token
            start_event = action_df[
                (action_df['type'] == 'L3_start') & 
                (action_df['value'] == token)
            ]
            if not start_event.empty:
                ordered_stops[stop_key]["startTime"] = start_event.iloc[0]['timestamp']
            
            # Find end event for this token
            end_event = action_df[
                (action_df['type'] == 'L3_end') & 
                (action_df['value'] == token)
            ]
            if not end_event.empty:
                ordered_stops[stop_key]["endTime"] = end_event.iloc[0]['timestamp']
    
    return ordered_stops

def create_action_summary_row(action_df: pd.DataFrame, session_id: str) -> pd.Series:
    """
    Create a summary row for a single action log file.
    
    Args:
        action_df (pd.DataFrame): Raw action log data
        session_id (str): Session identifier
        
    Returns:
        pd.Series: Summary row with metrics
        
    Note:
        Unlike the fixation processing, this function does not save individual files per session.
        All session data is collected and saved as a single summary file by the process_multiple_sessions function.
    """
    # Initialize with session ID
    summary = {
        'session_id': session_id,
        'validation_rounds': 0
    }
    
    # Extract linking ID from questionnaire-linking events
    linking_events = action_df[action_df['type'] == 'questionnaire-linking']
    if not linking_events.empty:
        summary['linking_id'] = linking_events.iloc[0]['value']
    else:
        summary['linking_id'] = None
    
    # Extract validation metrics
    validation_rounds, validation_data = extract_validation_metrics(action_df)
    
    # Add validation rounds count
    summary['validation_rounds'] = validation_rounds
    
    # If we have validation data, add metrics for each point
    if not validation_data.empty and len(validation_data) == 5:
        # Add individual point metrics
        for i, (_, row) in enumerate(validation_data.iterrows(), 1):
            summary[f'vali_point_{i}_accuracy'] = row['accuracy']
            summary[f'vali_point_{i}_precision'] = row['precision']
            summary[f'vali_point_{i}_points'] = row['points']
        
        # Add average metrics
        summary['vali_avg_accuracy'] = validation_data['accuracy'].mean()
        summary['vali_avg_precision'] = validation_data['precision'].mean()
        summary['vali_avg_points'] = validation_data['points'].mean()
    else:
        # Fill with NaN if we don't have complete validation data
        for i in range(1, 6):
            summary[f'vali_point_{i}_accuracy'] = float('nan')
            summary[f'vali_point_{i}_precision'] = float('nan')
            summary[f'vali_point_{i}_points'] = float('nan')
        summary['vali_avg_accuracy'] = float('nan')
        summary['vali_avg_precision'] = float('nan')
        summary['vali_avg_points'] = float('nan')
    
    # Extract L1 token data
    l1_token_data = extract_l1_token_data(action_df)
    
    # Add L1 token data to summary
    for token, data in l1_token_data.items():
        summary[f'L1__{token}__startTime'] = data['startTime']
        summary[f'L1__{token}__endTime'] = data['endTime']
        summary[f'L1__{token}__value'] = data['value']
    
    # Extract L2 token data
    l2_token_data = extract_l2_token_data(action_df)
    
    # Add L2 token data to summary
    for token, data in l2_token_data.items():
        summary[f'L2__{token}__startTime'] = data['startTime']
        summary[f'L2__{token}__endTime'] = data['endTime']
        summary[f'L2__{token}__value'] = data['value']
    
    # Extract L3 ordered stops data
    l3_ordered_stops = extract_l3_token_data(action_df)
    
    # Add L3 ordered stops to summary
    for stop_key, data in l3_ordered_stops.items():
        summary[f'L3__{stop_key}__L2token'] = data['token']  # The actual token number
        summary[f'L3__{stop_key}__startTime'] = data['startTime']
        summary[f'L3__{stop_key}__endTime'] = data['endTime']
        
        # Calculate duration if both times are available
        if data['startTime'] is not None and data['endTime'] is not None:
            duration = (data['endTime'] - data['startTime']).total_seconds()
            summary[f'L3__{stop_key}__duration'] = duration
        else:
            summary[f'L3__{stop_key}__duration'] = None
    
    return pd.Series(summary)

def standardize_datetime_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize datetime columns in a DataFrame to match the format used in process_fixation.py.
    
    Args:
        df (pd.DataFrame): DataFrame with potential datetime columns
        
    Returns:
        pd.DataFrame: DataFrame with standardized datetime format
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Find columns that likely contain datetime objects
    datetime_cols = []
    for col in result_df.columns:
        # Check if column name suggests it contains a timestamp
        if any(term in col.lower() for term in ['time', 'timestamp', 'date']):
            # Check the first non-null value
            non_null_vals = result_df[col].dropna()
            if len(non_null_vals) > 0:
                first_val = non_null_vals.iloc[0]
                if isinstance(first_val, (pd.Timestamp, datetime)):
                    datetime_cols.append(col)
    
    # Print debug info
    if datetime_cols:
        print(f"Standardizing datetime format for columns: {', '.join(datetime_cols)}")
    
    # Convert datetime columns to consistent format
    for col in datetime_cols:
        # Ensure columns are datetime objects first (in case they're strings)
        if result_df[col].dtype != 'datetime64[ns]':
            try:
                result_df[col] = pd.to_datetime(result_df[col])
            except:
                # If conversion fails, skip this column
                print(f"Warning: Could not convert column '{col}' to datetime")
                continue
    
    return result_df

def save_summary(summary_df: pd.DataFrame, base_name: str = "session_summaries") -> str:
    """
    Save the summary DataFrame to CSV with timestamp.
    
    Args:
        summary_df (pd.DataFrame): DataFrame to save
        base_name (str): Base name for the file
        
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)
    
    # Standardize datetime format to match process_fixation.py
    standardized_df = standardize_datetime_format(summary_df)
    
    # Save to CSV with datetime format matching process_fixation.py
    standardized_df.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S.%f')
    
    print(f"Saved summary with standardized datetime format to: {output_path}")
    return output_path

def process_multiple_sessions(session_ids: list[str], 
                            action_data_dict: Dict[str, pd.DataFrame],
                            save_output: bool = True,
                            base_name: str = "action_summary") -> pd.DataFrame:
    """
    Process multiple session files and create a summary table.
    
    Args:
        session_ids (list[str]): List of session IDs to process
        action_data_dict (Dict[str, pd.DataFrame]): Dictionary mapping session_ids to their action DataFrames
        save_output (bool, optional): Whether to save the output to CSV. Defaults to True.
        base_name (str, optional): Base name for the output file. Defaults to "action_summary".
        
    Returns:
        pd.DataFrame: Summary table where each row represents one session
        
    Note:
        Unlike fixation processing, action processing intentionally only creates a single
        summary file and NOT individual files per session. This design choice is made because:
        1. Action data is typically analyzed at the summary level
        2. The summary contains all necessary timing information for alignment with fixations
        3. This approach reduces file clutter when processing many sessions
    """
    summaries = []
    for session_id in session_ids:
        if session_id in action_data_dict:
            action_df = action_data_dict[session_id]
            summary_row = create_action_summary_row(action_df, session_id)
            summaries.append(summary_row)
    
    summary_df = pd.DataFrame(summaries)
    
    if not summary_df.empty:
        summary_df = summary_df.sort_values('session_id')
        
        # Save if requested - NOTE: We only save a single summary file, not individual files per session
        if save_output:
            output_path = save_summary(summary_df, base_name=base_name)
            print(f"\nSaved summary table to: {output_path}")
    
    return summary_df