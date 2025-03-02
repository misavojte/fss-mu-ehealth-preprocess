import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Union, List, Tuple


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
    validation_events['metrics'] = validation_events['value'].apply(json.loads)
    validation_events['accuracy'] = validation_events['metrics'].apply(lambda x: x['accuracy'])
    validation_events['precision'] = validation_events['metrics'].apply(lambda x: x['precision'])
    validation_events['points'] = validation_events['metrics'].apply(lambda x: x['gazePointCount'])
    
    # Count total validation attempts (divided by 5 points)
    total_validations = len(validation_events)
    validation_rounds = (total_validations // 5) if total_validations >= 5 else 0
    
    # Get the last 5 validation points
    last_validations = validation_events.tail(5)
    
    return validation_rounds, last_validations

def create_action_summary_row(action_df: pd.DataFrame, session_id: str) -> pd.Series:
    """
    Create a summary row for a single action log file.
    
    Args:
        action_df (pd.DataFrame): Raw action log data
        session_id (str): Session identifier
        
    Returns:
        pd.Series: Summary row with validation metrics
    """
    # Initialize with session ID
    summary = {
        'session_id': session_id,
        'validation_rounds': 0
    }
    
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
    
    return pd.Series(summary)

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
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    return output_path

def process_multiple_sessions(session_ids: list[str], 
                            action_data_dict: Dict[str, pd.DataFrame],
                            save_output: bool = True) -> pd.DataFrame:
    """
    Process multiple session files and create a summary table.
    
    Args:
        session_ids (list[str]): List of session IDs to process
        action_data_dict (Dict[str, pd.DataFrame]): Dictionary mapping session_ids to their action DataFrames
        save_output (bool, optional): Whether to save the output to CSV. Defaults to True.
        
    Returns:
        pd.DataFrame: Summary table where each row represents one session
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
        
        # Save if requested
        if save_output:
            output_path = save_summary(summary_df)
            print(f"\nSaved summary table to: {output_path}")
    
    return summary_df