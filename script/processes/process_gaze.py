import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import sys
import importlib.util
import traceback

# Import I2MC - we'll assume it's in the project root 
# or add it to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    import I2MC
except ImportError:
    print("Warning: I2MC module not found. Fixation detection will not work.")

def preprocess_gaze_data(gaze_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the ehealth-gaze data to make it compatible with I2MC.
    
    Args:
        gaze_df (pd.DataFrame): Raw gaze data from ehealth-gaze with columns:
            id, timestamp, ISOtimestamp, sessionId, gazeSessionId, 
            xL, xLScreenRelative, xR, xRScreenRelative, 
            yL, yLScreenRelative, yR, yRScreenRelative, 
            validityL, validityR, aoi
        
    Returns:
        pd.DataFrame: Preprocessed gaze data ready for fixation detection
    """
    # Make a copy to avoid modifying the original
    df = gaze_df.copy()
    
    # Convert columns to appropriate types
    df['timestamp'] = pd.to_datetime(df['ISOtimestamp'])
    
    # Calculate time in milliseconds from the start of recording
    start_time = df['timestamp'].min()
    df['time_ms'] = (df['timestamp'] - start_time).dt.total_seconds() * 1000
    
    # Average the left and right eye coordinates when both are valid
    # Otherwise use the valid eye's data
    df['validityL'] = pd.to_numeric(df['validityL'], errors='coerce')
    df['validityR'] = pd.to_numeric(df['validityR'], errors='coerce')
    
    # Create x and y coordinates (average of left and right when both valid)
    df['x'] = float('nan')  # Initialize with NaN
    df['y'] = float('nan')  # Initialize with NaN
    
    # Both eyes valid: use average
    both_valid = (df['validityL'] > 0) & (df['validityR'] > 0)
    df.loc[both_valid, 'x'] = (df.loc[both_valid, 'xL'].astype(float) + 
                              df.loc[both_valid, 'xR'].astype(float)) / 2
    df.loc[both_valid, 'y'] = (df.loc[both_valid, 'yL'].astype(float) + 
                              df.loc[both_valid, 'yR'].astype(float)) / 2
    
    # Only left eye valid: use left eye
    left_valid = (df['validityL'] > 0) & (df['validityR'] <= 0)
    df.loc[left_valid, 'x'] = df.loc[left_valid, 'xL'].astype(float)
    df.loc[left_valid, 'y'] = df.loc[left_valid, 'yL'].astype(float)
    
    # Only right eye valid: use right eye
    right_valid = (df['validityL'] <= 0) & (df['validityR'] > 0)
    df.loc[right_valid, 'x'] = df.loc[right_valid, 'xR'].astype(float)
    df.loc[right_valid, 'y'] = df.loc[right_valid, 'yR'].astype(float)
    
    # Create missing flag (True when both eyes are invalid)
    df['missing'] = ~(both_valid | left_valid | right_valid)
    
    # Ensure we have all required columns for I2MC
    required_cols = ['time_ms', 'x', 'y', 'missing']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing after preprocessing")
    
    # Select only required columns
    processed_df = df[required_cols].copy()
    
    return processed_df

def detect_fixations(gaze_df: pd.DataFrame, screen_params: Dict = None) -> pd.DataFrame:
    """
    Detect fixations in preprocessed gaze data using the I2MC algorithm.
    
    Args:
        gaze_df (pd.DataFrame): Preprocessed gaze data with columns:
            time_ms, x, y, missing
        screen_params (Dict, optional): Screen parameters like size and distance.
            Defaults to reasonable values if not provided.
            
    Returns:
        pd.DataFrame: Detected fixations with their properties
    """
    # Setup default screen parameters if not provided
    if screen_params is None:
        # Typical screen parameters - adjust as needed for your setup
        screen_params = {
            'resolution': [1920, 1080],  # Width, height in pixels
            'size': [53, 30],            # Width, height in cm
            'distance': 65,              # Participant distance from screen in cm
        }
    
    # Import I2MC function
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
    from I2MC import I2MC
    
    try:
        # Check if coordinates are fractional (0-1) or pixels
        x_max = gaze_df['x'].max()
        y_max = gaze_df['y'].max()
        
        # If coordinates are fractional (between 0-1), convert to pixels
        if x_max <= 1.0 and y_max <= 1.0:
            print("Detected fractional coordinates, converting to pixels...")
            gaze_df['x'] = gaze_df['x'] * screen_params['resolution'][0]
            gaze_df['y'] = gaze_df['y'] * screen_params['resolution'][1]
        
        # Replace NaN values with a specific missing value indicator (-1)
        missing_value = -1.0
        gaze_df['x'] = gaze_df['x'].fillna(missing_value)
        gaze_df['y'] = gaze_df['y'].fillna(missing_value)
        
        # Prepare data in the format expected by I2MC
        data_df = pd.DataFrame()
        
        # Add time column (expected to be in milliseconds)
        data_df['time'] = gaze_df['time_ms']
        
        # Add average eye position (this is what we have in our preprocessed data)
        data_df['average_X'] = gaze_df['x']
        data_df['average_Y'] = gaze_df['y']
        
        # Set missing flag
        data_df['missing'] = gaze_df['missing']
        
        # Print data quality stats
        valid_samples = (~gaze_df['missing']).sum()
        total_samples = len(gaze_df)
        print(f"Data quality: {valid_samples}/{total_samples} valid samples ({valid_samples/total_samples*100:.1f}%)")
        
        # Create options dictionary with exact parameter names from the I2MC code
        options = {
            # Required parameters
            'xres': float(screen_params['resolution'][0]),
            'yres': float(screen_params['resolution'][1]),
            'missingx': missing_value,
            'missingy': missing_value,
            'freq': 150.0,  # sampling rate in Hz
            
            # Optional parameters for visual angle calculations
            'scrSz': screen_params['size'],
            'disttoscreen': screen_params['distance'],
            
            # Additional parameters for the algorithm with correct parameter names
            'maxdisp': 0.8,               # maximum displacement during interpolation
            'windowtimeInterp': 0.1,      # time window for interpolation (seconds)
            'maxerrors': 100,             # maximum number of errors allowed in interpolation window
            'downsamples': [2, 5, 10],    # list of factors to use for downsampling
            'downsampFilter': 3,          # filter window size for downsampling
            'minFixDur': 40               # minimum fixation duration in ms
        }
        
        # Print the options being used
        print("Using I2MC with parameters:")
        for key, value in options.items():
            print(f"  {key}: {value}")
        
        # Call I2MC with DataFrame input
        fix_results = I2MC(data_df, options)
        
        # Print result type for debugging
        print(f"I2MC returned result of type: {type(fix_results)}")
        
        # Check if any fixations were detected - HANDLE TUPLE RETURN TYPE
        if fix_results is None:
            print("No fixation results returned by I2MC algorithm")
            return pd.DataFrame()
        
        # Handle tuple return type (fixation data, other info)
        if isinstance(fix_results, tuple):
            # Check length of tuple for safety
            if len(fix_results) > 0:
                # First element should be the fixation data dictionary
                fix_data = fix_results[0]
                print(f"First tuple element is of type: {type(fix_data)}")
                
                # If first element is a dictionary with fixation data
                if isinstance(fix_data, dict) and 'startT' in fix_data:
                    print(f"Detected {len(fix_data['startT'])} fixations")
                    
                    # Create fixation DataFrame
                    fixations = pd.DataFrame({
                        'start_time': fix_data['startT'],
                        'end_time': fix_data['endT'],
                        'duration': fix_data['dur'],
                        'x_position': fix_data['xpos'],
                        'y_position': fix_data['ypos']
                    })
                    
                    # Success! Return the fixations
                    return fixations
                else:
                    # First element didn't have expected structure
                    if isinstance(fix_data, dict):
                        print(f"Dictionary keys: {list(fix_data.keys())}")
                    return pd.DataFrame()
            else:
                # Empty tuple returned
                print("Empty tuple returned from I2MC")
                return pd.DataFrame()
        
        # Handle other return types as a fallback
        elif isinstance(fix_results, dict):
            # Original dictionary handling (in case return type changes in the future)
            if 'fix' in fix_results:
                fix_data = fix_results['fix']
                if isinstance(fix_data, dict) and 'startT' in fix_data:
                    print(f"Detected {len(fix_data['startT'])} fixations")
                    
                    # Create fixation DataFrame
                    fixations = pd.DataFrame({
                        'start_time': fix_data['startT'],
                        'end_time': fix_data['endT'],
                        'duration': fix_data['dur'],
                        'x_position': fix_data['xpos'],
                        'y_position': fix_data['ypos']
                    })
                    
                    return fixations
            return pd.DataFrame()
        
        # Unexpected return type - provide information for debugging
        else:
            print(f"Unexpected result type from I2MC: {type(fix_results)}")
            return pd.DataFrame()
        
    except Exception as e:
        print(f"Error in fixation detection: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def extract_fixation_metrics(fixations: pd.DataFrame) -> Dict[str, float]:
    """
    Extract summary metrics from detected fixations.
    
    Args:
        fixations (pd.DataFrame): DataFrame containing fixation data
        
    Returns:
        Dict[str, float]: Dictionary of summary metrics
    """
    metrics = {}
    
    if fixations.empty:
        # Return default values if no fixations detected
        metrics = {
            'total_fixations': 0,
            'mean_fixation_duration': 0,
            'median_fixation_duration': 0,
            'min_fixation_duration': 0,
            'max_fixation_duration': 0,
            'total_fixation_time': 0,
            'fixation_rate': 0  # fixations per second
        }
        return metrics
    
    # Ensure 'duration' column exists
    if 'duration' not in fixations.columns:
        if 'start_time' in fixations.columns and 'end_time' in fixations.columns:
            fixations['duration'] = fixations['end_time'] - fixations['start_time']
        else:
            print("Warning: Cannot calculate duration - missing start/end times")
            # Return limited metrics
            metrics['total_fixations'] = len(fixations)
            return metrics
    
    # Calculate basic metrics
    metrics['total_fixations'] = len(fixations)
    metrics['mean_fixation_duration'] = fixations['duration'].mean()
    metrics['median_fixation_duration'] = fixations['duration'].median()
    metrics['min_fixation_duration'] = fixations['duration'].min()
    metrics['max_fixation_duration'] = fixations['duration'].max()
    metrics['total_fixation_time'] = fixations['duration'].sum()
    
    # Calculate fixation rate (fixations per second)
    if metrics['total_fixation_time'] > 0:
        # Convert total_fixation_time from ms to seconds
        metrics['fixation_rate'] = metrics['total_fixations'] / (metrics['total_fixation_time'] / 1000)
    else:
        metrics['fixation_rate'] = 0
        
    # Calculate spatial metrics if position data is available
    if 'x_position' in fixations.columns and 'y_position' in fixations.columns:
        # Dispersion (standard deviation of fixation positions)
        metrics['x_dispersion'] = fixations['x_position'].std()
        metrics['y_dispersion'] = fixations['y_position'].std()
        
        # Calculate distances between consecutive fixations (saccade amplitudes)
        if len(fixations) > 1:
            x_diffs = np.diff(fixations['x_position'])
            y_diffs = np.diff(fixations['y_position'])
            saccade_amplitudes = np.sqrt(x_diffs**2 + y_diffs**2)
            
            metrics['mean_saccade_amplitude'] = np.mean(saccade_amplitudes)
            metrics['median_saccade_amplitude'] = np.median(saccade_amplitudes)
            metrics['max_saccade_amplitude'] = np.max(saccade_amplitudes)
            
    return metrics

def create_gaze_summary_row(gaze_df: pd.DataFrame, session_id: str) -> pd.Series:
    """
    Create a summary row for gaze data, including fixation metrics.
    
    Args:
        gaze_df (pd.DataFrame): Raw gaze data
        session_id (str): Session identifier
        
    Returns:
        pd.Series: Summary row with metrics
    """
    # Initialize with session ID
    summary = {
        'session_id': session_id
    }
    
    try:
        # Preprocess the gaze data
        preprocessed_df = preprocess_gaze_data(gaze_df)
        
        # Basic gaze data metrics
        summary.update({
            'gaze_sample_count': len(preprocessed_df),
            'gaze_missing_count': preprocessed_df['missing'].sum(),
            'gaze_missing_percent': (preprocessed_df['missing'].sum() / len(preprocessed_df)) * 100 if len(preprocessed_df) > 0 else float('nan')
        })
        
        # Detect fixations
        fixations = detect_fixations(preprocessed_df)
        
        # Extract fixation metrics
        fixation_metrics = extract_fixation_metrics(fixations)
        
        # Add fixation metrics to summary with 'fix_' prefix
        for key, value in fixation_metrics.items():
            summary[f'fix_{key}'] = value
        
        # Save fixations to CSV for later analysis
        output_dir = os.path.join("outputs", "fixations")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"fixations_{session_id}_{timestamp}.csv")
        fixations.to_csv(output_path, index=False)
        
    except Exception as e:
        print(f"Error processing gaze data for session {session_id}: {str(e)}")
        # Add error information to summary
        summary['processing_error'] = str(e)
    
    return pd.Series(summary)

def process_multiple_gaze_sessions(session_ids: List[str], 
                                  gaze_data_dict: Dict[str, pd.DataFrame],
                                  save_output: bool = True) -> pd.DataFrame:
    """
    Process multiple gaze data sessions and create a summary table.
    
    Args:
        session_ids (List[str]): List of session IDs to process
        gaze_data_dict (Dict[str, pd.DataFrame]): Dictionary mapping session_ids to their gaze DataFrames
        save_output (bool, optional): Whether to save the output to CSV. Defaults to True.
        
    Returns:
        pd.DataFrame: Summary table where each row represents one session
    """
    summaries = []
    for session_id in session_ids:
        if session_id in gaze_data_dict:
            gaze_df = gaze_data_dict[session_id]
            summary_row = create_gaze_summary_row(gaze_df, session_id)
            summaries.append(summary_row)
    
    summary_df = pd.DataFrame(summaries)
    
    if not summary_df.empty:
        summary_df = summary_df.sort_values('session_id')
        
        # Save if requested
        if save_output:
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"gaze_summary_{timestamp}.csv")
            
            summary_df.to_csv(output_path, index=False)
            print(f"\nSaved gaze summary table to: {output_path}")
    
    return summary_df
