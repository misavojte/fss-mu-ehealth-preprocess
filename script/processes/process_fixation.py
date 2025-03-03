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

def detect_fixations(gaze_df: pd.DataFrame, i2mc_options: Dict = None) -> pd.DataFrame:
    """
    Detect fixations in preprocessed gaze data using the I2MC algorithm.
    
    Args:
        gaze_df (pd.DataFrame): Preprocessed gaze data with columns:
            time_ms, x, y, missing
        i2mc_options (Dict, optional): Dictionary containing all I2MC algorithm parameters.
            If None, uses default values as specified below.
            
    Returns:
        pd.DataFrame: Detected fixations with their properties
    """
    # Setup default I2MC parameters if not provided
    if i2mc_options is None:
        i2mc_options = {
            # Required parameters
            'xres': 1920,              # screen width in pixels
            'yres': 1080,              # screen height in pixels
            'freq': 150.0,             # sampling rate in Hz
            'missingx': -1.0,          # value marking missing data
            'missingy': -1.0,          # value marking missing data
            'scrSz': [53, 30],         # screen size in cm [width, height]
            'disttoscreen': 65,        # distance to screen in cm
            
            # Interpolation parameters
            'windowtimeInterp': 0.1,   # max duration (s) for interpolation
            'edgeSampInterp': 2,       # samples needed at edges for interpolation
            'maxdisp': None,           # max displacement during interpolation (computed if None)
            
            # Clustering parameters
            'windowtime': 0.2,         # time window for 2-means clustering
            'steptime': 0.02,          # time window shift
            'downsamples': [2, 5, 10], # downsample levels (can be empty: [])
            'downsampFilter': False,   # use filter when downsampling
            'chebyOrder': 8,           # order of Chebyshev filter
            'maxerrors': 100,          # max clustering errors allowed
            
            # Fixation parameters
            'cutoffstd': 2.0,         # number of STD for fixation threshold
            'onoffsetThresh': 3.0,     # threshold for walk-back
            'maxMergeDist': 30.0,      # max distance (pixels) for merging
            'maxMergeTime': 30.0,      # max time (ms) for merging
            'minFixDur': 40.0,         # minimum fixation duration (ms)
        }
    else:
        # Ensure required parameters are present
        required_params = ['xres', 'yres', 'freq', 'missingx', 'missingy']
        for param in required_params:
            if param not in i2mc_options:
                raise ValueError(f"Required parameter '{param}' missing from i2mc_options")
    
    try:
        # Print data quality stats
        valid_samples = (~gaze_df['missing']).sum()
        total_samples = len(gaze_df)
        print(f"Data quality: {valid_samples}/{total_samples} valid samples ({valid_samples/total_samples*100:.1f}%)")
        
        # Print the real sampling rate
        mean_sampling_rate = 1000/np.mean(np.diff(gaze_df['time_ms']))
        print(f"Real Mean Sampling Rate: {mean_sampling_rate:.1f} Hz")
        
        # Prepare data in the format expected by I2MC
        data_df = pd.DataFrame()
        data_df['time'] = gaze_df['time_ms']
        data_df['average_X'] = gaze_df['x']
        data_df['average_Y'] = gaze_df['y']
        data_df['missing'] = gaze_df['missing']
        
        # Import I2MC function
        sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
        from I2MC import I2MC
        
        # Call I2MC with DataFrame input
        fix_results = I2MC(data_df, i2mc_options)
        
        # Handle tuple return type (fixation data, other info)
        if isinstance(fix_results, tuple):
            if len(fix_results) > 0:
                fix_data = fix_results[0]
                if isinstance(fix_data, dict) and 'startT' in fix_data:
                    print(f"Detected {len(fix_data['startT'])} fixations")
                    
                    # Create fixation DataFrame
                    fixations = pd.DataFrame({
                        'start_time': fix_data['startT'],
                        'end_time': fix_data['endT'],
                        'duration': fix_data['dur'],
                        'x_position': fix_data['xpos'],
                        'y_position': fix_data['ypos'],
                        'is_flanked_by_missing': fix_data['flankdataloss'],
                        'fraction_interpolated': fix_data['fracinterped']
                    })
                    
                    return fixations
        
        # If we get here, something went wrong
        print("No fixations detected or unexpected result format")
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
