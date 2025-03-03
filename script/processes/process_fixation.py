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

def add_hypothetical_elapsed_time(gaze_df: pd.DataFrame, sampling_freq_hz: float = 150.0) -> pd.DataFrame:
    """
    Adds a hypothetical elapsed time column based on the true frequency of data.
    This is useful when timestamps are affected by buffering or other inconsistencies.
    
    Args:
        gaze_df (pd.DataFrame): Gaze data DataFrame with an index
        sampling_freq_hz (float): The expected sampling frequency in Hz
            
    Returns:
        pd.DataFrame: DataFrame with added hypotheticalElapsedTime column in milliseconds
    """
    # Calculate time interval in milliseconds based on the sampling frequency
    time_interval_ms = 1000.0 / sampling_freq_hz
    
    # Create an array of hypothetical timestamps at regular intervals
    num_samples = len(gaze_df)
    hypothetical_time = np.arange(0, num_samples * time_interval_ms, time_interval_ms)
    
    # Add the column to the dataframe
    df = gaze_df.copy()
    df['hypotheticalElapsedTime'] = hypothetical_time[:num_samples]
    
    return df

def preprocess_gaze_data(gaze_df: pd.DataFrame, use_hypothetical_time: bool = False, sampling_freq_hz: float = None) -> pd.DataFrame:
    """
    Preprocess the ehealth-gaze data to make it compatible with I2MC.
    
    Args:
        gaze_df (pd.DataFrame): Raw gaze data from ehealth-gaze with columns:
            id, timestamp, ISOtimestamp, sessionId, gazeSessionId, 
            xL, xLScreenRelative, xR, xRScreenRelative, 
            yL, yLScreenRelative, yR, yRScreenRelative, 
            validityL, validityR, aoi
        use_hypothetical_time (bool): If True, use hypothetical elapsed time
            instead of actual timestamps for time_ms
        sampling_freq_hz (float, optional): The expected sampling frequency in Hz,
            used when use_hypothetical_time is True. If None, it will be
            calculated from the data.
        
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
    
    # Calculate the real frequency if not provided
    if sampling_freq_hz is None:
        sampling_freq_hz = calculate_real_frequency(df)
        print(f"Calculated sampling frequency: {sampling_freq_hz:.2f} Hz")
    
    # Add hypothetical elapsed time if requested
    if use_hypothetical_time:
        df = add_hypothetical_elapsed_time(df, sampling_freq_hz)
        # Replace time_ms with hypothetical time for use with I2MC
        df['time_ms'] = df['hypotheticalElapsedTime']
    
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
    
    # Select required columns and hypotheticalElapsedTime if it exists
    columns_to_keep = required_cols.copy()
    if 'hypotheticalElapsedTime' in df.columns:
        columns_to_keep.append('hypotheticalElapsedTime')
    
    processed_df = df[columns_to_keep].copy()
    
    # Store the calculated/provided frequency as an attribute
    processed_df.attrs['sampling_freq_hz'] = sampling_freq_hz
    
    return processed_df

def detect_fixations(gaze_df: pd.DataFrame, i2mc_options: Dict = None, use_hypothetical_time: bool = False) -> pd.DataFrame:
    """
    Detect fixations in preprocessed gaze data using the I2MC algorithm.
    
    Args:
        gaze_df (pd.DataFrame): Preprocessed gaze data with columns:
            time_ms, x, y, missing
        i2mc_options (Dict, optional): Dictionary containing all I2MC algorithm parameters.
            If None, uses default values as specified below.
        use_hypothetical_time (bool): If True, indicates that time_ms is based on
            hypothetical elapsed time rather than actual timestamps
            
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
            'downsamples': [],         # downsample levels (empty list to disable downsampling)
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
        if not use_hypothetical_time:
            mean_sampling_rate = 1000/np.mean(np.diff(gaze_df['time_ms']))
            print(f"Real Mean Sampling Rate: {mean_sampling_rate:.1f} Hz")
        else:
            print(f"Using hypothetical sampling rate: {i2mc_options['freq']} Hz")
        
        # Calculate suitable downsampling levels that are divisors of the sampling frequency
        freq = i2mc_options['freq']
        
        # Only generate downsampling levels if the original list is not empty
        if 'downsamples' not in i2mc_options or not isinstance(i2mc_options['downsamples'], list):
            i2mc_options['downsamples'] = []
        elif len(i2mc_options['downsamples']) > 0:
            # Find divisors of the sampling frequency
            possible_divisors = []
            for i in range(2, min(21, int(freq) + 1)):  # Check divisors up to 20 or freq
                if freq % i == 0:
                    possible_divisors.append(i)
            
            # If we found valid divisors, use them
            if possible_divisors:
                # Select up to 3 divisors, prioritizing those around 2, 5, 10
                target_values = [2, 5, 10]
                selected_divisors = []
                
                for target in target_values:
                    if target in possible_divisors:
                        selected_divisors.append(target)
                    else:
                        # Find closest divisor to this target
                        closest = min(possible_divisors, key=lambda x: abs(x - target))
                        if closest not in selected_divisors:
                            selected_divisors.append(closest)
                
                # Take up to 3 unique divisors
                i2mc_options['downsamples'] = sorted(list(set(selected_divisors)))[:3]
                print(f"Using downsampling levels: {i2mc_options['downsamples']}")
            else:
                # No suitable divisors found, disable downsampling
                i2mc_options['downsamples'] = []
                print("No suitable downsampling levels found. Downsampling disabled.")
        
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

def create_gaze_summary_row(gaze_df: pd.DataFrame, session_id: str, use_hypothetical_time: bool = False, sampling_freq_hz: float = None) -> pd.Series:
    """
    Creates a summary row for a gaze session with fixation metrics
    
    Args:
        gaze_df (pd.DataFrame): Raw gaze data from a single session
        session_id (str): The session identifier
        use_hypothetical_time (bool): If True, use hypothetical elapsed time for I2MC
        sampling_freq_hz (float, optional): The expected sampling frequency in Hz.
            If None, it will be calculated from the data.
        
    Returns:
        pd.Series: A single row with session_id and fixation metrics
    """
    try:
        # Calculate real frequency if not provided
        if sampling_freq_hz is None:
            sampling_freq_hz = calculate_real_frequency(gaze_df)
            print(f"Session {session_id} - Calculated sampling frequency: {sampling_freq_hz:.2f} Hz")
        
        # Preprocess the data
        preprocessed_df = preprocess_gaze_data(
            gaze_df, 
            use_hypothetical_time=use_hypothetical_time, 
            sampling_freq_hz=sampling_freq_hz
        )
        
        # Get the frequency from preprocessed_df attributes
        freq_hz = preprocessed_df.attrs.get('sampling_freq_hz', sampling_freq_hz)
        
        # Create I2MC options with the correct frequency
        i2mc_options = {
            # Required parameters
            'xres': 1920,
            'yres': 1080,
            'freq': freq_hz,
            'missingx': -1.0,
            'missingy': -1.0,
            'scrSz': [53, 30],
            'disttoscreen': 65,
            # Set downsamples to an empty list by default
            'downsamples': [],
        }
        
        # Detect fixations
        fixations = detect_fixations(
            preprocessed_df, 
            i2mc_options=i2mc_options,
            use_hypothetical_time=use_hypothetical_time
        )
        
        # If no fixations were detected, return a row with NaN values
        if len(fixations) == 0:
            metrics = {key: float('nan') for key in [
                'fixation_count', 'mean_fixation_duration', 'total_fixation_time',
                'mean_saccade_length', 'scanpath_length'
            ]}
            metrics['session_id'] = session_id
            metrics['sampling_freq_hz'] = freq_hz
            return pd.Series(metrics)
        
        # Extract metrics from fixations
        metrics = extract_fixation_metrics(fixations)
        
        # Add session_id and sampling frequency
        metrics['session_id'] = session_id
        metrics['sampling_freq_hz'] = freq_hz
        
        return pd.Series(metrics)
    
    except Exception as e:
        print(f"Error processing session {session_id}: {str(e)}")
        traceback.print_exc()
        
        # Return a row with the session_id and NaN for metrics
        metrics = {key: float('nan') for key in [
            'fixation_count', 'mean_fixation_duration', 'total_fixation_time',
            'mean_saccade_length', 'scanpath_length'
        ]}
        metrics['session_id'] = session_id
        metrics['sampling_freq_hz'] = sampling_freq_hz if sampling_freq_hz is not None else float('nan')
        return pd.Series(metrics)

def process_multiple_gaze_sessions(session_ids: List[str], 
                                  gaze_data_dict: Dict[str, pd.DataFrame],
                                  use_hypothetical_time: bool = False,
                                  sampling_freq_hz: float = None,
                                  save_output: bool = True) -> pd.DataFrame:
    """
    Process multiple gaze sessions and return summary metrics
    
    Args:
        session_ids (List[str]): List of session IDs to process
        gaze_data_dict (Dict[str, pd.DataFrame]): Dictionary mapping session IDs to gaze data
        use_hypothetical_time (bool): If True, use hypothetical elapsed time for I2MC
        sampling_freq_hz (float, optional): The expected sampling frequency in Hz.
            If None, it will be calculated individually for each session.
        save_output (bool): If True, save the results to a CSV file
        
    Returns:
        pd.DataFrame: DataFrame with one row per session containing fixation metrics
    """
    # Create empty results DataFrame
    results = []
    
    # Process each session
    for session_id in session_ids:
        print(f"\nProcessing session: {session_id}")
        
        if session_id not in gaze_data_dict:
            print(f"Warning: No data found for session {session_id}")
            continue
            
        # Get the gaze data for this session
        gaze_df = gaze_data_dict[session_id]
        
        # Calculate session-specific frequency or use the provided one
        session_freq = sampling_freq_hz
        if session_freq is None:
            # Calculate and show frequency for this session
            session_freq = calculate_real_frequency(gaze_df)
            print(f"Session {session_id} sampling frequency: {session_freq:.2f} Hz")
        
        # Create a summary row for this session
        summary_row = create_gaze_summary_row(
            gaze_df, 
            session_id, 
            use_hypothetical_time=use_hypothetical_time,
            sampling_freq_hz=session_freq
        )
        
        # Add to results
        results.append(summary_row)
    
    # Combine results
    results_df = pd.DataFrame(results)
    
    # Save results if requested
    if save_output and not results_df.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hypothetical_indicator = "_hypothetical" if use_hypothetical_time else ""
        filename = f"fixation_metrics{hypothetical_indicator}_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    return results_df

def calculate_real_frequency(gaze_df: pd.DataFrame) -> float:
    """
    Calculate the real frequency (Hz) based on timestamps and number of gaze points.
    
    Args:
        gaze_df (pd.DataFrame): Gaze data DataFrame that must contain 'ISOtimestamp' column
        method (str): Method to calculate frequency:
            'duration' - Use start/end timestamps and sample count
            'intervals' - Use median interval between consecutive samples
            'auto' - Try intervals first, fall back to duration if needed
        
    Returns:
        float: Calculated frequency in Hz (samples per second)
    """
    if 'ISOtimestamp' not in gaze_df.columns:
        raise ValueError("DataFrame must contain 'ISOtimestamp' column")

    
    # Duration-based calculation (original method)
    # Convert to datetime if not already
    timestamps = pd.to_datetime(gaze_df['ISOtimestamp'])
    
    # Get start and end timestamps
    start_time = timestamps.min()
    end_time = timestamps.max()
    
    # Calculate duration in seconds
    duration_seconds = (end_time - start_time).total_seconds()
    
    # Count number of samples
    num_samples = len(gaze_df)
    
    # Avoid division by zero
    if duration_seconds <= 0:
        return float('nan')
    
    # Calculate frequency (Hz)
    frequency = (num_samples - 1) / duration_seconds
    print(f"Frequency calculated using duration method: {frequency:.2f} Hz")
    
    return frequency