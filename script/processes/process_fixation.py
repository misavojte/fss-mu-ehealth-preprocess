import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import sys
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
    
    # Handle eHealth-specific coordinate formats (separate L/R eye coordinates)
    # Check if we have separate left/right eye coordinates instead of average
    if 'x' not in df.columns:
        # Check for left/right eye x coordinates
        has_left_x = any(col for col in df.columns if col.lower() in ['xl', 'xlscreenrelative', 'x_left'])
        has_right_x = any(col for col in df.columns if col.lower() in ['xr', 'xrscreenrelative', 'x_right'])
        
        if has_left_x and has_right_x:
            print("eHealth data: Found separate left/right eye coordinates, calculating average.")
            
            # CRITICAL: Use PIXEL coordinates, NOT screen-relative coordinates
            # For eHealth data, use xL and xR (pixel coordinates), NOT xLScreenRelative 
            left_x_col = 'xL'
            right_x_col = 'xR'
            
            # Calculate average x position (ignoring missing values)
            df['x'] = df[[left_x_col, right_x_col]].mean(axis=1, skipna=True)
            print(f"Created 'x' coordinate column as average of PIXEL coordinates {left_x_col} and {right_x_col}")
    
    # Same for y coordinates
    if 'y' not in df.columns:
        # Check for left/right eye y coordinates
        has_left_y = any(col for col in df.columns if col.lower() in ['yl', 'ylscreenrelative', 'y_left'])
        has_right_y = any(col for col in df.columns if col.lower() in ['yr', 'yrscreenrelative', 'y_right'])
        
        if has_left_y and has_right_y:
            # CRITICAL: Use PIXEL coordinates, NOT screen-relative coordinates
            # For eHealth data, use yL and yR (pixel coordinates), NOT yLScreenRelative
            left_y_col = 'yL'
            right_y_col = 'yR'
            
            # Calculate average y position (ignoring missing values)
            df['y'] = df[[left_y_col, right_y_col]].mean(axis=1, skipna=True)
            print(f"Created 'y' coordinate column as average of PIXEL coordinates {left_y_col} and {right_y_col}")
    
    # Handle eHealth-specific validity columns
    if 'missing' not in df.columns and 'validityL' in df.columns and 'validityR' in df.columns:
        # In eHealth data, a validity value of 1 means the data is valid
        # Convert the validity flags to a 'missing' column (True if either eye is invalid)
        df['missing'] = (df['validityL'] != 1) | (df['validityR'] != 1)
        print("Created 'missing' column based on eHealth validity flags (validity=1 means valid data)")
    
    # Create uniform column names regardless of data source
    for col in df.columns:
        if col.lower() in ['x', 'gaze_x', 'gazepoint_x', 'position_x']:
            df.rename(columns={col: 'x'}, inplace=True)
        elif col.lower() in ['y', 'gaze_y', 'gazepoint_y', 'position_y']:
            df.rename(columns={col: 'y'}, inplace=True)
        elif col.lower() in ['time', 'timestamp', 'isotimestamp', 'time_ms']:
            # Don't rename timestamp columns to preserve original names
            pass
    
    # Check if x and y columns exist
    if 'x' not in df.columns:
        raise ValueError("Missing x coordinate column. Please ensure the data contains a column for x coordinates.")
    if 'y' not in df.columns:
        raise ValueError("Missing y coordinate column. Please ensure the data contains a column for y coordinates.")
    
    # Add missing status column (True if data is missing)
    if 'missing' not in df.columns:
        df['missing'] = (df['x'].isna()) | (df['y'].isna()) | (df['x'] == -1) | (df['y'] == -1)
    
    # Debug logging for the missing data
    valid_samples = (~df['missing']).sum()
    total_samples = len(df)
    print(f"After preprocessing: {valid_samples}/{total_samples} valid samples ({valid_samples/total_samples*100:.1f}% valid)")
    
    # Create required columns list, preserving ALL timestamp columns
    required_cols = ['x', 'y', 'missing']
    
    # Add timestamp columns to required columns list to preserve them
    timestamp_cols = [col for col in df.columns if col.lower() in ['time', 'timestamp', 'isotimestamp', 'time_ms']]
    required_cols.extend(timestamp_cols)
    
    # Add eHealth-specific columns that might be useful for analysis
    ehealth_cols = [col for col in df.columns if col.lower() in ['sessionid', 'gazesessionid', 'aoi']]
    if ehealth_cols:
        print(f"Preserving eHealth-specific columns: {', '.join(ehealth_cols)}")
        required_cols.extend(ehealth_cols)
    
    # Check if we have all required columns
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing after preprocessing")
    
    # Select only required columns
    processed_df = df[required_cols].copy()
    
    return processed_df

def prepare_data_real_timestamps(
    gaze_df: pd.DataFrame, 
    timestamp_column: str = 'ISOtimestamp',
    x_column: str = 'x',
    y_column: str = 'y',
    missing_column: str = 'missing'
) -> Tuple[pd.DataFrame, datetime]:
    """
    Prepare gaze data using real timestamps, ordering by the specified timestamp column.
    
    Args:
        gaze_df (pd.DataFrame): Preprocessed gaze data
        timestamp_column (str): Name of the column containing timestamps
        x_column (str): Name of the column containing x coordinates
        y_column (str): Name of the column containing y coordinates
        missing_column (str): Name of the column indicating missing data
        
    Returns:
        Tuple[pd.DataFrame, datetime]: 
            - DataFrame with data prepared for I2MC (ordered by timestamp)
            - Original start time (datetime object)
    """
    # Make a copy to avoid modifying the original
    df = gaze_df.copy()
    
    # Check if the timestamp column exists
    if timestamp_column not in df.columns:
        raise ValueError(f"Specified timestamp column '{timestamp_column}' not found in data. Available columns: {', '.join(df.columns)}")
    
    # Ensure timestamp column is datetime
    if pd.api.types.is_string_dtype(df[timestamp_column]):
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Sort by timestamp
    df = df.sort_values(timestamp_column)
    
    # Get the start time
    start_time = df[timestamp_column].min()
    
    # Calculate time in milliseconds from the start
    df['time_ms'] = (df[timestamp_column] - start_time).dt.total_seconds() * 1000
    
    # Prepare data for I2MC
    data_df = pd.DataFrame()
    data_df['time'] = df['time_ms']
    
    # Handle x and y coordinates - convert missing values to I2MC's missing value (-1)
    data_df['average_X'] = df[x_column].copy()
    data_df['average_Y'] = df[y_column].copy()
    
    # Make sure missing column is created correctly
    # I2MC expects a boolean array where True = missing data
    # First ensure we have proper missing data flags
    missing_mask = df[missing_column].copy()
    
    # Also mark NaN values in x and y as missing
    missing_mask = missing_mask | df[x_column].isna() | df[y_column].isna()
    
    # Also check for extreme values if we're using pixel coordinates
    # For pixel coordinates, we need to check if they're outside the screen resolution
    if df[x_column].min() >= 0:  # Only check for positive coordinates
        # For pixel coordinates, check if they're outside the valid screen range
        x_outside_range = (df[x_column] <= 0) | (df[x_column] >= 1920)  # assuming 1920 is max X resolution
        y_outside_range = (df[y_column] <= 0) | (df[y_column] >= 1200)  # assuming 1200 is max Y resolution
        
        if (x_outside_range | y_outside_range).sum() > 0:
            print(f"Warning: Found {(x_outside_range | y_outside_range).sum()} samples with coordinates outside screen range. Marking them as missing.")
            missing_mask = missing_mask | x_outside_range | y_outside_range
    
    data_df['missing'] = missing_mask
    
    # Set missing x and y values to -1 as I2MC expects
    data_df.loc[missing_mask, 'average_X'] = -1
    data_df.loc[missing_mask, 'average_Y'] = -1
    
    # Log the actual missing data percentage being sent to I2MC
    missing_percent = (data_df['missing'].sum() / len(data_df)) * 100
    print(f"Data being sent to I2MC: {len(data_df) - data_df['missing'].sum()}/{len(data_df)} valid samples ({100-missing_percent:.1f}% valid)")
    
    # Pass original timestamp information for later use
    data_df['original_timestamp'] = df[timestamp_column]
    
    # Calculate the real sampling rate for logging purposes
    diffs = np.diff(df['time_ms'])
    if len(diffs) > 0:
        mean_diff = np.mean(diffs)
        real_rate = 1000 / mean_diff if mean_diff > 0 else None
        print(f"Real sampling rate from timestamps: {real_rate:.2f} Hz")
    
    return data_df, start_time

def prepare_data_fixed_frequency(
    gaze_df: pd.DataFrame,
    frequency: float = 150.0, 
    timestamp_column: str = 'ISOtimestamp',
    x_column: str = 'x',
    y_column: str = 'y',
    missing_column: str = 'missing'
) -> Tuple[pd.DataFrame, datetime]:
    """
    Prepare gaze data assuming fixed frequency sampling.
    First orders by timestamp, then assigns evenly spaced time points.
    
    Args:
        gaze_df (pd.DataFrame): Preprocessed gaze data
        frequency (float): Assumed sampling frequency in Hz
        timestamp_column (str): Name of the column containing timestamps
        x_column (str): Name of the column containing x coordinates
        y_column (str): Name of the column containing y coordinates
        missing_column (str): Name of the column indicating missing data
        
    Returns:
        Tuple[pd.DataFrame, datetime]: 
            - DataFrame with data prepared for I2MC (with fixed frequency time)
            - Original start time (datetime object)
    """
    # Make a copy to avoid modifying the original
    df = gaze_df.copy()
    
    # If timestamp column exists, use it for sorting and reference time
    start_time = None
    if timestamp_column in df.columns:
        # Ensure timestamp column is datetime
        if pd.api.types.is_string_dtype(df[timestamp_column]):
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Sort by timestamp
        df = df.sort_values(timestamp_column)
        
        # Get the start time
        start_time = df[timestamp_column].min()
    else:
        # If no timestamp column, use a placeholder
        start_time = pd.Timestamp('2000-01-01')
        print(f"Warning: Timestamp column '{timestamp_column}' not found. Using sample order instead.")
    
    # Create evenly spaced time points based on frequency
    sample_interval_ms = 1000.0 / frequency
    df['time_ms'] = np.arange(len(df)) * sample_interval_ms
    
    # Prepare data for I2MC
    data_df = pd.DataFrame()
    data_df['time'] = df['time_ms']
    
    # Handle x and y coordinates - convert missing values to I2MC's missing value (-1)
    data_df['average_X'] = df[x_column].copy()
    data_df['average_Y'] = df[y_column].copy()
    
    # Make sure missing column is created correctly
    # I2MC expects a boolean array where True = missing data
    # First ensure we have proper missing data flags
    missing_mask = df[missing_column].copy()
    
    # Also mark NaN values in x and y as missing
    missing_mask = missing_mask | df[x_column].isna() | df[y_column].isna()
    
    # Also check for extreme values if we're using pixel coordinates
    # For pixel coordinates, we need to check if they're outside the screen resolution
    if df[x_column].min() >= 0:  # Only check for positive coordinates
        # For pixel coordinates, check if they're outside the valid screen range
        x_outside_range = (df[x_column] <= 0) | (df[x_column] >= 1920)  # assuming 1920 is max X resolution
        y_outside_range = (df[y_column] <= 0) | (df[y_column] >= 1200)  # assuming 1200 is max Y resolution
        
        if (x_outside_range | y_outside_range).sum() > 0:
            print(f"Warning: Found {(x_outside_range | y_outside_range).sum()} samples with coordinates outside screen range. Marking them as missing.")
            missing_mask = missing_mask | x_outside_range | y_outside_range
    
    data_df['missing'] = missing_mask
    
    # Set missing x and y values to -1 as I2MC expects
    data_df.loc[missing_mask, 'average_X'] = -1
    data_df.loc[missing_mask, 'average_Y'] = -1
    
    # Log the actual missing data percentage being sent to I2MC
    missing_percent = (data_df['missing'].sum() / len(data_df)) * 100
    print(f"Data being sent to I2MC: {len(data_df) - data_df['missing'].sum()}/{len(data_df)} valid samples ({100-missing_percent:.1f}% valid)")
    
    # Store original timestamps if available
    if timestamp_column in df.columns:
        data_df['original_timestamp'] = df[timestamp_column]
    
    print(f"Using fixed sampling rate: {frequency} Hz")
    
    return data_df, start_time

def detect_fixations(
    gaze_df: pd.DataFrame, 
    i2mc_options: Dict = None,
    use_real_timestamps: bool = False,
    timestamp_column: str = 'ISOtimestamp',
    fixed_frequency: float = 150.0
) -> pd.DataFrame:
    """
    Detect fixations in preprocessed gaze data using the I2MC algorithm.
    
    Args:
        gaze_df (pd.DataFrame): Preprocessed gaze data with columns:
            time_ms, x, y, missing
        i2mc_options (Dict, optional): Dictionary containing all I2MC algorithm parameters.
            If None, uses default values as specified below.
        use_real_timestamps (bool): Whether to use real timestamps (True) or fixed frequency (False)
        timestamp_column (str): Name of the column containing timestamps (if use_real_timestamps=True)
        fixed_frequency (float): Assumed sampling frequency in Hz (if use_real_timestamps=False)
            
    Returns:
        pd.DataFrame: Detected fixations with their properties, including original start timestamps
    """
    # Setup default I2MC parameters if not provided
    if i2mc_options is None:
        i2mc_options = {
            # Required parameters - THESE MUST MATCH THE ORIGINAL IMPLEMENTATION VALUES EXACTLY!
            'xres': 1920,              # screen width in pixels (standard Full HD resolution)
            'yres': 1200,              # screen height in pixels for eHealth app
            'freq': fixed_frequency if not use_real_timestamps else None,  # sampling rate in Hz
            'missingx': -1.0,          # value marking missing data in x-coordinates
            'missingy': -1.0,          # value marking missing data in y-coordinates
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
            'cutoffstd': 2.0,          # number of STD for fixation threshold
            'onoffsetThresh': 3.0,     # threshold for walk-back
            'maxMergeDist': 30.0,      # max distance (pixels) for merging
            'maxMergeTime': 30.0,      # max time (ms) for merging
            'minFixDur': 40.0,         # minimum fixation duration (ms)
        }
    else:
        # Ensure required parameters are present
        required_params = ['xres', 'yres', 'missingx', 'missingy']
        for param in required_params:
            if param not in i2mc_options:
                raise ValueError(f"Required parameter '{param}' missing from i2mc_options")
        
        # Set frequency if not provided and using fixed frequency
        if not use_real_timestamps and 'freq' not in i2mc_options:
            i2mc_options['freq'] = fixed_frequency
    
    try:
        # Print data quality stats
        valid_samples = (~gaze_df['missing']).sum()
        total_samples = len(gaze_df)
        print(f"Data quality: {valid_samples}/{total_samples} valid samples ({valid_samples/total_samples*100:.1f}%)")
        
        # Check if we have the required timestamp column when using real timestamps
        has_timestamp = timestamp_column in gaze_df.columns
        if use_real_timestamps and not has_timestamp:
            print(f"Warning: Specified timestamp column '{timestamp_column}' not found. Falling back to fixed frequency approach.")
            use_real_timestamps = False
        
        # Prepare data based on timestamp approach
        if use_real_timestamps:
            # Use real timestamps, ordering by timestamp
            print("Using real timestamps approach")
            try:
                data_df, start_time = prepare_data_real_timestamps(
                    gaze_df, 
                    timestamp_column=timestamp_column,
                    x_column='x',
                    y_column='y',
                    missing_column='missing'
                )
                
                # Calculate the real sampling rate
                diffs = np.diff(data_df['time'])
                if len(diffs) > 0:
                    mean_diff = np.mean(diffs)
                    real_rate = 1000 / mean_diff if mean_diff > 0 else 150.0
                    print(f"Real sampling rate from timestamps: {real_rate:.2f} Hz")
                    # Set frequency based on actual data
                    if 'freq' not in i2mc_options or i2mc_options['freq'] is None:
                        i2mc_options['freq'] = real_rate
                        print(f"Setting I2MC frequency to {i2mc_options['freq']:.2f} Hz based on data")
                else:
                    i2mc_options['freq'] = 150.0
                    print(f"Using default I2MC frequency: {i2mc_options['freq']} Hz")
            except Exception as e:
                print(f"Error using real timestamps approach: {str(e)}")
                print("Falling back to fixed frequency approach")
                use_real_timestamps = False
                
        if not use_real_timestamps:
            # Use fixed frequency approach
            print(f"Using fixed frequency approach ({fixed_frequency} Hz)")
            data_df, start_time = prepare_data_fixed_frequency(
                gaze_df, 
                frequency=fixed_frequency,
                timestamp_column=timestamp_column,
                x_column='x',
                y_column='y',
                missing_column='missing'
            )
            
            # Ensure frequency is set correctly
            i2mc_options['freq'] = fixed_frequency
        
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
                        'start_time_ms': fix_data['startT'],
                        'end_time_ms': fix_data['endT'],
                        'duration': fix_data['dur'],  # Duration as calculated by I2MC algorithm
                        'x_position': fix_data['xpos'],
                        'y_position': fix_data['ypos'],
                        'is_flanked_by_missing': fix_data['flankdataloss'],
                        'fraction_interpolated': fix_data['fracinterped']
                    })
                    
                    # Map fixation start/end indices to original timestamps if available
                    if 'original_timestamp' in data_df.columns:
                        # Get the indices of fixation start and end points
                        fix_indices_start = fix_data['start']
                        fix_indices_end = fix_data['end']
                        
                        # Map these indices to the original timestamps
                        original_timestamps_start = data_df['original_timestamp'].values[fix_indices_start]
                        original_timestamps_end = data_df['original_timestamp'].values[fix_indices_end]
                        
                        # Add to fixations DataFrame
                        fixations['start_timestamp'] = original_timestamps_start
                        fixations['end_timestamp'] = original_timestamps_end
                    else:
                        # If no original timestamps available, use relative timestamps
                        fixations['start_timestamp'] = start_time + pd.to_timedelta(fixations['start_time_ms'], unit='ms')
                        fixations['end_timestamp'] = start_time + pd.to_timedelta(fixations['end_time_ms'], unit='ms')
                        print("Warning: Using relative timestamps because original timestamps are not available")
                    
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
        if 'start_time_ms' in fixations.columns and 'end_time_ms' in fixations.columns:
            fixations['duration'] = fixations['end_time_ms'] - fixations['start_time_ms']
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

def create_gaze_summary_row(
    gaze_df: pd.DataFrame, 
    session_id: str,
    use_real_timestamps: bool = False,
    timestamp_column: str = 'ISOtimestamp',
    fixed_frequency: float = 150.0
) -> pd.Series:
    """
    Create a summary row for gaze data, including fixation metrics.
    
    Args:
        gaze_df (pd.DataFrame): Raw gaze data
        session_id (str): Session identifier
        use_real_timestamps (bool): Whether to use real timestamps (True) or fixed frequency (False)
        timestamp_column (str): Name of the column containing timestamps (if use_real_timestamps=True)
        fixed_frequency (float): Assumed sampling frequency in Hz (if use_real_timestamps=False)
        
    Returns:
        pd.Series: Summary row with metrics
    """
    # Initialize with session ID
    summary = {
        'session_id': session_id
    }
    
    try:
        # Check if required timestamp column exists when using real timestamps
        if use_real_timestamps and timestamp_column not in gaze_df.columns:
            print(f"Warning: Timestamp column '{timestamp_column}' not found in gaze data.")
            print(f"Available columns: {', '.join(gaze_df.columns[:5])}...")
            print("Falling back to fixed frequency approach.")
            use_real_timestamps = False
        
        # Preprocess the gaze data
        preprocessed_df = preprocess_gaze_data(gaze_df)
        
        # Basic gaze data metrics
        summary.update({
            'gaze_sample_count': len(preprocessed_df),
            'gaze_missing_count': preprocessed_df['missing'].sum(),
            'gaze_missing_percent': (preprocessed_df['missing'].sum() / len(preprocessed_df)) * 100 if len(preprocessed_df) > 0 else float('nan')
        })
        
        # Detect fixations with specified approach
        fixations = detect_fixations(
            preprocessed_df, 
            use_real_timestamps=use_real_timestamps,
            timestamp_column=timestamp_column,
            fixed_frequency=fixed_frequency
        )
        
        # Extract fixation metrics
        fixation_metrics = extract_fixation_metrics(fixations)
        
        # Add fixation metrics to summary with 'fix_' prefix
        for key, value in fixation_metrics.items():
            summary[f'fix_{key}'] = value
        
        # Add timestamp approach to summary
        summary['timestamp_approach'] = 'real_timestamps' if use_real_timestamps else 'fixed_frequency'
        if use_real_timestamps:
            summary['timestamp_column'] = timestamp_column
        else:
            summary['fixed_frequency'] = fixed_frequency
        
        # Save fixations to CSV for later analysis
        output_dir = os.path.join("outputs", "fixations")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        approach = "real" if use_real_timestamps else f"fixed{int(fixed_frequency)}"
        output_path = os.path.join(output_dir, f"fixations_{session_id}_{approach}_{timestamp}.csv")
        
        if not fixations.empty:
            fixations.to_csv(output_path, index=False)
            print(f"Saved fixations to {output_path}")
        
    except Exception as e:
        print(f"Error processing gaze data for session {session_id}: {str(e)}")
        # Add error information to summary
        summary['processing_error'] = str(e)
    
    return pd.Series(summary)

def process_multiple_gaze_sessions(
    session_ids: List[str], 
    gaze_data_dict: Dict[str, pd.DataFrame],
    use_real_timestamps: bool = False,
    timestamp_column: str = 'ISOtimestamp',
    fixed_frequency: float = 150.0,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Process multiple gaze data sessions and create a summary table.
    
    Args:
        session_ids (List[str]): List of session IDs to process
        gaze_data_dict (Dict[str, pd.DataFrame]): Dictionary mapping session_ids to their gaze DataFrames
        use_real_timestamps (bool): Whether to use real timestamps (True) or fixed frequency (False)
        timestamp_column (str): Name of the column containing timestamps (if use_real_timestamps=True)
        fixed_frequency (float): Assumed sampling frequency in Hz (if use_real_timestamps=False)
        save_output (bool, optional): Whether to save the output to CSV. Defaults to True.
        
    Returns:
        pd.DataFrame: Summary table where each row represents one session
    """
    summaries = []
    for session_id in session_ids:
        if session_id in gaze_data_dict:
            gaze_df = gaze_data_dict[session_id]
            summary_row = create_gaze_summary_row(
                gaze_df, 
                session_id,
                use_real_timestamps=use_real_timestamps,
                timestamp_column=timestamp_column,
                fixed_frequency=fixed_frequency
            )
            summaries.append(summary_row)
    
    summary_df = pd.DataFrame(summaries)
    
    if not summary_df.empty:
        summary_df = summary_df.sort_values('session_id')
        
        # Save if requested
        if save_output:
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            approach = "real" if use_real_timestamps else f"fixed{int(fixed_frequency)}"
            output_path = os.path.join(output_dir, f"gaze_summary_{approach}_{timestamp}.csv")
            
            summary_df.to_csv(output_path, index=False)
            print(f"\nSaved gaze summary table to: {output_path}")
    
    return summary_df
