import os
import pandas as pd
import argparse
from gazeDataLoader import load_gaze_files
import numpy as np

def find_pattern_start(gazepoint_series, pattern_series, max_pattern_points=None):
    """
    Find the starting index of pattern_series in gazepoint_series for string data.

    Args:
        gazepoint_series (pd.Series): The larger string series to search within.
        pattern_series (pd.Series): The string sequence to find.
        max_pattern_points (int, optional): Limit pattern to first N points for faster matching.
                                           Default None uses full pattern.

    Returns:
        int: Starting index of the pattern or -1 if not found.
    """
    # Handle empty pattern or empty source
    if len(pattern_series) == 0 or len(gazepoint_series) < len(pattern_series):
        return -1
    
    # Limit pattern size if requested
    if max_pattern_points is not None and max_pattern_points > 0 and max_pattern_points < len(pattern_series):
        print(f"Using first {max_pattern_points} points of pattern (full length: {len(pattern_series)})")
        pattern_series = pattern_series.iloc[:max_pattern_points]
        
    # Convert to numpy arrays for faster operations
    pattern = pattern_series.to_numpy()
    gazepoint = gazepoint_series.to_numpy()
    
    pattern_length = len(pattern)
    series_length = len(gazepoint)
    
    # Pre-calculate first value of pattern for early termination
    first_val = pattern[0]
    
    # Sliding window with early termination optimization
    for i in range(series_length - pattern_length + 1):
        # Quick check of first value for early rejection
        if gazepoint[i] != first_val:
            continue
            
        # If first value matches, check the entire pattern
        match = True
        for j in range(pattern_length):
            if gazepoint[i + j] != pattern[j]:
                match = False
                break
                
        if match:
            return i
            
    return -1

def merge_gaze_data(gazepoint_df, ehealth_df, output_file=None):
    """
    Merge gazepoint and ehealth gaze data preserving temporal sequence.
    
    Args:
        gazepoint_df: DataFrame containing gazepoint data
        ehealth_df: DataFrame containing ehealth data
        output_file: Optional file path to save the merged data
        
    Returns:
        DataFrame containing the merged data
    """
    print("Merging gaze data...")
    
    # Verify that the required columns exist
    required_columns = {
        'gazepoint': ['BPOGX', 'BPOGY'],
        'ehealth': ['xLScreenRelative', 'yLScreenRelative', 'xRScreenRelative', 'yRScreenRelative']
    }
    
    for col in required_columns['gazepoint']:
        if col not in gazepoint_df.columns:
            print(f"Error: Gazepoint data missing required column ({col})")
            return None
            
    for col in required_columns['ehealth']:
        if col not in ehealth_df.columns:
            print(f"Error: eHealth data missing required column ({col})")
            return None

    gpXSeries = gazepoint_df['BPOGX']
    gpYSeries = gazepoint_df['BPOGY']
    ehealthXSeries = (ehealth_df['xLScreenRelative'] + ehealth_df['xRScreenRelative']) / 2
    ehealthYSeries = (ehealth_df['yLScreenRelative'] + ehealth_df['yRScreenRelative']) / 2

    # First replace string 'undefined' values with NaN
    # Fix to exactly 2 decimal places (not just round)
    gpXSeries = gpXSeries.apply(lambda x: '{:.2f}'.format(float(x))).astype(float)
    gpYSeries = gpYSeries.apply(lambda x: '{:.2f}'.format(float(x))).astype(float)
    ehealthXSeries = ehealthXSeries.apply(lambda x: '{:.2f}'.format(float(x))).astype(float)
    ehealthYSeries = ehealthYSeries.apply(lambda x: '{:.2f}'.format(float(x))).astype(float)

    # export csv of X only
    gpXSeries.to_csv('output/gazepoint_x.csv', index=False)
    ehealthXSeries.to_csv('output/ehealth_x.csv', index=False)

    # now, find the start index of the pattern in the gazepoint data
    start_index = find_pattern_start(gpXSeries, ehealthXSeries, max_pattern_points=10)
    print(f"Pattern found at gazepoint index: {start_index}")

def main():
    parser = argparse.ArgumentParser(description='Merge gazepoint and ehealth gaze data.')
    parser.add_argument('--sessionId', required=True, help='Session ID for the eHealth gaze file')
    parser.add_argument('--linkingId', required=True, help='Linking ID for the gazepoint file')
    parser.add_argument('--output', default="output/merged_gaze_{session_id}_{linking_id}.csv", 
                       help='Output file pattern')
    
    args = parser.parse_args()
    
    # Load the gaze data
    gazepoint_df, ehealth_df = load_gaze_files(
        args.sessionId,
        args.linkingId
    )
    
    if gazepoint_df is None or ehealth_df is None:
        print("Error loading gaze data files")
        return
    
    # Format the output filename
    output_file = args.output.format(session_id=args.sessionId, linking_id=args.linkingId)
    
    # Merge the data
    merged_df = merge_gaze_data(gazepoint_df, ehealth_df, output_file)
    
    if merged_df is not None:
        print(f"Successfully merged {len(merged_df)} rows of gaze data")

if __name__ == "__main__":
    main()
