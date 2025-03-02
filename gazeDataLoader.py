import os
import csv
import argparse
import pandas as pd

def load_gaze_files(session_id, linking_id, 
                   gazepoint_pattern="input/gazepoint/{linking_id}_all_gaze.csv",
                   ehealth_pattern="input/ehealth-gaze/multitask_gaze_{session_id}.csv"):
    """
    Load gaze data files based on session ID and linking ID.
    
    Returns:
        tuple: (gazepoint_df, ehealth_df) - pandas DataFrames containing the data
    """
    # Construct file paths
    gazepoint_file = gazepoint_pattern.format(linking_id=linking_id)
    ehealth_gaze_file = ehealth_pattern.format(session_id=session_id)
    
    # Check if files exist
    if not os.path.exists(gazepoint_file):
        print(f"Error: Gazepoint file not found: {gazepoint_file}")
        return None, None
        
    if not os.path.exists(ehealth_gaze_file):
        print(f"Error: eHealth gaze file not found: {ehealth_gaze_file}")
        return None, None
    
    # Load gazepoint file
    print(f"Loading gazepoint file: {gazepoint_file}")
    
    # Custom reading to handle potential line numbers
    with open(gazepoint_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        # Check if we have pipe-separated line numbers
        has_line_numbers = False
        if lines and '|' in lines[0]:
            has_line_numbers = True
        
        # Clean the lines
        cleaned_lines = []
        for line in lines:
            if has_line_numbers and '|' in line:
                line = line.split('|', 1)[1]
            cleaned_lines.append(line)
    
    # Create a pandas DataFrame
    from io import StringIO
    gazepoint_data = StringIO(''.join(cleaned_lines))
    gazepoint_df = pd.read_csv(gazepoint_data)
    
    # Load ehealth gaze file (this should be a standard CSV)
    print(f"Loading eHealth gaze file: {ehealth_gaze_file}")
    ehealth_df = pd.read_csv(ehealth_gaze_file)
    
    # # Filter out Questionnaire-Stage-1 and Questionnaire-Stage-2 from gazepoint data
    # if 'MEDIA_NAME' in gazepoint_df.columns:
    #     filtered_count = len(gazepoint_df)
    #     gazepoint_df = gazepoint_df[~gazepoint_df['MEDIA_NAME'].isin(['Questionnaire-Stage-1', 'Questionnaire-Stage-2'])]
    #     excluded_count = filtered_count - len(gazepoint_df)
    #     if excluded_count > 0:
    #         print(f"Excluded {excluded_count} rows with questionnaire stages from gazepoint data")
    
    # Print basic information
    print(f"\nGazepoint file summary:")
    print(f"- Number of rows: {len(gazepoint_df)}")
    print(f"- Columns: {', '.join(gazepoint_df.columns[:5])}...")
    
    print(f"\neHealth gaze file summary:")
    print(f"- Number of rows: {len(ehealth_df)}")
    print(f"- Columns: {', '.join(ehealth_df.columns[:5])}...")
    
    return gazepoint_df, ehealth_df

def load_ehealth_data(session_id, ehealth_pattern="input/ehealth-gaze/multitask_gaze_{session_id}.csv"):
    """
    Load only eHealth gaze data file based on session ID.
    
    This function is useful when you only need the eHealth gaze data without the Gazepoint data.
    It provides a simpler interface for loading just the eHealth gaze recordings.
    
    Args:
        session_id (str): The session identifier used in the eHealth gaze filename
        ehealth_pattern (str, optional): File pattern for ehealth files. 
                                       Default is "input/ehealth-gaze/multitask_gaze_{session_id}.csv"
    
    Returns:
        pandas.DataFrame: DataFrame containing the eHealth gaze data, or None if file not found
    """
    # Construct file path using the pattern
    ehealth_gaze_file = ehealth_pattern.format(session_id=session_id)
    
    # Check if file exists
    if not os.path.exists(ehealth_gaze_file):
        print(f"Error: eHealth gaze file not found: {ehealth_gaze_file}")
        return None
    
    # Load ehealth gaze file
    print(f"Loading eHealth gaze file: {ehealth_gaze_file}")
    ehealth_df = pd.read_csv(ehealth_gaze_file)
    
    # Print basic information about the loaded data
    print(f"\neHealth gaze file summary:")
    print(f"- Number of rows: {len(ehealth_df)}")
    print(f"- Columns: {', '.join(ehealth_df.columns[:5])}...")
    
    return ehealth_df

# Example of how to use in other files
def example_usage():
    # Load the data
    gazepoint_df, ehealth_df = load_gaze_files(
        session_id="1731932124378-2311",
        linking_id="1022"
    )
    
    # Now you can easily work with the data
    if gazepoint_df is not None:
        # Example: Calculate mean of a numeric column
        if 'FPOGX' in gazepoint_df.columns:
            mean_x = gazepoint_df['FPOGX'].mean()
            print(f"Mean gaze X position: {mean_x}")
        
        # Example: Filter data
        if 'FPOGY' in gazepoint_df.columns:
            filtered_df = gazepoint_df[gazepoint_df['FPOGY'] > 0.5]
            print(f"Rows with Y position > 0.5: {len(filtered_df)}")
    
    return gazepoint_df, ehealth_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load gaze data files based on session and linking IDs.')
    parser.add_argument('--sessionId', required=True, help='Session ID for the eHealth gaze file')
    parser.add_argument('--linkingId', required=True, help='Linking ID for the gazepoint file')
    parser.add_argument('--gazepoint-pattern', default="input/gazepoint/{linking_id}_all_gaze.csv", 
                        help='File pattern for gazepoint files')
    parser.add_argument('--ehealth-pattern', default="input/ehealth-gaze/multitask_gaze_{session_id}.csv", 
                        help='File pattern for ehealth files')
    
    args = parser.parse_args()
    
    # Load the files and get pandas DataFrames
    gazepoint_df, ehealth_df = load_gaze_files(
        args.sessionId, 
        args.linkingId,
        args.gazepoint_pattern,
        args.ehealth_pattern
    )