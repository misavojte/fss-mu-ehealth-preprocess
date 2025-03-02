import os
import pandas as pd
from io import StringIO

def load_gazepoint_data(linking_id, gazepoint_pattern="input/gazepoint/{linking_id}_all_gaze.csv"):
    """
    Load Gazepoint eye tracking data file based on linking ID.
    
    This function loads raw eye tracking data from the Gazepoint system.
    It handles the special case where the data might include line numbers
    separated by pipe characters.
    
    Args:
        linking_id (str): The linking identifier used in the Gazepoint filename
        gazepoint_pattern (str, optional): File pattern for gazepoint files.
                                         Default is "input/gazepoint/{linking_id}_all_gaze.csv"
    
    Returns:
        pandas.DataFrame: DataFrame containing the Gazepoint data, or None if file not found.
                        The DataFrame typically includes columns for:
                        - FPOGX, FPOGY (Fixation point of gaze)
                        - FPOGS (Fixation point of gaze state)
                        - RPOGX, RPOGY (Raw point of gaze)
                        - BPOGX, BPOGY (Best point of gaze)
                        - Other eye tracking metrics
    """
    # Construct file path
    gazepoint_file = gazepoint_pattern.format(linking_id=linking_id)
    
    # Check if file exists
    if not os.path.exists(gazepoint_file):
        print(f"Error: Gazepoint file not found: {gazepoint_file}")
        return None
    
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
    gazepoint_data = StringIO(''.join(cleaned_lines))
    gazepoint_df = pd.read_csv(gazepoint_data)
    
    # Print basic information
    print(f"\nGazepoint file summary:")
    print(f"- Number of rows: {len(gazepoint_df)}")
    print(f"- Columns: {', '.join(gazepoint_df.columns[:5])}...")
    
    return gazepoint_df 