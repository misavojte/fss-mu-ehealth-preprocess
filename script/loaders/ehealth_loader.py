import os
import pandas as pd

def load_ehealth_data(session_id, ehealth_pattern="input/ehealth-gaze/multitask_gaze_{session_id}.csv"):
    """
    Load eHealth gaze data file based on session ID.
    
    This function loads eye tracking data from the eHealth application format.
    The data typically contains gaze positions and task-related information
    from the eHealth application sessions.
    
    Args:
        session_id (str): The session identifier used in the eHealth gaze filename
        ehealth_pattern (str, optional): File pattern for ehealth files. 
                                       Default is "input/ehealth-gaze/multitask_gaze_{session_id}.csv"
    
    Returns:
        pandas.DataFrame: DataFrame containing the eHealth gaze data, or None if file not found.
                        The DataFrame typically includes columns for:
                        - Timestamps
                        - Gaze coordinates
                        - Task-specific information
                        - Session metadata
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