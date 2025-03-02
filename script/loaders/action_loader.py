import os
import pandas as pd

def load_action_data(session_id, action_pattern="input/ehealth-action/multitask_action_{session_id}.csv"):
    """
    Load eHealth action log data file based on session ID.
    
    This function loads action log data from the eHealth application format.
    The data contains user interactions, events, and actions recorded during
    the eHealth application sessions.
    
    Args:
        session_id (str): The session identifier used in the action log filename
        action_pattern (str, optional): File pattern for action log files. 
                                      Default is "input/ehealth-action/multitask_action_{session_id}.csv"
    
    Returns:
        pandas.DataFrame: DataFrame containing the action log data, or None if file not found.
                        The DataFrame typically includes columns for:
                        - id: Action identifier
                        - timestamp: When the action occurred
                        - sessionId: Session identifier
                        - type: Type of action/event
                        - value: Additional action data
    """
    # Construct file path using the pattern
    action_file = action_pattern.format(session_id=session_id)
    
    # Check if file exists
    if not os.path.exists(action_file):
        print(f"Error: Action log file not found: {action_file}")
        return None
    
    # Load action log file
    print(f"Loading action log file: {action_file}")
    action_df = pd.read_csv(action_file)
    
    # Print basic information about the loaded data
    print(f"\nAction log file summary:")
    print(f"- Number of rows: {len(action_df)}")
    print(f"- Columns: {', '.join(action_df.columns)}")
    
    # Optional: Print unique action types to understand the data
    if 'type' in action_df.columns:
        unique_actions = action_df['type'].unique()
        print(f"- Unique action types: {', '.join(unique_actions)}")
    
    return action_df 