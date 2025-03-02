import os
import re
from glob import glob

def load_session_ids(action_pattern="input/ehealth-action/multitask_action_*.csv"):
    """
    Find all unique session IDs from eHealth action log files.
    
    This function scans the action log directory and extracts session IDs
    from filenames matching the pattern.
    
    Args:
        action_pattern (str, optional): Glob pattern for action log files.
                                      Default is "input/ehealth-action/multitask_action_*.csv"
    
    Returns:
        list: List of unique session IDs found in the directory
    """
    # Get list of all files matching the pattern
    action_files = glob(action_pattern)
    
    if not action_files:
        print(f"Warning: No files found matching pattern: {action_pattern}")
        return []
    
    # Extract session IDs from filenames using regex
    # Pattern matches everything between 'multitask_action_' and '.csv'
    session_ids = []
    pattern = r"multitask_action_(.+)\.csv"
    
    for file_path in action_files:
        match = re.search(pattern, file_path)
        if match:
            session_id = match.group(1)
            session_ids.append(session_id)
    
    # Remove duplicates and sort
    unique_session_ids = sorted(list(set(session_ids)))
    
    # Print summary
    print(f"Found {len(unique_session_ids)} unique session IDs")
    for sid in unique_session_ids:
        print(f"- {sid}")
        
    return unique_session_ids 