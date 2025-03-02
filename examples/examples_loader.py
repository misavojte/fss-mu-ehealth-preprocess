import sys
import os
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders import load_gazepoint_data, load_ehealth_data, load_action_data, session_loader

def main():
    # Example session and linking IDs
    session_id = "1731932124378-2311"
    linking_id = "1022"
    
    # Load action data
    print("\n=== Loading action log data ===")
    action_df = load_action_data(session_id)
    
    if action_df is not None:
        print("\nSuccessfully loaded action log data")
        print("\nFirst few rows of action data:")
        print(action_df.head())
        
        # Example: Count actions by type
        if 'type' in action_df.columns:
            action_counts = action_df['type'].value_counts()
            print("\nAction type counts:")
            print(action_counts)
    
    # Load eHealth data
    print("\n=== Loading eHealth gaze data ===")
    ehealth_df = load_ehealth_data(session_id)
    
    if ehealth_df is not None:
        print("\nSuccessfully loaded eHealth data")
        print("\nFirst few rows of eHealth data:")
        print(ehealth_df.head())
    
    # Load Gazepoint data
    print("\n=== Loading Gazepoint data ===")
    gazepoint_df = load_gazepoint_data(linking_id)
    
    if gazepoint_df is not None:
        print("\nSuccessfully loaded Gazepoint data")
        # Example: Calculate mean gaze position
        if 'FPOGX' in gazepoint_df.columns and 'FPOGY' in gazepoint_df.columns:
            mean_x = gazepoint_df['FPOGX'].mean()
            mean_y = gazepoint_df['FPOGY'].mean()
            print(f"\nMean gaze position: X={mean_x:.3f}, Y={mean_y:.3f}")


    # Example: Load session IDs
    print("\n=== Loading session IDs ===")
    session_ids = session_loader.load_session_ids()
    print("\nSession IDs:")
    print(session_ids)

if __name__ == "__main__":
    main() 