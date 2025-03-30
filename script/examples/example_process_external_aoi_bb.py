import os
import sys
import pandas as pd
from pathlib import Path

# Add the parent directory to sys.path to import from script.processes
script_dir = Path(__file__).parent.parent.parent
sys.path.append(str(script_dir))

from script.processes.process_external_aoi_bb import process_external_aoi_bb

def main():
    """
    Example demonstrating how to process fixation insights with AOI bounding boxes.
    """
    print("Starting AOI bounding box processing example...")
    
    # Process the fixation insights and get the DataFrame
    df = process_external_aoi_bb(save_output=True)
    
    if df.empty:
        print("No data was processed. Please check if fixation insights files exist.")
        return
    
    # Display some basic statistics about the AOIs
    print("\nSummary of AOI distributions:")
    aoi_counts = df['aoi'].value_counts()
    print(aoi_counts)
    
    # Calculate percentage of fixations that fall within any AOI
    total_fixations = len(df)
    fixations_in_aoi = df['aoi'].notna().sum()
    aoi_percentage = (fixations_in_aoi / total_fixations) * 100
    
    print(f"\nTotal fixations: {total_fixations}")
    print(f"Fixations in AOIs: {fixations_in_aoi}")
    print(f"Percentage in AOIs: {aoi_percentage:.2f}%")
    
    # Show example of fixations for a specific stimulus
    print("\nExample fixations for first stimulus in the dataset:")
    first_stimulus = df['stimulus'].iloc[0]
    stimulus_data = df[df['stimulus'] == first_stimulus].head()
    print(stimulus_data[['stimulus', 'x_position', 'y_position', 'aoi']])

if __name__ == "__main__":
    main() 