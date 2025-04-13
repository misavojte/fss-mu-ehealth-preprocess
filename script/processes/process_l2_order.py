"""
Process L2 Order Validation Module

This module validates the reviewer assignments between EHealthL2ManualOrderLinking.xlsx
and stimuli_l2.xlsx files by checking if the reviewer_nick values match the positions
for each doctor.
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Tuple, Dict

def load_excel_files(input_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both Excel files from the input directory.
    
    Args:
        input_dir (str): Path to the input directory containing the Excel files
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Manual order linking df and stimuli df
    """
    # Construct file paths
    manual_order_path = os.path.join(input_dir, 'stimuli', 'EHealthL2ManualOrderLinking.xlsx')
    stimuli_path = os.path.join(input_dir, 'stimuli', 'stimuli_l2.xlsx')
    
    # Load both files
    manual_order_df = pd.read_excel(manual_order_path)
    stimuli_df = pd.read_excel(stimuli_path)
    
    return manual_order_df, stimuli_df

def get_reviewer_rows(stimuli_df: pd.DataFrame, doctor_name: str) -> pd.DataFrame:
    """
    Get all rows for a given doctor from the stimuli dataframe.
    
    Args:
        stimuli_df (pd.DataFrame): The stimuli dataframe
        doctor_name (str): Name of the doctor to find rows for
        
    Returns:
        pd.DataFrame: Rows containing reviews for the specified doctor
    """
    return stimuli_df[stimuli_df['name_doc'] == doctor_name]

def validate_reviewer_assignments(
    manual_order_df: pd.DataFrame,
    stimuli_df: pd.DataFrame
) -> List[Dict]:
    """
    Validate that reviewer assignments match between the two files.
    
    Args:
        manual_order_df (pd.DataFrame): Manual order linking dataframe
        stimuli_df (pd.DataFrame): Stimuli dataframe
        
    Returns:
        List[Dict]: List of validation errors found
    """
    validation_errors = []
    
    # Iterate through each row in manual order file
    for _, row in manual_order_df.iterrows():
        doctor_name = row['doctorName']
        doctor_id = row['Id']
        
        # Get actual reviewer positions from manual order file
        actual_reviewers = [
            row['position1'],
            row['position2'],
            row['position3']
        ]
        
        # Get actual reviewer rows for this doctor
        doctor_rows = get_reviewer_rows(stimuli_df, doctor_name)
        
        # If we don't find exactly 3 rows, that's an error
        if len(doctor_rows) != 3:
            validation_errors.append({
                'id': doctor_id,
                'doctor': doctor_name,
                'error': f'Expected 3 reviews, found {len(doctor_rows)}',
                'actual_reviewers': actual_reviewers,
                'expected_reviewers': doctor_rows['reviewer_nick'].tolist() if not doctor_rows.empty else []
            })
            continue
        
        # Get expected reviewer nicks from stimuli file
        expected_reviewers = doctor_rows['reviewer_nick'].tolist()
        
        # Check if the sets of reviewers match
        if set(actual_reviewers) != set(expected_reviewers):
            validation_errors.append({
                'id': doctor_id,
                'doctor': doctor_name,
                'error': 'Reviewer mismatch',
                'actual_reviewers': actual_reviewers,
                'expected_reviewers': expected_reviewers
            })
            
    return validation_errors

def main():
    """Main function to run the L2 order validation process."""
    # Get input directory path
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'input')
    
    try:
        # Load both Excel files
        print("Loading Excel files...")
        manual_order_df, stimuli_df = load_excel_files(input_dir)
        
        # Debug: Print column names
        print("\nManual Order DataFrame columns:")
        print(manual_order_df.columns.tolist())
        print("\nStimuli DataFrame columns:")
        print(stimuli_df.columns.tolist())
        
        # Validate reviewer assignments
        print("\nValidating reviewer assignments...")
        validation_errors = validate_reviewer_assignments(manual_order_df, stimuli_df)
        
        # Print results
        if validation_errors:
            print("\nValidation errors found:")
            for error in validation_errors:
                print(f"\nError for ID {error['id']} - {error['doctor']}:")
                print(f"Error type: {error['error']}")
                print(f"Actual reviewers: {error['actual_reviewers']}")
                print(f"Expected reviewers: {error['expected_reviewers']}")
        else:
            print("\nNo validation errors found. All reviewer assignments match!")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 