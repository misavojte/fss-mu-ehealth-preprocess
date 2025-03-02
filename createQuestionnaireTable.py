import os
import csv
import glob
import re

def process_ehealth_files():
    # Define input directory and output file
    input_dir = "input/ehealth-action"
    output_file = "output/session_linking.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Dictionary to store results (sessionId: linking)
    session_linking = {}
    
    # Process all CSV files in the input directory
    for file_path in glob.glob(f"{input_dir}/*.csv"):
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read all lines first to handle any format issues
            lines = file.readlines()
            
            # Skip header line if present
            start_idx = 1 if len(lines) > 0 and "id,timestamp" in lines[0] else 0
            
            for line in lines[start_idx:]:
                # Remove any line numbers if present
                if "|" in line and line.split("|", 1)[0].isdigit():
                    line = line.split("|", 1)[1]
                
                # Check if this line contains questionnaire-linking
                if "questionnaire-linking" in line:
                    # Extract session ID using regex (looking for pattern in quotes)
                    session_match = re.search(r'"([^"]+)"', line)
                    if session_match:
                        session_id = session_match.group(1)
                        
                        # Extract the linking value (the last field)
                        parts = line.strip().split(',')
                        linking_value = parts[-1].strip('"')
                        
                        # Store in dictionary
                        session_linking[session_id] = linking_value
    
    # Write results to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['sessionId', 'linking'])
        
        for session_id, linking in session_linking.items():
            writer.writerow([session_id, linking])
    
    print(f"Processed data written to {output_file}")
    print(f"Found {len(session_linking)} sessions with linking values")

if __name__ == "__main__":
    process_ehealth_files()