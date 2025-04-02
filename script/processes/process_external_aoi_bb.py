import os
import pandas as pd
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import re
from datetime import datetime
import traceback
import sys
import time
import json

# Define the bounding boxes for different stimuli
STIMULUS_BOUNDING_BOXES = {
    11: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    9: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    2: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    15: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    16: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    7: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 320)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 607)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    1: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    13: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    12: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    10: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 447)
        },
        "review_2": {
            "topLeft": (896, 447),
            "bottomRight": (1304, 607)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    4: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    3: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 335)
        },
        "review_1": {
            "topLeft": (896, 335),
            "bottomRight": (1304, 495)
        },
        "review_2": {
            "topLeft": (896, 495),
            "bottomRight": (1304, 655)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    5: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    6: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    8: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    },
    14: {
        "info": {
            "topLeft": (616, 151),
            "bottomRight": (872, 300)
        },
        "review_0": {
            "topLeft": (896, 151),
            "bottomRight": (1304, 311)
        },
        "review_1": {
            "topLeft": (896, 311),
            "bottomRight": (1304, 471)
        },
        "review_2": {
            "topLeft": (896, 471),
            "bottomRight": (1304, 631)
        },
        "likert": {
            "topLeft": (616, 801),
            "bottomRight": (1304, 993)
        }
    }
}

# Define bounding boxes for other stimuli (t_31_m, etc.)
OTHER_STIMULUS_BOUNDING_BOXES_OLD = {
    "likert": {"topLeft": (616, 721), "bottomRight": (1304, 969)},
    "image": {"topLeft": (776, 231), "bottomRight": (904, 359)},
    "title": {"topLeft": (936, 240), "bottomRight": (1144, 268)},
    "mainRating": {"topLeft": (936, 284), "bottomRight": (1144, 310)},
    "numberOfReviews": {"topLeft": (936, 326), "bottomRight": (1144, 350)},
    "stars_5_stars": {"topLeft": (846, 435), "bottomRight": (986, 461)},
    "stars_5_text": {"topLeft": (1002, 436), "bottomRight": (1074, 460)},
    "stars_4_stars": {"topLeft": (846, 477), "bottomRight": (986, 503)},
    "stars_4_text": {"topLeft": (1002, 478), "bottomRight": (1074, 502)},
    "stars_3_stars": {"topLeft": (846, 519), "bottomRight": (986, 545)},
    "stars_3_text": {"topLeft": (1002, 520), "bottomRight": (1074, 544)},
    "stars_2_stars": {"topLeft": (846, 561), "bottomRight": (986, 587)},
    "stars_2_text": {"topLeft": (1002, 562), "bottomRight": (1074, 586)},
    "stars_1_stars": {"topLeft": (846, 603), "bottomRight": (986, 629)},
    "stars_1_text": {"topLeft": (1002, 604), "bottomRight": (1074, 628)},
}

# these are already with tolerance!!!
OTHER_STIMULUS_BOUNDING_BOXES = {
    "likert": {"topLeft": (474, 654), "bottomRight": (1182, 922)},
    "image": {"topLeft": (634, 154), "bottomRight": (782, 312)},
    "mainRating": {"topLeft": (794, 217), "bottomRight": (1022, 263)},
    "numberOfReviews": {"topLeft": (794, 259), "bottomRight": (1022, 313)},
    "title": {"topLeft": (794, 153), "bottomRight": (1022, 221)},
    "distributionStars": {"topLeft": (704, 368), "bottomRight": (864, 582)},
    "distributionText": {"topLeft": (860, 369), "bottomRight": (952, 581)}
}


def is_point_in_bbox(point: Tuple[float, float], bbox: Dict[str, Tuple[int, int]], tolerance: int = 10) -> bool:
    """
    Check if a point is inside a bounding box with optional tolerance.
    
    Args:
        point (Tuple[float, float]): (x, y) coordinates of the point
        bbox (Dict[str, Tuple[int, int]]): Bounding box with topLeft and bottomRight coordinates
        tolerance (int): Number of pixels to expand the bounding box in all directions
        
    Returns:
        bool: True if point is inside the expanded bounding box, False otherwise
    """
    x, y = point
    top_left = bbox["topLeft"]
    bottom_right = bbox["bottomRight"]
    
    # Expand the bounding box by tolerance pixels in all directions
    expanded_top_left = (top_left[0] - tolerance, top_left[1] - tolerance)
    expanded_bottom_right = (bottom_right[0] + tolerance, bottom_right[1] + tolerance)
    
    return (expanded_top_left[0] <= x <= expanded_bottom_right[0] and 
            expanded_top_left[1] <= y <= expanded_bottom_right[1])

def get_aoi_for_point(point: Tuple[float, float], stimulus_id: str, tolerance: int = 10) -> Optional[List[str]]:
    """
    Determine which AOIs a point belongs to based on the stimulus ID and bounding boxes.
    
    Args:
        point (Tuple[float, float]): (x, y) coordinates of the point
        stimulus_id (str): ID of the stimulus
        tolerance (int): Number of pixels to expand bounding boxes in all directions
        
    Returns:
        Optional[List[str]]: List of AOI names if point is inside any bounding boxes, None otherwise
    """
    if not stimulus_id:
        return None
        
    try:
        # Convert stimulus_id to int for comparison with STIMULUS_BOUNDING_BOXES keys
        stimulus_int = int(stimulus_id)
        if stimulus_int in STIMULUS_BOUNDING_BOXES:
            bboxes = STIMULUS_BOUNDING_BOXES[stimulus_int]
        else:
            # For other stimuli (t_31_m, etc.)
            bboxes = OTHER_STIMULUS_BOUNDING_BOXES
    except (ValueError, TypeError):
        # If conversion fails, use OTHER_STIMULUS_BOUNDING_BOXES
        bboxes = OTHER_STIMULUS_BOUNDING_BOXES
    
    # Check each bounding box with tolerance and collect all intersecting AOIs
    intersecting_aois = []
    for aoi_name, bbox in bboxes.items():
        if is_point_in_bbox(point, bbox, tolerance):
            intersecting_aois.append(aoi_name)
    
    return intersecting_aois if intersecting_aois else None

def process_external_aoi_bb(save_output: bool = True, tolerance: int = 10) -> pd.DataFrame:
    """
    Process fixation insights to add AOI information based on bounding boxes.
    
    Args:
        save_output (bool): Whether to save the processed data to a CSV file
        tolerance (int): Number of pixels to expand bounding boxes in all directions
        
    Returns:
        pd.DataFrame: Processed fixation insights with AOI information
    """
    # Load the latest fixation insights file
    output_dir = "outputs"
    pattern = os.path.join(output_dir, "*fixation_insights*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Error: No fixation insights file found in {output_dir}")
        return pd.DataFrame()
    
    # Sort files by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]
    
    print(f"Loading latest fixation insights: {latest_file}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} fixation insights")
        
        # Process each row to add AOI information with tolerance
        df['aoi'] = df.apply(
            lambda row: get_aoi_for_point(
                (row['x_position'], row['y_position']),
                str(row['stimulus']) if pd.notna(row['stimulus']) else None,
                tolerance
            ),
            axis=1
        )
        
        # Add columns for individual AOIs
        df['aoi_count'] = df['aoi'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['aoi_list'] = df['aoi'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
        
        # Add individual AOI columns
        all_aois = set()
        for aois in df['aoi'].dropna():
            if isinstance(aois, list):
                all_aois.update(aois)
        
        for aoi in all_aois:
            df[f'aoi_{aoi}'] = df['aoi'].apply(lambda x: 1 if isinstance(x, list) and aoi in x else 0)
        
        if save_output:
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"fixation_insights_aoi_{timestamp}.csv")
            
            # Save the processed data
            df.to_csv(output_file, index=False)
            print(f"Saved processed data to {output_file}")
            
            # Print summary statistics
            print("\nAOI Intersection Statistics:")
            print(f"Total fixations: {len(df)}")
            print(f"Fixations with AOI intersections: {df['aoi_count'].gt(0).sum()}")
            print("\nAOI intersection counts:")
            for aoi in all_aois:
                count = df[f'aoi_{aoi}'].sum()
                print(f"{aoi}: {count} fixations")
        
        return df
        
    except Exception as e:
        print(f"Error processing fixation insights: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    process_external_aoi_bb() 