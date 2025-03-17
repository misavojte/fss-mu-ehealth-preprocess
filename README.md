# fss-mu-ehealth-preprocess

## Output Files for Data Analysis

The eHealth data processing pipeline generates several output files that can be used for subsequent analysis in R or other statistical packages. Below is a detailed description of each output type, its structure, and recommendations for analysis.

### 1. Fixation Files (folder `fixations/`)

**Description:** 
Individual CSV files for each session containing detected eye fixations. These are the most important files from an eye-tracking perspective, as they represent the processed gaze data with identified fixation points.

**File naming:**
- Format: `[session_id]_fixations.csv`
- Example: `1731932124378-2311_fixations.csv`

**Columns:**
- `start_time_ms`, `end_time_ms`: Time in milliseconds from the beginning of the recording when each fixation started and ended. These are the true eye movement data estimations based on the sampling frequency (150 Hz). These timestamps are essential for understanding the temporal sequence of fixations.
- `duration`: Duration of the fixation in milliseconds. This is a key metric for analyzing attention patterns.
- `x_position`, `y_position`: Screen coordinates of the fixation. These can be used for spatial analysis of visual attention.
- `is_flanked_by_missing`, `fraction_interpolated`: Technical quality indicators about the fixation data.
- `start_timestamp`, `end_timestamp`: Actual timestamps in 'YYYY-MM-DD HH:MM:SS.fff' format that correspond to when each fixation occurred. These are crucial for matching fixations with specific actions or tasks. You can use them to query specific time periods (e.g., to count fixations during a particular task, find the timestamp when it started/ended and filter accordingly).
- `aoi`: Area of Interest where the fixation occurred. Note that a fixation might be associated with multiple AOIs simultaneously if they overlap.

**Usage in R:**
```r
# Load a specific session's fixation data
fixations <- read.csv("outputs/fixations/1731932124378-2311_fixations.csv")

# Convert timestamps to proper datetime format
fixations$start_timestamp <- as.POSIXct(fixations$start_timestamp, format="%Y-%m-%d %H:%M:%OS")
fixations$end_timestamp <- as.POSIXct(fixations$end_timestamp, format="%Y-%m-%d %H:%M:%OS")

# Example: Filter fixations that occurred during a specific time window
task_fixations <- fixations[fixations$start_timestamp >= task_start_time & 
                            fixations$end_timestamp <= task_end_time, ]

# Example: Calculate average fixation duration
avg_duration <- mean(fixations$duration)
```

### 2. Action Summary File (`action_summary.csv`)

**Description:**
A single CSV file containing structured data about participant interactions with three experimental tasks (L1, L2, L3) and gaze validation across all sessions.

**Data Structure:**
1. **Session identification:**
   - `session_id`: Unique identifier for each session
   - `linking_id`: External ID linking to questionnaire data
   - `validation_rounds`: Number of validation rounds completed

2. **L1 Task Data (Face Rating):**
   - Format: `L1__t_XX_g__[startTime|endTime|value]`
   - XX: number from 01-32, g: gender (m/f)
   - value: participant's rating of the face
   - startTime: in the application timestamps, useful for linking with fixations' `start_timestamp` and `end_timestamp`

3. **L2 Task Data (Item Assessment):**
   - Format: `L2__X__[startTime|endTime|value]`
   - X: item number (1-16)
   - startTime: in the application timestamps, useful for linking with fixations' `start_timestamp` and `end_timestamp`
   - value: participant's response for the item

4. **L3 Task Data (Sequential Item Viewing):**
   - Format: `L3__OX__[L2token|startTime|endTime|duration]`
   - OX: order position (O1-O4)
   - L2token: which L2 item was shown
   - startTime: in the application timestamps, useful for linking with fixations' `start_timestamp` and `end_timestamp`
   - duration: seconds spent viewing the item

5. **Validation Metrics:**
   - `vali_point_X_[accuracy|precision|points]`: metrics for each validation point
   - `vali_avg_[accuracy|precision|points]`: average metrics across points

**Usage in R:**
```r
# Load the action summary data
action_summary <- read.csv("outputs/action_summary.csv")

# Convert timestamp columns to proper datetime format
timestamp_cols <- grep("Time", names(action_summary), value=TRUE)
for(col in timestamp_cols) {
  action_summary[[col]] <- as.POSIXct(action_summary[[col]], format="%Y-%m-%d %H:%M:%OS")
}

# Example: Extract L1 face rating data
l1_columns <- grep("L1__", names(action_summary), value=TRUE)
l1_data <- action_summary[, c("session_id", l1_columns)]

# Example: Calculate average viewing time for first item in L3 items
l3_durations <- grep("L3__O1__duration", names(action_summary), value=TRUE)
mean_viewing_time <- mean(action_summary[, l3_durations], na.rm=TRUE)

# Example: Join with fixation data for a specific session
session_id <- "1731932124378-2311"
session_actions <- action_summary[action_summary$session_id == session_id, ]
session_fixations <- read.csv(paste0("outputs/fixations/", session_id, "_fixations.csv"))
```

### 3. Gaze Summary File (`gaze_summary.csv`)

**Description:**
A single CSV file containing aggregated metrics for all processed sessions. This file serves primarily as a quality control tool for assessing eye-tracking data reliability across sessions before proceeding with detailed analyses.

**Columns:**
- `session_id`: Unique identifier for each session
- **Sample Quality Metrics:**
  - `gaze_sample_count`: Total number of gaze samples processed
  - `gaze_missing_count`: Number of missing gaze samples
  - `gaze_missing_percent`: Percentage of samples that were missing or invalid
  - `data_quality_percent`: Percentage of gaze samples that were valid (not missing). Note that positions outside the monitor range are filtered out.

- **Fixation Statistics:**
  - `fix_total_fixations`: Total number of fixations detected in the session
  - `fix_mean_fixation_duration`: Average duration of fixations in milliseconds
  - `fix_median_fixation_duration`: Median duration of fixations (useful for skewed distributions)
  - `fix_min_fixation_duration`: Shortest fixation duration detected
  - `fix_max_fixation_duration`: Longest fixation duration detected
  - `fix_total_fixation_time`: Sum of all fixation durations
  - `fix_fixation_rate`: Fixations per second (fixation frequency)

- **Spatial Distribution Metrics:**
  - `fix_x_dispersion`: Horizontal dispersion of fixations
  - `fix_y_dispersion`: Vertical dispersion of fixations
  - `fix_mean_saccade_amplitude`: Average distance between consecutive fixations
  - `fix_median_saccade_amplitude`: Median distance between consecutive fixations
  - `fix_max_saccade_amplitude`: Maximum distance between consecutive fixations

- **Processing Information:**
  - `timestamp_approach`: Method used for processing ("fixed_frequency" or "real_timestamps")
  - `fixed_frequency`: Sampling rate used for processing (typically 150 Hz)

### Notes

1. **Timestamp Standardization:**
   - All timestamps are in the format 'YYYY-MM-DD HH:MM:SS.fff'
   - In R, use `as.POSIXct()` with the format string `"%Y-%m-%d %H:%M:%OS"` for proper parsing

2. **Joining Data:**
   - Use the `session_id` field to join data across different files
   - Example: `combined <- merge(fixations, action_data[action_data$session_id == "1731932124378-2311", ], by.x="session_id", by.y="session_id")`

3. **Fixations vs. Actions:**
   - Fixation files provide detailed moment-by-moment gaze data
   - Action files provide information about user interactions with the interface
   - Combining both provides insights into visual attention during specific user actions