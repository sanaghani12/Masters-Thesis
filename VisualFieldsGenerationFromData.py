import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
# Load the data
file_path = 'D:\\sana\\Thesis\\Savir_Data\\QueryResults2.csv'
data = pd.read_csv(file_path)

# Create output directory for images
output_dir = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog'
os.makedirs(output_dir, exist_ok=True)

# Function to map dataset number to eye side
# Function to map dataset number to eye side
def map_eye(dataset):
    return "Right Eye" if dataset in [1, 3] else "Left Eye"

# Function to sanitize filenames
def sanitize_filename(value):
    return str(value).replace(" ", "_")

# Custom colormap with yellow for missing points
cmap = LinearSegmentedColormap.from_list('custom_gray', ['yellow', 'black', 'white'], N=256)

# Convert TestDate to datetime and sort the data
data['TestDate'] = pd.to_datetime(data['TestDate'], dayfirst=True)
data.sort_values(by=['TestDate', 'ResultSet'], inplace=True)

# Group by UniquePatID, dataset, and TestDate
grouped = data.groupby(['UniquePatID', 'dataset', data['TestDate'].dt.date])

# Iterate through each group
for (unique_pat_id, dataset, test_date), group in grouped:
    eye = map_eye(dataset)
    result_sets = group['ResultSet'].unique()

    # Get patient details
    first_name = group['FirstName'].iloc[0]
    last_name = group['LastName'].iloc[0]
    sanitized_first_name = sanitize_filename(first_name)
    sanitized_last_name = sanitize_filename(last_name)
    
    # Plot individual grids
    for result_set in result_sets:
        subset = group[group['ResultSet'] == result_set]
        #max_x = subset['xpos'].max() 
        #max_x = sorted(set(subset['xpos']))[-2]  + 1
        #max_y = subset['ypos'].max()
        #max_y = sorted(set(subset['ypos']))[-2]  + 1
        max_x = sorted(set(subset['xpos'].astype(int)))[-2] + 1
        max_y = sorted(set(subset['ypos'].astype(int)))[-2] + 1
        print("max x", max_x)
        print("max y", max_y)
        grid = np.full((max_y, max_x), -1)  # -1 for missing points
        stim_min = subset['stim_minReactionTime'].iloc[0]
        stim_max = subset['stim_maxReactionTime'].iloc[0]

        for _, row in subset.iterrows():
            try:
                x, y = int(row['xpos']), int(row['ypos'])
                response = row['responseTime']

                # Determine visibility
                if response > stim_min and response < stim_max:
                    grid[y, x] = 1  # White for visible
                elif response == 0 or response < stim_min or response > stim_max:
                    grid[y, x] = 0  # Black for non-visible
            except IndexError:
                # Ignore out-of-bounds values and continue
                print(f"Skipping out-of-bounds position (x={x}, y={y}) for patient: {unique_pat_id}")
                continue

###############################################################
        # Fetch the fixation row once per test
        fixation_row = subset[(subset['xpos'] == subset['xpos'].max()) & (subset['ypos'] == subset['ypos'].max())]

        if not fixation_row.empty:  # Ensure fixation_row is found
            response_time = fixation_row['responseTime'].iloc[0]
            fixp_min = fixation_row['fixp_minReactionTime'].iloc[0]
            fixp_max = fixation_row['fixp_maxReactionTime'].iloc[0]

            print(f"Fixation row for patient {unique_pat_id}: xpos={fixation_row['xpos'].iloc[0]}, ypos={fixation_row['ypos'].iloc[0]}")
        # Iterate over all missing points in the grid
            for y in range(max_y):
                for x in range(max_x):
                    if grid[y, x] == -1:  # Missing point
                        # Determine visibility based on fixation response time
                        if fixp_min <= response_time <= fixp_max:
                            grid[y, x] = 1  # White for visible
                        else:
                            grid[y, x] = 0  # Black for non-visible
        else:
            print(f"Fixation row not found for patient: {unique_pat_id}, skipping missing points.")


###############################################################################

        # Plot the individual grid
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap=cmap, origin='upper', extent=[0, max_x, max_y, 0], vmin=-1, vmax=1)
        plt.title(
            f"Patient: {first_name} {last_name}\n"
            f"Dataset: {eye}, ResultSet: {result_set}, Date: {test_date}\n"
            f"Stim Min: {stim_min} ms, Stim Max: {stim_max} ms",
            fontsize=10
        )
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.xticks(range(max_x))
        plt.yticks(range(max_y))
        plt.grid(visible=True)

        # Save individual grid
        output_file_individual = f'{output_dir}/grid_{sanitized_first_name}_{sanitized_last_name}_{unique_pat_id}_{eye}_resultset_{result_set}.png'
        plt.savefig(output_file_individual)
        plt.close()

        print(f"Individual Grid saved: {output_file_individual}")

    # Handle more than two tests for one eye on the same day: keep only the last two
    if len(result_sets) > 2:
        group = group[group['ResultSet'].isin(result_sets[-2:])]

    # Skip if only one test exists (no combined grid needed)
    if len(result_sets) < 2:
        print(f"Skipping combined grid for {first_name} {last_name}, {eye}, {test_date}: Only one test exists.")
        continue

    # Combine the last two grids
    grids = []
    #max_x = group['xpos'].max()
    max_x = sorted(set(group['xpos'].astype(int)))[-2]  + 1
    #max_y = group['ypos'].max()
    max_y = sorted(set(group['ypos'].astype(int)))[-2]  + 1

    # Fetch the fixation row once for missing points logic
    fixation_row = group[(group['xpos'] == group['xpos'].max()) & (group['ypos'] == group['ypos'].max())]
    if not fixation_row.empty:
        fixation_response_time = fixation_row['responseTime'].iloc[0]
        fixp_min = fixation_row['fixp_minReactionTime'].iloc[0]
        fixp_max = fixation_row['fixp_maxReactionTime'].iloc[0]
    else:
        print(f"In combined grid logic: Fixation row not found for patient: {unique_pat_id}, skipping missing points logic.")
        fixation_response_time = None
        fixp_min = None
        fixp_max = None


    for result_set in result_sets[-2:]:     # Iterate through the last two tests
        subset = group[group['ResultSet'] == result_set]
        grid = np.full((max_y, max_x), -1)  # -1 for missing points
        stim_min = subset['stim_minReactionTime'].iloc[0]
        stim_max = subset['stim_maxReactionTime'].iloc[0]

        fixation_responses = []  # To store fixation response times

        for _, row in subset.iterrows():
            try:
                x, y = int(row['xpos']), int(row['ypos'])
                response = row['responseTime']
                if response > stim_min and response < stim_max:
                    grid[y, x] = 1  # White for visible
                elif response == 0 or response < stim_min or response > stim_max:
                    grid[y, x] = 0  # Black for non-visible
            except IndexError:
                # Ignore out-of-bounds values and continue
                print(f"Skipping out-of-bounds position (x={x}, y={y}) for patient: {unique_pat_id}")
                continue
            
        # Handle missing points (-1) based on fixation response time
        for y in range(max_y):
            for x in range(max_x):
                if grid[y, x] == -1 and fixation_response_time is not None:  # Missing point and valid fixation data
                    if fixp_min <= fixation_response_time <= fixp_max:
                        grid[y, x] = 1  # White if fixation response time is in range
                    else:
                        grid[y, x] = 0  # Black if not in range

        grids.append(grid)

    combined_grid = np.mean(grids, axis=0)

    # Plot the combined grid
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_grid, cmap=cmap, origin='upper', extent=[0, max_x, max_y, 0], vmin=-1, vmax=1)
    plt.title(
        f"Patient: {first_name} {last_name}\n"
        f"Eye: {eye}, Combined ResultSets: {result_sets[-2:]}, Date: {test_date}\n"
        f"Stim Range: Min={stim_min} ms, Max={stim_max} ms",
        fontsize=10
    )
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.xticks(range(max_x))
    plt.yticks(range(max_y))
    plt.grid(visible=True)

    # Save the combined grid
    output_file_combined = f'{output_dir}/grid_{sanitized_first_name}_{sanitized_last_name}_{unique_pat_id}_{eye}_combinedresultsets_{result_sets[-2:]}_date_{test_date}.png'
    plt.savefig(output_file_combined)
    plt.close()

    print(f"Combined Grid saved: {output_file_combined}")

print(f"All grids saved in {output_dir}")
