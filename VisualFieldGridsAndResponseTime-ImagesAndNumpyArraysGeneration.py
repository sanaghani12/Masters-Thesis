import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

# Load the data
file_path = 'D:\\sana\\Thesis\\Savir_Data\\QueryResults2.csv'
data = pd.read_csv(file_path)

# Create output directories for images and arrays
output_dir = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog'
array_output_dir = os.path.join(output_dir, 'arrays')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(array_output_dir, exist_ok=True)

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

    # Plot and save individual grids
    for result_set in result_sets:
        subset = group[group['ResultSet'] == result_set]
        max_x = sorted(set(subset['xpos'].astype(int)))[-2] + 1
        max_y = sorted(set(subset['ypos'].astype(int)))[-2] + 1

        grid = np.full((max_y, max_x), -1, dtype=np.float32)  # -1 for missing points
        response_time_grid = np.zeros((max_y, max_x), dtype=np.float32)

        stim_min = subset['stim_minReactionTime'].iloc[0]
        stim_max = subset['stim_maxReactionTime'].iloc[0]

        # Fetch fixation row
        fixation_row = subset[(subset['xpos'] == subset['xpos'].max()) & (subset['ypos'] == subset['ypos'].max())]
        if not fixation_row.empty:
            fixation_response_time = fixation_row['responseTime'].iloc[0]
            fixp_min = fixation_row['fixp_minReactionTime'].iloc[0]
            fixp_max = fixation_row['fixp_maxReactionTime'].iloc[0]
        else:
            fixation_response_time, fixp_min, fixp_max = None, None, None

        for _, row in subset.iterrows():
            x, y = int(row['xpos']), int(row['ypos'])
            response = row['responseTime']

            if 0 <= x < max_x and 0 <= y < max_y:  # Handle out-of-bounds points
                if stim_min <= response <= stim_max:
                    grid[y, x] = 1  # Visible
                    response_time_grid[y, x] = response
                else:
                    grid[y, x] = 0  # Non-visible

        # Handle missing points based on fixation response time
        if fixation_response_time is not None:
            for y in range(max_y):
                for x in range(max_x):
                    if grid[y, x] == -1:
                        if fixp_min <= fixation_response_time <= fixp_max:
                            grid[y, x] = 1  # Assume visible
                        else:
                            grid[y, x] = 0  # Assume non-visible

        # Save individual grid arrays
        np.save(os.path.join(array_output_dir, f"grid_{sanitized_first_name}_{sanitized_last_name}_{unique_pat_id}_{eye}_resultset_{result_set}.npy"), grid)
        np.save(os.path.join(array_output_dir, f"response_{sanitized_first_name}_{sanitized_last_name}_{unique_pat_id}_{eye}_resultset_{result_set}.npy"), response_time_grid)

        # Save grid as an image
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

        output_file_individual = f'{output_dir}/grid_{sanitized_first_name}_{sanitized_last_name}_{unique_pat_id}_{eye}_resultset_{result_set}.png'
        plt.savefig(output_file_individual)
        plt.close()

        print(f"Individual Grid saved: {output_file_individual}")

    # Handle combined grids for the last two tests
    if len(result_sets) > 2:
        group = group[group['ResultSet'].isin(result_sets[-2:])]

    if len(result_sets) < 2:
        print(f"Skipping combined grid for {first_name} {last_name}, {eye}, {test_date}: Only one test exists.")
        continue

    grids = []
    response_times = []

    for result_set in result_sets[-2:]:
        subset = group[group['ResultSet'] == result_set]
        grid = np.full((max_y, max_x), -1, dtype=np.float32)
        response_time_grid = np.zeros((max_y, max_x), dtype=np.float32)

        for _, row in subset.iterrows():
            x, y = int(row['xpos']), int(row['ypos'])
            response = row['responseTime']

            if 0 <= x < max_x and 0 <= y < max_y:  # Handle out-of-bounds points
                if stim_min <= response <= stim_max:
                    grid[y, x] = 1
                    response_time_grid[y, x] = response
                else:
                    grid[y, x] = 0

        # Handle missing points based on fixation response time
        if fixation_response_time is not None:
            for y in range(max_y):
                for x in range(max_x):
                    if grid[y, x] == -1:
                        if fixp_min <= fixation_response_time <= fixp_max:
                            grid[y, x] = 1  # Assume visible
                        else:
                            grid[y, x] = 0  # Assume non-visible

        grids.append(grid)
        response_times.append(response_time_grid)

    combined_grid = np.mean(grids, axis=0)
    mean_response_time_grid = np.mean(response_times, axis=0)

    # Save combined grid arrays
    np.save(os.path.join(array_output_dir, f"combined_grid_{sanitized_first_name}_{sanitized_last_name}_{unique_pat_id}_{eye}_date_{test_date}.npy"), combined_grid)
    np.save(os.path.join(array_output_dir, f"combined_response_{sanitized_first_name}_{sanitized_last_name}_{unique_pat_id}_{eye}_date_{test_date}.npy"), mean_response_time_grid)

    # Save combined grid as an image
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

    output_file_combined = f'{output_dir}/combined_grid_{sanitized_first_name}_{sanitized_last_name}_{unique_pat_id}_{eye}_date_{test_date}.png'
    plt.savefig(output_file_combined)
    plt.close()

    print(f"Combined Grid saved: {output_file_combined}")

print(f"All grids and arrays saved in {output_dir} and {array_output_dir}")
