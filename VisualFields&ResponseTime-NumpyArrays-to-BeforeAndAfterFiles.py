import os
import re
from datetime import datetime
from shutil import copy

# Define input folders containing the files
input_folder_grids = 'D:\\sana\\Thesis\\Savir_Data\\output_gridsog'
input_folder_arrays = os.path.join(input_folder_grids, 'arrays')

# Define output folders for images and npy files
output_folders = {
    "combined_grid": {
        "before": os.path.join(input_folder_grids, 'before_grids'),
        "after": os.path.join(input_folder_grids, 'after_grids')
    },
    "combined_response": {
        "before": os.path.join(input_folder_grids, 'before_responses'),
        "after": os.path.join(input_folder_grids, 'after_responses')
    },
    "png": {
        "before": os.path.join(input_folder_grids, 'before_images'),
        "after": os.path.join(input_folder_grids, 'after_images')
    }
}

# Create output folders if they do not exist
for folder_type in output_folders.values():
    for folder in folder_type.values():
        os.makedirs(folder, exist_ok=True)

# Regex to parse filenames for grids, responses, and images
file_pattern = re.compile(
    r"(?P<type>(combined_grid|combined_response))_(?P<name>.+?)_date_(?P<date>\d{4}-\d{2}-\d{2})\.(?P<extension>npy|png)"
)


# Function to process `combined_grid` files
def process_combined_grids():
    grid_files = {}

    # Gather all combined grid files
    for folder in [input_folder_arrays]:
        for filename in os.listdir(folder):
            match = file_pattern.match(filename)
            if match and match.group('type') == 'combined_grid':
                patient_name = match.group('name')
                file_date = datetime.strptime(match.group('date'), '%Y-%m-%d')
                file_extension = match.group('extension')
                file_details = (filename, file_date, file_extension, folder)
                grid_files.setdefault(patient_name, []).append(file_details)

    # Process files
    for patient, files in grid_files.items():
        files.sort(key=lambda x: x[1])  # Sort by date

        if len(files) > 2:
            print(f"\nPatient '{patient}' with type 'combined_grid' has more than 2 files ({len(files)} files):")
            for file_info in files:
                print(f"  - {file_info[0]} (Date: {file_info[1]}, Extension: {file_info[2]})")

        if len(files) >= 2:  # Ensure at least two files exist
            before_file = files[0]
            after_file = files[-1]

            # Copy files
            copy(os.path.join(before_file[3], before_file[0]), os.path.join(output_folders['combined_grid']['before'], before_file[0]))
            copy(os.path.join(after_file[3], after_file[0]), os.path.join(output_folders['combined_grid']['after'], after_file[0]))

            print(f"Copied '{before_file[0]}' to '{output_folders['combined_grid']['before']}'")
            print(f"Copied '{after_file[0]}' to '{output_folders['combined_grid']['after']}'")


# Function to process `combined_response` files
def process_combined_responses():
    response_files = {}

    # Gather all combined response files
    for folder in [input_folder_arrays]:
        for filename in os.listdir(folder):
            match = file_pattern.match(filename)
            if match and match.group('type') == 'combined_response':
                patient_name = match.group('name')
                file_date = datetime.strptime(match.group('date'), '%Y-%m-%d')
                file_extension = match.group('extension')
                file_details = (filename, file_date, file_extension, folder)
                response_files.setdefault(patient_name, []).append(file_details)

    # Process files
    for patient, files in response_files.items():
        files.sort(key=lambda x: x[1])  # Sort by date

        if len(files) > 2:
            print(f"\nPatient '{patient}' with type 'combined_response' has more than 2 files ({len(files)} files):")
            for file_info in files:
                print(f"  - {file_info[0]} (Date: {file_info[1]}, Extension: {file_info[2]})")

        if len(files) >= 2:  # Ensure at least two files exist
            before_file = files[0]
            after_file = files[-1]

            # Copy files
            copy(os.path.join(before_file[3], before_file[0]), os.path.join(output_folders['combined_response']['before'], before_file[0]))
            copy(os.path.join(after_file[3], after_file[0]), os.path.join(output_folders['combined_response']['after'], after_file[0]))

            print(f"Copied '{before_file[0]}' to '{output_folders['combined_response']['before']}'")
            print(f"Copied '{after_file[0]}' to '{output_folders['combined_response']['after']}'")


# Function to process `.png` image files
def process_images():
    image_files = {}

    # Gather all image files
    for folder in [input_folder_grids]:
        for filename in os.listdir(folder):
            match = file_pattern.match(filename)
            if match and match.group('extension') == 'png':
                patient_name = match.group('name')
                file_date = datetime.strptime(match.group('date'), '%Y-%m-%d')
                file_extension = match.group('extension')
                file_details = (filename, file_date, file_extension, folder)
                image_files.setdefault(patient_name, []).append(file_details)

    # Process files
    for patient, files in image_files.items():
        files.sort(key=lambda x: x[1])  # Sort by date

        if len(files) > 2:
            print(f"\nPatient '{patient}' with type 'png' has more than 2 files ({len(files)} files):")
            for file_info in files:
                print(f"  - {file_info[0]} (Date: {file_info[1]}, Extension: {file_info[2]})")

        if len(files) >= 2:  # Ensure at least two files exist
            before_file = files[0]
            after_file = files[-1]

            # Copy files
            copy(os.path.join(before_file[3], before_file[0]), os.path.join(output_folders['png']['before'], before_file[0]))
            copy(os.path.join(after_file[3], after_file[0]), os.path.join(output_folders['png']['after'], after_file[0]))

            print(f"Copied '{before_file[0]}' to '{output_folders['png']['before']}'")
            print(f"Copied '{after_file[0]}' to '{output_folders['png']['after']}'")


# Run all three functions
print("\nProcessing combined_grid files:")
process_combined_grids()

print("\nProcessing combined_response files:")
process_combined_responses()

print("\nProcessing .png image files:")
process_images()

print("\nFiles have been copied into 'before' and 'after' folders for grids, responses, and images.")
