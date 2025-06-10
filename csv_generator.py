"""
Artery Vein Network (AV-Net) CSV Generator.

This script generates `train.csv` and `test.csv` files based on image names
found in a specified base directory. It assumes all input channels
(e.g., 'oct', 'octa', 'gt') share the same filenames.

"""

import pandas as pd
from pathlib import Path

def generate_image_csvs(base_dir_for_filenames="dataset_dir/oct", train_ratio=0.7):
    """
    Generates train.csv and test.csv files containing image names.

    Args:
        base_dir_for_filenames: String or Path, the base directory to scan for image filenames.
                                This directory should contain a representative set of all image names.
        train_ratio: Float, the proportion of data to use for the training set (e.g., 0.7 for 70%).
    """
    image_names = []
    basepath = Path(base_dir_for_filenames)

    if not basepath.is_dir():
        print(f"Error: Base directory '{base_dir_for_filenames}' not found.")
        return

    # Collect all file names in the base directory
    for item in basepath.iterdir():
        if item.is_file():
            image_names.append(item.name)
    
    if not image_names:
        print(f"Warning: No image files found in '{base_dir_for_filenames}'. No CSVs generated.")
        return

    # Create a DataFrame from the collected names
    dataframe = pd.DataFrame({'image_names': image_names})
    
    # Shuffle the DataFrame for a more robust split
    dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the DataFrame into training and testing sets
    n_length = int(len(dataframe) * train_ratio)
    dataframe_train = dataframe.iloc[:n_length]
    dataframe_test = dataframe.iloc[n_length:]

    # Save to CSV files
    dataframe_train.to_csv("train.csv", index=False)
    dataframe_test.to_csv("test.csv", index=False)
    print(f"Generated train.csv with {len(dataframe_train)} entries.")
    print(f"Generated test.csv with {len(dataframe_test)} entries.")

if __name__ == "__main__":
    # Example usage:
    # Make sure 'dataset_dir/oct' exists and contains your image files
    generate_image_csvs(base_dir_for_filenames="dataset_dir/oct", train_ratio=0.8) # Consistent with main.py's split