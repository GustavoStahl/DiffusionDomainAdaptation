import os
import scipy.io
import argparse
from tqdm import tqdm

def create_splits(mat_file_path, input_dir, output_dir):
    # Load the MATLAB file
    mat_file = scipy.io.loadmat(mat_file_path)

    # Define split names
    split_keys = ["trainIds", "valIds", "testIds"]
    split_names = ["train", "val", "test"]

    for folder_name in ["images", "labels"]:
        source_folder = os.path.join(input_dir, folder_name)
        
        for split_key, split_name in zip(split_keys, split_names):
            split_dir = os.path.join(output_dir, folder_name, split_name)
            os.makedirs(split_dir, exist_ok=True)

            # Get split indexes
            split_indexes = mat_file[split_key].flatten()

            # Move files with matching indices from source to destination
            for index in tqdm(split_indexes, desc=split_name):
                file_name = f"{index:04d}.png"
                source_file_path = os.path.abspath(os.path.join(source_folder, file_name))
                destination_file_path = os.path.abspath(os.path.join(split_dir, file_name))

                # Move the file if it exists
                if os.path.exists(source_file_path):
                    os.symlink(source_file_path, destination_file_path)

    print("Splits and symbolic links created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset splits with symbolic links")
    parser.add_argument("mat_file_path", type=str, help="Path to the MATLAB file")
    parser.add_argument("input_dir", type=str, help="Input directory containing 'images' and 'labels' folders")
    parser.add_argument("output_dir", type=str, help="Output directory where splits will be created")
    args = parser.parse_args()

    create_splits(args.mat_file_path, args.input_dir, args.output_dir)
