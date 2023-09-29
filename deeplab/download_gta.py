import threading
import requests
from tqdm import tqdm
import argparse
import os
import ntpath

def download_file(url, output_dir):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    # Extract the file name from the URL and remove the extension
    file_name = ntpath.basename(url)
    t = tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name)

    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

def main():
    parser = argparse.ArgumentParser(description="Parallel file downloader with a progress bar.")
    parser.add_argument("output_dir", type=str, help="Directory where downloaded files will be saved")
    args = parser.parse_args()

    # List of URLs to download
    url = "https://download.visinf.tu-darmstadt.de/data/from_games/data/{:02d}_{}.zip"
    urls = [url.format(part, content) for part in range(1,11) for content in ("images", "labels")]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create threads for downloading
    threads = []
    for url in urls:
        thread = threading.Thread(target=download_file, args=(url, args.output_dir))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("All downloads completed.")

if __name__ == "__main__":
    main()
