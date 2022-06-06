import urllib.request
from urllib.error import HTTPError
import os


def download_web_weights(config):
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in config.PRETRAINED_MODELS_FILES:
        file_path = os.path.join(config.CHECKPOINT_PATH, file_name)
        if not os.path.isfile(file_path):
            file_url = config.PRETRAINED_MODELS_BASE_URL + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please try to download the file from the GDrive folder, " +
                      "or contact the author with the full output including the following error:\n", e)
    print("Done Downloading Pre-Trained Models!")
