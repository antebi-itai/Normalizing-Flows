import urllib.request
from urllib.error import HTTPError
import os
from config import CHECKPOINT_PATH, PRETRAINED_MODELS_BASE_URL, PRETRAINED_MODELS_FILES


def download_web_weights():
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in PRETRAINED_MODELS_FILES:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if not os.path.isfile(file_path):
            file_url = PRETRAINED_MODELS_BASE_URL + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please try to download the file from the GDrive folder, " +
                      "or contact the author with the full output including the following error:\n", e)
    print("Done Downloading Pre-Trained Models!")
