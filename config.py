import os
import torch

device = torch.device("cuda:0")  # "cuda:0" / "cpu"

## Paths / URLs
DATASET_PATH = "/home/itaian/group/datasets"  # Path to the folder where the datasets are/should be downloaded
CHECKPOINT_PATH = "./saved_models"  # Path to the folder where the pretrained models are saved
PL_TRAINER_PATH = "./pl_trainer"
PRETRAINED_MODELS_BASE_URL = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial11/"  # Github URL where saved models are stored for this tutorial
PRETRAINED_MODELS_FILES = ["MNISTFlow_simple.ckpt", "MNISTFlow_vardeq.ckpt", "MNISTFlow_multiscale.ckpt"]  # Files to download

## Model
model_name = "MNISTFlow_multiscale"  # MNISTFlow_simple / MNISTFlow_vardeq / MNISTFlow_multiscale
train = False
use_web_weights = False
assert not (train and use_web_weights), "Training and Using the web weights are mutually exclusive"
epochs = 200
trained_filepath = os.path.join(CHECKPOINT_PATH, model_name +
                                ("" if use_web_weights else f"_<{epochs}_epochs>") + ".ckpt")

num_samples_to_show = 10
