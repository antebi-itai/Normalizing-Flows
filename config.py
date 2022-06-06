import os
import torch
from types import SimpleNamespace


config_dict = {
    # GPU
    'device': torch.device("cuda:0"),
    'gpu_num': 0,

    # Paths / URLs
    'DATASET_PATH': "/home/itaian/group/datasets",  # Path to the folder where the datasets are/should be downloaded
    'RESULTS_PATH': "/home/itaian/data/results/NF_tutorial",
    'CHECKPOINT_PATH': "./saved_models",  # Path to the folder where the pretrained models are saved
    'PL_TRAINER_PATH': "./pl_trainer",
    'PRETRAINED_MODELS_BASE_URL': "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial11/",  # Github URL where saved models are stored for this tutorial
    'PRETRAINED_MODELS_FILES': ["MNISTFlow_simple.ckpt", "MNISTFlow_vardeq.ckpt", "MNISTFlow_multiscale.ckpt"],  # Files to download

    # Model
    'train': False,
    'size': 28,  # 10 / 28
    'model_name': "MNISTFlow_vardeq",  # MNISTFlow_simple / MNISTFlow_vardeq / MNISTFlow_multiscale
    'epochs': 5,  # 5 / 200

    'use_web_weights': False,
    'num_samples_to_show': 10
}


# dict to namespace
config = SimpleNamespace(**config_dict)

# validate config
assert not (config.train and config.use_web_weights), "Training and Using the web weights are mutually exclusive"

# add config-dependent configurations
unique_filename = config.model_name + \
                  ("" if config.use_web_weights else f"_size_{config.size}") + \
                  ("" if config.use_web_weights else f"_epochs_{config.epochs}")
config.trained_filepath = os.path.join(config.CHECKPOINT_PATH, unique_filename + ".ckpt")
config.results_filepath = os.path.join(config.RESULTS_PATH, unique_filename + ".png")
config.log_dir = os.path.join(config.PL_TRAINER_PATH, config.model_name)
