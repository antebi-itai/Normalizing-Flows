import os
import torch
from types import SimpleNamespace


config_dict = {
    # GPU
    'device': torch.device("cuda:0"),
    'gpu_num': 0,

    # Paths / URLs
    'DATASET_PATH': "/home/itaian/group/datasets",  # Path to the folder where the datasets are/should be downloaded
    'DATA_IMAGES_PATH': "/home/itaian/data/images/Normalizing_Flow",
    'RESULTS_PATH': "/home/itaian/data/results/NF_tutorial",
    'CHECKPOINT_PATH': "./saved_models",  # Path to the folder where the pretrained models are saved
    'PL_TRAINER_PATH': "./pl_trainer",

    # Data
    'dataset': "PEIP",  # MNIST / PEIP / <image_name> like NATURE / CHEETAH / ...

    # Model
    'train': True,
    'size': 10,  # 5 / 10 / 28
    'model_name': "vardeq",  # simple / vardeq / multiscale
    'epochs': 5,  # 5 / 200

    # Sample
    'num_samples': 64,
    'save_samples': True,
    'show_samples': False
}


# dict to namespace
config = SimpleNamespace(**config_dict)

# add config-dependent configurations
unique_filename = f"{config.dataset}_{config.model_name}_size_{config.size}_epochs_{config.epochs}"
config.trained_filepath = os.path.join(config.CHECKPOINT_PATH, unique_filename + ".ckpt")
config.results_filepath = os.path.join(config.RESULTS_PATH, unique_filename + ".png")
config.log_dir = os.path.join(config.PL_TRAINER_PATH, config.model_name)
config.c = 1 if config.dataset == "MNIST" else 3  # input data's number of channels
