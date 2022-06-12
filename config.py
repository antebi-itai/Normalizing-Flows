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
    'PL_LOG_PATH': "./pl_trainer",

    # Data
    'dataset': "PEIP",  # MNIST / PEIP / <image_name> like NATURE / CHEETAH / ...

    # Model
    'train': True,
    'size': 10,  # 5 / 10 / 28
    'model_name': "long",  # simple / vardeq / long / linear / multiscale
    'epochs': 5,  # 5 / 200

    # Sample
    'num_samples': 64,
    'save_samples': True,
    'show_samples': False
}


# dict to namespace
config = SimpleNamespace(**config_dict)

# add config-dependent configurations
config.unique_filename = f"{config.dataset}_{config.model_name}_size_{config.size}_epochs_{config.epochs}"
config.trained_filepath = os.path.join(config.CHECKPOINT_PATH, config.unique_filename + ".ckpt")
config.results_filepath = os.path.join(config.RESULTS_PATH, config.unique_filename + ".png")
config.c = 1 if config.dataset == "MNIST" else 3  # input data's number of channels


def print_running_config(config, print_keys=('dataset', 'train', 'size', 'model_name', 'epochs')):
    print()
    print("="*20 + "  CONFIG  " + "="*20)
    max_len_key = max([len(key) for key in print_keys])
    for key in print_keys:
        print(f"{key}:{' '*(max_len_key + 2 - len(key))}{config.__dict__[key]}")
    print("=" * 50)
    print()
