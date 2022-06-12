import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
import os
from ResizeRight.resize_right import resize
from images import imread


############################## MNIST ##############################


# Return a transform that resizes the image to shape [1, size, size]
def get_resize_transform(out_size):
    def resize_transform(tensor_image):
        assert tensor_image.shape == torch.Size([1, 28, 28])
        return resize(input=tensor_image, scale_factors=out_size / 28, out_shape=[1, out_size, out_size]).clip(min=0, max=1)
    return resize_transform


# Convert images from 0-1 to 0-255 (integers)
def discretize(sample):
    return (sample * 255).to(torch.int32)


def get_mnist_datasets(config):
    # Transformations applied on each image => make them a tensor and discretize
    transform = transforms.Compose([transforms.ToTensor(),
                                    get_resize_transform(out_size=config.size),
                                    discretize])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = MNIST(root=config.DATASET_PATH, train=True, transform=transform, download=True)
    pl.seed_everything(42)
    train_set, val_set = data.random_split(train_dataset, [50000, 10000])
    # Loading the test set
    test_set = MNIST(root=config.DATASET_PATH, train=False, transform=transform, download=True)

    return train_set, val_set, test_set


############################## PATCHES ##############################


def get_positional_encoding_image():
    (x, y) = torch.meshgrid(torch.linspace(0, 1, 256), torch.linspace(0, 1, 256))
    red = 0.5 + 0.5 * torch.sin(75 * torch.pi * ((x - 0.5).pow(2) + (y - 0.5).pow(2)))  # circles around center
    green = torch.linspace(0, 1, 256).view(1, -1).repeat(256, 1)  # x
    blue = torch.linspace(0, 1, 256).view(-1, 1).repeat(1, 256)  # y
    image = torch.stack([red, green, blue])
    return image


class PatchesDataset(data.Dataset):
    def __init__(self, image, config):
        patches = F.unfold(input=image.unsqueeze(0), kernel_size=config.size).squeeze().t()
        self.patches = patches.reshape(patches.size(0), 3, config.size, config.size)  # num_patches * 3 * k * k
        self.transform = transforms.Compose([discretize])

    def __len__(self):
        return self.patches.size(0)

    def __getitem__(self, idx):
        return self.transform(self.patches[idx]), 0


def get_patches_dataset(image, config):
    # image to dataset of patches
    dataset = PatchesDataset(image=image, config=config)

    # split dataset to train (5/7), val (1/7) and test (1/7)
    val_len = test_len = len(dataset) // 7
    train_len = len(dataset) - val_len - test_len
    train_set, val_set, test_set = data.random_split(dataset, [train_len, val_len, test_len])

    return train_set, val_set, test_set


############################## GENERAL ##############################


def get_data_loaders(train_set, val_set, test_set):
    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=False, num_workers=8)
    val_loader = data.DataLoader(val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    return train_loader, val_loader, test_loader


def get_data(config):
    if config.dataset == "MNIST":
        train_set, val_set, test_set = get_mnist_datasets(config=config)
    elif config.dataset == "PEIP":
        image = get_positional_encoding_image()
        train_set, val_set, test_set = get_patches_dataset(image=image, config=config)
    else:
        image_path = os.path.join(config.DATA_IMAGES_PATH, f"{config.dataset}.png")
        assert os.path.isfile(image_path), f"Image file {image_path} not found"
        image = imread(fname=image_path).squeeze()
        train_set, val_set, test_set = get_patches_dataset(image=image, config=config)

    train_loader, val_loader, test_loader = get_data_loaders(train_set=train_set, val_set=val_set, test_set=test_set)

    print("Done Loading Dataset!")
    return train_set, val_set, test_set, train_loader, val_loader, test_loader
