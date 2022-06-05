import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from config import DATASET_PATH


# Convert images from 0-1 to 0-255 (integers)
def discretize(sample):
    return (sample * 255).to(torch.int32)


def get_dataset_loaders():
    # Transformations applied on each image => make them a tensor and discretize
    transform = transforms.Compose([transforms.ToTensor(),
                                    discretize])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
    # Loading the test set
    test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    # Note that for actually training a model, we will use different data loaders
    # with a lower batch size.
    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=False, num_workers=8)
    val_loader = data.DataLoader(val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
    print("Done Loading Dataset!")
    return train_set, val_set, test_set, train_loader, val_loader, test_loader
