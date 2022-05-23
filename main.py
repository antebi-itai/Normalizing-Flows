print("Starting main...")
import matplotlib_inline.backend_inline
import matplotlib
import seaborn as sns
import torch
import pytorch_lightning as pl
from config import device, model_name, pretrained, trained_filepath, num_samples_to_show
from pretrained_models import download_pretrained_models
from dataset import get_dataset_loaders
from network import create_flow
from train import train_flow
from tools import show_samples
print("Done Importing!")

# CONFIGURATIONS
matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')
matplotlib.rcParams['lines.linewidth'] = 2.0
sns.reset_orig()
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
# Setting the seed
pl.seed_everything(42)


def main():
    # download pre-trained models
    download_pretrained_models()

    # load MNIST datasets and data-loaders
    train_set, val_set, test_set, train_loader, val_loader, test_loader = get_dataset_loaders()

    # build network
    net, sample_shape_factor = create_flow(model_name=model_name, device=device)

    # train network on dataset
    flow, result = train_flow(flow=net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                              model_name=model_name, pretrained=pretrained)

    # save newly trained model
    if not pretrained:
        torch.save({'state_dict': flow.state_dict(), 'result': result}, trained_filepath)
        print("Done Saving Network!")

    # show some samples of the flow
    show_samples(flow=flow, img_shape=test_set[0][0].shape, sample_shape_factor=sample_shape_factor, num_samples_to_show=num_samples_to_show)

    print("Done Main!")


if __name__ == "__main__":
    main()
