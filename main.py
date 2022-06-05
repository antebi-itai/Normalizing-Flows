print("Starting main...")
import matplotlib_inline.backend_inline
import matplotlib
import seaborn as sns
import torch
import pytorch_lightning as pl
from config import device, model_name, train, trained_filepath, num_samples_to_show
from pretrained_models import download_web_weights
from dataset import get_dataset_loaders
from network import create_flow
from train import train_flow, load_flow
from tools import show_samples, print_result
print("Done Importing!")

# CONFIGURATIONS
matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')
matplotlib.rcParams['lines.linewidth'] = 2.0
sns.reset_orig()
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Setting the seed
pl.seed_everything(42)


def main():
    # download weights from the web
    download_web_weights()

    # load MNIST datasets and data-loaders
    train_set, val_set, test_set, train_loader, val_loader, test_loader = get_dataset_loaders()

    # build network
    flow, sample_shape_factor = create_flow(model_name=model_name, device=device)

    if train:
        # train network on dataset
        flow, result = train_flow(flow=flow, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, model_name=model_name)

        # save newly trained model
        torch.save({'state_dict': flow.state_dict(), 'result': result}, trained_filepath)
        print(f"Done Saving Network to {trained_filepath}!")

    else:
        # load network weights
        flow, result = load_flow(flow)
        print_result(result)

        # show some samples of the flow
        show_samples(flow=flow, img_shape=test_set[0][0].shape, sample_shape_factor=sample_shape_factor, num_samples_to_show=num_samples_to_show)

    print("Done Main!")


if __name__ == "__main__":
    main()