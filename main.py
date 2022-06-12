print("Starting main...")
import matplotlib_inline.backend_inline
import matplotlib
import seaborn as sns
import torch
import pytorch_lightning as pl
from config import config
from dataset import get_data
from network import create_flow
from train import train_flow, load_flow
from tools import sample_save_show, print_result, make_cuda_visible
print("Done Importing!")

# Environment Settings
matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')
matplotlib.rcParams['lines.linewidth'] = 2.0
sns.reset_orig()
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Setting the seed
pl.seed_everything(42)
# GPU
make_cuda_visible(gpu_num=config.gpu_num)


def main():
    # load datasets and data-loaders
    train_set, val_set, test_set, train_loader, val_loader, test_loader = get_data(config=config)

    # build network
    flow, sample_shape_factor = create_flow(config=config)

    if config.train:
        # train network on dataset
        flow, result = train_flow(flow=flow, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, config=config)

        # save newly trained model
        torch.save({'state_dict': flow.state_dict(), 'result': result}, config.trained_filepath)
        print(f"Done Saving Network to {config.trained_filepath}!")

    else:
        # load network weights
        flow, result = load_flow(flow=flow, config=config)

    # print train, val and test bpd
    print_result(result)

    # save & show some samples of the flow
    sample_save_show(flow=flow, img_shape=test_set[0][0].shape, sample_shape_factor=sample_shape_factor, config=config)

    print("Done Main!")


if __name__ == "__main__":
    main()
