import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math


# Plotting images


def show_imgs(imgs, title=None, row_size=4):
    # Form a grid of pictures (we use max. 8 columns)
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
    is_int = imgs.dtype == torch.int32 if isinstance(imgs, torch.Tensor) else imgs[0].dtype == torch.int32
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs/nrow))
    imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128 if is_int else 0.5)
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()


def show_samples(flow, img_shape, sample_shape_factor, num_samples_to_show):
    batched_img_shape = torch.cat((torch.tensor([num_samples_to_show]), torch.tensor(img_shape)))
    sample_shape = torch.Size((batched_img_shape * sample_shape_factor).int())
    samples = flow.sample(sample_shape=sample_shape).cpu()
    show_imgs(samples.cpu(), row_size=num_samples_to_show)


# Plotting histogram


def num_bins(x):
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    bins = round((x.max() - x.min()) / bin_width)
    return bins


def plot_hist(x):
    bins = num_bins(x)
    plt.hist(x, bins=bins)
    plt.show()


# Masks for Coupling Layer


def create_checkerboard_mask(h, w, invert=False):
    h_range, w_range = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    hh, ww = torch.meshgrid(h_range, w_range, indexing='ij')
    mask = torch.fmod(hh + ww, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if invert:
        mask = 1 - mask
    return mask


def create_channel_mask(c_in, invert=False):
    mask = torch.cat([torch.ones(c_in//2, dtype=torch.float32),
                      torch.zeros(c_in-c_in//2, dtype=torch.float32)])
    mask = mask.view(1, c_in, 1, 1)
    if invert:
        mask = 1 - mask
    return mask


def visualize_masks():
    checkerboard_mask = create_checkerboard_mask(h=8, w=8).expand(-1, 2, -1, -1)
    channel_mask = create_channel_mask(c_in=2).expand(-1, -1, 8, 8)

    show_imgs(checkerboard_mask.transpose(0, 1), "Checkerboard mask")
    show_imgs(channel_mask.transpose(0, 1), "Channel mask")


def print_num_params(model):
    num_params = sum([np.prod(p.shape) for p in model.parameters()])
    print("Number of parameters: {:,}".format(num_params))


@torch.no_grad()
def interpolate(model, img1, img2, num_steps=8):
    """
    Inputs:
        model - object of ImageFlow class that represents the (trained) flow model
        img1, img2 - Image tensors of shape [1, 28, 28]. Images between which should be interpolated.
        num_steps - Number of interpolation steps. 8 interpolation steps mean 6 intermediate pictures besides img1 and img2
    """
    imgs = torch.stack([img1, img2], dim=0).to(model.device)
    z, _ = model.encode(imgs)
    alpha = torch.linspace(0, 1, steps=num_steps, device=z.device).view(-1, 1, 1, 1)
    interpolations = z[0:1] * alpha + z[1:2] * (1 - alpha)
    interp_imgs = model.sample(interpolations.shape[:1] + imgs.shape[1:], z_init=interpolations)
    show_imgs(interp_imgs, row_size=8)
