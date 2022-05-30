import torch
from flow_template import ImageFlow
from flow_dequantization import Dequantization, VariationalDequantization
from flow_models import CouplingLayer, SqueezeFlow, SplitFlow
from nn_layers import GatedConvNet
from tools import create_checkerboard_mask, create_channel_mask


def create_simple_flow(use_vardeq=True):
    flow_layers = []
    if use_vardeq:
        vardeq_layers = [CouplingLayer(network=GatedConvNet(c_in=2, c_out=2, c_hidden=16),
                                       mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                                       c_in=1) for i in range(4)]
        flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]
    else:
        flow_layers += [Dequantization()]

    for i in range(8):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=1, c_hidden=32),
                                      mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                                      c_in=1)]

    flow_model = ImageFlow(flow_layers)
    sample_shape_factor = torch.tensor([1, 1, 1, 1])
    return flow_model, sample_shape_factor


def create_multiscale_flow():
    flow_layers = []

    vardeq_layers = [CouplingLayer(network=GatedConvNet(c_in=2, c_out=2, c_hidden=16),
                                   mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                                   c_in=1) for i in range(4)]
    flow_layers += [VariationalDequantization(vardeq_layers)]

    flow_layers += [CouplingLayer(network=GatedConvNet(c_in=1, c_hidden=32),
                                  mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                                  c_in=1) for i in range(2)]
    flow_layers += [SqueezeFlow()]
    for i in range(2):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=4, c_hidden=48),
                                      mask=create_channel_mask(c_in=4, invert=(i % 2 == 1)),
                                      c_in=4)]
    flow_layers += [SplitFlow(),
                    SqueezeFlow()]
    for i in range(4):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=8, c_hidden=64),
                                      mask=create_channel_mask(c_in=8, invert=(i % 2 == 1)),
                                      c_in=8)]

    flow_model = ImageFlow(flow_layers)
    sample_shape_factor = torch.tensor([1, 8, 0.25, 0.25])
    return flow_model, sample_shape_factor


def create_flow(model_name, device):
    if model_name == "MNISTFlow_simple":
        net, sample_shape_factor = create_simple_flow(use_vardeq=False)
    elif model_name == "MNISTFlow_vardeq":
        net, sample_shape_factor = create_simple_flow(use_vardeq=True)
    elif model_name == "MNISTFlow_multiscale":
        net, sample_shape_factor = create_multiscale_flow()
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")

    net = net.to(device=device)
    print(f"Done Creating Network! (model: {model_name})")
    return net, sample_shape_factor
