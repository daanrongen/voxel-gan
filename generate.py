import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from model import _3DGAN, Generator, Discriminator
import argparse
import numpy as np
import tqdm
import os
import os.path
from visualisation.util import *
from visualisation.util_vtk import visualisation
from matplotlib import cm
import sys
import scipy.ndimage

def get_voxels(voxels):
    dims = voxels.shape

    if len(dims) == 5:
        assert dims[1] == 1
        dims = (dims[0],) + tuple(dims[2:])
    elif len(dims) == 3:
        dims = [1] + list(dims)
    else:
        assert len(dims) == 4

    result = np.reshape(voxels, dims)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="""Visualizing .mat voxel file. """)
    parser.add_argument(
        "-n",
        "--network",
        default="models/G_iter_000500.pth",
        type=str,
        help="Specify network",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="Seed",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        metavar="threshold",
        type=float,
        default=0.17,
        help="voxels with confidence lower than the threshold are not displayed",
    )
    parser.add_argument(
        "-o", "--out_file", metavar="out_file", type=str, required=True
    )
    parser.add_argument(
        "-i",
        "--index",
        metavar="index",
        type=int,
        default=1,
        help="the index of objects in the inputfile that should be rendered (one based)",
    )
    parser.add_argument(
        "-df",
        "--downsample-factor",
        metavar="factor",
        type=int,
        default=1,
        help="downsample objects via a max pooling of step STEPSIZE for efficiency. Currently supporting STEPSIZE 1, 2, and 4.",
    )
    parser.add_argument(
        "-dm",
        "--downsample-method",
        metavar="downsample_method",
        type=str,
        default="max",
        help="downsample method, where mean stands for average pooling and max for max pooling",
    )
    parser.add_argument(
        "-u",
        "--uniform-size",
        metavar="uniform_size",
        type=float,
        default=1.0,
        help="set the size of the voxels to BLOCK_SIZE",
    )
    parser.add_argument(
        "-cm",
        "--colormap",
        default=cm.get_cmap("prism", 10),
        action="store_true",
        help="whether to use a colormap to represent voxel occupancy, or to use a uniform color",
    )
    parser.add_argument(
        "-mc",
        "--max-component",
        metavar="max_component",
        type=int,
        default=3,
        help="whether to keep only the maximal connected component, where voxels of distance no larger than `DISTANCE` are considered connected. Set to 0 to disable this function.",
    )
    args = parser.parse_args()
    print(args)

    matname = "instance"
    threshold = args.threshold
    ind = args.index - 1  # matlab use 1 base index
    downsample_factor = args.downsample_factor
    downsample_method = args.downsample_method
    uniform_size = args.uniform_size
    use_colormap = args.colormap
    connect = args.max_component

    assert downsample_method in ("max", "mean")

    # read file
    print(f"==> Reading from seed: {args.seed}")

    model = Generator()
    model.load_state_dict(torch.load(args.network))
    model.eval()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    z = Variable(torch.rand(16, 200, 1, 1, 1))
    X = model(z)
    voxels = X.detach().numpy()
    print("Done")

    voxels_raw = get_voxels(voxels)
    voxels = voxels_raw[ind]

    # keep only max connected component
    print("Looking for max connected component")
    if connect > 0:
        voxels_keep = voxels >= threshold
        voxels_keep = max_connected(voxels_keep, connect)
        voxels[np.logical_not(voxels_keep)] = 0

    # downsample if needed
    if downsample_factor > 1:
        print(
            "==> Performing downsample: factor: "
            + str(downsample_factor)
            + " method: "
            + downsample_method
        )
        voxels = downsample(voxels, downsample_factor, method=downsample_method)
        print("Done")

    visualisation(
        voxels,
        threshold,
        title=args.out_file,
        uniform_size=uniform_size,
        use_colormap=use_colormap,
        snapshot=True
    )
