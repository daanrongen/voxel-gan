import os
import sys
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from visualisation.util import read_tensor
import argparse
import numpy as np
from scipy import ndimage, io

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--attribute", type=str, help="Specify category for training."
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="custom",
        help="Specify target for training.",
    )
    args = parser.parse_args()
    print(args)

    os.makedirs(f"volumetric_data/{args.out_dir}", exist_ok=True)

    prefix = os.path.join("volumetric_data", args.attribute, "30")
    data_dir = prefix + "/train"

    filenames = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".mat")
    ]

    for file in filenames[0:2]:
        filename = os.path.basename(file)

        print("==> Reading input voxel file: " + file)
        voxels_raw = read_tensor(file, "instance")
        print("Done")

        voxels = voxels_raw[0]
        voxels = ndimage.zoom(voxels, 2.15)
        duplicate = np.flip(voxels, axis=2)
        result = np.bitwise_or(voxels, duplicate)

        assert result.shape == (64, 64, 64)
        voxels = result
        print(file)
        print(voxels.shape)
        io.savemat(f"volumetric_data/{args.out_dir}/{filename}", {'data': voxels})
