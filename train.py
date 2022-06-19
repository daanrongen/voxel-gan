import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from model import _3DGAN, Generator, Discriminator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--attribute", type=str, help="Specify category for training."
    )
    parser.add_argument(
        "-s", "--snap", type=int, default=100, help="Model save interval"
    )
    parser.add_argument(
        "-g", "--gpu", default=[], nargs="+", type=int, help="Specify GPU ids."
    )
    parser.add_argument(
        "-r",
        "--restore",
        default=None,
        action="store",
        type=int,
        help="Specify checkpoint id to restore.",
    )
    parser.add_argument(
        "-m", "--mode", default="train", type=str, choices=["train", "test"]
    )
    args = parser.parse_args()
    print(args)

    G = Generator()
    # G = Generator().cuda(0)
    D = Discriminator()
    # D = Discriminator().cuda(0)
    G = torch.nn.DataParallel(G, device_ids=[0, 1])
    D = torch.nn.DataParallel(D, device_ids=[0, 1])

    # z = Variable(torch.rand(16,512,4,4,4))
    # m = nn.ConvTranspose3d(512, 256, 4, 2, 1)
    z = Variable(torch.rand(16, 200, 1, 1, 1))
    # z = Variable(torch.rand(16, 200, 1, 1, 1)).cuda(1)
    X = G(z)
    m = nn.Conv3d(1, 64, 4, 2, 1)
    D_X = D(X)
    print(X.shape, D_X.shape)

    model = _3DGAN(args)
    print(model)

    model.train()
