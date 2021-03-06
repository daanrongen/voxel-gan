from utils import config, ShapeNet

import os, argparse
import torch
import numpy as np
import scipy.io as sio

# from tensorboardX import SummaryWriter
from itertools import chain
import torch
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(200, 512, 4, 2, 0),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x's size: batch_size * hidden_size
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(1, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, 4, 2, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, 4, 2, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
            nn.Conv3d(512, 1, 4, 2, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x's size: batch_size * 1 * 64 * 64 * 64
        x = self.main(x)
        return x.view(-1, x.size(1))


class _3DGAN(object):
    def __init__(self, args, config=config):
        self.args = args
        self.attribute = args.attribute
        self.gpu = args.gpu
        self.mode = args.mode
        self.restore = args.restore
        self.snap = args.snap

        # init dataset and networks
        self.config = config
        self.dataset = ShapeNet(self.attribute)
        self.G = Generator()
        self.D = Discriminator()

        self.adv_criterion = torch.nn.BCELoss()

        self.set_mode_and_gpu()
        self.restore_from_file()

    def set_mode_and_gpu(self):
        if self.mode == "train":
            self.G.train()
            self.D.train()
            if self.gpu:
                with torch.cuda.device(self.gpu[0]):
                    self.G.cuda()
                    self.D.cuda()
                    self.adv_criterion.cuda()

            if len(self.gpu) > 1:
                self.G = torch.nn.DataParallel(self.G, device_ids=self.gpu)
                self.D = torch.nn.DataParallel(self.D, device_ids=self.gpu)

        elif self.mode == "test":
            self.G.eval()
            self.D.eval()
            if self.gpu:
                with torch.cuda.device(self.gpu[0]):
                    self.G.cuda()
                    self.D.cuda()

            if len(self.gpu) > 1:
                self.G = torch.nn.DataParallel(self.G, device_ids=self.gpu)
                self.D = torch.nn.DataParallel(self.D, device_ids=self.gpu)

        else:
            raise NotImplementationError()

    def restore_from_file(self):
        if self.restore is not None:
            ckpt_file_G = os.path.join(
                self.config.model_dir, "G_iter_{:06d}.pth".format(self.restore)
            )
            assert os.path.exists(ckpt_file_G)
            self.G.load_state_dict(torch.load(ckpt_file_G))

            if self.mode == "train":
                ckpt_file_D = os.path.join(
                    self.config.model_dir, "D_iter_{:06d}.pth".format(self.restore)
                )
                assert os.path.exists(ckpt_file_D)
                self.D.load_state_dict(torch.load(ckpt_file_D))

            self.start_step = self.restore + 1
        else:
            self.start_step = 1

    def save_log(self):
        scalar_info = {
            "loss_D": self.loss_D,
            "loss_G": self.loss_G,
            "G_lr": self.G_lr_scheduler.get_lr()[0],
            "D_lr": self.D_lr_scheduler.get_lr()[0],
        }
        for key, value in self.G_loss.items():
            scalar_info["G_loss/" + key] = value

        for key, value in self.D_loss.items():
            scalar_info["D_loss/" + key] = value

        for tag, value in scalar_info.items():
            self.writer.add_scalar(tag, value, self.step)

    def save_img(self, save_num=5):
        for i in range(save_num):
            mdict = {"instance": self.fake_X[i, 0].data.cpu().numpy()}
            sio.savemat(
                os.path.join(
                    self.config.img_dir, "{:06d}_{:02d}.mat".format(self.step, i)
                ),
                mdict,
            )

    def save_model(self):
        torch.save(
            {key: val.cpu() for key, val in self.G.state_dict().items()},
            os.path.join(self.config.model_dir, "G_iter_{:06d}.pth".format(self.step)),
        )
        torch.save(
            {key: val.cpu() for key, val in self.D.state_dict().items()},
            os.path.join(self.config.model_dir, "D_iter_{:06d}.pth".format(self.step)),
        )

    def train(self):
        self.writer = SummaryWriter(self.config.log_dir)
        self.opt_G = torch.optim.Adam(
            self.G.parameters(), lr=self.config.G_lr, betas=(0.5, 0.999)
        )
        self.opt_D = torch.optim.Adam(
            self.D.parameters(), lr=self.config.D_lr, betas=(0.5, 0.999)
        )
        self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt_G, step_size=self.config.step_size, gamma=self.config.gamma
        )
        self.D_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt_D, step_size=self.config.step_size, gamma=self.config.gamma
        )

        # start training
        for step in range(self.start_step, 1 + self.config.max_iter):
            self.step = step
            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            self.real_X = next(self.dataset.gen(True))
            self.noise = torch.randn(self.config.nchw[0], 200)
            if len(self.gpu):
                with torch.cuda.device(self.gpu[0]):
                    self.real_X = self.real_X.cuda()
                    self.noise = self.noise.cuda()

            self.fake_X = self.G(self.noise)

            # update D
            self.D_real = self.D(self.real_X)
            self.D_fake = self.D(self.fake_X.detach())
            self.D_loss = {
                "adv_real": self.adv_criterion(
                    self.D_real, torch.ones_like(self.D_real)
                ),
                "adv_fake": self.adv_criterion(
                    self.D_fake, torch.zeros_like(self.D_fake)
                ),
            }
            self.loss_D = sum(self.D_loss.values())

            self.opt_D.zero_grad()
            self.loss_D.backward()
            self.opt_D.step()

            # update G
            self.D_fake = self.D(self.fake_X)
            self.G_loss = {
                "adv_fake": self.adv_criterion(
                    self.D_fake, torch.ones_like(self.D_fake)
                )
            }
            self.loss_G = sum(self.G_loss.values())
            self.opt_G.zero_grad()
            self.loss_G.backward()
            self.opt_G.step()

            print(
                "step: {:06d}, loss_D: {:.6f}, loss_G: {:.6f}".format(
                    self.step,
                    self.loss_D.data.cpu().numpy(),
                    self.loss_G.data.cpu().numpy(),
                )
            )

            if self.step % 100 == 0:
                self.save_log()

            if self.step % self.snap == 0:
                self.save_img()
                self.save_model()

        print("Finished training!")
        self.writer.close()
