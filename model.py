from fb import calculate_FB_bases
from Conv_DCFD import Conv_DCFD
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math
import cv2


def bases_list(ks, num_bases):
    len_list = ks // 2
    b_list = []

    for i in range(len_list):
        kernel_size = (i + 1) * 2 + 1
        normed_bases, _, _ = calculate_FB_bases(i + 1)
        normed_bases = normed_bases.transpose().reshape(-1, kernel_size, kernel_size).astype(np.float32)[:num_bases, ...]
        pad = len_list - (i + 1)
        bases = torch.Tensor(normed_bases)
        bases = torch.nn.functional.pad(bases, (pad, pad, pad, pad, 0, 0)).view(num_bases, ks * ks)
        b_list.append(bases)

    return torch.cat(b_list, 0)


def get_gabor_bases(ks, num_bases):
    sizes = ks // 2
    filters = []

    for m in range(sizes):
        for n in range(num_bases):
            omega = ((math.pi / 2) * (math.sqrt(2) ** (-m + 1)))
            theta = n / num_bases * math.pi
            sigma = math.pi / omega
            lambd = 2 * math.pi / omega
            filters.append(cv2.getGaborKernel((ks, ks), sigma, theta, lambd, 1, 0).flatten())

    return torch.from_numpy(np.asarray(filters))


def get_random_bases(ks, num_bases):
    sizes = ks // 2
    filters = []

    for m in range(sizes):
        for n in range(num_bases):
            f = np.random.randn(ks, ks).flatten()
            f = f - np.mean(f) / np.std(f)
            filters.append(f)

    return torch.from_numpy(np.asarray(filters))


class ADConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_bases=6, padding=0, bias=True, bases_type='FB'):
        super(ADConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_bases = num_bases
        self.padding = padding

        if bases_type == 'FB':
            bases = bases_list(kernel_size, num_bases)
        elif bases_type == 'Gabor':
            bases = get_gabor_bases(kernel_size, num_bases)
        elif bases_type == 'Random':
            bases = get_random_bases(kernel_size, num_bases)

        self.register_buffer('bases', bases.float())
        self.total_bases = len(bases)
        bc_dim = len(bases) * num_bases

        inter = max(64, bc_dim // 2)
        self.bases_net = nn.Sequential(
            nn.Conv2d(in_channels, inter, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(inter),
            nn.Tanh(),
            nn.Conv2d(inter, bc_dim, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(bc_dim),
            nn.Tanh()
        )

        self.to_out = nn.Linear(in_channels * num_bases, out_channels, bias=bias)

    def forward(self, x):
        N, C, H, W = x.shape

        bases_coef = self.bases_net(x).view(N, self.num_bases, self.total_bases, H, W).permute(0, 3, 4, 1, 2).contiguous()
        atoms = (bases_coef.unsqueeze(-1) * self.bases).sum(-2).transpose(-1, -2).contiguous()  # (N, H, W, kernel_size * kernel_size, num_bases)
        x = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding).view(N, self.in_channels, self.kernel_size**2, H, W).permute(0, 3, 4, 1, 2).contiguous()
        bases_out = (x.unsqueeze(-1) * atoms.unsqueeze(-3)).sum(-2).view(N, H, W, self.in_channels * self.num_bases)
        out = self.to_out(bases_out).permute(0, 3, 1, 2).contiguous()
        return out


class ADLeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ADLeNet, self).__init__()
        self.conv1 = ADConv(3, 6, 5, num_bases=5, padding=2, bases_type='FB')
        self.conv2 = ADConv(6, 16, 5, num_bases=5, padding=2, bases_type='FB')
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2, stride=2)
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class LeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, 2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2, stride=2)
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
