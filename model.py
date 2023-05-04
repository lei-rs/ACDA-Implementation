from fb import calculate_FB_bases
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


class ADConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_fa, num_bases=6, padding=0, bias=True, bases_type='FB'):
        super(ADConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_fa = num_fa
        self.padding = padding

        if bases_type == 'FB':
            bases = bases_list(kernel_size, num_bases)
        elif bases_type == 'Gabor':
            bases = get_gabor_bases(kernel_size, num_bases)

        self.register_buffer('bases', bases.float())
        self.total_bases = len(bases)
        bc_dim = len(bases) * num_fa

        inter = max(64, bc_dim // 2)
        self.bases_net = nn.Sequential(
            nn.Conv2d(in_channels, inter, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(inter),
            nn.Tanh(),
            nn.Conv2d(inter, bc_dim, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(bc_dim),
            nn.Tanh()
        )

        self.coef = nn.Parameter(torch.Tensor(out_channels, in_channels * num_fa, 1, 1))

    def forward(self, x):
        N, C, H, W = x.shape

        bases_coef = self.bases_net(x).transpose(1, -1).view(N, H, W, self.num_fa, self.total_bases)
        atoms = (bases_coef.unsqueeze(-1) * self.bases).sum(-2).transpose(-1, -2)  # (N, H, W, kernel_size * kernel_size, num_fa)
        x = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding).transpose(-1, -2).view(N, H, W, self.in_channels, -1)
        bases_out = (x.unsqueeze(-1) * atoms.unsqueeze(-3)).sum(-2).view(N, H, W, -1).transpose(1, -1)
        out = F.conv2d(bases_out, self.coef)

        return out


class ADLeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ADLeNet, self).__init__()
        self.conv1 = ADConv(3, 6, 5, 4, padding=2)
        self.conv2 = ADConv(6, 16, 5, 4, padding=2)
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


if __name__ == '__main__':
    x = torch.randn(10, 3, 224, 224)
    conv = ADConv(3, 10, 5, 4, padding=2)
    print(conv(x).shape)
