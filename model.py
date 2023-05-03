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
    def __init__(self, in_channels, out_channels, num_bases, kernel_size, stride=1, padding=0, bases_type='FB'):
        super(ADConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bases = num_bases
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if bases_type == 'FB':
            bases = bases_list(kernel_size, num_bases)
        elif bases_type == 'Gabor':
            bases = get_gabor_bases(kernel_size, num_bases)

        self.register_buffer('bases', bases.float())
        self.tem_size = len(bases)

        bases_size = num_bases * len(bases)
        inter = max(64, bases_size // 2)
        self.bases_net = nn.Sequential(
            nn.Conv2d(in_channels, inter, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inter),
            nn.Tanh(),
            nn.Conv2d(inter, bases_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(bases_size),
            nn.Tanh()
        )

        self.coef = nn.Parameter(torch.Tensor(out_channels, in_channels * num_bases, 1, 1))

    def forward(self, x):
        N, C, H, W = x.shape
        H = H // self.stride
        W = W // self.stride

        bases = self.bases_net(x).view(N, self.num_bases, self.tem_size, H, W)
        bases = torch.einsum('bmkhw, kl->bmlhw', bases, self.bases)

        x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).view(
            N, self.in_channels, self.kernel_size * self.kernel_size, H, W
        )
        bases_out = torch.einsum('bmlhw, bclhw->bcmhw', bases.view(N, self.num_bases, -1, H, W), x).reshape(
            N, self.in_channels * self.num_bases, H, W
        )
        out = F.conv2d(bases_out, self.coef)

        return out


if __name__ == '__main__':
    layer = ADConv(3, 10, 6, 3, padding=1, stride=2)
    data = torch.randn(10, 3, 224, 224)
    print(layer(data).shape)
