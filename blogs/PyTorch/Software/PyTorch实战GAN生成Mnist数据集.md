---
title: 基本的生成对抗网络
date: 2026-04-24
categories:
  - AI
tags:
  - PyTorch
  - 深度学习
---

﻿# PyTorch实战GAN生成Mnist数据集

## 项目解读
使用GAN生成Mnist数据集，对抗生成网络的关键在于损失函数的设计，相关理论、代码见[https://blog.csdn.net/qq_41605740/article/details/127816320](https://blog.csdn.net/qq_41605740/article/details/127816320)即下面的BCEloss.py
项目结构：
!(https://i-blog.csdnimg.cn/blog_migrate/735ea96a2f7870e2ac4da28645144500.png#pic_center)
主要代码在gan.py模块，数据不需要你提前准备，运行gan.py自动下载数据
## 代码解读
### 1. 导入所需的包

```python
# 基本的生成对抗网络
import argparse
from ast import parse
from email import generator
from email.policy import default
from importlib.metadata import requires
from locale import normalize
import os
from turtle import forward
from imageio import save
import numpy as np
import math
from sklearn.utils import shuffle

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
```
### 2.设定Mnist图片生成路径

```python
os.makedirs("images", exist_ok=True)
```
### 3.设置参数配置

```python
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--n_cpu", type=int, default=8)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--img_size", type=int, default=28)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--sample_interval", type=int, default=400)
opt = parser.parse_args()
print(opt)
```
图像形状，输入通道1（黑白），28*28（长宽）

```python
img_shape = (opt.channels, opt.img_size, opt.img_size)
```
是否用GPU训练
```python
cuda = True if torch.cuda.is_available() else False
```
### 4.定义生成器网络结构

```python
class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            # in_feat为100，自生产空白特征，第一个一个隐藏层out_feat为128
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            # 得到的特征个数要与原始输入一致
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
```
### 5.定义判别器网络结构

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # 输入一张图784
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
```
选定损失函数

```python
adversarial_loss = torch.nn.BCELoss()
```
实例化生成器与判别器

```python
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
```
### 6.配置数据集

```python
os.makedirs("./data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
```
选定优化器

```python
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
```
### 7.进行训练，打印训练过程，保存生成的Mnist图片到images文件夹下

```python
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # 定义真假标签
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        # 训练生成器
        optimizer_G.zero_grad()
        # 随机构建一个batch向量64*100
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # 生产一个batch图像
        gen_imgs = generator(z)
        # 用生成结果骗判别器，valid为全1
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # 训练判别器
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
```
查看结果，最开始生成的图片，几乎无法辨别
!(https://i-blog.csdnimg.cn/blog_migrate/405d67272909192c622327d5f2a8736a.png#pic_center)
经过100个周期后生成的图片，已经可以看出来7，9了
!(https://i-blog.csdnimg.cn/blog_migrate/c4ee646466a90ffa49799809dabd033d.png#pic_center)

