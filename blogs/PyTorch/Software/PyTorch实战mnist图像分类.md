---
title: 项目结构
date: 2026-04-24
categories:
  - AI
tags:
  - PyTorch
  - 深度学习
---

﻿(PyTorch实战mnist图像分类)
# 项目结构
项目结构如图，代码都放在mnistclassify.py里面，data数据是代码执行过程中自己下载的。
!(https://i-blog.csdnimg.cn/blog_migrate/0675f770f41f215a02947112bdbe2268.png#pic_center)

# 项目代码
1. 导入包，构建训练集测试集
```python
from random import shuffle
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义超参数
input_size = 28
num_classes = 10
num_epoches = 3
batch_size = 64

# 训练集
train_dateset = datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True)

# 测试集
test_dateset = datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor())

# 构建batch数据
train_loader = torch.utils.data.DataLoader(dataset=train_dateset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dateset,batch_size=batch_size,shuffle=True)
```
2. 构建神经网络
```python
# 构建网络
class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              # 灰度图
                out_channels=16,            # 输出特征图个数
                kernel_size=5,              # 卷积核大小
                stride=1,                   # 步长
                padding=2,                  # 边缘填充，如果stride=1，希望卷积后的图像和原来的图像一样大则设置padding=(kernal_size-1)/2
            ),                              # 输出特征图为(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # 2*2最大池化，结果为(16,14,14)
        )
        self.conv2 = nn.Sequential(         # 输入(16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),     # 输出(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                # 输出(32,7,7)
        )
        self.out = nn.Linear(32 * 7 *7, 10) # 全连接得到结果

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # 将结果转换为向量，方便下一步全连接(32*7*7)
        output = self.out(x)
        return output
```
3. 实例化网络开始训练
```python
# 预测准确率
def accuracy(predictins, labels):
    pred = torch.max(predictins.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

# 实例化神经网络
net = CNN()
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 开始训练循环
for epoch in range(num_epoches):
    # 保存当前epoch结果
    train_rights = []
    for batch_idx, (data, target) in enumerate(train_loader):
        net.train()
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = accuracy(output, target)
        train_rights.append(right)

        if batch_idx % 100 == 0:
            net.eval()
            val_rights = []

            for (data, target) in test_loader:
                output = net(data)
                right = accuracy(output, target)
                val_rights.append(right)

            # 准确率计算
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            print('当前epoch：{} [{}/{}({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集准确率： {:.2f}%'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data,
                100. * train_r[0].numpy() / train_r[1],
                100. * val_r[0].numpy() / val_r[1],
            ))
```
4. 训练结果

```python
当前epoch：0 [0/60000(0%)]      损失: 2.290263  训练集准确率: 6.25%     测试集准确率： 11.39%
当前epoch：0 [6400/60000(11%)]  损失: 0.222888  训练集准确率: 76.14%    测试集准确率： 90.28%
当前epoch：0 [12800/60000(21%)] 损失: 0.275965  训练集准确率: 84.60%    测试集准确率： 94.70%
当前epoch：0 [19200/60000(32%)] 损失: 0.071834  训练集准确率: 88.24%    测试集准确率： 95.60%
当前epoch：0 [25600/60000(43%)] 损失: 0.029019  训练集准确率: 90.25%    测试集准确率： 96.68%
当前epoch：0 [32000/60000(53%)] 损失: 0.159890  训练集准确率: 91.48%    测试集准确率： 97.08%
当前epoch：0 [38400/60000(64%)] 损失: 0.080257  训练集准确率: 92.39%    测试集准确率： 97.00%
当前epoch：0 [44800/60000(75%)] 损失: 0.100067  训练集准确率: 93.11%    测试集准确率： 97.57%
当前epoch：0 [51200/60000(85%)] 损失: 0.105826  训练集准确率: 93.66%    测试集准确率： 97.84%
当前epoch：0 [57600/60000(96%)] 损失: 0.042444  训练集准确率: 94.11%    测试集准确率： 98.05%
当前epoch：1 [0/60000(0%)]      损失: 0.169493  训练集准确率: 93.75%    测试集准确率： 98.01%
当前epoch：1 [6400/60000(11%)]  损失: 0.033878  训练集准确率: 98.04%    测试集准确率： 97.87%
当前epoch：1 [12800/60000(21%)] 损失: 0.108467  训练集准确率: 98.05%    测试集准确率： 98.01%
当前epoch：1 [19200/60000(32%)] 损失: 0.007603  训练集准确率: 97.97%    测试集准确率： 98.35%
当前epoch：1 [25600/60000(43%)] 损失: 0.202825  训练集准确率: 98.04%    测试集准确率： 98.49%
当前epoch：1 [32000/60000(53%)] 损失: 0.113783  训练集准确率: 98.11%    测试集准确率： 98.47%
当前epoch：1 [38400/60000(64%)] 损失: 0.027782  训练集准确率: 98.11%    测试集准确率： 98.46%
当前epoch：1 [44800/60000(75%)] 损失: 0.034398  训练集准确率: 98.12%    测试集准确率： 98.51%
当前epoch：1 [51200/60000(85%)] 损失: 0.013913  训练集准确率: 98.18%    测试集准确率： 98.51%
当前epoch：1 [57600/60000(96%)] 损失: 0.021681  训练集准确率: 98.19%    测试集准确率： 98.91%
当前epoch：2 [0/60000(0%)]      损失: 0.052889  训练集准确率: 96.88%    测试集准确率： 98.72%
当前epoch：2 [6400/60000(11%)]  损失: 0.070504  训练集准确率: 98.95%    测试集准确率： 98.86%
当前epoch：2 [12800/60000(21%)] 损失: 0.104337  训练集准确率: 98.67%    测试集准确率： 98.85%
当前epoch：2 [19200/60000(32%)] 损失: 0.028965  训练集准确率: 98.72%    测试集准确率： 98.70%
当前epoch：2 [25600/60000(43%)] 损失: 0.048499  训练集准确率: 98.70%    测试集准确率： 98.82%
当前epoch：2 [32000/60000(53%)] 损失: 0.021659  训练集准确率: 98.70%    测试集准确率： 98.80%
当前epoch：2 [38400/60000(64%)] 损失: 0.002921  训练集准确率: 98.72%    测试集准确率： 98.95%
当前epoch：2 [44800/60000(75%)] 损失: 0.015612  训练集准确率: 98.70%    测试集准确率： 98.92%
当前epoch：2 [51200/60000(85%)] 损失: 0.043291  训练集准确率: 98.71%    测试集准确率： 99.08%
当前epoch：2 [57600/60000(96%)] 损失: 0.033159  训练集准确率: 98.72%    测试集准确率： 99.01%
```

如有代码不懂或者报错请评论区留言，博主帮忙解答、调试。

