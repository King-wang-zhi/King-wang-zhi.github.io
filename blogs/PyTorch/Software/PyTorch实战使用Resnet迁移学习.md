---
title: 项目结构
date: 2026-04-24
categories:
  - AI
tags:
  - PyTorch
  - 深度学习
---

﻿(PyTorch实战使用Resnet迁移学习)
# 项目结构
1. 数据集存放在flower_data文件夹
2. cat_to_name.json是makejson文件运行生成的
3. TorchVision文件主要存放本项目实战代码
!(https://i-blog.csdnimg.cn/blog_migrate/c6807d4c4cc9696173073bb6342091b0.png#pic_center)

# 项目任务
项目描述：对花进行分类
项目数据集：102种花的图片，数据集下载[https://www.kaggle.com/datasets/eswarkamineni/flower-data](https://www.kaggle.com/datasets/eswarkamineni/flower-data)
算法：使用迁移学习Resnet152，冻结所有卷积层，更改全连接层并进行训练
# 项目代码
makejson.py文件代码如下，给每一种分类花一个名字，因为我们下载的数据集中每个分类文件命名为1，2，3，所以我们对应给个名称。

```python
import json
f = open("cat_to_name.json","w")
f.write('{"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt\'s foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}')
f.close()
```
TorchVision.py文件代码如下
```python
# torchvision模块实战
# torchvision.datasets模块包括数据集，数据加载方法
# torchvision.models模块包括一些经典的网络架构
# torchvision.transforms模块包括一些预处理图像增强方法
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
# import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

# 处理流程

# 数据预处理部分：
# 数据增强：torchvision.transforms
# 数据预处理：torchvisivon.transforms
# DataLoader读取batch数据

# 数据读取与预处理操作
data_dir = './flower_data'
train_dir = data_dir + 'train'
valid_dir = data_dir + 'valid'

# data_transforms中指定了所有图像的预处理操作
# ImageFolder假设所有文件按文件夹保存好，每个文件夹下面存储同一类别图片，文件夹的名字为分类的名字
data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),                           # 随机旋转，-45到45度之间随机旋转
            transforms.CenterCrop(224),                                                   # 从中心开始裁剪
            transforms.RandomHorizontalFlip(p=0.5),                                       # 随机水平翻转 选择一个翻转概率
            transforms.RandomVerticalFlip(p=0.5),                                         # 随机垂直翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),# 亮度，对比度，饱和度，色相
            transforms.RandomGrayscale(p=0.025),                                          # 概率转换为灰度率
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            # 标准化((x-均值)/标准差)，均值，标准差，拿别人的预训练数据的均值和标准差为了使训练效果更好
    ]),
    'valid': transforms.Compose([transforms.Resize(256),                                  # 我们的训练集比较小所以没有resize
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            # 大小和标准化必须和训练集一样
    ]),
}
batch_size = 8 # 显存不够把batch_size调小
# 构建数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']} # 传路径和预处理流程
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
print(dataloaders)
print(datasets_sizes)

# 读取标签和对应的实际名字
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(cat_to_name)

# 展示数据
# 因为我们已经对数据做了处理，所以先将tensor数据转化为numpy格式，然后还原回标准化的结果
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0) # torch颜色通道被放到了第一位，我们利用transpose将h,w,c还原回去
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406)) # 去除标准化，成标准差加均值
    image = image.clip(0, 1)

    return image

fig = plt.figure(figsize=(20, 12))
columns = 4
rows = 2

dataiter = iter(dataloaders['valid']) # 迭代一次取一个batch数据
inputs, classes = dataiter.next()     # 数据，标签

for idx in range(columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
    plt.imshow(im_convert(inputs[idx]))
plt.show()

# 迁移学习
# 拿别人的卷积层，方案1在此卷积层的基础上继续训练，方案2冻结此卷积层作为我们的特征提取工具
# 全连接层都是要重写重训练的

# 网络模块设置：
# 加载预训练模型：torchvision.models，可以调用训练好的权重参数来继续训练，也就是所谓的迁移学习
# 注意：需要把最后一层改一下，改成我们自己的任务
# 训练时可以全部重头训练，也可以只训练我们的任务层，因为前几层都是做特征提取的，本质任务目标是一致的

model_name = 'resnet' # 可选择的模型比较多{'alexnet', 'vgg', 'resnet', 'squeezenet', 'densenet', 'inception'}
# 是否用人家训练好的特征来做
feature_extract = True

# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# 加载模型
model_ft = models.resnet152()
print(model_ft)

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同的模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0
    
    if model_name =="resnet":
        """Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)  # 下载预训练的模型参数
        set_parameter_requires_grad(model_ft, feature_extract)  # 冻住卷积层
        num_ftrs = model_ft.fc.in_features                      # 拿到最后全连接层输入
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102), nn.LogSoftmax(dim=1)) # 重写全连接层

        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU计算
model_ft = model_ft.to(device)

# 指定保存模型的名字
filename = 'checkpoint.pth'

# 是否训练所有层,一般策略是先学习自己的层，在观察学习全部层
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# 看一下现在的网络层
print(model_ft)

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 学习率每7个epoch衰减为原来的0.1
# 最后一层LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename=filename):
    since = time.time()
    best_acc = 0
    """
    """
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1) # 获取概率最大的类别

                    # 训练更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好次好的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)

            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_group[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当作模型最终结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=10, is_inception=False, filename=filename)
# 模型保存可以带有选择性，例如在验证集中如果效果好则保存
```
模型训练结果，在这里我用的,Linux系统，3090显卡。在上述代码的基础上，将batch_size改为了64，并在DataLoader最后加入了num_workers=8。如果你使用的是windows系统则不能设置num_workers，会报错，我本地电脑，windows+1050显卡，一个周期要7m11s。差距太大了，所以我建议大家还是租显卡，或者买张显卡。
```python
Params to learn:
	 fc.0.weight
	 fc.0.bias
Epoch 0/49
----------
Time elapsed 0m 16s
train Loss: 6.2446 Acc: 0.4113
Time elapsed 0m 20s
valid Loss: 2.4209 Acc: 0.5513
Optimizer learning rate : 0.0100000

Epoch 1/49
----------
Time elapsed 0m 36s
train Loss: 1.5336 Acc: 0.6784
Time elapsed 0m 40s
valid Loss: 2.2870 Acc: 0.5929
Optimizer learning rate : 0.0100000

Epoch 2/49
----------
Time elapsed 0m 57s
train Loss: 1.4210 Acc: 0.7121
Time elapsed 1m 1s
valid Loss: 2.8051 Acc: 0.5452
Optimizer learning rate : 0.0100000

Epoch 3/49
----------
Time elapsed 1m 17s
train Loss: 1.5040 Acc: 0.7346
Time elapsed 1m 21s
valid Loss: 2.8843 Acc: 0.6174
Optimizer learning rate : 0.0100000

Epoch 4/49
----------
Time elapsed 1m 38s
train Loss: 1.3956 Acc: 0.7511
Time elapsed 1m 41s
valid Loss: 2.5887 Acc: 0.6308
Optimizer learning rate : 0.0100000

Epoch 5/49
----------
Time elapsed 1m 57s
train Loss: 1.4859 Acc: 0.7486
Time elapsed 2m 1s
valid Loss: 3.4337 Acc: 0.6015
Optimizer learning rate : 0.0100000

Epoch 6/49
----------
Time elapsed 2m 17s
train Loss: 1.3451 Acc: 0.7764
Time elapsed 2m 21s
valid Loss: 3.3750 Acc: 0.6455
Optimizer learning rate : 0.0100000

Epoch 7/49
----------
Time elapsed 2m 37s
train Loss: 1.5002 Acc: 0.7746
Time elapsed 2m 41s
valid Loss: 3.5827 Acc: 0.5856
Optimizer learning rate : 0.0100000

Epoch 8/49
----------
Time elapsed 2m 57s
train Loss: 1.3252 Acc: 0.7863
Time elapsed 3m 0s
valid Loss: 3.1641 Acc: 0.6516
Optimizer learning rate : 0.0100000

Epoch 9/49
----------
Time elapsed 3m 17s
train Loss: 1.4956 Acc: 0.7856
Time elapsed 3m 20s
valid Loss: 3.5925 Acc: 0.6015
Optimizer learning rate : 0.0100000
```


# 网络模型测试
```python

# 加载训练好的模型
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU模式
model_ft = model_ft.to(device)

# 保存文件的名字
filename = 'checkpoint.pth'

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

# 测试数据预处理
def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize，thumbnail方法只能进行缩小，所以进行判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000,256))
    else:
        img.thumbnail((256,10000))
    # 裁剪操作
    left_margin = (img.width - 224)/2
    bottom_margin = (img.height - 224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # 相同的归一化，正则化
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std

    # 注意颜色通道放到第一个位置
    img = img.transpose((2, 0 ,1))

    return img

def imshow(image, ax=None, title=None):
    """展示数据"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # 颜色通道还原
    image = np.array(image).transpose((1, 2, 0))

    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax

image_path = './flower_data/test/1/image_05087.jpg'
img = process_image(image_path)
imshow(img)
print(img.shape)
```
取一个图片看看
!(https://i-blog.csdnimg.cn/blog_migrate/c656e61e43a624483e23bd44bdb6baf8.png#pic_center)
取一个batch_size数据进行测试
```python
# 得到一个batch的测试数据
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.next()

model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

print(output.shape)
# torch.Size([8, 102])

# 得到概率最大的那个
_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
print(preds)

# 预测结果展示
fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 2

for idx in range (columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]), 
    color = ("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))

plt.show()
```
我们可以看到只有一个花预测错误，其他都预测正确。
!(https://i-blog.csdnimg.cn/blog_migrate/f1a067154307c163ff930687a809004e.png#pic_center)
!(https://i-blog.csdnimg.cn/blog_migrate/2943d7752febcb5b68730bebd87583e2.png#pic_center)






