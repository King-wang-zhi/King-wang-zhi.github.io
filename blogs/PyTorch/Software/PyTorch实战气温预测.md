---
title: 任务
date: 2026-04-24
categories:
  - AI
tags:
  - PyTorch
  - 深度学习
---

﻿(PyTorch实战气温预测)
注意：仅记录学习过程，如有侵权联系删除
# 任务
本次任务是进行气温预测，数据集链接[https://www.kaggle.com/datasets/ns0720/tempscsv](https://www.kaggle.com/datasets/ns0720/tempscsv)，数据集下载有困难的评论区留言，作为全面学习PyTorch实战的第一章，我们会使用比较原始的方法写整个训练过程，除了反向传播由PyTorch代码调用自行计算。

# 数据集介绍

数据集是csv文件，他饱含9列，按顺序分别是year，month，day，week，temp_1，temp_2，average，actual，friend。我们的训练数据集为除了actual的所有列，训练数据集的标签为actual。数据的预处理我们展示在代码中。
!(https://i-blog.csdnimg.cn/blog_migrate/5d365119fbd7a8c9acb793cea4a0a924.png#pic_center)

# 项目目录
![项目目录](https://i-blog.csdnimg.cn/blog_migrate/d3cec3de76c46758fe9a13ebbfff6afc.png#pic_center)
注意代码执行环境要在PredictionTemps目录下，否则会报temps.csv文件找不到。

# 训练代码

```cpp
from ast import increment_lineno
from audioop import bias
from calendar import month
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline

#数据读取
features = pd.read_csv('temps.csv')

#查看数据
print(features.head())
print("数据维度", features.shape)

#处理数据，转换时间类型
import datetime

#年，月，日
years = features['year']
months = features['month']
days = features['day']

# datetime格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 查看数据格式
print(dates[:5])

# week列为字符串不是数值，利用独热编码，将数据中非字符串转换为数值，并拼接到数据中
features = pd.get_dummies(features)
# 看独热编码的效果
print(features.head(5))

# 标签
labels = np.array(features['actual'])

# 去掉标签用作特征
features = features.drop('actual', axis=1)

# 保存列名用于展示
features_list = list(features.columns)

# 转换为合适的格式
features = np.array(features)
print(features.shape)

# 数据标准化
from sklearn import preprocessing
input_features = preprocessing.StandardScaler().fit_transform(features)

# 看一下数字标准化的效果
print(input_features[0])
```
接下来构建神经网络模型,首先使用原始的方法
```
# 将输入和预测转为tensor
x = torch.tensor(input_features, dtype=float)
y = torch.tensor(labels,dtype=float)

# 权重参数初始化
weights = torch.randn((14, 128), dtype= float, requires_grad= True)
biases = torch.randn(128, dtype=float, requires_grad= True)
weights2 = torch.randn((128, 1), dtype=float, requires_grad= True)
biases2 = torch.randn(1, dtype=float, requires_grad=True)

learning_rate = 0.001
losses = []

for i in range(1000):
    # 前向传播
    # 计算隐藏层
    hidden = x.mm(weights) + biases
    # 加入激活函数
    hidden = torch.relu(hidden)
    # 预测结果
    predictions = hidden.mm(weights2) + biases2
    # 计算损失
    loss = torch.mean((predictions - y)**2)
    losses.append(loss.data.numpy())

    # 打印损失
    if i % 100 == 0:
        print('loss:', loss)
    # 反向传播
    loss.backward()
    # 更新参数
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)

    # 梯度清零
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()
```
训练结果

```python
loss: tensor(3511.3141, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(154.7521, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(146.5845, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(144.1342, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(142.9047, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(142.1384, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(141.5937, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(141.1904, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(140.8811, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(140.6381, dtype=torch.float64, grad_fn=<MeanBackward0>)
```
loss稳步下降

或者我们使用简化的方法

```python
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)

# 指定损失函数
cost = torch.nn.MSELoss(reduction='mean')

# 指定优化器
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

# 训练网络
losses = []
for i in range(1000):
    batch_loss = []
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        prediction = my_nn(xx)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))
```
最终我们进行预测，并以图片的形式展示
```
# 预测结果
x = torch.tensor(input_features, dtype=torch.float)
predict = my_nn(x).data.numpy() # 转化为numpy格式，tensor格式画不了图

# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来保存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

# 再创建一个来存日期和其对应的模型预测值
months = features[:, features_list.index('month')]
days = features[:, features_list.index('day')]
years = features[:, features_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
test_dates = dates

predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='vertical');
plt.legend()

# 图名
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.show()
```
结果展示：
!(https://i-blog.csdnimg.cn/blog_migrate/35bb314a76fc198296575e85ef70e5c9.png#pic_center)

说明：代码执行中所需要的包请自行pip install xx下载





