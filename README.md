# ADE20K多标签场景分类

这个项目实现了基于ADE20K数据集的多标签场景分类模型。通过多标签分类，模型能够识别图像中出现的所有物体类别，从而提供更全面的场景理解。**其中，损失函数使用的是[Asymmetric Loss For Multi-Label Classification](https://arxiv.org/abs/2009.14119)论文所提出的Asymmetric Loss，专门针对多标签分类的场景下。


## 数据集

[ADE20K数据集](https://groups.csail.mit.edu/vision/datasets/ADE20K/)是一个包含大量场景图像的数据集，其中有：
- 20K+ 训练图像
- 2K+ 验证图像
- 每张图像都有像素级标注
- 150个物体类别

本项目使用ADE20K数据集进行多标签分类，即识别图像中出现的所有物体类别。

## 项目结构

```
├── train.py               # 主训练脚本
├── dataloader_ade.py      # ADE20K数据集加载器
├── checkpoints/           # 模型保存目录
└── ADEChallengeData2016/  # 数据集目录
    ├── images/            # 图像目录
    ├── annotations/       # 标注目录
    └── ade_class.yaml     # 类别映射文件
```
==其中你需要自行下载ADE数据集==
## 模型架构

该项目使用预训练的ResNet152作为骨干网络，并添加了自定义的分类头来实现多标签分类。关键特性：

- 使用预训练的ResNet152提取特征
- 自定义的分类头进行多标签预测
- 使用Asymmetric Loss (ASL)作为损失函数，更适合多标签分类问题
- 使用SwanLab进行实验追踪和可视化

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- torchvision
- scikit-learn
- tqdm
- swanlab
- PyYAML

## 使用方法

### 1. 准备数据集
```shell
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```
确保ADE20K数据集已下载(需要挂梯子）并解压到`ADEChallengeData2016`目录下。数据集结构应如下：

```
ADEChallengeData2016/
├── images/
│   ├── training/
│   └── validation/
├── annotations/
│   ├── training/
│   └── validation/
├── ade_class.yaml
└── objectInfo150.txt
```

### 2. 训练模型

执行以下命令开始训练：

```bash
python train.py
```

训练参数在`train.py`中可以调整，包括：
- 批次大小
- 学习率
- 训练轮数
- 是否冻结backbone等

### 3. 训练结果

训练过程中，模型会自动保存在`checkpoints`目录下。同时，SwanLab会记录训练过程中的各种指标，包括：
- 训练和验证损失
- 精确率（Precision）
- 召回率（Recall）
- F1分数

## 性能指标

模型在多标签分类任务上评估使用以下指标：
- Sample-based Precision
- Sample-based Recall
- Sample-based F1

这些指标适合评估多标签分类任务，其中每个样本可能具有多个正确标签。

## 损失函数

项目使用Asymmetric Loss (ASL)作为损失函数，它对多标签分类问题特别有效，能够更好地处理类别不平衡问题。ASL通过对正样本和负样本使用不同的聚焦参数来提高性能。 