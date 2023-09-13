import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch
import math
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from PIL import ImageFile
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from PIL import ImageFile
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
#from wwwpos import get_pos
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import matplotlib.pyplot as plt
#import os
from tqdm import tqdm
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
# 自定义PyTorch分类器
class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 添加全局平均池化层
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.global_avg_pool(x)  # 全局平均池化
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
"""

# 定义SPP层
class SpatialPyramidPooling(nn.Module):
    def __init__(self, previous_conv_size, out_pool_size):
        super(SpatialPyramidPooling, self).__init__()
        self.previous_conv_size = previous_conv_size
        self.out_pool_size = out_pool_size
        self.pool_layers = nn.ModuleList()
        for i in range(len(out_pool_size)):
            h_wid = int(math.ceil(self.previous_conv_size[0] / self.out_pool_size[i]))
            w_wid = int(math.ceil(self.previous_conv_size[1] / self.out_pool_size[i]))
            h_pad = int((h_wid * self.out_pool_size[i] - self.previous_conv_size[0] + 1) / 2)
            w_pad = int((w_wid * self.out_pool_size[i] - self.previous_conv_size[1] + 1) / 2)
            pool_layer = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            self.pool_layers.append(pool_layer)
    def forward(self, previous_conv, num_sample):
        pool_outputs = []
        i=0
        for pool_layer in self.pool_layers:
            x = pool_layer (previous_conv)
            if (i == 0):
                spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
            else:
            # print("size:",spp.size())
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
            i=i+1
        return spp

# 定义SE注意力机制的类
class se_block(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(se_block, self).__init__()

        # 属性分配
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # relu激活
        self.relu = nn.ReLU()
        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):  # inputs 代表输入特征图

        # 获取输入特征图的shape
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs




# Resnet 18/34使用此残差块
class BasicBlock(nn.Module):  # 卷积2层，F(X)和X的维度相等
    # expansion是F(X)相对X维度拓展的倍数
    expansion = 1  # 残差映射F(X)的维度有没有发生变化，1表示没有变化，downsample=None

    # in_channel输入特征矩阵的深度(图像通道数，如输入层有RGB三个分量，使得输入特征矩阵的深度是3)，out_channel输出特征矩阵的深度(卷积核个数)，stride卷积步长，downsample是用来将残差数据和卷积数据的shape变的相同，可以直接进行相加操作。
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN层在conv和relu层之间

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out=F(X)+X
        out += identity
        out = self.relu(out)

        return out


# Resnet 50/101/152使用此残差块
class Bottleneck(nn.Module):  # 卷积3层，F(X)和X的维度不等
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    """
    # expansion是F(X)相对X维度拓展的倍数
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # 此处width=out_channel

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # downsample是用来将残差数据和卷积数据的shape变的相同，可以直接进行相加操作。
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out=F(X)+X
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,  # 使用的残差块类型
                 blocks_num, # 每个卷积层，使用残差块的个数
                 dropout_rate,
                 num_classes=7,  # 训练集标签的分类个数
                 include_top=True,  # 是否在残差结构后接上pooling、fc、softmax
                 groups=1,
                 width_per_group=64,in_channel1=512):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 第一层卷积输出特征矩阵的深度，也是后面层输入特征矩阵的深度

        self.groups = groups
        self.width_per_group = width_per_group

        # 输入层有RGB三个分量，使得输入特征矩阵的深度是3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # _make_layer(残差块类型，残差块中第一个卷积层的卷积核个数，残差块个数，残差块中卷积步长)函数：生成多个连续的残差块的残差结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.avgpool6 = nn.AdaptiveAvgPool2d((1, 1))  # avgpool层 不知道对了没哈
        self.bn11=nn.BatchNorm2d(256)
        self.bn22 = nn.BatchNorm2d(512)
        self.bn33 = nn.BatchNorm2d(1024)
        self.bn44 = nn.BatchNorm2d(2048)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.fla = nn.Flatten(1)
        self.Linear1 = nn.Linear(512, 256)
       # self.Linear1 = nn.Linear(28, 512)
        self.Dropout = nn.Dropout(dropout_rate)
        self.Linear2 = nn.Linear(256, 67)
        # Lateral layers  1*1转换通道
        self.latlayer4 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.se1 = se_block(in_channel=in_channel1).to(device)
      #  self.SE=se_block(in_channel=in_channel1).to(device)
        self.spp = SpatialPyramidPooling([28, 28], [4, 2, 1])
        self.fla = nn.Flatten(1)
  #      if self.include_top:  # 默认为True，接上pooling、fc、softmax
   #         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化下采样，无论输入矩阵的shape为多少，output size均为的高宽均为1x1
            # 使矩阵展平为向量，如（W,H,C）->(1,1,W*H*C)，深度为W*H*C
    #        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层，512 * block.expansion为输入深度，num_classes为分类类别个数

        for m in self.modules():  # 初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # _make_layer()函数：生成多个连续的残差块，(残差块类型，残差块中第一个卷积层的卷积核个数，残差块个数，残差块中卷积步长)
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # 寻找：卷积步长不为1或深度扩张有变化，导致F(X)与X的shape不同的残差块，就要对X定义下采样函数，使之shape相同
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        # layers用于顺序储存各连续残差块
        # 每个残差结构，第一个残差块均为需要对X下采样的残差块，后面的残差块不需要对X下采样
        layers = []
        # 添加第一个残差块，第一个残差块均为需要对X下采样的残差块
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion
        # 后面的残差块不需要对X下采样
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # 以非关键字参数形式，将layers列表，传入Sequential(),使其中残差块串联为一个残差结构
        return nn.Sequential(*layers)

    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.latlayer4(x4)
        x3 = self.latlayer3(x3)
        x2 = self.latlayer2(x2)
        x1 = self.latlayer1(x1)

        x=self._upsample_add(x1,x2)
        x=self._upsample_add(x3,x)
        x=self._upsample_add(x4,x)
       # SE = se_block(in_channel=in_channel).to(DEVICE)
#        x = self.se1(x)

 #       num_sample = x.shape[0]

  #      x = self.spp(x, num_sample)

        # print("6")
        x=self.avgpool6(x)
        # x=torch.cat([x,y],1)    #去掉
     #   print(x.shape)
        x=self.fla(x)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Dropout(x)
        x = self.Linear2(x)
        x = self.LogSoftmax(x)
        # print(x.shape)
     #论文复现
      #  result = torch.cat([x1, x2,x3,x4], 1)
      #  result = self.Linear1(result)
       # result = self.relu(result)
       # result = self.Dropout(result)
       # result = self.Linear2(result)
        #result=self.LogSoftmax(result)
        #论文复现
       # return result
        return x


def resnet50he(dropout,num_classes=7, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,dropout_rate=dropout)


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)



image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

train_dataset = datasets.ImageFolder(root=r'C:\Users\可可豆子\Desktop\mit67\train', transform=image_transforms['train'])
#datasets.ImageFolder(root=r'C:\Users\可可豆子\Desktop\t3_pic\train', transform=image_transforms['valid'])


# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True)

# MNIST数据集
#train_dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
#test_dataset = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

# 网格搜索不同的超参数组合
learning_rates = [0.0001,0.0002]
dropout_rates = [0.2,0.4,0.6]

param_combinations = []
for lr in learning_rates:
    for dropout in dropout_rates:
        param_combinations.append((lr, dropout))

best_accuracy = 0.0
best_params = None

for lr, dropout in param_combinations:
    print(f"Training with lr={lr}, dropout={dropout}")
  #  model = SimpleCNN(dropout)
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    # 加载model
    resnet50 = torchvision.models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False
    cnn = resnet50he(dropout=dropout,num_classes=7).to(device)

  #  print(resnet50)
   # print("==============================")
   # print(cnn)
    # 读取参数
    pretrained_dict = resnet50.state_dict()
    model_dict = cnn.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    for k in pretrained_dict:
        pretrained_dict[k].requires_grad = False
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    cnn.load_state_dict(model_dict)
    # print(resnet50)
    # print(cnn)
  #  model = Net(resnet50,dropout)
  #  model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    accuracies = []
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=32, sampler=val_sampler)

        # 训练模型
        num_epochs = 100
        for epoch in range(num_epochs):
            cnn.train()
    #        print(model)
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = cnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Fold {fold+1}, Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        # 在验证集上评估模型
        cnn.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = cnn(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f"Mean Accuracy: {mean_accuracy:.2f}%")

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_params = (lr, dropout)

# 打印搜索结果
print("\nBest Hyperparameters:")
print(f"Learning Rate: {best_params[0]}, Dropout: {best_params[1]}, Accuracy: {best_accuracy:.2f}%")
