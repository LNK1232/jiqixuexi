import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import os
import csv
from torch import nn, optim
from torch.utils.data import Dataset

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_dir = './fashion_mnist'  # 设置存放位置
transform = transforms.ToTensor()
# 读取训练集数据
train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, transform=transform, download=True)
print("训练集的条数", len(train_dataset))
test = pd.read_csv('fashion-mnist_test_data.csv')
print(test.shape)
print(test.head(10))

class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28).astype(float)
        #取出从第二列开始的所有图片数据并reshape
        del data  # 结束data对数据的引用,节省空间
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item = self.X[idx]
        return (item)
# 读取测试集数据

test_dataset = FashionMNISTDataset('fashion-mnist_test_data.csv')
print("测试集的条数", len(test_dataset))

# 按批次封装FashionMNIST数据集
batch_size = 10 #
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义类别名称
classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_Boot')

# 定义模型类
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.Sigmoid())  # 16, 28, 28
        self.pool1 = nn.MaxPool2d(2)  # 16, 14, 14
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.Sigmoid())  # 32, 12, 12
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.Sigmoid ())  # 64, 10, 10
        self.pool2 = nn.MaxPool2d(2)  # 64, 5, 5
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.pool1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.pool2(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    network = Net()  # 生成自定义模块的实例化对象
    # 指定设备
    device = torch.device("cuda:0")
    print(device)
    network.to(device)
    # 损失函数与优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    # 训练模型
    rounds=200
    count = 0
    a = np.zeros(rounds * 6)
    for epoch in range(rounds):  # 数据集迭代20次
        running_loss = 0.0
        for i, data in enumerate(train_loader):  # 循环取出批次数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 清空之前的梯度
            outputs = network(inputs)
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss /1000 ))
                a[count]=running_loss/1000
                count=count+1
                running_loss = 0.0
    print(a)
    plt.plot(a)
    plt.show()
    print('Finished Training')
    # 保存模型
    torch.save(network.state_dict(), './models/CNNFashionMNist.PTH')
    # 加载模型，并且使用该模型进行预测
    network.load_state_dict(torch.load('./models/CNNFashionMNist.PTH'))  # 加载模型
    # 获取测试数据
    dataiter = iter(test_loader)
    images = dataiter.next()
    inputs = images.to(device)
    # 测试模型
    list = []
    with torch.no_grad():
        x = 0
        for images in test_loader:  # 遍历测试数据集
            inputs = images.float().to(device)
            outputs = network(inputs)  # 将每个批次的数据输入模型
            _, predicted = torch.max(outputs, 1)  # 计算预测结果,_返回最大可信度，predicted返回预测的标签
            predicted = predicted.to(device)
            print(predicted)
            for i in range(10):
                label = str(predicted[i])[7]
                list.append([str(x*10+i)+'.jpg',label])
            x += 1

    with open("result.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(list)
