import torchvision
import torchvision.transforms as transforms
import pylab
import torch
from matplotlib import pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import os
import csv

from torch import nn, optim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_dir = './fashion_mnist'  # 设置存放位置
transform = transforms.Compose([transforms.ToTensor()])  # 可以自动将图片转化为Pytorch支持的形状[通道，高，宽]，同时也将图片的数值归一化

# 读取训练集数据
train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, transform=transform, download=True)
print("训练集的条数", len(train_dataset))

# 读取测试集数据
val_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=transform)
#  pd.read_csv('fashion-mnist_test_data.csv')
print("测试集的条数", len(val_dataset))

# 按批次封装FashionMNIST数据集
batch_size1 = 10  # 训练集批次大小
batch_size2 = 10000  # 测试集批次大小
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size1, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size2, shuffle=False)

# 定义类别名称
classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_Boot')

# 定义模型类
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义卷积层
        self.conva1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.conva2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
        self.convb1 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.convb2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        # 定义全连接层
        self.fc1 = torch.nn.Linear(in_features=32 * 4 * 4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=60)
        self.out = torch.nn.Linear(in_features=60, out_features=10)  # 10是固定的，因为必须要和模型所需要的分类个数一致

    def forward(self, t):
        # 第一层卷积
        t = self.conva1(t)
        # 第二层卷积
        t = self.conva2(t)
        # 激活
        t = F.relu(t)
        # 池化
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # 第三层卷积
        t = self.convb1(t)
        # 第四层卷积
        t = self.convb2(t)
        # 激活
        t = F.relu(t)
        # 池化
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # 搭建全连接网络，第一层全连接
        t = t.reshape(-1, 32 * 4 * 4)  # 将卷积结果由4维变为2维
        t = self.fc1(t)
        t = F.relu(t)
        # 第二层全连接
        t = self.fc2(t)
        t = F.relu(t)
        # 第三层全连接
        t = self.out(t)
        return t

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
    for epoch in range(2):  # 数据集迭代2次
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):  # 循环取出批次数据 使用enumerate()函数对循环计数，第二个参数为0，表示从0开始
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
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    # 保存模型
    torch.save(network.state_dict(), './models/CNNFashionMNist.PTH')
    # 加载模型，并且使用该模型进行预测
    network.load_state_dict(torch.load('./models/CNNFashionMNist.PTH'))  # 加载模型
    # 获取测试数据
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    inputs, labels = images.to(device), labels.to(device)
    # 测试模型
    class_correct = list(0. for i in range(10))  # 定义列表，收集每个类的正确个数
    class_total = list(0. for i in range(10))  # 定义列表，收集每个类的总个数
    with torch.no_grad():
        for data in test_loader:  # 遍历测试数据集
            images, labels = data
            inputs, labels = images.to(device), labels.to(device)
            outputs = network(inputs)  # 将每个批次的数据输入模型
            _, predicted = torch.max(outputs, 1)  # 计算预测结果,_返回最大可信度，predicted返回预测的标签
            predicted = predicted.to(device)
            c = (predicted == labels).squeeze()  # 统计正确的个数
            for i in range(10000):  # 遍历所有类别
                label = labels[i]
                class_correct[label] = class_correct[label] + c[i].item()  # 若该类别正确则+1
                class_total[label] = class_total[label] + 1  # 根据标签中的类别，计算类的总数
    sumacc = 0
    for i in range(10):  # 输出每个类的预测结果
        Accuracy = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (classes[i], Accuracy))
        sumacc = sumacc + Accuracy
    print('Accuracy of all : %2d %%' % (sumacc / 10.))  # 输出最终的准确率

list = []
for i in range(10000):
        out = str(predicted[i])[7]
        list.append([str(i)+".jpg", out])
with open("result.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
    # 写入多行用writerows
        writer.writerows(list)
