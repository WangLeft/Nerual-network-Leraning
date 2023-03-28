import gzip
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np

Batch_Size = 256

class DealDataset(Dataset):

    # 类的实例化操作会自动调用该方法
    def __init__(self, dataset_folder, dataset_data, dataset_label, transform) -> None:

        (train_set, train_label) = load_data(dataset_folder, dataset_data, dataset_label)
        self.train_set = train_set
        self.train_label = train_label
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_label[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_label)


def load_data(dataset_folder, dataset_data, dataset_label):

    # with 语句用于在代码块执行完毕后自动执行清理操作，例如关闭文件、释放锁等等
    # with 语句使用上下文管理器对象,可以避免遗漏关闭文件、释放锁等操作，提高可靠性和安全性

    # os.path.join用于拼接文件路径
    with gzip.open(os.path.join("./RNN_MNIST_Identify", dataset_folder, dataset_label), 'rb') as labelpath:
        # np.frombuffer 可以将一个字符串或字节对象转换成 NumPy 数组
        y_train = np.frombuffer(labelpath.read(), np.uint8, offset=8)
    with gzip.open(os.path.join("./RNN_MNIST_Identify", dataset_folder, dataset_data), 'rb') as imgpath:
        # 一起使用可以将一个字符串或字节对象转换为指定形状的 NumPy 数组
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    # x自变量-图片 ; y结果-标签
    return (x_train, y_train)

def  dataInit(dataset_folder, dataset_data,
              dataset_label, datasettrain_data, datasettrain_label, isPrint = False):
    trainDataset = DealDataset(dataset_folder, dataset_data, dataset_label, transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=Batch_Size, # 一个批次可以认为是一个包，每个包中含有10张图片
        shuffle=False,
    )

    testDataset = DealDataset("Dataset", datasettrain_data, datasettrain_label, transform = transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=Batch_Size, # 一个批次可以认为是一个包，每个包中含有10张图片
        shuffle=False,
    )


    if(isPrint):
        print("训练集样本和标签的大小:", trainDataset.train_set.shape,trainDataset.train_label.shape)
        # #查看数据，例如训练集中第一个样本的内容和标签
        print("训练集中第一个样本的内容:", trainDataset.train_set[0])
        print("训练集中第一个样本的标签:", trainDataset.train_label[0])

    return (train_loader, test_loader)


