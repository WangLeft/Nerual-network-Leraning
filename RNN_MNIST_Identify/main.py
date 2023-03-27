import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np

import gzip
import os

#如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DealDataset(Dataset):

    # 类的实例化操作会自动调用该方法
    def __init__(self, dataset_folder, dataset_data, dataset_label, transform) -> None:

        (train_set, train_label) = load_data(dataset_folder, dataset_data, dataset_label)
        self.train_set = train_set
        self.train_label = train_label
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_label)


def load_data(dataset_folder, dataset_data, dataset_label):

    # with 语句用于在代码块执行完毕后自动执行清理操作，例如关闭文件、释放锁等等
    # with 语句使用上下文管理器对象,可以避免遗漏关闭文件、释放锁等操作，提高可靠性和安全性

    # os.path.join用于拼接文件路径
    with gzip.open(os.path.join(".", dataset_folder, dataset_label), 'rb') as labelpath:
        # np.frombuffer 可以将一个字符串或字节对象转换成 NumPy 数组
        y_train = np.frombuffer(labelpath.read(), np.uint8, offset=4)
    with gzip.open(os.path.join(".", dataset_folder, dataset_data), 'rb') as imgpath:
        # 一起使用可以将一个字符串或字节对象转换为指定形状的 NumPy 数组
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    # x自变量-图片 ; y结果-标签
    return (x_train, y_train)

# trainDataset = DealDataset("Dataset", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

# train_loader = torch.utils.data.DataLoader(
#     dataset=trainDataset,
#     batch_size=10, # 一个批次可以认为是一个包，每个包中含有10张图片
#     shuffle=False,
# )

# # 实现单张图片可视化
# images, labels = next(iter(train_loader))
# img = torchvision.utils.make_grid(images)

# img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
# print(labels)
# plt.imshow(img)
# plt.show()

# path = os.path.join("./Others", "RMI_IF.png")

if os.path.exists("README_RMI.md"):
    print("File or directory exists")
else:
    print("File or directory does not exist")

# script_dir = os.path.dirname(os.path.abspath(__file__))
# print(script_dir)