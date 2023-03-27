import torch
import torchvision
import matplotlib.pyplot as plt

import datainit as datainit
import main_CNN as cnn
from datainit import Batch_Size


#如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_loader, test_loader = datainit.dataInit('Dataset','train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz')

#########  全局  #########
Batch_Size = Batch_Size
EPOCHS = 4
##########################

net = cnn.Net().to(device)

# cnn.oneEpoch(EPOCHS, net, train_loader, test_loader, device)
pth = torch.load("./RNN_MNIST_Identify/netpar.pth")
net.load_state_dict(pth)
# net.load_state_dict({k.replace('module.',''):v for k,v in torch.load("./RNN_MNIST_Identify/netpar.pth")})
net.train(False)
correct, testAccuracy = 0, 0
with torch.no_grad():
    for batch_idx, (testImgs, labels) in enumerate(test_loader):
        if batch_idx == 1:
            testImgs = testImgs.to(device)
            labels = labels.to(device)
            outputs = net(testImgs)
            predictions = torch.argmax(outputs,dim = 1)

            correct += torch.sum(predictions == labels)

            testAccuracy = correct/(Batch_Size )

            img = torchvision.utils.make_grid(testImgs)
            print(predictions)
            print(labels)

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# with torch.no_grad():
#   output = network(example_data)
# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Prediction: {}".format(
#     output.data.max(1, keepdim=True)[1][i].item()))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()













# 实现单张图片可视化
# 下面用到了__getitem__
# images, labels = next(iter(train_loader))
# img = torchvision.utils.make_grid(images)

# img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
# print(labels)
# plt.imshow(img)
# plt.show()