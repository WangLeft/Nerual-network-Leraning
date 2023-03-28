import torch
from datainit import Batch_Size
from tqdm import tqdm
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(
            #The size of the picture is 28x28
            torch.nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),

            #The size of the picture is 14x14
            torch.nn.Conv2d(in_channels = 16,out_channels = 32,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2,stride = 2),

            #The size of the picture is 7x7
            torch.nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 7 * 7 * 64,out_features = 128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 128,out_features = 10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self,input):
        output = self.model(input)
        return output


def oneEpoch(EPOCHS, net, trainDataLoader, testDataLoader, device):
    history = {'Test Loss':[],'Test Accuracy':[]}

    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(1,EPOCHS + 1):
        processBar = tqdm(trainDataLoader, unit = 'step')

        net.train()
        for batch_idx, (data, target) in enumerate(processBar):
            data, target = data.to(device), target.to(device)

            # 避免前一次迭代中计算的梯度累积起来
            optimizer.zero_grad()
            outputs = net(data)
            loss = lossF(outputs, target)  # 这里使用交叉熵代价函数
            loss.backward()  # 反向传播，计算损失相对于模型参数的梯度
            optimizer.step()  # 更新模型参数

            predictions = torch.argmax(outputs, dim = 1)
            accuracy = torch.sum(predictions == target)/target.shape[0]

            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                    (epoch,EPOCHS,loss.item(),accuracy.item()))


            if batch_idx == len(processBar)-1:
                correct,totalLoss = 0,0
                net.train(False)
                with torch.no_grad():
                    for testImgs,labels in testDataLoader:
                        testImgs = testImgs.to(device)
                        labels = labels.to(device)
                        outputs = net(testImgs)
                        loss = lossF(outputs,labels)
                        predictions = torch.argmax(outputs,dim = 1)

                        totalLoss += loss
                        correct += torch.sum(predictions == labels)

                        testAccuracy = correct/(Batch_Size * len(testDataLoader))
                        testLoss = totalLoss/len(testDataLoader)
                    history['Test Loss'].append(testLoss.item())
                    history['Test Accuracy'].append(testAccuracy.item())

                    processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                    (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
                processBar.close()

    drawline(history)
    netsave(net)

def drawline(history):
    plt.plot(history['Test Loss'],label = 'Test Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

def netsave(net):
    torch.save(net.state_dict(), './RNN_MNIST_Identify/netCNNstate.pth')

# def train( model, device, train_loader, optimizer, epoch):
#     """训练网络:
#     model(net) 为定义的模型； device 为用到的设备； train_loader 为训练数据；
#     optimizer 为权值更新方式； epoch 为训练的轮数"""

#     lossF = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters())

#     model.train()
#     # PyTorch 只接收 batch 作为输入数据，即输入是一个四维数组（nSamples*nChannels*Height*Weight）
#     # 每一次循环就是一个batch
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # 数据给到指定设备 CPU/GPU
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = lossF(output, target)  # 这里使用交叉熵代价函数
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新
#         # print(batch_idx)
#          # 输出日志
#         if batch_idx == len(train_loader.dataset)/(Batch_Size)-1:
#             print(
#                 "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}".format(
#                     epoch,
#                     batch_idx * len(data),
#                     len(train_loader.dataset),
#                     100.0 * batch_idx / len(train_loader),
#                     loss.item(),
#                 )
#             )

# def test(model, device, test_loader, history):
#     """测试网络
#     args 命令行参数； model 定义的模型； device 用到的设备；n test_loader 测试数据；
#     """
#     model.eval()  # 指定模型为测试模式
#     test_loss = 0
#     correct = 0

#     # 无梯度模式，具体看 PyTorch 的自动求导机制文档
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += lossF(output, target)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     history['Test Loss'].append(test_loss)
#     history['Test Accuracy'].append(correct)
#     print(
#         "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
#             test_loss,
#             correct,
#             len(test_loader.dataset),
#             100.0 * correct / len(test_loader.dataset),
#         )
#     )