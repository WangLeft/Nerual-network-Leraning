import torch
from datainit import Batch_Size
from tqdm import tqdm
import tools

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


def oneEpoch(EPOCHS, net, trainDataLoader, testDataLoader, device, savepath):
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

    tools.drawline(history)
    tools.netsave(net, savepath)
