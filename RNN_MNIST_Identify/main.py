import torch
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm

import datainit as datainit
import mainCNN as cnn
import mainRNN as rnn
from datainit import Batch_Size


#如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_loader, test_loader = datainit.dataInit('Dataset','train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz')

#########  全局  #########
Batch_Size = Batch_Size
EPOCHS = 4
##########################

# net = cnn.Net().to(device)

# cnn.oneEpoch(EPOCHS, net, train_loader, test_loader, device)


def cnnPredict():
    pth = torch.load("./RNN_MNIST_Identify/netpar.pth")
    net.load_state_dict(pth)
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
                print(testAccuracy)

# cnnPredict()


#########  全局  #########
Batch_Size = Batch_Size
EPOCHS = 4
##########################

net = rnn.RNNNet(input_size=28, hidden_size=32, batch_size=64, num_layers=1)
# print(net)



optimizer = torch.optim.Adam(net.parameters())   # optimize all cnn parameters
loss_func = torch.nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCHS):

    net.train()
    processBar = tqdm(train_loader, unit = 'step')

    for step, (b_x, b_y) in enumerate(processBar):        # gives batch data
        b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        output = net(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        predictions = torch.argmax(output, dim = 1)
        accuracy = torch.sum(predictions == b_y)/b_y.shape[0]

        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                    (epoch,EPOCHS,loss.item(),accuracy.item()))

        # if step % 50 == 0:
        #     test_output = rnn(test_loader)                   # (samples, time_step, input_size)
        #     pred_y = torch.max(test_output, 1)[1].data.numpy()
        #     accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
        #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
        if step == len(processBar)-1:
                correct,totalLoss = 0,0
                net.train(False)
                with torch.no_grad():
                    for testImgs,labels in test_loader:
                        # testImgs = testImgs.to(device)
                        # labels = labels.to(device)
                        outputs = net(testImgs)
                        loss = loss_func(outputs,labels)
                        predictions = torch.argmax(outputs,dim = 1)

                        totalLoss += loss
                        correct += torch.sum(predictions == labels)

                        testAccuracy = correct/(Batch_Size * len(test_loader))
                        testLoss = totalLoss/len(test_loader)
                    # history['Test Loss'].append(testLoss.item())
                    # history['Test Accuracy'].append(testAccuracy.item())

                    processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                    (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
                processBar.close()






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