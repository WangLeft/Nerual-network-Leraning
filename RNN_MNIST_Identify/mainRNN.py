import torch
from datainit import Batch_Size
from tqdm import tqdm
import tools

TIME_STEP = 16
INPUT_SIZE = 1
HIDDEN_SIZE = 32

class RNNNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(RNNNet, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.model = torch.nn.RNN(
            input_size = self.input_size, hidden_size = self.hidden_size,
            num_layers = self.num_layers, batch_first = True)

        self.out = torch.nn.Linear( self.hidden_size , 10)

    def forward(self, input):  # 这里的input： batch_size * seg_len * input_size
        # input shape (batch, time_step, input_size)

        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        # 初始化hidden，即h0: num_layers * batch_size * hidden_size
        # hidden = torch.zeros(self.batch_size, self.num_layers,  self.hidden_size)
        r_out, hiddenN = self.model(input, None)  # out: batch_size * seg_len * hidden_size
        # return out.view(-1, self.hidden_size)  # return: seg_len * batch_size 返回一个矩阵
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out



def startEpoch(EPOCHS,net, trainDataLoader, testDataLoader, device, batch_size, savepath):

    history = {'Test Loss':[],'Test Accuracy':[]}

    optimizer = torch.optim.Adam(net.parameters())   # optimize all cnn parameters
    loss_func = torch.nn.CrossEntropyLoss()                       # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCHS):

        net.train()
        processBar = tqdm(trainDataLoader, unit = 'step')

        for step, (dataImg, dataLabel) in enumerate(processBar):        # gives batch data
            dataImg, dataLabel = dataImg.to(device), dataLabel.to(device)
            dataImg = dataImg.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

            output = net(dataImg)                               # rnn output
            loss = loss_func(output, dataLabel)                   # cross entropy loss
            optimizer.zero_grad()                           # clear gradients for this training step
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients

            predictions = torch.argmax(output, dim = 1)
            accuracy = torch.sum(predictions == dataLabel)/dataLabel.shape[0]

            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                        (epoch,EPOCHS,loss.item(),accuracy.item()))

            if step == len(processBar)-1:
                    correct,totalLoss = 0,0
                    net.train(False)
                    with torch.no_grad():
                        for testImgs,labels in testDataLoader:
                            testImgs = testImgs.to(device)
                            labels = labels.to(device)
                            testImgs = testImgs.view(-1, 28, 28)
                            outputs = net(testImgs)
                            loss = loss_func(outputs,labels)
                            predictions = torch.argmax(outputs,dim = 1)

                            totalLoss += loss
                            correct += torch.sum(predictions == labels)

                            testAccuracy = correct/(batch_size * len(testDataLoader))
                            testLoss = totalLoss/len(testDataLoader)
                        history['Test Loss'].append(testLoss.item())
                        history['Test Accuracy'].append(testAccuracy.item())

                        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                        (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
                    processBar.close()

    tools.drawline(history)
    tools.netsave(net, savepath)