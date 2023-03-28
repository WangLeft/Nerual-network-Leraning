import gzip
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm


# Hyper Parameters
EPOCH = 4               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate


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


trainDataset = DealDataset('Dataset','train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz', transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=BATCH_SIZE, # 一个批次可以认为是一个包，每个包中含有10张图片
        shuffle=False,
    )

testDataset = DealDataset('Dataset','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz', transform = transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=BATCH_SIZE, # 一个批次可以认为是一个包，每个包中含有10张图片
        shuffle=False,
    )


class RNNNet(torch.nn.Module):
    def __init__(self):
        super(RNNNet, self).__init__()

        self.batch_size = BATCH_SIZE
        self.input_size = INPUT_SIZE
        self.hidden_size = 64
        self.num_layers = 1

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
        hidden = torch.zeros(self.num_layers, self.batch_size,  self.hidden_size)
        r_out, h_state = self.model(input, hidden)  # out: batch_size * seg_len * hidden_size
        # return out.view(-1, self.hidden_size)  # return: seg_len * batch_size 返回一个矩阵
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

net = RNNNet()
print(net)

EPOCHS = 4

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
                        testImgs = testImgs.view(-1, 28, 28)
                        outputs = net(testImgs)
                        loss = loss_func(outputs,labels)
                        predictions = torch.argmax(outputs,dim = 1)

                        totalLoss += loss
                        correct += torch.sum(predictions == labels)

                        testAccuracy = correct/(BATCH_SIZE * len(test_loader))
                        testLoss = totalLoss/len(test_loader)
                    # history['Test Loss'].append(testLoss.item())
                    # history['Test Accuracy'].append(testAccuracy.item())

                    processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                    (epoch,EPOCHS,loss.item(),accuracy.item(),testLoss.item(),testAccuracy.item()))
                processBar.close()