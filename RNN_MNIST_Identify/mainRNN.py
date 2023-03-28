import torch
from datainit import Batch_Size
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        hidden = torch.zeros(self.batch_size, self.num_layers,  self.hidden_size)
        r_out, hiddenN = self.rnn(input, hidden)  # out: batch_size * seg_len * hidden_size
        # return out.view(-1, self.hidden_size)  # return: seg_len * batch_size 返回一个矩阵
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out



# def oneEpoch(EPOCHS, net, trainDataLoader, testDataLoader, device):

#     lossF = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(net.parameters())

#     for epoch in range(EPOCHS):
#         for step, (b_x, b_y) in enumerate(trainDataLoader):        # gives batch data
#             b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

#             output = net(b_x)                               # rnn output
#             loss = loss_func(output, b_y)                   # cross entropy loss
#             optimizer.zero_grad()                           # clear gradients for this training step
#             loss.backward()                                 # backpropagation, compute gradients
#             optimizer.step()                                # apply gradients

#             if step % 50 == 0:
#                 test_output = rnn(test_x)                   # (samples, time_step, input_size)
#                 pred_y = torch.max(test_output, 1)[1].data.numpy()
#                 accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
#                 print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

