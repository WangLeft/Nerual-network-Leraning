import torch
import torchvision
import datainit as datainit
import mainCNN as cnn
import mainRNN as rnn
from datainit import Batch_Size
import matplotlib.pyplot as plt


#如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_loader, test_loader = datainit.dataInit('Dataset','train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz')

#########  全局  #########
Batch_Size = Batch_Size
EPOCHS = 20
BATCH_SIZE = 64

TIME_STEP = 28          # rnn time step / image height
HIDDEN_SIZE = 28
INPUT_SIZE = 28         # rnn input size / image width
NUM_LAYER = 1
LR = 0.01               # learning rate
##########################

def netMain(isRNN = True, isTrain = True):
    if isRNN:
        net = rnn.RNNNet(INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE, NUM_LAYER).to(device)
        if isTrain:
            rnn.startEpoch(EPOCHS,net, train_loader, test_loader, device, BATCH_SIZE, "./RNN_MNIST_Identify/netrnn.pth")
        rnnPredict(net, "./RNN_MNIST_Identify/netrnn.pth", 4)

    else:
        net = cnn.Net().to(device)
        if isTrain:
            cnn.oneEpoch(EPOCHS, net, train_loader, test_loader, device, "./RNN_MNIST_Identify/netcnn.pth")
        cnnPredict(net, "./RNN_MNIST_Identify/netcnn.pth", 4)

def cnnPredict(net, path, num):
    pth = torch.load(path)
    net.load_state_dict(pth)
    net.train(False)
    correct, testAccuracy = 0, 0

    with torch.no_grad():
        for batch_idx, (testImgs, labels) in enumerate(test_loader):
            if batch_idx == num:
                testImgs = testImgs.to(device)
                labels = labels.to(device)
                outputs = net(testImgs)
                predictions = torch.argmax(outputs,dim = 1)
                correct += torch.sum(predictions == labels)
                testAccuracy = correct/(Batch_Size )

                imgs = testImgs.cpu()
                y = predictions.cpu()
                fig, axs = plt.subplots(8, 8, figsize=(10, 10))
                fig.subplots_adjust(hspace=0.4)

                for i in range(64):
                    row = i // 8
                    col = i % 8
                    axs[row, col].imshow(imgs[i,0], cmap='gray')
                    axs[row, col].set_title(str(y[i].item()))
                    if labels[i] == predictions[i]:
                        axs[row, col].set_title(str(labels[i].item()))
                    else:
                        axs[row, col].set_title('{} (true: {})'.format(predictions[i].item(), labels[i].item()), color='red')
                    axs[row, col].axis('off')

                # n_errors = torch.count_nonzero(torch.eq(labels, predictions) == False)
                fig.text(0.5, 0.05, 'Accuracy: {:.2f} '.format(testAccuracy), ha='center')

                plt.show()
                print(testAccuracy)
                break
def rnnPredict(net, path, num):
    pth = torch.load(path)
    net.load_state_dict(pth)
    net.train(False)
    correct, testAccuracy = 0, 0

    with torch.no_grad():
        for batch_idx, (testImgs, labels) in enumerate(test_loader):
            if batch_idx == num:
                testImgs = testImgs.to(device)
                labels = labels.to(device)
                testImg = testImgs.view(-1, 28, 28)
                outputs = net(testImg)
                predictions = torch.argmax(outputs,dim = 1)
                correct += torch.sum(predictions == labels)
                testAccuracy = correct/(Batch_Size )

                imgs = testImgs.cpu()
                y = predictions.cpu()
                fig, axs = plt.subplots(8, 8, figsize=(10, 10))
                fig.subplots_adjust(hspace=0.4)

                for i in range(64):
                    row = i // 8
                    col = i % 8
                    axs[row, col].imshow(imgs[i,0], cmap='gray')
                    axs[row, col].set_title(str(y[i].item()))
                    if labels[i] == predictions[i]:
                        axs[row, col].set_title(str(labels[i].item()))
                    else:
                        axs[row, col].set_title('{} (true: {})'.format(predictions[i].item(), labels[i].item()), color='red')
                    axs[row, col].axis('off')

                # n_errors = torch.count_nonzero(torch.eq(labels, predictions) == False)
                fig.text(0.5, 0.05, 'Accuracy: {:.2f} '.format(testAccuracy), ha='center')

                plt.show()
                print(testAccuracy)
                break


if __name__ == '__main__':
    netMain(isRNN = True, isTrain = False)
