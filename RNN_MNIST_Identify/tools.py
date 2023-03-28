import torch
import matplotlib.pyplot as plt

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

def netsave(net, path):
    torch.save(net.state_dict(), path)