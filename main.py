from net import Net
import os
import struct
import numpy as np

def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def onehot(targets, num):
    result = np.zeros((len(targets), num))
    for i in range(len(targets)):
        result[i][targets[i]] = 1
    return result

train = 60000
iteratin = 20
test = 10000

train_images, train_labels = load_mnist_train('ANN\\mnist', 'train')
train_images = train_images/255
train_goals = onehot(train_labels, 10)
test_images, test_labels = load_mnist_train('ANN\\mnist', 't10k')
test_images = test_images/255
test_goals = onehot(test_labels, 10)

recognizeFigure = Net([784, 14, 14, 10])
print('Training: 0.00%')
for j in range(iteratin):
    for i in range(train):
        train_results = recognizeFigure.train(train_images[i], train_goals[i])
        if i == train-1:
            loss = np.sqrt(np.sum((train_results-train_goals[i])**2)/10)
            print('Training: {:.2f}%----Loss: {:.4f}'.format((j+1)*100/iteratin, loss))

precison = 0
test_results = np.empty(test, dtype=int)
for i in range(test):
    test_results[i] = np.argmax(recognizeFigure.recognize(test_images[i]))
    if test_results[i] == test_labels[i]:
        precison += 1
print(test_labels[:20])
print(test_results[:20])
print('Precision: {:.2f}%'.format(precison/test*100))
