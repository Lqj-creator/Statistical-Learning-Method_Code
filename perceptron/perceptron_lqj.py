import numpy as np
import time

def loadData(fileName):

    print('start to read data')

    dataArr = []; labelArr = []
    file = open(fileName, 'r')

    for line in file.readlines():
        line = line.strip().split(',')
        if int(line[0]) >= 5:
            labelArr.append(1)
        else:
            labelArr.append(0)
        dataArr.append([int(num)/255 for num in line[1:]])

    return dataArr, labelArr

def perceptron(dataArr, labelArr, iter):

    print('start to trans')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T

    m, n = np.shape(dataMat)
    w = np.zeros((1, n))
    b = 0
    h = 0.0001

    for k in range(iter):
        for i in range(m):
            x = labelMat[i] * (w * dataMat[i].T + b)
            if x <= 0:
                w = w + labelMat[i] * dataMat[i] * h
                b = b + labelMat[i] * h
    print('Round %d:%d traning' % (k, iter))
    return w, b


def model_test(dataArr, labelArr, w, b):

    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T

    m, n = np.shape(dataMat)

    for i in range(m):
        x = labelMat[i] * (dataMat[i] * w.T + b)
        num = 0
        if x <= 0:
            num += 1

    accu = 1 - (num / m)

    return accu


if __name__ == '__main__':

    start = time.time()

    trainData, trainlabel = loadData('../Mnist/mnist_train1.csv')
    testData, testlabel = loadData('../Mnist/mnist_test1.csv')

    w, b = perceptron(trainData, trainlabel, iter = 30)

    accu = model_test(testData, testlabel, w, b)

    end = time.time()

    print('accuracy rate is:%f' % (accu))
    print('time: %f' % (end - start))



