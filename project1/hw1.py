import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def cosDistance(a, b):
    denom = np.sqrt(sum(np.square(a)) * sum(np.square(b)))
    return 1 - np.dot(a, b) / denom


def knnForTraining(k, dist, trainingData, trainingLabel):
    errSum = 0
    for i in range(trainingData.shape[0]):
        temp = np.array([[dist[i][j], trainingLabel[j]] for j in range(trainingData.shape[0])])
        temp2 = np.lexsort(temp[:, ::-1].T)
        temp = temp[temp2]
        hotCnt = 0
        for l in range(k):
            if temp[l][1] == 1:
                hotCnt += 1
        if 2*hotCnt >= k:
            judge = 1
        else:
            judge = 2
        if judge != trainingLabel[i]:
            errSum += 1
    return errSum / trainingData.shape[0]


def knnForTesting(k, trainingData, trainingLabel, testingData, testingLabel):
    errSum = 0
    for i in range(testingData.shape[0]):
        temp = np.array([[cosDistance(testingData[i], trainingData[j]), trainingLabel[j]]
                            for j in range(trainingData.shape[0])])
        temp2 = np.lexsort(temp[:, ::-1].T)
        temp = temp[temp2]
        hotCnt = 0
        for l in range(k):
            if temp[l][1] == 1:
                hotCnt += 1
        if 2*hotCnt >= k:
            judge = 1
        else:
            judge = 2
        if judge != testingLabel[i]:
            errSum += 1
    return errSum / testingData.shape[0]

if __name__ == '__main__':
    data = sio.loadmat('faces.mat')
    preProcessDist = np.array([float('Inf') for i in range(data['traindata'].shape[0] * data['traindata'].shape[0])])\
        .reshape(280, 280)
    for i in range(preProcessDist.shape[0]):
        for j in range(preProcessDist.shape[1]):
            if preProcessDist[j][i] != float('Inf'):
                preProcessDist[i][j] = preProcessDist[j][i]
            else:
                preProcessDist[i][j] = cosDistance(data['traindata'][i], data['traindata'][j])

    trainResArr = [knnForTraining(1, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(10, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(20, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(30, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(40, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(50, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(60, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(70, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(80, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(90, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(100, preProcessDist, data['traindata'], data['trainlabels'])]

    testResArr = [knnForTesting(1, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(10, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(15, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(20, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(30, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(40, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(50, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(60, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(70, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(80, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(90, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
                  knnForTesting(100, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels'])]

    x = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.plot(x, trainResArr, 'b', x, testResArr, 'r')
    plt.show()
