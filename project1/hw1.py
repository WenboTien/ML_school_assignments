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

    trainResArr = [knnForTraining(2, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(4, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(6, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(8, preProcessDist, data['traindata'], data['trainlabels']),
                   knnForTraining(10, preProcessDist, data['traindata'], data['trainlabels'])]
    #                knnForTraining(30, preProcessDist, data['traindata'], data['trainlabels']),
    #                knnForTraining(40, preProcessDist, data['traindata'], data['trainlabels']),
    #                knnForTraining(50, preProcessDist, data['traindata'], data['trainlabels']),
    #                knnForTraining(60, preProcessDist, data['traindata'], data['trainlabels']),
    #                knnForTraining(70, preProcessDist, data['traindata'], data['trainlabels']),
    #                knnForTraining(80, preProcessDist, data['traindata'], data['trainlabels']),
    #                knnForTraining(90, preProcessDist, data['traindata'], data['trainlabels']),
    #                knnForTraining(100, preProcessDist, data['traindata'], data['trainlabels'])]
    #
    # testResArr = [knnForTesting(1, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(5, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(10, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(15, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(20, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(30, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(40, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(50, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(60, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(70, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(80, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(90, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels']),
    #               knnForTesting(100, data['traindata'], data['trainlabels'], data['testdata'], data['testlabels'])]
    #
    print(trainResArr)
    # print(testResArr)


    # x = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # y_train = [0.0, 0.3, 0.37857142857142856, 0.39285714285714285, 0.42857142857142855, 0.45357142857142857,
    #            0.4607142857142857, 0.46785714285714286, 0.475, 0.48928571428571427, 0.48928571428571427, 0.5,
    #            0.49642857142857144]
    # y_test = [0.4583333333333333, 0.49166666666666664, 0.475, 0.425, 0.4083333333333333, 0.43333333333333335, 0.475,
    #           0.49166666666666664, 0.49166666666666664, 0.5, 0.5, 0.5, 0.5]
    # plt.plot(x,y_train, 'b', x,y_test, 'r')
    # plt.show()