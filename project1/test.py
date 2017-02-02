import scipy.io as sio
import numpy as np
import time

data = sio.loadmat('faces.mat')
# print(type(data))
# print(data['trainlabels'].shape)
# print(data['testlabels'])
# print(data['traindata'][0])
# print(data['testdata'])
# print(data['evaldata'])

# denom = np.sqrt(sum(np.square(data['traindata'][0])) * sum(np.square(data['traindata'][1])))

# denom = np.sqrt(np.sum(i**2 for i in data['traindata'][0]) * np.sum(i**2 for i in data['traindata'][1]))
# print(1 - np.dot(data['traindata'][0], data['traindata'][1]) / denom)
# print(np.square([1,2]))


# def cosDistance(a, b):
#     denom = np.sqrt(sum(np.square(data['traindata'][a])) * sum(np.square(data['traindata'][b])))
#     return 1 - np.dot(data['traindata'][a], data['traindata'][b]) / denom

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

preProcessData = np.array([float('Inf') for i in range(data['traindata'].shape[0] * data['traindata'].shape[0])])\
    .reshape(280, 280)
for i in range(preProcessData.shape[0]):
    for j in range(preProcessData.shape[1]):
        if preProcessData[j][i] != float('Inf'):
            preProcessData[i][j] = preProcessData[j][i]
        else:
            preProcessData[i][j] = cosDistance(data['traindata'][i], data['traindata'][j])
