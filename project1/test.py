import scipy.io as sio
import numpy as np

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


def cosDistance(a, b):
    denom = np.sqrt(sum(np.square(data['traindata'][a])) * sum(np.square(data['traindata'][b])))
    return 1 - np.dot(data['traindata'][a], data['traindata'][b]) / denom


print (cosDistance(0,1))
print (cosDistance(1,2))
