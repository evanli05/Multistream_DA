"""
@author: yifan
@data: June 7, 2018
@objective: apply domain adaptation to Debo's paper
"""

import logging
import numpy as np
import pandas as pd
from sklearn import svm
from mSDA import MSDA
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

kRange = [2, 4, 6, 8]
bRange = [0.2, 0.2]

k = 2 # define the dimension of latent feature space
b = 4 # define the similarity
r = 1 # ratio of data seleted

# initialization:
# the data we use here are packet data, source data are from Windows, and target
# data are from Linux

sampling = 0 # sampling switch, 0 stands for not sampling, 1 stands for sampling

# first: source
# df = pd.read_csv('/home/yifan/Dropbox/workspace/debo/linux2.csv')
df = pd.read_csv('windows2.csv')
print('source size:', df.shape)
# print(df['class'])
df['class'] = df['class'].map({'webpage1':1, 'webpage2':2, 'webpage3':3,
                'webpage4':4, 'webpage5':5, 'webpage6':6})
source = df.as_matrix()

rows = max(np.shape(source))
sourceIndex = int(rows * r)


################################################

# from here select source samples
if sampling == 1:     
    perms = np.arange(source.shape[0])
    np.random.shuffle(perms)
    source = source[perms]


    source = source[:sourceIndex, :]
################################################


train_feature = source[:450, :-1]
train_label = source[:450, -1]

########################################
# # old code
# train_feature = source[:450, :-1]
# train_label = source[:450, -1]
# # print(train_label)
print('train shape:', np.shape(train_feature))
########################################


# print(source[:, -1])

# second: target
# df = pd.read_csv('/home/yifan/Dropbox/workspace/debo/windows2.csv')
df = pd.read_csv('linux2.csv')
print('target size:', df.shape)

# print(df['class'])
df['class'] = df['class'].map({'webpage1':1, 'webpage2':2, 'webpage3':3,
                'webpage4':4, 'webpage5':5, 'webpage6':6})
target = df.as_matrix()

#######################################################

# select target samples
if sampling == 1:
    permt = np.arange(target.shape[0])
    np.random.shuffle(permt)
    target = target[permt]
    target = target[:sourceIndex, :]

#######################################################

test_feature = target[:450, :-1]
test_label = target[:450, -1]


# old code
# test_feature = target[:450, :-1]
# test_label = target[:450, -1]
print('test shape:', np.shape(test_feature))


msda = MSDA(train_feature, test_feature, k, beta= b)

m = msda.genAMatrix()
w, v, u = msda.topK()
# print(u)

Bt, Bs = msda.genBtBs(u)
Bt = Bt.real
Bs = Bs.real
# print(np.shape(Bt))
# print(np.shape(Bs))

print('Bs shape:', np.shape(Bs))
print('Bt shape:', np.shape(Bt))
print('Label shape:', np.shape(train_label))
# print(train_label)


# # visualization of mapping data
# for row in range(len(Bs)):
#     if train_label[row] == 1:
#         plt.scatter(Bs[row, 0], Bs[row, 1], color = 'b', marker = '.')
#     elif train_label[row] == 2:
#         plt.scatter(Bs[row, 0], Bs[row, 1], color = 'g', marker = ',')
#     elif train_label[row] == 3:
#         plt.scatter(Bs[row, 0], Bs[row, 1], color = 'r', marker = 'o')
#     elif train_label[row] == 4:
#         plt.scatter(Bs[row, 0], Bs[row, 1], color = 'c', marker = 'v')
#     elif train_label[row] == 5:
#         plt.scatter(Bs[row, 0], Bs[row, 1], color = 'm', marker = '^') 
#     else:
#         plt.scatter(Bs[row, 0], Bs[row, 1], color = 'y', marker = '<') 

# plt.show()




###################################################################

# # using ANN
# clf = MLPClassifier(solver='adam', alpha=1e-5,
#                     hidden_layer_sizes=(50, 10), random_state=5)

# clf.fit(Bs, train_label)
# test_predict = clf.predict(Bt)
# # print(test_predict)



###################################################################

# using SVM

# clf = svm.SVC(probability=True, C=131072, gamma=0.0000019073486328125, kernel="rbf")
clf = svm.SVC()
clf.fit(Bs, train_label)

# test_predict = np.zeros((1, 5))


test_predict = clf.predict(Bt)
# print(test_predict)


###################################################################

result = [test_label, test_predict]

logging.basicConfig(filename = 'sen_logger', level = logging.DEBUG)
logger = logging.getLogger()

corPrediction = 0
for i in range(450):
    if test_label[i] == test_predict[i]:
        corPrediction += 1

accuracy = corPrediction / 450

output = str(('k is:', k, 'b is:', b, 'Accuracy is:', accuracy))
# print(output)

logger.info(output)