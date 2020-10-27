"""
@author: yifan
@data: June 7, 2018
@objective: apply domain adaptation to Debo's paper
"""


import numpy as np
import pandas as pd
from sklearn import svm
from mSDA import MSDA
from sklearn.neural_network import MLPClassifier


# initialization:
# the data we use here are packet data, source data are from Windows, and target
# data are from Linux

# first: source
# df = pd.read_csv('/home/yifan/Dropbox/workspace/CIKM2018/debo/windows1.csv')
df = pd.read_csv('linux2.csv')
# print('source size:', df.shape)
# print(df['class'])
df['class'] = df['class'].map({'webpage1':1, 'webpage2':2, 'webpage3':3,
                'webpage4':4, 'webpage5':5, 'webpage6':6})

source = df.as_matrix()
# print(source)
perm = np.arange(source.shape[0])
np.random.shuffle(perm)
source = source[perm]



# print(source)

rows = min(np.shape(source))

trainIndex = int(rows*0.8)
training = source[:trainIndex, :]
testing = source[trainIndex:, :]
testing_label = testing[:, -1]

print(np.shape(training))
print(np.shape(testing))

###########################################################

# SVM

clf = svm.SVC()
clf.fit(training[:, :-1], training[:, -1])
test_predict = clf.predict(testing[:, :-1])

# print(test_predict)

###########################################################

###########################################################

# # using ANN
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(10, 100), random_state=1)

# clf.fit(Bs, train_label)
# test_predict = clf.predict(Bt)
# # print(test_predict)

###########################################################

corPrediction = 0
for i in range(96):
    if testing_label[i] == test_predict[i]:
        corPrediction += 1

accuracy = corPrediction/96

print('Accuracy is:', accuracy)




# print(source[:,-1])





# train_feature = source[:, 0:-1]
# train_label = source[:, -1]

# print(train_label)
# print(np.shape(train_feature))

# print(source[:, -1])
