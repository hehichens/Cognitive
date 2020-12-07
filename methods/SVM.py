"""
SVM classification
"""
from sklearn.svm import SVC
import sys; sys.path.append("..")
from utils.options import opt
import numpy as np 


def Accuracy(y_hat, y):
    pred = y_hat.max(1)[1]
    return (pred == y).sum().item() / len(pred)



## load data
X_train, X_test, y_train, y_test = np.load("../datasets/data/train_data.npy"), \
    np.load("../datasets/data/test_data.npy"), np.load("../datasets/data/train_label.npy"), np.load("../datasets/data/test_label.npy")

X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)

print("train data shape: {}, train label shape: {}".format(X_train.shape, y_train.shape))
print("test data shape: {}, test label shape: {}".format(X_test.shape, y_test.shape))


model = SVC(C=1.0, kernel='rbf', degree=10, 
                coef0=0.0, shrinking=True, probability=False, tol=0.001, 
                cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
                decision_function_shape='ovr', random_state=None)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("=="*20)
print("Accuracy: ", (y_test==pred).mean())