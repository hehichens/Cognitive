"""
split the label to train and test
edit by hichens

Example
>>> python split.py --data_path ./data/small_data.npy 
>>> python split.py --data_path ./data/data.npy
"""


import numpy as np 
from sklearn.model_selection import train_test_split

import sys; sys.path.append("..")
from utils.options import opt


"""
data option
>>> --data_path ./data/small_data.npy # num_dim=101
>>> --data_path  ./data/data.npy# num_dim=7680

label options
>>> --label_path ./data/arousal_labels.npy
>>> --label_path ./data/valence_labels.npy
"""


# load data
data = np.load(opt.data_path)
label = np.load(opt.label_path)

# combine 0 and 1 dimension
data = np.concatenate(data, axis=0) # 1280X40Xnum_dim
label = np.concatenate(label, axis=0) # 1280

# split
X_train, X_test, y_train, y_test = train_test_split(data, label, \
    test_size=opt.test_size)

# save 
np.save("./data/train_data.npy", X_train)
np.save("./data/train_label.npy", y_train)
np.save("./data/test_data.npy", X_test)
np.save("./data/test_label.npy", y_test)

# print some info
print("train data shape: {}, train label shape: {}".format(X_train.shape, y_train.shape))
print("test data shape: {}, test label shape: {}".format(X_test.shape, y_test.shape))
print("split done !")