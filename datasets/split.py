"""
split the data to train and test
edit by hichens

Example
>>> python split.py --data_path ./data/small_data.npy
or 
>>> python split.py --data_path ./data/data.npy
"""
import numpy as np 
from sklearn.model_selection import train_test_split

import sys; sys.path.append("..")
from utils.options import opt

"""
data option
>>> --opt.data_path ./data/small_data.npy # num_dim=101
>>> --opt.data_path  ./data/data.npy# num_dim=7680
"""

data = np.load(opt.data_path)
label = np.load(opt.label_path)

X_train, X_test, y_train, y_test =  train_test_split(data, label, \
    test_size=opt.test_size, random_state=opt.seed)

np.save("./data/train_data.npy", X_train)
np.save("./data/train_label.npy", y_train)
np.save("./data/test_data.npy", X_test)
np.save("./data/test_label.npy", y_test)

print("split done !")