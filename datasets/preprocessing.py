"""
edit by hichens
data preprocessing
"""

import numpy as np
import pandas as pd 
import scipy.io as sio
import argparse
import os
from tqdm import tqdm

## feature extract
def calc_features(data):
    result = []
    result.append(np.mean(data))
    result.append(np.median(data))
    result.append(np.max(data))
    result.append(np.min(data))
    result.append(np.std(data))
    result.append(np.var(data))
    result.append(np.max(data)-np.min(data))
    result.append(pd.Series(data).skew())
    result.append(pd.Series(data).kurt())
    return result


def calc_subject_featured_data(data, flag):
    data = data[:, :, 128*3:] #(40, 40, 7680)
    featured_data = np.zeros([40,40,101])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(10):
                featured_data[i,j,k*9:(k+1)*9] = calc_features(data[i,j,k*128*6:(k+1)*128*6])
                featured_data[i,j,10*9:11*9] = calc_features(data[i,j,:])
                featured_data[i,j,99] = j
                featured_data[i,j,100] = flag
    return featured_data


# def process_labels(labels):
#     result = []
#     for i in range(len(labels)):
#         if(labels[i] <= 5):
#             result.append(0)
#         else:
#         result.append(1)
#     result = np.array(result)
#     return result


def process_labels(labels, split_index=5):
    result = np.array(labels)
    result[result <= split_index] = 0
    result[result > split_index] = 1
    return result

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./deapmatdatas", help="data directory", type=str)
parser.add_argument("--extract", default=False, help="wheather to extract features", type=bool)
args = parser.parse_args()

data_dir = args.data_dir
files = os.listdir(data_dir)
files = files[2:] # remove hidden file

for i in tqdm(range(len(files))):
    mat = sio.loadmat(os.path.join(data_dir, str(files[i])))
    data[i,:,:,:] = calc_subject_featured_data(mat['data'], i+1)
    valence_labels[i,:] = process_labels(mat["labels"][:,0])
    arousal_labels[i,:] = process_labels(mat["labels"][:,1])


## show the result
print(data[1,1,1,:])
print(arousal_labels[13,:])