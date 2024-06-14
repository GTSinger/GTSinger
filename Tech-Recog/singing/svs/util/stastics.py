import os
import random

import numpy as np
from utils.commons.indexed_datasets import IndexedDataset
from tqdm import tqdm
import torch
import math
import csv
import matplotlib.pyplot as plt
import copy

def min_max(list):
    # list of [t, c]
    # channel 维度保留
    min = []
    max = []
    for arr in list:
        cur_min = np.min(arr, axis=0, keepdims=True)
        cur_max = np.max(arr, axis=0, keepdims=True)
        min.append(cur_min)
        max.append(cur_max)
    min = np.stack(min, axis=0)
    max = np.stack(max, axis=0)
    min = np.min(min, axis=0, keepdims=True)
    max = np.max(max, axis=0, keepdims=True)
    return min, max

def mel_minmax():
    mels = []
    binary_data_dir = "data/binary/rms"
    indexed_ds = IndexedDataset(f'{binary_data_dir}/train')
    data_size = len(indexed_ds)
    for idx in tqdm(range(data_size)):
        mel = indexed_ds[idx]["mel"]
        mels.append(mel)
    min, max = min_max(mels)
    # max = np.round(max, 4)
    print((min.tolist()))
    print((max.tolist()))

if __name__ == "__main__":
    mel_minmax()