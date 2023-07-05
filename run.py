import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from TModel import TData, TModel
import argparse
import numpy as np
import random
import pickle

# 为了复现
random.seed(6689)
np.random.seed(6689)
torch.manual_seed(6689)

# 训练参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument("--pt", type=str)
parser.add_argument('--device', choices=['cuda', 'cpu'], default= 'cuda')
parser.add_argument('--dropout', type=float, default= 0.4)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_worker', type=int, default=16)
parser.add_argument('--input_size', type=int, default=9)
parser.add_argument('--hidden_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--label_cnt', type=int, default=20)
parser.add_argument('--maxLen', type=int, default=45)
args = parser.parse_args()

if args.mode == "train":
    trainData, testData, TScaler = TData("./dataset/aligned/data.csv").read()
    model = TModel(args, trainData, testData, TScaler)
    # model.model.load_state_dict(torch.load("./pt/TModel_1688543700_TomatoModel_train_0.032.pt"))
    model.train()
else:
    trainData, testData, TScaler = TData("./dataset/aligned/data.csv").read()
    model = TModel(args, trainData, testData, TScaler)
    model.model.load_state_dict(torch.load(args.pt))
    model.test()
