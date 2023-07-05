import time
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import TomatoModel
from utils import readImg
import random

class TReader(Dataset):
    def __init__(self, args, imgs, data, input_size, maxLen):
        super(TReader, self).__init__()
        self.args = args
        self.input_size = input_size
        self.maxLen = maxLen
        self.data = data
        self.imgs = imgs
        self.trainTransform = transforms.Compose([
                    transforms.Resize((120, 120)),  # 缩放
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])
        
    def __getitem__(self, item):
        _input = [element for l in self.data[item][0] for element in l]
        _label = [element for l in self.data[item][1] for element in l]
        _imgs_ptah = [element for l in self.data[item][2] for element in l]
        dataLen = len(_input)
        if dataLen<self.maxLen:
            _input += [[0 for _ in range(self.input_size)] for _ in range(self.maxLen-dataLen)]
            _imgs_ptah += [ _imgs_ptah[dataLen-1] for _ in range(self.maxLen-dataLen)]
        else:
            _input = _input[:45]
            _imgs_ptah = _imgs_ptah[:45]
            dataLen = 45
        if len(_label)< self.args.label_cnt:
            _label += [0 for _ in range(self.args.label_cnt-len(_label))]
        else:
            _label = _label[:self.args.label_cnt]
        _input = np.array(_input,dtype=np.float32)
        _label = np.array(_label,dtype=np.float32)
        imgs = None
        for i in _imgs_ptah:
            img = self.imgs[i]
            # img = self.trainTransform(img)
            if imgs == None:
                imgs = img.unsqueeze(0)
            else:
                imgs = torch.vstack((imgs, img.unsqueeze(0)))
        _input = torch.tensor(_input, dtype=torch.float)
        _label = torch.tensor(_label, dtype=torch.float)
        return _input, _label, imgs, dataLen
        
    def __len__(self):
        return len(self.data) 
    
class TData:
    def __init__(self, filePath):
        self.filePath = filePath

    def divideData(self, data):
        selected_col = ['土壤水分%', '土壤温度℃', '叶面积', '叶夹角','高度', 'RVI', 'LNC', 'LNA', 'LAI']
        inputD=[]
        labelD=[]
        img_paths = []
        tmpData = []
        tmpLabel = []
        tmpPaths = []
        for _k, _v in data:
            _d = _v.groupby('year_month_day')
            for key, value in _d:
                tmpData.append(value[selected_col].values)
                tmpLabel.append(value["LAI"].values)
                tmpPaths.append(value['img_path'].values)
                if(len(tmpData)==10):
                    inputD.append(tmpData[0:5])
                    img_paths.append(tmpPaths[0:5])
                    labelD.append(tmpLabel[5:10])
                    # tmpData = []
                    # tmpLabel = []
                    tmpData = tmpData[1:10]
                    tmpLabel = tmpLabel[1:10]
                    tmpPaths = tmpPaths[1:10]

        return inputD, labelD, img_paths
    
    def read(self):
        data = pd.read_csv(self.filePath, parse_dates=['时间'], index_col='时间', 
                           date_parser=lambda x:pd.to_datetime(x,format='%Y-%m-%d %H:%M:%S'))
        data['year'] = data.index.year
        data['month'] = data.index.month
        data['hour'] = data.index.hour
        data['year_month_day'] = data.index.year.astype(str) + '_' + data.index.month.astype(str) + '_' + data.index.day.astype(str)
        selected_col = ['空气温度℃', '相对湿度%', '光照klux', '二氧化碳ppm', '土壤水分%', '土壤温度℃', '叶面积', '叶夹角',
                        '高度', 'RVI', 'LNC', 'LNA', 'LAI', 'LDW']
        # 归一化
        TScaler = []
        for col in selected_col:
            scaler = MinMaxScaler()
            data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
            if col == "LAI":
                TScaler.append(scaler)
        # 划分训练集和测试集
        groupD = data.groupby("编号")
        _input, _label, _imgs = self.divideData(groupD)
        train_group = []
        test_group = []
        sample = random.sample(range(0, len(_input)), 50)
        print(sample)
        for i in range(0, len(_input)):
            if i in sample:
                test_group.append((_input[i], _label[i], _imgs[i]))
            else:
                train_group.append((_input[i], _label[i], _imgs[i]))

        return train_group, test_group, TScaler

class TModel(nn.Module):
    def __init__(self, args, trainData, testData, TScaler):
        super(TModel, self).__init__()
        # 定义参数
        self.args = args
        self.lr = args.lr
        self.batchSize = args.batch_size
        self.device = torch.device(args.device)
        self.labelCnt = args.label_cnt
        self.testAvgDeVal = 100
        self.trainAvgDeVal = 100
        self.TScaler = TScaler
        self.imgs = readImg("./dataset/aligned/data.csv")
        
        self.trainLoader = DataLoader(
            dataset=TReader(args, self.imgs, trainData, self.args.input_size, self.args.maxLen),
            num_workers=args.num_worker,
            batch_size=args.batch_size,
            shuffle=True
        )
        self.testLoader = DataLoader(
            dataset=TReader(args, self.imgs, testData, self.args.input_size, self.args.maxLen),
            num_workers=args.num_worker,
            batch_size=1,
            shuffle=True
        )
        
        # 定义模型、优化器
        self.model = TomatoModel(args.input_size, self.args.label_cnt, args.hidden_size, self.device, isBidirectional = False).to(self.device)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss().to(self.device)
        self.scheduler = MultiStepLR(self.optim, milestones=[30, 60, 90], gamma=0.5) 
        
    def train(self):
        print("{}开始训练{}".format("*"*15,"*"*15))
        for e in range(self.args.epoch):
            lossVal = 0
            deVal = []
            self.model.train()
            for datas, labels, imgs, dataLens in tqdm(self.trainLoader):
                datas = datas.to(self.args.device)
                labels = labels.to(self.args.device)
                imgs = imgs.to(self.args.device)
                y, hidden = self.model(datas, imgs, dataLens)
                loss = torch.sum(torch.abs(y-labels))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                lossVal += loss.item()
                for i in range(y.shape[0]):
                    for j in range(self.args.label_cnt):
                        deVal.append(abs(y[i][j].item()-labels[i][j].item()))
            deVal = np.array(deVal)
            avgDeVal = np.mean(deVal)
            medianDeVal = np.median(deVal)
            if avgDeVal < self.trainAvgDeVal:
                self.trainAvgDeVal = avgDeVal
                self.save(1)
            if e %5 == 0 and e!=0:
                self.test(False)
            if e%31 == 0 and e!=0:
                self.test(True)
            self.scheduler.step()
            print("第{}次训练，loss={:.4f} avgDeVal={:.4f} medianDeVal={}".format(e, lossVal, avgDeVal, medianDeVal))
    
    def test(self, isDraw = True):
        self.model.eval()
        deVal = []
        truthVal = []
        predVal = []
        with torch.no_grad():
            for id, (datas, labels, imgs, dataLens) in enumerate(tqdm(self.testLoader)):
                truthVal = []
                predVal = []
                datas = datas.to(self.args.device)
                labels = labels.to(self.args.device)
                imgs = imgs.to(self.args.device)
                y, hidden = self.model(datas, imgs, dataLens)
                y = y.cpu()
                labels = labels.cpu()
                for i in range(y.shape[0]):
                    for j in range(dataLens[0].item()):
                        truthVal.append(datas[i][j][8].item())
                        predVal.append(datas[i][j][8].item())
                    for j in range(self.args.label_cnt//2):
                        deVal.append(abs(y[i][j].item()-labels[i][j].item()))
                        truthVal.append(labels[i][j].item())
                        predVal.append(y[i][j].item())
                truthVal = self.TScaler[0].inverse_transform(np.array(truthVal).reshape(-1, 1))
                predVal = self.TScaler[0].inverse_transform(np.array(predVal).reshape(-1, 1))
                if isDraw==True:
                    plt.plot(predVal, 'g', label="Predict")
                    plt.plot(truthVal, 'r', label="Truth")
                    plt.legend()
                    # plt.show()
                    plt.savefig("./png/{}_{}_{}.png".format(str(self.model), int(time.time()), id),dpi=500)
                    plt.cla()
            deVal = np.array(deVal)
            avgDeVal = np.mean(deVal)
            medianDeVal = np.median(deVal)
            if avgDeVal < self.testAvgDeVal:
                self.testAvgDeVal = avgDeVal
                self.save(0)
            print("测试集 avgDeVal={:.4f} medianDeVal={}".format(avgDeVal, medianDeVal))
        
    def save(self, id):
        if id == 0:
            torch.save(self.model.state_dict(),"./pt/TModel_{}_{}_test_{:.3f}.pt".format(int(time.time()),str(self.model),self.testAvgDeVal))
        else:
            torch.save(self.model.state_dict(),"./pt/TModel_{}_{}_train_{:.3f}.pt".format(int(time.time()),str(self.model),self.trainAvgDeVal))
        