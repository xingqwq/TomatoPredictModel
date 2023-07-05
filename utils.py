import scipy
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from torchvision import transforms

# Utils工具包初始化
axisNameToNum = {}
numToAxisName = {}

def utilsInit():
    cnt = 0
    for i in "编号,起止时间差,横向,纵向,叶面积,叶夹角,高度,NVI,RVI,LNC,LNA,LAI,LDW".split(",")[1:]:
        axisNameToNum[i] = cnt
        numToAxisName[cnt] = i
        cnt += 1

utilsInit()

# 数据预处理 
def splitedExcellToCsv(fileName):
    """
    要求输入的Excel文件只有一个标头，同时按照时间进行分类，时间格式也不能够错误。
    """
    data = pd.ExcelFile(fileName)
    dataGroupByTime = pd.read_excel(data, "Sheet4").groupby('采集时间')
    startTime = pd.Timestamp(2023,5,1)
    for key, value in dataGroupByTime:
        with open("./dataset/csv/data_{}.csv".format(key.strftime("%Y_%m_%d")), 'w', encoding='utf-8') as file:
            file.write("编号,起止时间差,横向,纵向,叶面积,叶夹角,高度,NVI,RVI,LNC,LNA,LAI,LDW\n")
            for i in range(len(value)):
                file.write("{},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    value.iloc[i, 1],
                    (key-startTime).days*24+int((key-startTime).seconds/3600),
                    value.iloc[i, 2],
                    value.iloc[i, 3],
                    value.iloc[i, 4],
                    value.iloc[i, 5],
                    value.iloc[i, 6],
                    (0 if pd.isnull(value.iloc[i, 8]) else value.iloc[i, 8]),
                    (0 if pd.isnull(value.iloc[i, 9]) else value.iloc[i, 9]/(0 if pd.isnull(value.iloc[i, 8]) else value.iloc[i, 8])),
                    (0 if pd.isnull(value.iloc[i, 10]) else value.iloc[i, 10]/(0 if pd.isnull(value.iloc[i, 8]) else value.iloc[i, 8])),
                    (0 if pd.isnull(value.iloc[i, 11]) else value.iloc[i, 11]/(0 if pd.isnull(value.iloc[i, 8]) else value.iloc[i, 8])),
                    (0 if pd.isnull(value.iloc[i, 12]) else value.iloc[i, 12]/(0 if pd.isnull(value.iloc[i, 8]) else value.iloc[i, 8])),
                    (0 if pd.isnull(value.iloc[i, 13]) else value.iloc[i, 13]/(0 if pd.isnull(value.iloc[i, 8]) else value.iloc[i, 8])),
                ))

def readDataGroupById():
    """
    读取数据并将其按照编号进行分组，以便后续的插值操作
    """
    fileLists = os.listdir("./dataset/csv")
    dataDict = {}
    for fileName in fileLists:
        tmp = pd.read_csv("./dataset/csv/{}".format(fileName),sep=",")
        for i in range(len(tmp)):
            if pd.isnull(tmp.iloc[i,0]):
                continue
            if tmp.iloc[i,0] in dataDict:
                dataDict[str(tmp.iloc[i,0])].append(tmp.iloc[i,1:].to_list())
            else:
                dataDict[str(tmp.iloc[i,0])] = []
                dataDict[str(tmp.iloc[i,0])].append(tmp.iloc[i,1:].to_list())
    return dataDict

def printData(data):
    print("-----打印数据-----")
    for id in data:
        print("ID: {}".format(id))
        for i in data[id]:
            print("\t{}".format(i))
        
def interData(data, ID, axis, totalLen = 61*24):
    """
    使用Scipy中的插值算法对数据进行插值操作
    Args:
        data (_type_): _description_
        ID (_type_): _description_
        axis (_type_): _description_
        totalLen (int, optional): _description_. Defaults to 35.
    """
    # Axis处理
    if type(axis) == str:
        axis = axisNameToNum[axis]
    dataID = np.array(data[ID])
    x = dataID[:,0]
    y = dataID[:,axis]
    # 拉格朗日插值
    fillX = range(int(np.min(x)), int(np.max(x)), 1)
    fillFun = interpolate.interp1d(x, y)
    fillY = fillFun(fillX)
    # 结果dict
    result = {fillX[i]:fillY[i] for i in range(0, len(fillX))}
    return result
    
    # 绘制结果
    # plt.title("ID:{} Axis:{} Trutn Num:{}".format(ID, axis, totalLen))
    # plt.scatter(x, y, color='r', s=20, marker='*')
    # plt.plot(fillX, fillY, 'blue')
    # # plt.show()
    # plt.savefig("./png/{}_{}_{}.png".format(ID, numToAxisName[axis], totalLen), dpi=600)
    # plt.close()

def splitedPic(key, solve:dict, fx = 0.3, fy = 0.3, width = 500, height = 400):
    print("正在准备{}的图片数据...".format(key), flush=True)
    for i in solve:
        if not os.path.exists("./dataset/png/{}".format(i)):
            os.mkdir("./dataset/png/{}".format(i))
    for i in tqdm(os.listdir("./data/{}".format(key))):
        tmp = i.split("_")[0]
        time_str = time.strftime('%m_%d_%H', time.localtime(int(tmp)))
        img = cv.imread("./data/{}/{}".format(key,i))
        for solve_item in solve:
            x = solve[solve_item][0] / fx
            y = solve[solve_item][1] / fy
            # wide Left & Right
            wl = max(0, round(x - width/2))
            wr = min(img.shape[1], round(x + width/2))
            # height Top & Bottom
            ht = max(0, round(y - height/2))
            hb = min(img.shape[0], round(y + height/2))
            roi = img[ht:hb, wl:wr]
            cv.imwrite("./dataset/png/{}/{}.jpg".format(solve_item, time_str), roi)
            
def prepareDataset():
    # 植株中心坐标，注意缩放因子
    fx = 0.3
    fy = 0.3
    # 处理三
    solve3 = {'314':(250, 161), '315':(254, 360), '316':(232, 564), '324':(411, 64), '325':(419, 262), 
            '326':(404, 469), 'Ck13':(839, 176), 'Ck12':(832, 377), 'Ck11':(838, 571), 'Ck23':(985, 69), 
            'Ck22':(992, 270), 'Ck21':(992, 482)}

    # 处理五
    solve5 = {'Ck14':(230, 179), 'Ck15':(233, 391), 'Ck16':(238, 590), 'Ck24':(387, 90), 'Ck25':(374, 301), 
            'Ck26':(372, 488)}

    # 处理八
    solve8 = {'517':(196, 167), '518':(203, 387), '519':(182, 602), '527':(369, 70), '528':(375, 276), '529':(377, 500), 
            '614':(821, 186), '615':(815, 379), '616':(804, 588), '624':(991, 63), '625':(988, 273),'626': (985, 484)}

    # 处理二
    solve2 = {'214':(198, 59), '215':(199, 279), '216':(199, 472), '224':(387, 143), '225':(386, 353), '226':(371, 560), 
            '311':(822, 36), '312':(815, 245), '313':(818, 455) ,'321':(1031, 165), '322':(1019, 368), '323':(1013, 589)}

    # 处理七
    solve7 = {'514':(179, 210), '515':(160, 439), '516':(151, 629), '524':(366, 319), 
            '525':(335, 540), '611':(776, 242), '612':(782, 460), '621':(960, 345), '622':(958, 558)}

    # 处理一
    solve1 = {'111':(230, 57), '112':(230, 260), '113':(226, 477), '122':(420, 177), '123':(413, 366), 
            '211':(849, 201), '212':(854, 413), '213':(843, 634), '221':(1031, 48), '223':(1027, 480)}

    # 处理二
    solve6 = {'411':(189, 84), '412':(189, 282), '413':(186, 488), '421':(257, 165), '422':(346, 393), 
            '423':(376, 612), '511':(806, 232), '512':(782, 240), '513':(779, 610), '521':(974,98)}

    # solve
    solve = {
        '（处理1）':solve1,'（处理2）':solve2,'（处理3）':solve3,'（处理5）':solve5,
        '（处理6）':solve6,'（处理7）':solve7,'（处理8）':solve8,
    }
    for i in solve:
        splitedPic(i, solve[i])
    
def interImg():
    print("开始准备数据增广，08-16时间段每小时一张图片...", flush=True)
    for imgDir in tqdm(os.listdir("./dataset/png")):
        paths = os.listdir("./dataset/png/{}".format(imgDir))
        paths.sort()
        for i in range(0, len(paths)-1):
            img1_path = "./dataset/png/{}/{}".format(imgDir, paths[i])
            img2_path = "./dataset/png/{}/{}".format(imgDir, paths[i+1])
            tmp1 = paths[i].split(".")[0].split("_")
            tmp2 = paths[i+1].split(".")[0].split("_")
            if tmp2[1] != tmp1[1]:
                continue
            img1 = cv.imread(img1_path)
            img2 = cv.imread(img2_path)
            img1_beta = np.mean(img1)
            img2_beta = np.mean(img2)
            div = int(tmp2[2])-int(tmp1[2])
            for i in range(div-1):
                img3 = cv.convertScaleAbs(img1, beta=(i+1)*(img2_beta-img1_beta)/div, alpha=1)
                cv.imwrite("./dataset/png/{}/{}_{}_{:0>2}.jpg".format(imgDir, tmp1[0], tmp1[1], int(tmp1[2])+i+1), img3)
                
                
def readImg(filePath):
    data = pd.read_csv(filePath, parse_dates=['时间'], index_col='时间', 
                           date_parser=lambda x:pd.to_datetime(x,format='%Y-%m-%d %H:%M:%S'))
    imgs = {}
    trainTransform = transforms.Compose([
            transforms.Resize((120, 120)),  # 缩放
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)])
    for i in tqdm(data['img_path']):
        img = Image.open(i)
        img = trainTransform(img)
        imgs.update({i:img})
    
    return imgs