from utils import splitedExcellToCsv, readDataGroupById, printData, interData, splitedPic
from tqdm import tqdm


# # 从Excel中切分数据
# splitedExcellToCsv("./数据采集结果_0701.xlsx")
# 按照编号分组
data = readDataGroupById()
# 数据插值
for i in tqdm(data):
    interData(data, i, "LAI")
# printData(data)