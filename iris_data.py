"""
加载iris数据，并做初步处理，包括归一化，标准化。对类别用one-hot进行映射
"""
import numpy as np
import random
def load_data():
    f = open(r"iris.txt","r")
    Dataset = f.readlines()  #加载数据集
    f.close()

    #打乱数据集，随机分布，得到一个随机序列号index
    index = [i for i in range(len(Dataset)-1)]  #Dataset中最后一行是一行空行，所以生成序列时不引用这一行
    random.shuffle(index)

    # 把字符串形式的数据集分成属性和label。
    x = []
    y = []
    for i in range(len(Dataset)-1):
        data = Dataset[index[i]]
        data = data.rstrip("\n")
        x.append(data.split(",")[0:4])
        x[i] = list(map(float,x[i]))
        #print(data)
        if "setosa" in data:
           y.append([1,0,0])
        elif "versicolor" in data:
            y.append([0,1,0])
        else:
            y.append([0,0,1])

    #化为矩阵形式，并检查维度
    x = np.array(x)
    y = np.array(y)
    #print(x.shape,y.shape)  x.shape = (150，4）,y.shape=(150, 3)


    #进行归一化

    #x = np.true_divide(x, np.sum(x, axis=1).reshape((150,1)))  #利用了numpy的广播机制，每个样本的属性除以其总和，使其无量纲
    x = x - np.mean(x,axis =1).reshape((150,1))
    x = np.true_divide(x,(np.max(x,axis =1)-np.min(x,axis =1)).reshape((150,1)))
    #print(np.sum(x,axis=1))
    return x,y


