"""
进行数据处理，归一化,使用自助法划分数据集，把标签改为-1和1
"""
import random
import numpy as np

def load_data():
    """
    加载数据
    返回类型统一都是list
         test_set_x: 测试集数据
         test_set_y: 测试集标签
         train_set_x:  训练集数据
         train_set_y:  训练集标签
    """
    #打开文件

    with open("pima-indians-diabetes.txt","r") as f:
       Data = f.readlines()

    Len = len(Data)

    #分隔数据，并数字化
    X = []
    Y = []
    for i in range(Len):
        data = Data[i].split(",")
        data = list(map( float, data))
        X.append(data[:8])
        Y.append(data[8])
        if Y[i]==0:
            Y[i] = -1

    X = np.array(X)
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0],1)
    #print(X.shape,Y.shape)
    #归一化
    X = X - np.mean(X,axis =1).reshape((768,1))
    X = np.true_divide(X,(np.max(X,axis =1)-np.min(X,axis =1)).reshape((768,1)))


    index =int( Len / 7)
    #print(index)
    train_set_x = X[index:,:]
    train_set_y = Y[index:,:]
    test_set_x = X[0:index,:]
    test_set_y = Y[0:index,:]
    #print(np.shape(test_set_x))
    return train_set_x,train_set_y,test_set_x,test_set_y


