"""
进行数据处理，归一化，并且使用k-折交叉验证集,
"""
import random
import numpy as np

def load_data(n = 1, bool =0 ):
    """
    加载数据
    :param n: 把数据分为k折，n表示第n折用作验证集
           bool: 是一个布尔量，为0表示不返回测试集，只返回训练集和验证集，为1表示只返回测试集
    :return:
    返回类型统一都是list
         test_set_x: 测试集数据
         test_set_y: 测试集标签
         train_set_x:  训练集数据
         train_set_y:  训练集标签
         varificate_set_x:  验证集数据
         varificate_set_y:   验证集标签

    """
    #打开文件
    k = 10

    with open("pima-indians-diabetes.txt","r") as f:
       Data = f.readlines()

    Len = len(Data)
    #打乱数据集，随机分布，得到一个随机序列号index
    index = [i for i in range(Len)]  #Dataset中最后一行是一行空行，所以生成序列时不引用这一行
    random.shuffle(index)


    #分隔数据，并数字化
    X = []
    Y = []
    for i in range(Len):
        data = Data[index[i]].split(",")
        data = list(map( float, data))
        X.append(data[:8])
        Y.append(data[8])

    X = np.array(X)
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0],1)
    #print(X.shape,Y.shape)
    #归一化
    X = X - np.mean(X,axis =1).reshape((768,1))
    X = np.true_divide(X,(np.max(X,axis =1)-np.min(X,axis =1)).reshape((768,1)))


    #划分测试集，测试集占20%
    index = int(Len / 5)
    test_set_x = X[: index]
    test_set_y = Y[: index]

    #划分训练集和验证集
    X = X[index :]
    Y = Y[index :]
    Len1 = len(X)
    index1 = int(Len1 / k * (n - 1))
    index2 = int(Len1 / k * n )
    #print(index1,index2)
    varificate_set_x = X[index1:index2]
    varificate_set_y = Y[index1:index2]
    train_set_x = np.concatenate((X[:index1],X[index2:]),axis = 0)
    train_set_y = np.concatenate((Y[:index1],Y[index2:]),axis = 0)
    #print(varificate_set_y.shape)
    if bool:
        return test_set_x,test_set_y
    else:
        return train_set_x,train_set_y,varificate_set_x,varificate_set_y

