"""
进行数据处理
"""
import random

def load_data():
    """
    加载数据
    :param
    :return:
    返回类型统一都是list
         test_set_x: 测试集数据
         test_set_y: 测试集标签
         D : 样本集
    """
    #打开文件
    with open("pima-indians-diabetes.txt","r") as f:
       Data = f.readlines()

    Len = len(Data)
    #print(Len)
    #打乱数据集，随机分布，得到一个随机序列号index
    index = [i for i in range(Len)]
    random.shuffle(index)

    #print(type(Data[0]))
    #分隔数据，并数字化
    D = []
    test_set_x = []
    test_set_y = []
    Index = int(Len / 5)
    for i in range( Len ):
        data = Data[index[i]].split(",")
        data = list(map( float, data))
        if( i>= Index):
            D.append(data)
        else:
            test_set_x.append(data[:8])
            test_set_y.append(data[8])

    #划分测试集，测试集占20%

    #print(len(test_set_y))
    #划分训练集和验证集
    #print(index1,index2)
    #print(len(test_set_x))

    return D,test_set_x,test_set_y

