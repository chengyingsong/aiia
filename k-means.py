"""
k-mean算法的主要程序，实际上作为聚类函数是不需要标签数据集的，但是处理数据的函数是统一写的，所以还是load一下。
k-mean算法有几个步骤：
1.随机生成k个聚类中心，鉴于数据集，这里k = 3。
2.根据距离聚类，一般使用欧氏距离。【曼哈顿距离，余弦距离】
3.对每个类重置聚类中心为其从属点的平均值，重新进行聚类
4. 重复3，直到聚类中心不再变化
使用多次随机初始化中心的方法避免陷入局部最优
"""
import iris_data
import numpy as np
import random
import operator

def L2(point1,point2):
    """
    计算L2距离，也就是欧氏距离
    :param point1: 点1
    :param point2: 点2
    :return: 返回距离值
    """
    L = np.sqrt(np.sum((point1 - point2)* (point1 - point2)))
    return L

def loss(ClusterCentre,C,k):
    """
    计算损失函数
    :param ClusterCentre: 聚类中心
    :param C: 蔟集
    :return: 损失函数值
    """
    Loss = 0
    for i in range(k):
        loss = 0
        for point in  C[i]:
            loss += L2(point, ClusterCentre[i])
        Loss += loss / len(C[i])
    return Loss

def main():
    SIZE = 150  # 数据集大小
    MAX_STEP = 10000  # 最大迭代次数，这是一个阈值
    RANDOM_STEP = 100
    # 载入数据，聚类算法，没有验证集之分
    X, Y = iris_data.load_data()
    # print(X,Y)
    r = 0
    Min = 10000
    while r< RANDOM_STEP:
        #随机生成3个聚类中心
        r += 1
        k = 0
        Index =[]  #记录随机聚类中心的序列号
        ClusterCentre = []  #聚类中心集合
        while k < 3:
            index = random.randint(0,SIZE-1)
            if index not in Index:
                Index.append(random.randint(0,SIZE))
                ClusterCentre.append(X[index])
                k += 1

        #print(ClusterCentre)
        #开始聚类过程
        step = 0
        C = [[],[],[]]  #记录蔟的点集
        Index = [[],[],[]]  #记录蔟中点的序列号
        while step<MAX_STEP:
            step += 1
            for i in range(k):  #首先创建三个空蔟集合
                C[i] = []
                Index[i] = []
            for i in range(SIZE):  #对每个样本计算其和三个聚类中心的欧氏距离，并将其归入最近的蔟
                min = 10000
                for j in range(k):
                    distance = L2(X[i], ClusterCentre[j])
                    if distance < min:
                        min = distance
                        max_index = j
                C[max_index].append(X[i])
                Index[max_index].append(i)

            #计算新的聚类中心
            NewClusterCentre =[]
            bool = 1  #状态量，判断聚类是否收敛
            for i in range(k):
                if len(C[i]):
                    C[i] = np.array(C[i])
                    NewClusterCentre.append(np.mean(C[i],axis=0))
                else:   #如果某个聚类中心没有点跟随，就抛弃这个聚类中心，随机生成一个
                    NewClusterCentre.append(X[random.randint(0, SIZE-1)])
                if (NewClusterCentre[i]!=ClusterCentre[i]).any():
                   bool = 0
            if bool:
                #print("!")
                break
            else:
                ClusterCentre  = NewClusterCentre

        Loss = loss(NewClusterCentre, C,k)
        if Loss < Min:
            Min = Loss
            FinalC = C
            FinalIndex = Index

    for i in range(k):
        print("第%d个类别:%d"%(i+1,len(FinalIndex[i])))
        #for index in FinalIndex[i]:
         #   print(Y[index])


if __name__ == "__main__":
    main()