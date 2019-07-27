"""
SVM问题使用SMO算法，使用核函数进行映射。
由数学推导，使用拉格朗日乘数法优化目标变为只有alpha的凸二次规划问题。
优化函数 max[sum(alpha) - 0.5 * sum(y[i]* y[j]*alpha[i]*alpha[j]*<x[i],x[j]>)]
使用软间隔，约束条件为  C>= alpha >= 0 , sum(alpha[i] * y[i]) = 0
SMO算法的思想是固定住其他的alpha，只剩下两个参数的凸优化二次规划问题，又由于约束，变为只有一个参数的二次规划问题。
关键在于选择两个参数。
第一个参数为外循环，通过遍历整个参数集和遍历非边界点交替进行的方式选择
第二个参数为内循环，建立一个全局的缓存保存误差值E，然后选择步长最大的
选择好两个参数之后，更新参数，计算L，H和eta，更新两个alpha和b
步骤
1. 外循环遍历第一个参数
2. 在内循环中找到第二个参数，并更新参数【在计算E和更新参数使用核函数】
预测函数sign为单元阶跃函数，即函数值大于0为1，小于0为-1
"""
import pima_data
import numpy as np

def kernerTrans( X, A, kTup):
    """
    计算核技巧矩阵
    :param X: 数据集X
    :param A:  数据集A
    :param kTup:核函数信息，元祖，第一个元素说明核函数类型，第二个是某个核函数的参数
    :return: 核函数矩阵的列
    """
    m= np.shape(X)[0]
    K = np.zeros((m,1))
    #print(np.shape(X))
    if kTup[0]=="rbf":  #径向基核函数
        deltaRow = X - A
        K = np.dot(deltaRow,deltaRow.T)
        K = np.exp((-K) /(0.5 * kTup[1]**2))
        #print(np.shape(K))
    elif kTup[0] == "lin":   #线性核函数
        K = X * A.T
    else:
        raise NameError("没有写入该核函数类型")
    return K



class optStruct:
    def __init__(self,X,Y,C,tolerance,kTup):
        """
        直接用一个类来传输信息
        """
        self.X = X     #属性值
        self.Y = Y     #标签集
        self.C = C     #超参数C
        self.tolerance = tolerance  #KKT条件精度
        self.m = len(X)         #数据集规模
        self.alphas = np.zeros((self.m,1))   #学习参数alpha
        self.b = 0       #偏置值
        self.E = np.zeros((self.m,1))   #E值
        self.K = np.zeros((self.m,self.m))  #核函数矩阵，K_ij = K(X[i], X[j])
        self.kTup = kTup


def predict(os, x):
    """
    对样本进行预测
    :param os:模型对象
    :param x: 数据样本
    :return:  预测值
    """
    f = 0
    #print(np.shape(x))
    for j in range(os.m):
        f += os.Y[j]*os.alphas[j]* kernerTrans(os.X[j], x, os.kTup)
        #print(kernerTrans(os.X[j],x,os.kTup))
    f += os.b
    #print(f)
    return f

def findsecond(os,i):
    """
    在确定了第一个参数之后，通过启发式搜索确定第二个参数
    :param os: 模型对象
    :param i:  第一个参数的标号
    :return:   选择的参数
    """
    #第二个参数选择迭代步长最大的
    maxj = -1
    maxDeltaE = 0
    for j in range(os.m):
        if j==i:
            continue
        deltaE = abs(os.E[i] - os.E[j])
        if deltaE>maxDeltaE:
            maxDeltaE = deltaE
            maxj = j
    return maxj

def update(os,i):
    """
    进行参数的更新
    :param os: 模型对象
    :param i:  第一个参数下标
    :return:   有无成功更新参数
    """
    #选择第二个参数
    j = findsecond(os,i)
    #记录旧值
    oldalphai = os.alphas[i]
    oldalphaj = os.alphas[j]
    #计算L，H
    if(os.Y[j] != os.Y[i]):
        L = max(0, os.alphas[j] - os.alphas[i])
        H = min(os.C, os.C + os.alphas[j] - os.alphas[i] )
    else:
        L = max(0,os.C + os.alphas[j] - os.alphas[i] )
        H = min(os.C, os.alphas[j] - os.alphas[i])
    #更新第二个参数
    Lambda = os.K[i][i] + os.K[j][j] - 2 * os.K[i][j]
    os.alphas[j] += os.Y[j] * (os.E[i] - os.E[j])/ Lambda
    os.alphas[j] = min(os.alphas[j], H)  #if os.alphas[j]>=H: os.alphas[j]=H
    os.alphas[j] = max(os.alphas[j], L)

    os.E[j] = predict(os,os.X[j]) - os.Y[j]

    #os.E[j] = predict(os,os.X[j]) - os.Y[j]
    #if (abs(os.alphas[j] - oldalphaj)<0.0000001):
        #print("改变量太小了")
        #return 0
    #更新第一个参数
    os.alphas[i] += os.Y[i]*os.Y[j]*(oldalphaj - os.alphas[j])

    #更新b
    b1 = - os.E[i] - os.Y[i]*(os.alphas[i] - oldalphai)* os.K[i][i] -\
         os.Y[j]*(os.alphas[j]-oldalphaj)* os.K[i][j] + os.b
    b2 = - os.E[j] - os.Y[i]*(os.alphas[i] - oldalphai)* os.K[i][j] -\
         os.Y[j]*(os.alphas[j]-oldalphaj)* os.K[j][j] + os.b
    if (os.alphas[i]< os.C and os.alphas[i]>0):
        os.b = b1
    elif (os.alphas[j]< os.C and os.alphas[j]>0):
        os.b = b2
    else:
        os.b = (b1 + b2)/2
    os.E[i] = predict(os,os.X[i]) - os.Y[i]
    return 1

def sign(y):
    if y>0:
        return 1
    else:
        return -1



def main():
    #引入数据
    train_set_x,train_set_y,test_set_x,test_set_y = pima_data.load_data()
    #设定超参数,并初始化对象
    C = 0.4
    tolerance = 0.001
    kTup = ["rbf" , 7]
    maxiter = 100
    #print(type(kTup))
    os = optStruct(train_set_x, train_set_y, C, tolerance, kTup)
    #print(os.X[1])
    #初始化误差项
    for j in range(os.m):
        os.E[j] = predict(os,os.X[j,:]) - os.Y[j,:]
        #print(os.E[j])
        for i in range(os.m):
            os.K[i,j] = kernerTrans(os.X[i,:], os.X[j, :], kTup)

    #寻找参数对，并更新参数
    #寻找第一个乘子使用启发式搜索，首先更新所有违背KKT条件的，第二轮更新非边界点
    iter = 0
    alphachanged = 0
    entireSet = True
    while (iter < maxiter) and ((alphachanged > 0) or (entireSet)):
        alphachanged = 0
        #第一遍循环对所有违反kkt的参数更新
        if entireSet:
            for i in range(os.m):
                r = os.E[i] * os.Y[i]
                if (r < -os.tolerance and os.alphas[i] <C) or (r>os.tolerance and os.alphas[i]>0):
                    alphachanged += update(os,i)
            iter += 1
            entireSet = False
        else:  #第二遍对非界点的参数更新
            for i in range(os.m):
                if (os.alphas[i]>0 and os.alphas[i]< C):
                    alphachanged += update(os, i)
            iter += 1
            if alphachanged==0:
                entireSet = True

    #测试
    count = 0
    for i in range(os.m):
        if sign(predict(os,os.X[i]))==os.Y[i]:
            count += 1
    Accuracy = float(count) / os.m*100
    print("训练集正确率为：%d"%Accuracy,"%")
    count = 0
    for i in range(len(test_set_y)):
        if sign(predict(os,test_set_x[i]))==test_set_y[i]:
            count += 1
    Accuracy = float(count) / len(test_set_y) *100
    print("测试集正确率为：%d"%Accuracy,"%")

if __name__ =="__main__":
    main()