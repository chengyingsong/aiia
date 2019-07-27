'''
逻辑回归练习
1. 载入数据
2. 数据处理，图集降维，特征规范化
3. 随机初始化参数
4. 计算代价函数【正向传播】，并计算梯度【反向传播】，
5. 迭代更新参数 【梯度下降】
6. 模型评估
激活函数：sigmoid函数
算法的评估：留一法，k折法，自助法
梯度下降： 小批次梯度下降
使用正则化方法
采用训练集，交叉验证集，测试集的划分数据方式
'''
import numpy as np
import matplotlib.pyplot as plt
import h5py   #是与H5文件中存储的数据集进行交互的常用软件包
from lr_utils import load_dataset
import random
#加载数据集和测试集
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes  = load_dataset()


m_train  = train_set_y.shape[1]  #训练集的规模，图片数
m_test  = test_set_y.shape[1]    #测试集的规模
num_px = train_set_x_orig.shape[1]   #图片的规模，大小均为64*64
"""
#现在看一看我们加载的东西的具体情况
print ("训练集的数量: m_train = " + str(m_train))
print ("测试集的数量 : m_test = " + str(m_test))
print ("每张图片的宽/高 : num_px = " + str(num_px))
print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print ("训练集_标签的维数 : " + str(train_set_y.shape))
print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
print ("测试集_标签的维数: " + str(test_set_y.shape))
"""
#要把numpy数组(a,b,c,d)规模转化为（b*c*d,a),即使用reshape的-1参数，也就是从维数和其他参数中推出剩余参数
train_set_x_flatten  = train_set_x_orig.reshape((train_set_x_orig.shape[0],-1)).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#print('降维之后的测试集：'+str(test_set_x_flatten.shape))


#标准化数据集，特征规范化归一化处理,因为图像数据都是0到255的数据，所以除以255让数据在0到1之内
train_set_x = train_set_x_flatten[:,0:150] / 255
verify_set_x = train_set_x_flatten[:,150:209]/255
test_set_x = test_set_x_flatten / 255

verify_set_y  = train_set_y[:,150:209]
train_set_y  = train_set_y[:,0:150]
#计算激活函数 ，sigmond函数
def sigmoid(z):
    '''参数：z-任何大小的标量或者numpy数组。实际上z= W.T*x+b
    返回值：s = sigmoid(z)'''
    s = 1 / ( 1+ np.exp(-z))
    return s

'''
#测试激活函数
a = np.array([[0,1,9.2,-9.2,100]])
print('sigmoid(a):'+str(sigmoid(a)))
print('log(z):'+str(np.log(sigmoid(a))))
exit()
'''
#初始化参数
def initialize_with_zeros(dim):
    '''此函数为w创建一个维度为（dim,1）的随机向量，并把b初始化为1
    参数： dim  -  我们想要的w矢量的大小（或者这种情况下的参数数量）
    返回：  w-维度为（dim，1）的随机向量，b-初始化的标量，对应于偏差'''
    w = np.random.rand(dim, 1)*0.01+0.0001
    b = 0
    # 使用断言来确保我要的数据是正确的
    assert (w.shape == (dim, 1))  # w的维度是(dim,1)
    assert (isinstance(b, float) or isinstance(b, int))  # b的类型是float或者是int


    return (w,b)


#前向传播和后向传播，计算代价函数和梯度
def propagate(w,b,X,Y,Lambda):
    '''实现前向和后向传播的成本函数及其梯度
    参数：
    w  - 权重，大小不等的数组（num_px * num_px * 3 ,1）
    b  - 偏差，一个标量
    X  - 矩阵类型为（num_px*num_px*3,训练数量）
    Y  - 真正的标签矢量，即监督的正确答案
    Lambda - 正则化系数，是一个超参数
    返回：cost: 逻辑回归的负对数似然成本
          dw :  相对于w的损失梯度
          db：  相对于b的损失梯度'''
    #使用小批次梯度下降,批次为32
    m  = 32
    random_index = random.randint(0,X.shape[1]-m)
    x =X[:,random_index:random_index+m]
    y =Y[:,random_index:random_index+m]
    #正向传播
    A = sigmoid(np.dot(w.T,x)+b) #计算激活函数值，即y
    cost  = (- 1 / m) * np.sum(y * np.log( A )+ ( 1 - y ) *(np.log( 1 - A ))) #计算代价函数值
    #添加正则化项
    cost += (Lambda/(2*m))*np.sum(w * w)
    #反向传播
    #print(x.shape,y.shape,w.shape)
    dw = (1 / m) * np.dot(x,( A - y ).T) + (Lambda / m)* w
    db = (1 / m) * np.sum( A - y)

    # 使用断言确保我的数据是正确的
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost) #降维，变成数
    assert (cost.shape == ())

    #创建字典，保存梯度
    grads  = {'dw':dw,'db':db}

    return (grads ,cost)
'''
#测试传播过程
print('===========test  propagate ===============')
#初始化参数
w,b,X,Y = np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
#2个样本，2个特征值
grads ,cost  =  propagate(w,b,X,Y)
print('dw='+str(grads['dw']))
print('db='+str(grads['db']))
print('cost='+str(cost))
'''

#更新参数，梯度下降
def optimize(w,b,X,Y,Lambda, num_iterations, learning_rate, print_cost =False):
    '''
    此函数通过运行梯度下降算法来优化w和b
    参数：
      w  -权重，和样本特征值规模相当
      b  -偏差，一个标量
      X  - 维度为（num_px*num_px*3,训练数据的数量）的数组
      Y  - 真正的’标签’矢量
      Lambda - 正则化系数
      num_iterations  -优化循环的迭代次数
      learning_rate    -梯度下降更新的学习率
      print_cost   -每一千步打印一次损失值，绘图

    返回：
      params   - 包含权重w和偏差b的字典
      grads    - 包含权重w和偏差下降梯度的字典
      costs    - 成本-优化期计算的所有成本列表，用于绘制学习曲线
    '''
    costs = []
    for i in range(num_iterations):
        grads,cost =propagate(w,b,X,Y,Lambda)

        dw = grads['dw']
        db = grads['db']
        #更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        #记录成本
        if i%1000 == 0:
            costs.append(cost)
        #打印成本数据
        if (print_cost) and (i%1000 == 0) :
            print('迭代的次数：%i,误差值：%f'%(i,cost))
        #用字典记录参数和梯度
        params  ={'w':w,'b':b}
        grads   = {'dw':dw,'db':db}

    return (params,grads,costs)
'''
#测试optimize
print("====================测试optimize====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
params , grads , costs = optimize(w , b , X , Y , num_iterations=100 , learning_rate = 0.009 , print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
'''

#输出预测值
def predict(w,b,X):
    '''
    使用学习逻辑回归参数预测 标签
    :param w:  权重
    :param b:  偏差
    :param X:   训练集数据
    :return:   Y_prediction 包含X中所有图片的所有预测
    '''
    m  = X.shape[1]  #图片的数量
    Y_prediction  = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    #记预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T ,X)+b)
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    assert(Y_prediction.shape == (1,m))

    return Y_prediction
'''
#测试predict
print("====================测试predict====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print("predictions = " + str(predict(w, b, X)))'''

def model(X_train,Y_train,X_test,Y_test,X_verify,Y_verify, learning_rate,Lambda,num_iterations=2000,print_cost =False):
    '''
    通过调用之间实现的函数来构建逻辑回归模型
    :param X_train: 训练集
    :param Y_train: 训练标签集
    :param X_test: 测试集
    :param Y_test:  测试标签集
    :param num_iterations: 迭代次数，超参数
    :param learning_rate:   学习率，超参数
    :param print_cost:   设置为true以每100次迭代打印成本
    :return: d--包含有关模型信息的字典
    '''
    #初始化参数
    w ,b =  initialize_with_zeros(X_train.shape[0])
    #通过学习得到参数，梯度和损失
    parameters , grads , costs = optimize(w,b,X_train,Y_train,Lambda,
                                          num_iterations,learning_rate,print_cost)

    #从字典参数中检索参数w和b
    w,b = parameters['w'],parameters['b']

    #预测测试集/训练集的例子
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_verify = predict(w,b,X_verify)
    Y_prediction_train = predict(w, b, X_train)
    # 打印训练后的准确性
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("交叉验证集准确性：", format(100 - np.mean(np.abs(Y_prediction_verify - Y_verify)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")
    correct = 1 - np.mean(np.abs(Y_prediction_verify - Y_verify))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
         "correct":correct}
    return d
print("====================测试model====================")

Lambda =12
d = model(train_set_x, train_set_y, test_set_x, test_set_y,verify_set_x,verify_set_y,learning_rate=0.007,
                   Lambda=0.5, num_iterations = 20000,  print_cost = True)

#绘制图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per thousands)')
plt.title("Learning rate =" + str(d["learning_rate"])+"\nlambda ="+str(Lambda))
plt.show()
