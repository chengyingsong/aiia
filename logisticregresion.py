"""
  逻辑回归程序，使用pima_indian数据集。
  可以把逻辑回归看成只有一个隐藏层，一个隐藏单元，激活函数是sigmond的浅层神经网络
  程序实现步骤：
  1. 载入数据，并随机初始化参数
  2. 前向传播，对数据进行线性运算再输入激活函数sigmond中计算输出
  3. 反向传播，计算代价函数值和梯度值
  4. 进行梯度下降
  5. 进行k-折交叉验证，返回k次学习的验证结果以调整学习率等超参数
  加入正则化损失
  使用Adam优化算法
"""
import pima_data
import numpy as np

def initialize(dim):
    """
    初始化参数
    :return: W,b
    """
    w = np.zeros((dim,1))
    b = 0

    assert(w.shape==(dim,1))
    return w,b

def sigmond(w,b, x):
    """
    计算sigmond函数
    :param w: 参数矩阵w
    :param b: 偏置单元
    :param x: 样本矩阵
    :return:  sigmond函数值
    """
    #print(x.shape,w.shape)
    y = np.dot(x, w)+ b
    #print(y.shape)
    return (1 /(1 + np.exp(- y)))

def back(x, y, y_, m,Lambda,w):
    """
    计算梯度
    :param x: 样本集
    :param y: 正确标签
    :param y_: 预测值
    :param m:  样本数
    :return:  dw,db，梯度值
    """
    #print(y.shape,x.shape)
    dw = (1 / m) * np.dot(x.T,(y_ - y)) + (Lambda / m)* w
    db = (1 / m) * np.sum(y_ - y)

    #assert(dw.shape == w.shape)
    return dw, db
    
def loss(y, y_, m, Lambda, w):
    """
    计算损失函数值
    :param y:  正确标签
    :param y_: 预测值
    :param m:   样本数
    :return: 损失函数值
    """
    try:
        cost = ((- 1 / m) * np.sum(y * np.log(y_) + (1 - y) * (np.log(1 - y_))))
        cost += (Lambda/(2*m))*np.sum(w * w)
        cost = np.squeeze(cost)
    except:RunitimeWarning
    return cost



def main():
    """
    主函数体
    :return:
    """
    #首先载入测试集数据
    test_set_x, test_set_y = pima_data.load_data(bool=1)
    LEARNING_RATE = 0.1 # 设置学习率
    k = 10                #10折验证集
    Loss = 0              #初始化总损失
    Accuracy = 0          #初始化正确率
    Train_step = 50000    #训练轮数
    dim = 8               #属性个数
    Lambda = 12     #正则化系数
    W,B = initialize(dim)
    Min = 100
    Vdw = 0
    Vdb = 0
    Sdw = 0
    Sdb = 0
    beta1 = 0.9         #滑动平均的衰减率
    beta2 = 0.999       #RMSprop参数
    for i in range(k):
        #加载训练集和交叉验证集
        train_set_x , train_set_y, varificate_set_x,varificate_set_y = pima_data.load_data(n = k)
        #初始化参数
        m = train_set_x.shape[0]
        w, b = initialize(dim)
        for j in range(Train_step):
            #前向传播，计算输出值
            train_set_y_ = sigmond(w,b,train_set_x)
            #反向传播,计算梯度值
            dw,db = back(train_set_x,train_set_y,train_set_y_,m, Lambda,w)
            #梯度下降
            Vdw = beta1 * Vdw + (1 - beta1) * dw
            Vdb = beta1 * Vdb + (1 - beta1) * db
            Sdw = beta2 * Sdw + (1 - beta2) * dw * dw
            Sdb = beta2 * Sdb + (1 - beta2) * db * db
            bias_correction1 = 1 / (1 - pow(beta1, j+1))
            bias_correction2 = 1 / (1 - pow(beta2, j+1))
            w = w - LEARNING_RATE * (Vdw * bias_correction1) / (pow(Sdw * bias_correction2 , 0.5) + pow(10, -8))
            b = b - LEARNING_RATE * (Vdb * bias_correction1) / (pow(Sdb * bias_correction2 , 0.5) + pow(10, -8))


        #在验证集上验证
        varificate_set_y_ = sigmond(w, b,varificate_set_x)
        #计算损失
        cost = loss(train_set_y, train_set_y_, m, Lambda,w)    #计算该模型的损失，10折中的模型
        Loss += loss(varificate_set_y,varificate_set_y_, m, Lambda,w)  #累加验证集损失，以供选择超参数
        #得出预测值
        varificate_set_y_ = varificate_set_y_.round()
        #计算正确率,0-1Loss
        accuracy = 100 - np.mean(np.abs(varificate_set_y_ - varificate_set_y)) * 100
        Accuracy += accuracy
        if accuracy<Min:
            W = w
            B = b
            Min = accuracy
        print("第%d折交叉验证集准确性："%(i+1) , format(100 - np.mean(np.abs(varificate_set_y_ - varificate_set_y)) * 100), "%")

    Loss = Loss / k
    Accuracy = Accuracy / k
    print("最终模型损失为:%f" % Loss)
    print("最终模型验证集正确率为：%f" % Accuracy)
    test_set_y_ = sigmond(W, B, test_set_x)
    test_set_y_ = test_set_y_.round()
    print("测试集准确性：", format(100 - np.mean(np.abs(test_set_y_ - test_set_y)) * 100), "%")


if __name__ == "__main__":
    main()