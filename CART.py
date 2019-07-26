"""
实现CART决策树算法，使用基尼指数来选择划分属性，并进行后剪枝
步骤：
  1. load数据集
  2. 递归创建树，递归边界为样本集D中样本全部属于同一类别，属性集A为空或D中样本在A在取值相同
     还有就是当前节点所含样本集为空。
  3. 后剪枝，是要验证集精度为指标
  4. 模型评估和验证
使用类来实现树结构
"""

import pima_data

class Tree(object):
    def __init__(self,smaller_branch= None,larger_branch=None,cloumn_index =None,
           value = None, result = None, summary = None, data = []):
        self.smaller_branch = smaller_branch     #比离散二分边界小的分支子树
        self.larger_branch = larger_branch      #比离散二分边界大的分支子树
        self.cloumn_index = cloumn_index        #该节点的划分属性编号
        self.value = value                       #该节点划分属性的离散划分边界
        self.result = result                    #该节点划分类
        self.summary = summary                    #该节点的Gini值和划分类
        self.data = data                      #该节点的数据集


def spilt_data(D,value,cloumns):
    """
    把样本集划分为两个子集
    :param D: 样本集
    :param value:  离散属性划分的侯选划分点
    :param cloumns:  使用的属性编号
    :return: listsmaller,listlarger两个子集
    """
    listsmaller = []
    listlarger = []
    for data in D:
        if data[cloumns] > value:
            listlarger.append(data)
        else:
            listsmaller.append(data)

    return listsmaller,listlarger


def find_class(data_set):
    """
    计算该数据集中不同类别的个数，由于二分类，使用了一个列表表示，第一列表示类别为0，第二列表示类别为1
    :param data_set: 数据集
    :return: result
    """
    results = [0,0]  #初始化
    for data in data_set:
        if data[-1] == 0:
            results[0] += 1
        else:
            results[1] += 1

    if(results[0] > results[1]):
        result = 0
    else:
        result = 1
    return result,results


def Gini(data_set):
    """
    计算一个集合的Gini属性值,也就是在数据集中随机抽取两个样本，其类别标记不一致的概率
    :param data_set:
    :return:Gini值
    """
    length = float(len(data_set))
    #if length==0:

    _ ,results = find_class(data_set)
    Gini = 1 - (results[0] /length )* (results[0] /length ) - (results[1] /length )*(results[1] /length )
    return Gini


#递归创建树
def TreeGenerate(D):
    """
    使用递归构造一棵决策树，递归边界是Gini值为0
    :param D: 数据集
    :return: 树的根节点
    """

    #Total_Gini = Gini(D)  #当前数据集的总Gini值
    lengthD = float(len(D))   #当前数据集的样本数
    #print(lengthD)
    cloumn_length = 8   #当前数据集的属性数

    Best_value = 0      #最佳划分离散值
    Best_index = 0      #最佳划分属性
    Best_Gini = 10000   #最佳划分属性的Gini值
    Best_set_larger = []
    Best_set_smaller = []
    #选择划分属性
    for index in range( cloumn_length  ):
        #对每一个属性，把每个候选点作为离散划分，求出Gini值，选出Gini值最小的
        value_set = sorted(list(set(data[index] for data in D )))  #使用集合数据类型，避免重复候选点
        for i in range(len(value_set) - 2):
            value_set[i] = (value_set[i] + value_set[i+1]) / 2.0
        value_set = value_set[:len(value_set)-1]
        #候选点为该属性取值之间的中心点
        best_Gini = 10000
        best_value = 0
        for value in value_set:
            #对属性index的value候选点计算Gini值
            listsmaller,listlarger = spilt_data(D, value, index)  #做划分
            #计算Gini值
            current_Gini = len(listsmaller)/lengthD * Gini(listsmaller) + len(listlarger)/lengthD * Gini(listlarger)
            #更新最小Gini，并记录节点幸喜
            if current_Gini < best_Gini:
                best_Gini = current_Gini
                best_value = value
                best_set_smaller = listsmaller
                best_set_larger = listlarger

        if best_Gini < Best_Gini:
            Best_Gini = best_Gini
            Best_index = index
            Best_set_smaller= best_set_smaller
            Best_set_larger = best_set_larger
            Best_value = best_value
        summary = {"Gini":Best_Gini,"sample":lengthD}

    result,_= find_class(D)
    if Best_Gini > 0:
        #递归创建左子树和右子树
        if len(Best_set_larger)==0:
            return Tree(result = result)
        else:
            larger_branch = TreeGenerate(Best_set_larger)
        if len(Best_set_smaller)==0:
            return Tree(result = result)
        else:
            smaller_branch = TreeGenerate(Best_set_smaller)
        return Tree(larger_branch = larger_branch,smaller_branch = smaller_branch,cloumn_index= Best_index,
                    value = Best_value, summary = summary)
    else:
        return Tree(result = result, summary= summary,data = D)


def classify(data , tree):
    """
    用已经完成的决策树做分类工作
    :param data: 数据
    :param tree:  决策树的根节点
    :return: 分类结果
    """
    if tree.result != None:  #到达叶结点
        return tree.result
    else:
        branch = None
        v = data[tree.cloumn_index]
        if v >= tree.value:
            branch = tree.larger_branch
        else:
            branch = tree.smaller_branch

        return classify(data, branch)


def prune(tree):
    """
    剪枝
    :param tree:
    :param mini_gain:
    :return:
    """
    if tree.larger_branch.result == None:  #不是叶结点
        prune(tree.larger_branch)
    if tree.smaller_branch.result == None:
        prune(tree.smaller_branch)
    if tree.larger_branch.result != None and tree.smaller_branch.result != None:  #左孩子右孩子都是叶节点，是倒数第二层
        len1 = len(tree.larger_branch.data)
        len2 = len(tree.smaller_branch.data)
        if len1 and len2:
            #进行后剪枝，合并前后该节点数据集的验证集精度比较决定是否剪枝
            count1 = 0
            for i in range(len1):
                if(tree.larger_branch.result == tree.larger_branch.data[i][-1]):
                    count1 += 1
            for i in range(len2):
                if(tree.smaller_branch.result == tree.smaller_branch.data[i][-1]):
                    count1 += 1
            result = find_class(tree.larger_branch.data + tree.smaller_branch.data)
            count = 0
            for i in range(len1 ):
                if(result == tree.larger_branch.data[i][-1]):
                    count += 1
            for i in range(len2):
                if(result == tree.smaller_branch.data[i][-1]):
                    count += 1
            if count1 < count:
                tree.data = tree.larger_branch.data + tree.smaller_branch.data
                tree.result,_ = find_class(tree.data)
                tree.true_branch = None
                tree.false_branch = None




def test(test_set_x,test_set_y,tree):

    count = 0
    Len = len(test_set_x)

    for i in range( Len  ):
        #print(classify(test_set_x[i],tree))
        if (classify(test_set_x[i],tree)==test_set_y[i]):
            count += 1

    Accuracy = float(count) / Len * 100
    return Accuracy

def main():
    #load数据
    D,test_set_x,test_set_y = pima_data.load_data()
    #D = np.array(D)
    #属性集就是样本集的列数，第i列表示第i种属性
    tree = TreeGenerate(D)
    print("done!!!!!!!!!!!")
    #检验决策树
    train_set_x = []
    train_set_y = []
    for data in D:
        train_set_x.append(data[:8])
        train_set_y.append(data[8])
    Accuracy = test(train_set_x , train_set_y, tree)
    print("在训练集上的正确率为%f"%Accuracy,"%")

    Accuracy = test(test_set_x,test_set_y,tree)
    print("在测试集上的正确率为%f"%Accuracy,"%")
    prune(tree)
    Accuracy = test(test_set_x,test_set_y,tree)
    print("剪枝后的正确率为%f"%Accuracy,"%")




if __name__ == "__main__":
    main()

