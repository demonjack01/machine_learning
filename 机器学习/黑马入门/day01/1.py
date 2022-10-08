
from cgi import print_exception
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import jieba
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#pd.set_option("display.max_rows", None)#显示全部行
#pd.set_option("display.max_columns", None)#显示全部列

#sklearn数据集使用
def datasets_demo():   
    iris = load_iris()  # 获取数据集
    print("数据集: \n", iris)
    print("数据集描述: \n", iris["DESCR"])
    print("查看特征值的名字: \n", iris.feature_names)

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)  # 数据集划分
    print("训练集的特征值：\n", x_train, x_train.shape)
    return None

#字典特征提取
def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{"city":"北京", "temperature": 100}, {"city": "上海", "temperature": 60}, {"city": "深圳", "temperature": 30}]
    #step1、实例化一个转换器类
    transfer = DictVectorizer(sparse=True)
    #step2、调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("特征名字：\n", transfer.get_feature_names_out())
    print("data_new:\n", data_new.toarray(), type(data_new))
    return None

# 代码3：文本特征提取：CountVecotrizer
def count_demo():
    """
    文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    # step1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["is", "too"])
    # step2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())  #toarray方法是将稀疏矩阵转为二维矩阵
    print("特征名字：\n", transfer.get_feature_names_out())
    return None

#中文文本特征抽取：CountVecotrizer
def count_Chinese_demo():
    """
    中文文本特征抽取：CountVecotrizer
    :return:
    """
    data = ["我 爱 北京 天安门", "天安门 上 太阳 升"]
    # step1、实例化一个转换器类
    transfer = CountVectorizer()

    # step2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())

    return None

#中文分词
def cut_word(text):
    """
    进行中文分词
    :param text:
    :return:
    """
    text=" ".join(jieba.lcut(text))
    return text

#中文文本特征抽取2
def count_Chinsens_demo2():
    """
    中国文本特征抽取，自助分词
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))   #append函数往列表末尾追加元素
    print(data_new)
    # step1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一种", "所以"])

    # step2、调用fit_transform
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())
    print("特征名字：\n", transfer.get_feature_names_out())
    return None

def minmax_demo():
    """
    归一化
    """
    #step1.获取数据
    data=pd.read_csv("day01\dating.txt")  #读取文本文件
    print("data:\n",data)
    data=data.iloc[:,:3]
    #step2.实例化一个转化器类
    transfer = MinMaxScaler(feature_range=(0,1))
    #stepstep3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None

def stand_demo():
    """
    标准化
    """
    #step1.获取数据
    data=pd.read_csv("day01\dating.txt")  #读取文本文件
    print("data:\n",data)
    data=data.iloc[:,:3]
    #step2.实例化一个转化器类
    transfer = StandardScaler()
    #step3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None

def variance_demo():
    """
    过滤低方差特征
    """
    #step1.获取数据
    data=pd.read_csv("day01\\factor_returns.csv")
    data=data.iloc[:,1:-2]
    print("data:\n",data)
    #step2.实例化一个转化器
    transfer=VarianceThreshold(threshold=10)
    #step3.调用fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new.shape)
    #step4.计算某两个量之间的相关系数
    r=pearsonr(data["pe_ratio"],data["pb_ratio"])
    print("相关系数:\n",r)
    return None

def pca_demo():
    """
    PCA降维
    """
    data=[[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    transfer=PCA(n_components=0.9)                 #!!整数表示剩余特征数，小数表示保留信息率
    data_new=transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return None


if __name__ == "__main__":
    # 代码1：sklearn数据集使用
    # datasets_demo()
    # 代码2：字典特征提取
    dict_demo()
    # 代码3：文本特征提取：CountVecotrizer
    #count_demo()
    # 代码4：中文文本特征抽取：CountVecotrizer
    #count_Chinese_demo()
    #代码5：中文文本特征抽取2
    #count_Chinsens_demo2()
    #代码6：中文分词
    #print(cut_word("我爱龚雅南"))
    #代码8：归一化
    #minmax_demo()
    #代码9：标准化
    #stand_demo()
    #代码10：低方差特征过滤
    #variance_demo()
    #代码11：PCA降维
    #pca_demo()