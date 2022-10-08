##D3和echarts
import matplotlib.pyplot as plt
import random
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

def matplotlib_demo():
    #创建画布
    plt.figure(figsize=(15,8),dpi=80)
    #绘制图像
    plt.plot([1,0,9],[4,5,6])
    #保存图像
    #plt.savefig(path)
    #展示图像
    plt.show()
    return None

def zhexian_demo():
    # 1、准备数据 x y
    x = range(60)
    y = [random.uniform(15, 18) for i in x]

    # 2、创建画布
    plt.figure(figsize=(15, 8), dpi=80)

    # 3、绘制图像
    plt.plot(x, y)

    # 修改x、y刻度
    # 准备x的刻度说明
    x_label = ["11点{}分".format(i) for i in x]
    plt.xticks(x[::5], x_label[::5])
    plt.yticks(range(0, 40, 5))

    # 添加网格显示
    plt.grid(linestyle="--", alpha=0.5)

    # 添加描述信息
    plt.xlabel("时间变化")
    plt.ylabel("温度变化")
    plt.title("某城市11点到12点每分钟的温度变化状况")

    # 4、显示图
    plt.show()
    return None

    

if __name__ == "__main__":
    # 代码1：简单演示matplotlib
    #matplotlib_demo()
    # 代码2：完善直线图
    zhexian_demo()










