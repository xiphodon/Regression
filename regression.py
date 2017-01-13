#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/11 18:20
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : regression.py
# @Software: PyCharm

# 线性回归

import numpy as np

def loadDataSet(fileName):
    '''
    读取数据
    :param fileName: 文件路径
    :return: 数据集，标签集
    '''
    # 获得特征数量
    numFeat = len(open(fileName).readline().split('\t')) - 1
    # 数据集
    dataMat = []
    # 标签集
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    '''
    标准回归函数
    计算训练集的回归系数（优点：简单方便，缺点：有时会计算不出来）
    :param xArr: 训练集
    :param yArr: 标签集
    :return: 可使代价函数最小的回归系数向量
    '''
    # 数据矩阵
    xMat = np.mat(xArr)
    # 转成列向量
    yMat = np.mat(yArr).T
    # 计算xTx
    xTx = xMat.T*xMat
    # 计算xTx行列式值，从而判断是否可逆
    if np.linalg.det(xTx) == 0.0:
        # xTx为奇异矩阵，不可逆
        print("This matrix is singular, cannot do inverse")
        return
    # 计算回归系数【(xTx)^(-1)*xT*y】
    ws = xTx.I * (xMat.T*yMat)
    return ws



def lwlr(testPoint,xArr,yArr,k=1.0):
    '''
    局部加权线性回归
    :param testPoint: 某条数据
    :param xArr: 数据集
    :param yArr: 标签集
    :param k: 高斯核函数参数k
    :return: 加权回归系数
    '''
    xMat = np.mat(xArr)
    # 转成列向量
    yMat = np.mat(yArr).T
    # 训练集矩阵行数（即样本数据量）
    m = np.shape(xMat)[0]
    # 创建对角矩阵
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        # 权重值大小以指数级衰减
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        # 奇异矩阵，不可逆
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    '''
    为数据集中每条数据点调用lwlr(),有助于求解k的大小
    :param testArr: 测试数据集
    :param xArr: 数据集
    :param yArr: 标签集
    :param k: 高斯核参数k
    :return: 计算值（预计值）
    '''
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat



def step01():
    xArr,yArr = loadDataSet('ex0.txt');
    # 展示前两条数据
    print(xArr[:2])
    ws = standRegres(xArr,yArr)
    print(ws)
    xMat = np.mat(xArr) # 训练集矩阵
    yMat = np.mat(yArr) # 标签集矩阵
    yHat = xMat * ws # 回归值矩阵

    import matplotlib.pyplot as plt

    # 绘制原始数据
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制散点图（X数据向量，Y数据向量）
    # 【mat.flatten(),使矩阵所有元素拼接为行向量 -> .A,行矩阵转成二维数组 -> .A[0]转成一维数组】
    # eg: [[1,2],[3,4]] -> [[1,2,3,4,]] -> [1,2,3,4]
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])

    # 绘制回归直线
    xCopy = xMat.copy()
    # 0是列内排序，1是行内排序
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

    # 计算预测值和真实值的相关系数
    yHat = xMat * ws  # 回归值矩阵
    # 确保二者都为行向量
    print(np.corrcoef(yHat.T, yMat))


def step02():
    xArr, yArr = loadDataSet('ex0.txt');
    # 第一条训练集的标签
    print(yArr[0])
    # 第一条训练集局部加权后的预测值
    print(lwlr(xArr[0],xArr,yArr,1.0))
    print(lwlr(xArr[0],xArr,yArr,0.001))

    # 得到所有点的估计值
    yHat = lwlrTest(xArr,xArr,yArr,0.01)
    xMat = np.mat(xArr)
    # 获得排序后的索引
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]

    # 绘图
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()



if __name__ == "__main__":
    # step01();
    step02();