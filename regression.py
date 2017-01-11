#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/1/11 18:20
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : regression.py
# @Software: PyCharm

# 线性回归

from numpy import *

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
    xMat = mat(xArr)
    # 转成列向量
    yMat = mat(yArr).T
    # 计算xTx
    xTx = xMat.T*xMat
    # 计算xTx行列式值，从而判断是否可逆
    if linalg.det(xTx) == 0.0:
        # 奇异矩阵，不可逆
        print("This matrix is singular, cannot do inverse")
        return
    # 计算回归系数【(xTx)^(-1)*xT*y】
    ws = xTx.I * (xMat.T*yMat)
    return ws


def step01():
    xArr,yArr = loadDataSet('ex0.txt');
    # 展示前两条数据
    print(xArr[:2])
    ws = standRegres(xArr,yArr)
    print(ws)


if __name__ == "__main__":
    step01();