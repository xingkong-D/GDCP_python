import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter,minimum_filter,maximum_filter

import GDCP.Imtool as Imtool


def Strch(x, r0, r1):
    return (x - np.min(x)) * ((r1 - r0) / (np.max(x) - np.min(x))) + r0


# 获取梯度图
def getGrad(I, win):
    r0 = 0
    r1 = 1

    I = I.astype(np.float32)
    # 转换为YUV颜色空间
    imYUV = cv2.cvtColor(I, cv2.COLOR_BGR2YCrCb)

    # 计算亮度图像的梯度
    Gmag_x = cv2.Sobel(imYUV[:, :, 0], cv2.CV_64F, 1, 0, ksize=3)
    Gmag_y = cv2.Sobel(imYUV[:, :, 0], cv2.CV_64F, 0, 1, ksize=3)
    Gmag = np.sqrt(Gmag_x**2 + Gmag_y**2)

    # 创建方形结构元素
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    # 对梯度图像进行膨胀操作
    imG = cv2.dilate(Gmag, se)
    # 对梯度图像进行填充操作并归一化
    Grad = Strch(imG, r0, r1)

    return Grad


# 获取深度图和梯度图
def GetDepth(I, win):
    height, width, _ = I.shape
    imsize = height * width
    GradMap = getGrad(I, win)  # 利用Sobel算子求得梯度图
    RoughDepthMap = 1 - GradMap  # 粗略估计的深度图Dr = 1 - Fs

    DepVec = RoughDepthMap.reshape(imsize, 1)  # 深度图转为列向量
    ImVec = I.reshape(imsize, 3)  # 原图转为一个imsize*3的矩阵

    A = np.hstack((DepVec ** 0, DepVec ** 1))  # 深度图的0次方和1次方
    w = np.zeros((2, 3))  # 2*3的0矩阵

    tc = ['b', 'g', 'r']
    plt.figure()
    for ind in range(3):
        w[:, ind] = np.linalg.lstsq(A, ImVec[:, ind], rcond=None)[0]  # A逆*原图像=w

        c = np.zeros(3)
        c[2 - ind] = 1
        plt.subplot(1, 3, ind + 1)
        plt.scatter(DepVec, ImVec[:, ind], s=1, color=c, alpha=0.7)
        plt.plot(np.unique(DepVec), w[0, ind] + w[1, ind] * np.unique(DepVec), 'k-', linewidth=2)
        plt.axis([0, 1, 0, 1])
        plt.title(tc[ind])
        if ind == 0:
            plt.ylabel('Intensity')
        elif ind == 1:
            plt.xlabel('Depth')
    plt.show(block=False)
    ws = np.tanh(4 * np.abs(w[1, :]))
    s = np.double(w[1, :] <= 0)

    min_r_im = ws[0] * minimum_filter(np.abs(s[0] - I[:, :, 0]), size=win) + (1 - ws[0])
    min_g_im = ws[1] * minimum_filter(np.abs(s[1] - I[:, :, 1]), size=win) + (1 - ws[1])
    min_b_im = ws[2] * minimum_filter(np.abs(s[2] - I[:, :, 2]), size=win) + (1 - ws[2])

    rgb_im = np.stack((min_r_im, min_g_im, min_b_im), axis=2)
    DepthMap = np.min(rgb_im, axis=2)

    return DepthMap, GradMap


# 获得三通道的环境光值
def atmLight(I, DepthMap):
    height, width, _ = I.shape
    imsize = width * height
    JDark = DepthMap
    numpx = int(imsize / 1000)     # 求取原图像的千分之一是多少个像素（即PD0.1%）
    JDarkVec = np.reshape(JDark, (imsize, 1))    # 将深度图转化的列向量JDarkVec
    ImVec = np.reshape(I, (imsize, 3))    # 将原图像转化的列向量ImVec
    indices = np.argsort(JDarkVec, axis=0)    # 返回JDarkVec中不同元素出现时的索引,从小到大排序
    indices = indices[imsize - numpx:]    # 选择元素索引的后面部分

    # 获取各通道的全局环境光
    Id = np.zeros_like(I)
    for i in range(3):
        Id[:, :, i] = DepthMap
    A = np.mean(ImVec[indices, :], axis=0)
    A = A[0]

    # 查看环境光的选取位置
    h_index = np.floor(indices / width).astype(int)        # 求行数，floor() 函数用于向下取整。
    w_index = np.mod(indices, width).astype(int)       # 求列数，mod() 函数返回两个数整除后的余数。
    I2 = I.copy()
    I2[h_index, w_index, :] = 1
    Imtool.imshow('I2', I2)

    return A


# 获取透射图
def calcTrans(I, BL, win):
    h, w, _ = I.shape
    BLmap = np.zeros_like(I)
    for ind in range(3):
        BLmap[:, :, ind] = BL[ind] * np.ones((h, w))
    DLmap = np.abs(BLmap - I)
    maxDL = np.zeros_like(I)
    DLmapNor = np.zeros_like(I)
    for ind in range(3):
        Bm = max(BL[ind], 1 - BL[ind])
        DLmapNor[:, :, ind] = DLmap[:, :, ind] / Bm
        maxDL[:, :, ind] = maximum_filter(DLmap[:, :, ind] / Bm, size=(win, win), mode='mirror')
    maxDL1 = np.amax(DLmapNor, axis=2)
    T = median_filter(maxDL1, size=(win, win), mode='mirror')
    T = np.clip(T, 0, 1)

    return T
