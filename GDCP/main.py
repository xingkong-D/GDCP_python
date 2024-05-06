import numpy as np
from tkinter import Tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt

import GDCP.Generalization as gen
import GDCP.Color_Correction as cc
import GDCP.Imtool as Imtool

win = 13
t0 = 0.2
r0 = t0 * 1.5
sc = 1


# def stretch(x):
#     min_val = np.min(x)
#     max_val = np.max(x)
#     stretched = (x - min_val) * (1 / (max_val - min_val))
#     return stretched

def main():
    # 读取图像
    root = Tk()  # 创建根窗口
    root.withdraw()  # 隐藏根窗口
    filename = filedialog.askopenfilename(filetypes=[('Image Files', ['*.bmp', '*.jpg', '*.png'])])  # 打开文件选择窗口
    # 读入选择的图像文件,imread()不支持中文路径
    im = np.float32(cv2.imread(r"{}".format(filename))) / 255.0  # 读取图像转成double类型数据

    # 将图像转化为列数480的图像
    width = im.shape[1]  # 获得图像的列数
    sc = 1
    if width != 480:
        sc = 480 / width
    I = cv2.resize(im, (0, 0), fx=sc, fy=sc)

    #  #偏色纠正系数
    s = cc.CC(I)

    #  #获得深度图，并根据深度图求得A
    DepthMap, GradMap = gen.GetDepth(I, win)  # 获得深度图和梯度图
    A = gen.atmLight(I, DepthMap)  # 求环境光A

    Imtool.imtool(DepthMap,cmap='gray')       # 显示深度图
    plt.title('DepthMap')

    # 求透射率t
    T = gen.calcTrans(I, A, win)  # 公式9的实现
    maxT = np.max(T)
    minT = np.min(T)
    T_pro = ((T - minT) / (maxT - minT)) * (maxT - t0) + t0  # 将透射率拉伸到0.2-maxT

    # #求还原图像
    Jc = np.zeros_like(I)  # 创建和原图像一样规格的矩阵
    for ind in range(3):
        Am = A[ind] / s[ind]
        Jc[:, :, ind] = Am + (I[:, :, ind] - Am) / np.maximum(T_pro, r0)
    Jc[Jc < 0] = 0
    Jc[Jc > 1] = 1

    # 将两张图像水平拼接
    combined_image = cv2.hconcat([I, Jc])
    # 在窗口中显示拼接后的图像
    cv2.imshow('XiaoGuoDuiBi', combined_image)
    cv2.waitKey(0)

# Imtool.imtool(im)
# cv2.waitKey(0)


if __name__ == '__main__':
    main()
