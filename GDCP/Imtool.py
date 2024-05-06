import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from pandastable import Table
import cv2
import threading


# 将BGR图像转化为RGB图像
def bgr_to_rgb(image):
    # 获取图像的通道数
    num_channels = image.shape[-1]

    if num_channels == 3:
        # 提取每个通道
        blue_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        red_channel = image[:, :, 2]

        # 创建一个新的 RGB 图像
        rgb_image = np.zeros_like(image)
        rgb_image[:, :, 0] = red_channel
        rgb_image[:, :, 1] = green_channel
        rgb_image[:, :, 2] = blue_channel

        return rgb_image
    else:
        return image


# 使用OpenCV时没有matlab中的imtool()函数，此函数实现了类似的效果
def imtool(image,cmap='viridis'):
    # 读取图像并将其从 BGR 转换为 RGB
    image = bgr_to_rgb(image)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap)
    plt.show(block=False)


# 将图像以矩阵形式进行显示，便于查看数据层面的执行效果
def display_jz(matrix):
    dimensions = matrix.ndim  # 获取矩阵的维数

    if dimensions >= 4:
        raise ValueError("Matrix dimension should be less than 4.矩阵尺寸应小于4。")

    if dimensions == 1:
        matrix = matrix.reshape(1, -1)
    elif dimensions == 2:
        pass
    elif dimensions == 3:
        num_tables = matrix.shape[2]
        if num_tables <= 4:
            for i in range(num_tables):
                df = pd.DataFrame(matrix[:, :, i])
                root = tk.Tk()
                root.title(f'Matrix Display - Table {i + 1}')

                frame = tk.Frame(root)
                frame.pack(fill='both', expand=True)

                table = Table(frame, dataframe=df, showtoolbar=False, showstatusbar=False)
                table.show()

            root.mainloop()
            return
        else:
            print("第三维度超过4层，不是图像")

    df = pd.DataFrame(matrix)
    root = tk.Tk()
    root.title('Matrix Display')

    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)

    table = Table(frame, dataframe=df, showtoolbar=False, showstatusbar=False)
    table.show()

    root.mainloop()


# 复写cv2中的imshow()
def imshow(a,b):
    t = threading.Thread(cv2.imshow(a,b))
    t.start()