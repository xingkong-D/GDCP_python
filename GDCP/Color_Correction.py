import numpy as np


# BGR色彩空间转Lab色彩空间
def getBGR2Lab(B, G, R):
    if R.ndim == 3:
        R = R[:, :, 0]
        G = R[:, :, 1]
        B = R[:, :, 2]

    if np.max(R) > 1.0 or np.max(G) > 1.0 or np.max(B) > 1.0:
        R = R / 255.0
        G = G / 255.0
        B = B / 255.0

    T = 0.008856

    s = R.size
    RGB = np.array([R.flatten(), G.flatten(), B.flatten()])

    # BGR to XYZ
    MAT = np.array([[0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227]])
    XYZ = np.dot(MAT, RGB)

    # Normalize for D65 white point
    X = XYZ[0, :] / 0.950456
    Y = XYZ[1, :]
    Z = XYZ[2, :] / 1.088754

    XT = X > T
    YT = Y > T
    ZT = Z > T

    Y3 = Y ** (1 / 3)

    fX = XT * X ** (1 / 3) + (~XT) * (7.787 * X + 16 / 116)
    fY = YT * Y3 + (~YT) * (7.787 * Y + 16 / 116)
    fZ = ZT * Z ** (1 / 3) + (~ZT) * (7.787 * Z + 16 / 116)

    L = np.reshape(YT * (116 * Y3 - 16.0) + (~YT) * (903.3 * Y), R.shape)
    a = np.reshape(500 * (fX - fY), R.shape)
    b = np.reshape(200 * (fY - fZ), R.shape)

    return L, a, b


# 求色彩的方差
def getColorCast(im):
    L, a, b = getBGR2Lab(im[:, :, 0], im[:, :, 1], im[:, :, 2])  # 转换色彩空间
    var_ansa = np.var(a.flatten())  # 求方差
    var_ansb = np.var(b.flatten())  # 求方差

    var_sq = np.sqrt(var_ansa + var_ansb)
    u = np.sqrt(np.mean(a.flatten()) ** 2 + np.mean(b.flatten()) ** 2)
    D = u - var_sq
    Dl = D / var_sq

    return D, Dl


def CC(I):
    h, w, _ = I.shape

    imsize = h * w
    Ivec = np.reshape(I, (imsize, 3))  # 将图像变成 imsize 行 * 3 列的矩阵
    avgIC = np.mean(Ivec, axis=0)  # 对二维矩阵沿列求平均值
    _, Dl = getColorCast(I)  # 返回偏色情况


    if Dl <= 0:
        sc = [1, 1, 1]
    else:
        sc_c = (1 / max(np.sqrt(Dl), 1))
        sc_a = max(max(avgIC), 0.1)
        sc_b = np.maximum(avgIC, 0.1)
        sc = (sc_a / sc_b) ** sc_c
    return sc
