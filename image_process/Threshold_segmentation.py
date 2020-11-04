# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
font = FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf')
warnings.filterwarnings('ignore')

img_data = cv2.imread('./40.jpg', 0)  # 以灰度模式加载图片，可以直接写0。
T = [10, 60, 120]
# plt.title('yuantu')
# cv2.imshow('yuantu', img_data)
# cv2.imwrite('./40/OriginPicture.jpg', img_data)
x, y = img_data.shape
print(x, y)

dst = cv2.Laplacian(img_data, cv2.CV_16S, ksize=3)
#dst = cv2.convertScaleAbs(gray_lap)  # 将原图片转换为uint8类型

sobelx = cv2.Sobel(img_data, cv2.CV_64F, 1, 0, ksize=3)  # 默认ksize=3
sobely = cv2.Sobel(img_data, cv2.CV_64F, 0, 1)
gm = cv2.sqrt(sobelx ** 2 + sobely ** 2)

arr1 = img_data.flatten()
n, bins, patches = plt.hist(arr1, 256)  # 直方图
plt.title('原直方图', FontProperties=font)
plt.show()


def huizhi(T):
    img_data2 = np.zeros((x, y), dtype=np.uint8)
    for i in range(0, x):
        for j in range(0, y):
            img_data2[i][j] = 0

    image1 = img_data2
    jiaochadian = img_data2
    sss = []
    sss1 = []
    image2 = img_data
    for i in range(1, x-1):
        for j in range(1, y-1):
            if (gm[i][j] > T) & ((dst[i-1][j] > 0) & (dst[i][j] < 0)):
                sss.append(img_data[i][j])
                image2[i][j] = (int(img_data[i-1][j-1]) + int(img_data[i-1][j]) + int(img_data[i-1][j+1]) + int(img_data[i][j-1]) + int(img_data[i][j]) + int(img_data[i][j+1]) + int(img_data[i+1][j-1]) + int(img_data[i+1][j]) + int(img_data[i+1][j + 1]))/9
                sss1.append(image2[i][j])
                jiaochadian[i][j] = img_data[i][j]
            elif (gm[i][j] > T) & ((dst[i-1][j] < 0) & (dst[i][j] > 0)):
                sss.append(img_data[i][j])
                image2[i][j] = (int(img_data[i - 1][j - 1]) + int(img_data[i - 1][j]) + int(img_data[i - 1][j + 1]) + int(
                    img_data[i][j - 1]) + int(img_data[i][j]) + int(img_data[i][j + 1]) + int(img_data[i + 1][j - 1]) + int(
                    img_data[i + 1][j]) + int(img_data[i + 1][j + 1])) / 9
                sss1.append(image2[i][j])
                jiaochadian[i][j] = img_data[i][j]
            elif (gm[i][j] > T) & ((dst[i][j-1] > 0) & (dst[i][j] < 0)):
                sss.append(img_data[i][j])
                image2[i][j] = (int(img_data[i - 1][j - 1]) + int(img_data[i - 1][j]) + int(img_data[i - 1][j + 1]) + int(
                    img_data[i][j - 1]) + int(img_data[i][j]) + int(img_data[i][j + 1]) + int(img_data[i + 1][j - 1]) + int(
                    img_data[i + 1][j]) + int(img_data[i + 1][j + 1])) / 9
                sss1.append(image2[i][j])
                jiaochadian[i][j] = img_data[i][j]
            elif (gm[i][j] > T) & ((dst[i][j] > 0) & (dst[i][j-1] < 0)):
                sss.append(img_data[i][j])
                image2[i][j] = (int(img_data[i - 1][j - 1]) + int(img_data[i - 1][j]) + int(img_data[i - 1][j + 1]) + int(
                    img_data[i][j - 1]) + int(img_data[i][j]) + int(img_data[i][j + 1]) + int(img_data[i + 1][j - 1]) + int(
                    img_data[i + 1][j]) + int(img_data[i + 1][j + 1])) / 9
                sss1.append(image2[i][j])
                jiaochadian[i][j] = img_data[i][j]
            elif (gm[i][j] > T) & ((dst[i-1][j-1] > 0) & (dst[i][j] < 0)):
                sss.append(img_data[i][j])
                image2[i][j] = (int(img_data[i - 1][j - 1]) + int(img_data[i - 1][j]) + int(img_data[i - 1][j + 1]) + int(
                    img_data[i][j - 1]) + int(img_data[i][j]) + int(img_data[i][j + 1]) + int(img_data[i + 1][j - 1]) + int(
                    img_data[i + 1][j]) + int(img_data[i + 1][j + 1])) / 9
                sss1.append(image2[i][j])
                jiaochadian[i][j] = img_data[i][j]
            elif (gm[i][j] > T) & ((dst[i-1][j-1] < 0) & (dst[i][j] > 0)):
                sss.append(img_data[i][j])
                image2[i][j] = (int(img_data[i - 1][j - 1]) + int(img_data[i - 1][j]) + int(img_data[i - 1][j + 1]) + int(
                    img_data[i][j - 1]) + int(img_data[i][j]) + int(img_data[i][j + 1]) + int(img_data[i + 1][j - 1]) + int(
                    img_data[i + 1][j]) + int(img_data[i + 1][j + 1])) / 9
                sss1.append(image2[i][j])
                jiaochadian[i][j] = img_data[i][j]
            elif (gm[i][j] > T) & ((dst[i][j-1] > 0) & (dst[i-1][j] < 0)):
                sss.append(img_data[i][j])
                image2[i][j] = (int(img_data[i - 1][j - 1]) + int(img_data[i - 1][j]) + int(img_data[i - 1][j + 1]) + int(
                    img_data[i][j - 1]) + int(img_data[i][j]) + int(img_data[i][j + 1]) + int(img_data[i + 1][j - 1]) + int(
                    img_data[i + 1][j]) + int(img_data[i + 1][j + 1])) / 9
                sss1.append(image2[i][j])
                jiaochadian[i][j] = img_data[i][j]
            elif (gm[i][j] > T) & ((dst[i][j-1] < 0) & (dst[i-1][j] > 0)):
                sss.append(img_data[i][j])
                image2[i][j] = (int(img_data[i - 1][j - 1]) + int(img_data[i - 1][j]) + int(img_data[i - 1][j + 1]) + int(
                    img_data[i][j - 1]) + int(img_data[i][j]) + int(img_data[i][j + 1]) + int(img_data[i + 1][j - 1]) + int(
                    img_data[i + 1][j]) + int(img_data[i + 1][j + 1])) / 9
                sss1.append(image2[i][j])
                jiaochadian[i][j] = img_data[i][j]
            elif (gm[i][j] > T) & (dst[i][j] == 0):
                sss.append(img_data[i][j])
                image2[i][j] = (int(img_data[i - 1][j - 1]) + int(img_data[i - 1][j]) + int(img_data[i - 1][j + 1]) + int(
                    img_data[i][j - 1]) + int(img_data[i][j]) + int(img_data[i][j + 1]) + int(img_data[i + 1][j - 1]) + int(
                    img_data[i + 1][j]) + int(img_data[i + 1][j + 1])) / 9
                sss1.append(image2[i][j])
                jiaochadian[i][j] = img_data[i][j]
    sum = 0
    num = 0
    for i in sss1:
        sum += i
        num += 1
    ping = sum/num +5
    print(ping)
    # print(num)
    return sss1, jiaochadian, ping

# cv2.imshow('jiaochadian', jiaochadian)
# cv2.imwrite('./40/bianyuanjianceT=%d.jpg' % T, jiaochadian)


plt.subplot(131)
sss1, jiaochadian1, ping1 = huizhi(T[0])
sss1 = np.array(sss1)
arr3 = sss1.flatten()
n, bins, patches = plt.hist(arr3, 256)  # 直方图
plt.title('T=40', FontProperties=font)
plt.subplot(132)
sss1, jiaochadian2, ping2 = huizhi(T[1])
sss1 = np.array(sss1)
arr3 = sss1.flatten()
n, bins, patches = plt.hist(arr3, 256)  # 直方图
plt.title('T=100', FontProperties=font)
plt.subplot(133)
sss1, jiaochadian3, ping3 = huizhi(T[2])
sss1 = np.array(sss1)
arr3 = sss1.flatten()
n, bins, patches = plt.hist(arr3, 256)  # 直方图
plt.title('T=160', FontProperties=font)
plt.show()
# print(sss)

# sum = 0
# num = 0
# for i in sss:
#     sum += i
#     num += 1
# ping = sum/num
# print(ping)
# # print(num)

# sum = 0
# num = 0
# for i in range(0, x):
#     for j in range(0, y):
#         sum += img_data[i][j]
#         num += 1
# ping = sum/num
# print(ping)

ret1, thresh1 = cv2.threshold(img_data, ping1-2, 255, cv2.THRESH_BINARY)  # binary （黑白二值）
ret2, thresh2 = cv2.threshold(img_data, ping2+2, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(img_data, ping3+6, 255, cv2.THRESH_BINARY)
# 这个函数有四个参数，第一个是原图像矩阵，第二个是进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数

# cv2.imshow('分割图', thresh1)
# cv2.imwrite('./40/yuzhifengeT=%d.jpg' % T, thresh1)
#
titles = ['T=40', 'T=100', 'T=160']
images1 = [jiaochadian1, jiaochadian2, jiaochadian3]
for i in range(3):
    plt.subplot(1, 3, i + 1), plt.imshow(images1[i], 'gray')
    plt.title(titles[i], FontProperties=font)
    plt.xticks([]), plt.yticks([])
plt.show()

images2 = [thresh1, thresh2, thresh3]
for i in range(3):
    plt.subplot(1, 3, i + 1), plt.imshow(images2[i], 'gray')
    plt.title(titles[i], FontProperties=font)
    plt.xticks([]), plt.yticks([])
plt.show()


