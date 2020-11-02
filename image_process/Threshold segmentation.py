# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
font = FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf')
warnings.filterwarnings('ignore')

img_data = cv2.imread('./40.bmp', 0)  # 以灰度模式加载图片，可以直接写0。
T = 90
plt.title('yuantu')
cv2.imshow('yuantu', img_data)
cv2.imwrite('./40/OriginPicture.jpg', img_data)
x, y = img_data.shape
print(x, y)

gray_lap = cv2.Laplacian(img_data, cv2.CV_16S, ksize=3)
dst = cv2.convertScaleAbs(gray_lap)  # 将原图片转换为uint8类型

sobelx = cv2.Sobel(img_data, cv2.CV_64F, 1, 0, ksize=3)  # 默认ksize=3
sobely = cv2.Sobel(img_data, cv2.CV_64F, 0, 1)
gm = cv2.sqrt(sobelx ** 2 + sobely ** 2)


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
        if (gm[i][j] > T) & (dst[i][j] == 0):
            sss.append(img_data[i][j])
            image2[i][j] = (int(img_data[i-1][j-1]) + int(img_data[i-1][j]) + int(img_data[i-1][j+1]) + int(img_data[i][j-1]) + int(img_data[i][j]) + int(img_data[i][j+1]) + int(img_data[i+1][j-1]) + int(img_data[i+1][j]) + int(img_data[i+1][j + 1]))/9
            sss1.append(image2[i][j])
            jiaochadian[i][j] = img_data[i][j]

cv2.imshow('jiaochadian', jiaochadian)
cv2.imwrite('./40/bianyuanjianceT=%d.jpg' % T, jiaochadian)

plt.subplot(121)
arr1 = img_data.flatten()
n, bins, patches = plt.hist(arr1, 256)  # 直方图
plt.title('原直方图',FontProperties=font)
plt.subplot(122)
sss1 = np.array(sss1)
arr3 = sss1.flatten()
n, bins, patches = plt.hist(arr3, 256)  # 直方图
plt.title('后直方图', FontProperties=font)
plt.show()

sum = 0
num = 0
for i in sss1:
    sum += i
    num += 1
ping = sum/num
print(ping)

ret, thresh1 = cv2.threshold(img_data, ping, 255, cv2.THRESH_BINARY)  # binary （黑白二值）
# 这个函数有四个参数，第一个是原图像矩阵，第二个是进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数
print(ret)
cv2.imshow('分割图', thresh1)
cv2.imwrite('./40/yuzhifengeT=%d.jpg' % T, thresh1)

titles = ['原图', ' 边缘检测 ', "分割图"]
images = [img_data, jiaochadian, thresh1]

for i in range(3):
    plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i], FontProperties=font)
    plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

