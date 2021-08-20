import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('E:/opencv_picture/222.jpg', cv2.IMREAD_REDUCED_COLOR_2)
img1 = cv2.imread('E:/opencv_picture/7.jpg', cv2.IMREAD_GRAYSCALE)
open_1 = cv2.imread('E:/opencv_picture/PS-2.jpg', cv2.IMREAD_REDUCED_COLOR_2)
open_2 = cv2.imread('E:/opencv_picture/PS-1.jpg', cv2.IMREAD_REDUCED_COLOR_2)
fx = cv.imread('E:/opencv_picture/fx.jpg')
fx_1 = cv.imread('E:/opencv_picture/fx_1.jpg')
mh_1 = cv.imread('E:/opencv_picture/mh.jpg')
mh_2 = cv.imread('E:/opencv_picture/mh_2.jpg')
cap = cv.VideoCapture('E:/opencv_picture/test.mp4', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape[:2]
# '图片展示，并截取指定位置'
# cat = img[200:900, 200:900]
# cv2.imshow('Imag', cat)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# 图像移动 图像混合 图像缩放  完成
# img_1 = cv.addWeighted(img, 0.2, img, 0.9, 0)  图像混合
# resl = cv.resize(img, None, fx=0.5, fy=0.5)  相对图像大小 缩小
# rows, cols = img.shape[:2] 获取 图像宽高
# m = np.float32([[1, 0, 200], [0, 1, 150]])  图像移动 位置
# resl2 = cv.warpAffine(img, m, (cols, rows)) 图像移动
# cv2.imshow("Im", resl2) 展示效果
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# 图像旋转   完成

# m = cv.getRotationMatrix2D((cols/2, rows/2), 60, 1)
# res3 = cv.warpAffine(img, m, (cols, rows))
# cv2.imshow("Im", res3)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()


# 图像仿射  完成
# 创建变换矩阵
# ptsl = np.float32([[50, 50], [200, 50], [50, 200]])
# 变换后的位置
# ptsl2 = np.float32([[100, 100], [200, 100], [100, 200]])
# m = cv.getAffineTransform(ptsl, ptsl2)
# show = cv.warpAffine(img, m, (cols, rows))
# cv2.imshow("Im", show)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# 图像透射
# ptsl = np.float32([[56, 56], [368, 52], [28, 387], [389, 398]])
# ptsl2 = np.float32([[100, 145], [300, 100], [80, 290], [310, 300]])
# m = cv.getPerspectiveTransform(ptsl, ptsl2)
# dst = cv.warpPerspective(img, m, (cols, rows))
# cv2.imshow("Im", dst)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# 图像金字塔     可无限上采样 画中画效果
# 上采样 可无限上采样 画中画效果
# up_img = cv.pyrUp(img)
# 下采样 可无限下采样 画中画效果
# donw_img = cv.pyrDown(img)
# 图形显示
# cv.imshow('enlarge', up_img)
# cv.imshow('ys', img)
# cv.imshow('shrink', donw_img)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# 图像腐蚀 与 膨胀
# 创建核结构
# kenel = np.ones((15, 15), np.uint8)
# # 腐蚀
# # img_1 = cv.erode(img, kenel)
# # 膨胀
# img_1 = cv.dilate(img, kenel)
# cv.imshow('shrink', img_1)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# 图像开闭运算
# kenel = np.ones((10, 10), np.uint8)
# 开运算   清楚图像外部 噪点
# cvopen = cv.morphologyEx(open_2, cv.MORPH_OPEN, kenel)
# 闭运算  清除图像内部 噪点
# cvclose = cv.morphologyEx(open_1, cv.MORPH_CLOSE, kenel)
# cv.imshow('shrink', cvopen)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# 黑帽 礼貌 运算
# kenel = np.ones((2, 2), np.uint8)
# 礼貌运算
# top = cv.morphologyEx(fx_1, cv.MORPH_TOPHAT, kenel)
# 黑帽运算
# black = cv.morphologyEx(fx_1, cv.MORPH_BLACKHAT, kenel)
# up_img = cv.pyrUp(top)
# yesup = cv.pyrUp(up_img)
# cv.imshow('LmSum', up_img)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()



#  **** 腐蚀膨胀 开闭运算  礼貌 黑帽 组合使用    开运算 对应: 礼貌   闭运算对应: 黑帽



# 图像 噪声 平滑处理

# 均值滤波   降低椒盐噪声
# jzlb = cv.blur(mh_2, (5, 5))
# cv.imshow('fx', jzlb)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# 高斯滤波
# gaos = cv.GaussianBlur(mh_2, (1, 1), 1)
# cv.imshow('fx', gaos)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# 中值滤波    获取周围橡树平均值 显示图像
# midv = cv.medianBlur(mh_2, 5)
# cv.imshow('fx', midv)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()


#  直方图 掩模
# 蒙版
# mask = np.zeros(fx_1.shape[:2], np.uint8)
# 全黑蒙版
# mask[400:650, 200:500] = 1
# masked_img = cv.bitwise_and(fx_1, fx_1, mask=mask)
# mask_histr = cv.calcHist([fx_1], [0], mask, [256], [1, 256])
# cv.imshow('mask', mask_histr)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()



# 图像 模糊处理   双边滤波
# kenel = np.ones((2, 2), np.uint8)
# cvopen = cv.morphologyEx(mh_2, cv.MORPH_OPEN, kenel)
# mhs = cv.bilateralFilter(mh_2, 2, 100, 2)
# midv = cv.medianBlur(mhs, 5)
# cv.imshow('mask', midv)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()


# 视频播放
# cap = cv.VideoCapture('E:/opencv_picture/test.mp4', cv2.IMREAD_GRAYSCALE)
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         cv.imshow('frame', frame)
#         if cv.waitKey(25) & 0xFF == ord('q'):
#             break
#             cap.release()
#             cv2.destroyAllWindows()

# 视频截取保存本地   设置 视频格式 默认 FPS 正常速度 25 越高播放速度越快
# width = int(cap.get(3))
# height = int(cap.get(4))
# out = cv.VideoWriter('E://opencv_picture//out.avi', cv.VideoWriter_fourcc("M", "J", "P", "G"), 10, (width, height))
# while(True):
#     ret, frame = cap.read()
#     if ret == True:
#         out.write(frame)
#     else:
#         break
#         cap.release()
#         out.release()
#         cv.destroyAllWindows()


# 图像边缘轮廓
v1 = cv2.Canny(fx_1, 80, 100)
v2 = cv2.Canny(fx_1, 50, 100)
cv.imshow('frame', v1)
while(True):
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# 人脸识别算法 

# cap_1 = cv.VideoCapture(0, cv2.IMREAD_GRAYSCALE)
# kenel = np.ones((2, 2), np.uint8)
# width = int(cap_1.get(3))
# height = int(cap_1.get(4))
# out = cv.VideoWriter('E://opencv_picture//out_new.avi', cv.VideoWriter_fourcc("M", "J", "P", "G"), 25, (width, height))
# while(cap_1.isOpened()):
#     ret, frame = cap_1.read()
#     if ret == True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # cvopen = cv.morphologyEx(frame, cv.MORPH_OPEN, kenel) COLOR_BGR2GRAY
#         # mhs = cv.bilateralFilter(cvopen, 2, 100, 2)
#         # gaos = cv.GaussianBlur(mhs, (1, 1), 1)
#         # jzlb = cv.blur(gaos, (5, 5))
#         # # #轮廓描边
#         # v2 = cv2.Canny(gray, 30, 100)
#         face_cas = cv.CascadeClassifier('E:/opencv_picture/haarcascade_frontalface_default.xml')
#         face_cas.load('E:/opencv_picture/haarcascade_frontalface_default.xml')
#         eyes_cas = cv.CascadeClassifier('E:/opencv_picture/haarcascade_eye.xml')
#         eyes_cas.load('E:/opencv_picture/haarcascade_eye.xml')
#         faceRects = face_cas.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))
#         for faceRect in faceRects:
#             x, y, w, h = faceRect
#             cv.rectangle(gray, (x, y), (x + h, y + w), (0, 255, 0), 3)
#             roi_color = gray[y:y+h, x:x+w]
#             # roi_gray = gray[y:y+h, x:x+w]
#             eyes = eyes_cas.detectMultiScale(roi_color)
#             for(ex, ey, ew, eh) in eyes:
#                 cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
#         cv2.imshow('My', gray)
#         out.write(gray)
#         if cv.waitKey(25) & 0xFF == ord('q'):
#             break
#             cap.release()
#             out.release()
#             cv2.destroyAllWindows()


