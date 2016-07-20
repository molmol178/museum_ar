#-*- coding: utf-8 -*-
import numpy as np
import cv2

img = cv2.imread('risou_en.png', 1)
rows,cols,ch = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imwrite('rotate90_risou.png', dst)
