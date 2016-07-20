# -*- coding: utf-8 -*-
import numpy as np
import cv2

img = cv2.imread('risou_en.png')
rows,cols,ch = img.shape

#拡大
#res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_AREA)

#縮小
res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

cv2.imwrite('scale_small_risou.png',res)
