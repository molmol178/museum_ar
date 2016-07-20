# -*- coding: utf-8 -*-
import numpy as np
import cv2

img = cv2.imread('risou_en.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imwrite('affin_risou.png',dst)
