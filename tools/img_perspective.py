# -*- coding: utf-8 -*-
import numpy as np
import cv2

img = cv2.imread('risou_en.png')

rows,cols,ch = img.shape

pts1 = np.float32([[23,35],[530,26],[16,790],[512,769]])
pts2 = np.float32([[0,0],[540,0],[0,800],[540,800]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(800,540))



cv2.imwrite('perspective_risou.png',dst)
