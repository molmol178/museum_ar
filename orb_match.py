# -*- coding: utf-8 -*-
import numpy as np
import cv2
#from matplotlib import pyplot as plt

def drawMatches(img1, kp1, img2, kp2, matches):

  rows1 = img1.shape[0]
  cols1 = img1.shape[1]
  rows2 = img2.shape[0]
  cols2 = img2.shape[1]

  out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
  out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])
  out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

  for mat in matches:

    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt

    cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
    cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
    cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

  cv2.imshow('Matched Features', out)



cap = cv2.VideoCapture(0)
img = cv2.imread('marlboro.png')
#img2 = cv2.imread('marlboro3.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("image", img)

while(True):
#  cv2.imshow('frame' ,frame)
  ret, frame = cap.read()
  gray2= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

  orb = cv2.ORB()
  kp1 ,des1 = orb.detectAndCompute(gray,None)
  kp2 ,des2 = orb.detectAndCompute(gray2,None)

  # create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # Match descriptors.
  matches = bf.match(des1,des2)

  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)

  # Draw first 10 matches.
  img3 = drawMatches(gray,kp1,gray2,kp2,matches[:30])

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
