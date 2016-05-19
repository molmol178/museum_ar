# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pylab as plt
import math

input_img = cv2.imread('graf3.png',1)
template = cv2.imread('graf1.png',1)

#RGBからHSVに変換
def rgb2Hsv(input_img, template):
  input_hsv = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
  template_hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
  return (input_hsv, template_hsv)

#templateの領域をしきい値に応じて分割
def template_splitRegion(template_hsv):
  #画像のV値を用いてヒストグラムを作成(特定のV値が何個あるか集計)
  '''
  template_hist = plt.hist(template_hsv[:,:,2].ravel(),256,range=(0,255), fc='r')
  plt.xlim([0,256])
  plt.show()
  '''

  template_x ,template_y = template_hsv.shape[:2]
  HSV = cv2.split(template_hsv)
  v_value = HSV[2]
  v_count_list = np.zeros(256)
  for y in xrange(template_y):
    for x in xrange(template_x):
      v_count_list[v_value[x,y]] += 1
      #print 'v_value = ' + str(v_value[y,x])
      #print 'v_count_list' + str(v_count_list)
  #print v_count_list

  #0-255のそれぞれについて分離度を求めて、１次元配列に格納
  separation_list = np.zeros(256)
  threshold = 10
  for i in xrange(0 + threshold ,256 - threshold):
    sum_k = 0
    sum_j = 0
    sum_l = 0
    sum_o1_numerator = 0
    sum_o2_numerator = 0
    #print '----------start loop-----------\ni =' + str(i)
    for k in xrange(i-threshold,i+threshold):
      sum_k += v_count_list[k]
      #print 'k = ' + str(k)
      #print 'v_count_list[k] = '+str(v_count_list[k])
    #print 'sum_k = ' + str(sum_k)
    for j in xrange(i-threshold,i):
      sum_j += v_count_list[j]
      #print 'j = ' + str(j)
      #print 'v_count_list[j] = '+str(v_count_list[j])
    #print 'sum_j = ' + str(sum_j)
    for l in xrange(i,i+threshold):
      sum_l += v_count_list[l]
      #print 'l = ' + str(l)
      #print 'v_count_list[l] = '+str(v_count_list[l])
    #print 'sum_l = ' + str(sum_l)
    w1 = sum_j / sum_k
    u1 = sum_j / threshold
    w2 = sum_l / sum_k
    u2 = sum_l /threshold
    for j4o1 in xrange(i-threshold,i):
      sum_o1_numerator += (v_count_list[j4o1] - u1)**2
    for l4o2 in xrange(i,i+threshold):
      sum_o2_numerator += (v_count_list[l4o2] - u2)**2
    o1 = sum_o1_numerator / threshold
    o2 = sum_o2_numerator / threshold
    o12 = w1 * w2 * (u1 - u2) **2 + (w1 * o1 **2 + w2 * o2 **2)
    n = w1 * w2 * (u1 - u2) **2 / o12
    separation_list[i] = n

    print '-------------------------------'
    print 'w1 = ' + str(w1)
    print 'u1 = ' + str(u1)
    print 'w2 = ' + str(w2)
    print 'u2 = ' + str(u2)
    print 'o1 = ' + str(o1)
    print 'o2 = ' + str(o2)
    print 'o12 = ' + str(o12)
    print 'n = ' + str(n)
    #print 'separation_list['+str(i)+'] = ' + str(separation_list[i])

  #領域分割するために0-255の分離点(極大値&&2/π)を複数求める
  for a in xrange(0 + 1, 256 - 1):
    #print 'separation_list['+str(a)+'] = ' + str(separation_list[a])
    if separation_list[a] > separation_list[a-1] and separation_list[a] > separation_list[a+1] :
      print 'get separation point ' + str(a) + ' value = ' + str(separation_list[a])





  #分割したらそれぞれの１次元配列の領域にラベルをつける
  #ラベル配列をルックアップテーブルとして用い、領域ごとに濃淡地を分けた濃淡画像として出力する。

#作成した領域ラベルを用いてinputを領域分割
#def input_splitRegion():

#fastの16点がどのラベルと重なっているか。ラベルトラベルの接点は何個あるか。ラベル数:3,接点数:3の点をキーポイントとする。
def getHsvValue(input_hsv, template_hsv):
  input_x, input_y = input_hsv.shape[:2]
  tempalte_x, template_y = template_hsv.shape[:2]
  for y in range(input_y):
    for x in range(input_x):
      print 'rgb'
      print input_img[y,x]
      print 'hsv'
      print input_hsv[y,x]




#inputとtemplateのバイナリ化
'''
全てのラベルの内、パッチ内にどのラベルがあるかでバイナリ化する。
'''
#def featureDescription():



#マッチング
'''
inputとtemplateのバイナリのハミング距離を取ってマッチング
'''
#def featureMatching():



#main
rgb2Hsv(input_img, template)
input_hsv, template_hsv = rgb2Hsv(input_img, template)
template_splitRegion(template_hsv)
#getHsvValue(input_hsv, template_hsv)
'''
while(True):
  cv2.imshow("input", input_hs)
  cv2.imshow("template", template_hs)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cv2.destroyAllWindows()
'''
