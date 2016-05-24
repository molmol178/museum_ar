# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pylab as plt
import math
import csv

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
  template_x ,template_y = template_hsv.shape[:2]
  template_HSV = cv2.split(template_hsv)
  v_value = template_HSV[2]
  v_count_list = np.zeros(256)
  for y in xrange(template_y):
    for x in xrange(template_x):
      v_count_list[v_value[x,y]] += 1
      #print 'v_value = ' + str(v_value[y,x])
      #print 'v_count_list' + str(v_count_list)
  #print 'v_count_list\n' + str(v_count_list)
  f = open('template_hist.csv', 'w')
  csvWriter = csv.writer(f)
  csvWriter.writerow(v_count_list)

  #0-255のそれぞれについて分離度を求めて、１次元配列に格納
  separation_list = np.zeros(256)
  tmp_range = 25
  for i in xrange(0 + tmp_range ,255 - tmp_range):
    sum_k = 0
    sum_j = 0
    sum_l = 0
    sum_vj = 0
    sum_vl = 0
    sum_o1_numerator = 0
    sum_o2_numerator = 0
    #print '----------start loop-----------\ni =' + str(i)
    for k in xrange(i-tmp_range,i+tmp_range):
      if k != i:
        sum_k += v_count_list[k]
      #print 'k = ' + str(k)
      #print 'v_count_list[k] = '+str(v_count_list[k])
    #print 'sum_k = ' + str(sum_k)
    for j in xrange(i-tmp_range,i - 1):
      sum_j += v_count_list[j]
      sum_vj += v_count_list[j] * j
      #print 'j = ' + str(j)
      #print 'v_count_list[j] = '+str(v_count_list[j])
    #print 'sum_j = ' + str(sum_j)
    for l in xrange(i + 1,i+tmp_range):
      sum_l += v_count_list[l]
      sum_vl += v_count_list[l] * l
      #print 'l = ' + str(l)
      #print 'v_count_list[l] = '+str(v_count_list[l])
    #print 'sum_l = ' + str(sum_l)
    w1 = sum_j / sum_k
    u1 = sum_vj / sum_j
    w2 = sum_l / sum_k
    u2 = sum_vl / sum_l
    for j4o1 in xrange(i-tmp_range,i +1):
      sum_o1_numerator += v_count_list[j4o1] * (j4o1 - u1)**2
    for l4o2 in xrange(i -1,i+tmp_range):
      sum_o2_numerator += v_count_list[l4o2] * (l4o2 - u2)**2
    o1 = sum_o1_numerator / sum_j
    o2 = sum_o2_numerator / sum_l
    o12 = w1 * w2 * (u1 - u2) **2 + (w1 * o1 + w2 * o2)
    n = w1 * w2 * (u1 - u2) **2 / o12
    separation_list[i] = n
    '''
    print '-------------------------------'
    print 'w1 = ' + str(w1)
    print 'u1 = ' + str(u1)
    print 'w2 = ' + str(w2)
    print 'u2 = ' + str(u2)
    print 'o1 = ' + str(o1)
    print 'o2 = ' + str(o2)
    print 'o12 = ' + str(o12)
    '''
  #print 'separation_list\n' + str(separation_list)
  '''
  print '@:templateのヒストグラム、*:分離度'
  col = 80
  add_sep = 10000000000000
  for list_row in xrange(0, 255):
    print ' '
    if  col * separation_list[list_row] * add_sep > col * v_count_list[list_row] / int(max(v_count_list)):
      for v_list in xrange(0, col * int(v_count_list[list_row]) / int(max(v_count_list))):
        print '@',
      for sep_list in xrange(0, col * int(separation_list[list_row]) *add_sep - col * int(v_count_list[list_row]) / int(max(v_count_list))):
        print '*',
    elif col * int(separation_list[list_row]) * add_sep < col * int(v_count_list[list_row]) / int(max(v_count_list)):
      for sep_i_list in xrange(0, col * int(separation_list[list_row]) * add_sep):
        print '*',
      for v_i_list in xrange(0, col * int(v_count_list[list_row]) / int(max(v_count_list)) - col * int(separation_list[list_row]) * add_sep):
        print '@',
  '''
  f_separation = open('separation_list.csv', 'w')
  csvWriter = csv.writer(f_separation)
  csvWriter.writerow(separation_list)

  #領域分割するために0-255の分離点(極大値&&2/π)を複数求める
  label_list = np.zeros(256)
  b = 1
  for a in xrange(0 + 1, 255 - 1):
    label_list[a] = b
    if separation_list[a] > separation_list[a-1] and \
       separation_list[a] > separation_list[a+1] and \
       separation_list[a] > 2 / math.pi:
       print 'get separation point ' + str(a) + ' value = ' + str(separation_list[a])
       #分割したらそれぞれの１次元配列の領域にラベルをつける
       b += 10
       label_list[a] = b
  print 'label_list\n' + str(label_list)
  f_label = open('label_list.csv', 'w')
  csvWriter = csv.writer(f_label)
  csvWriter.writerow(label_list)

  #ラベル配列をルックアップテーブルとして用い、領域ごとに濃淡値を分けた濃淡画像として出力する。
  label_template_img = np.zeros((template_x, template_y, 1), np.uint8)
  for ty in xrange(template_y):
    for tx in xrange(template_x):
      label_template_img[tx,ty] = label_list[v_value[tx,ty]]
  cv2.imwrite('label_template_img.tif', label_template_img)
  return (v_count_list)

#inputのラベル画像生成
#inputのV値をtemplateのV値に補正する
def createInputLabelImage(input_hsv, v_count_list):
  input_x ,input_y = input_hsv.shape[:2]
  input_HSV = cv2.split(input_hsv)
  input_v_value = input_HSV[2]

  #inputのヒストグラム作成
  input_v_count_list = np.zeros(256)
  for y in xrange(input_y):
    for x in xrange(input_x):
      input_v_count_list[input_v_value[x,y]] += 1
  print 'input_v_count_list\n' + str(input_v_count_list)
  f_pre_in = open('pre_input_hist.csv', 'w')
  csvWriter = csv.writer(f_pre_in)
  csvWriter.writerow(input_v_count_list)

  #inputのヒストグラムをtemplateに合わせる処理
  #templateの標準偏差と全体の平均を求める
  sum_t = 0
  sum_i = 0
  sum_vt = 0
  sum_vi = 0
  sum_o_numerator = 0
  sum_oi_numerator = 0
  for t in xrange(0 ,256):
    sum_vt += v_count_list[t] * t
    sum_t += v_count_list[t]
  ave_t = sum_vt / sum_t
  for tt in xrange(0, 256):
    sum_o_numerator += v_count_list[tt] * (tt - ave_t)**2
  ot = sum_o_numerator / sum_t
  template_std_deviation = math.sqrt(ot)

  #inputの標準偏差と全体の平均を求める
  for i in xrange(0, 256):
    sum_vi += input_v_count_list[i] * i
    sum_i += input_v_count_list[i]
  ave_i = sum_vi / sum_i
  for ii in xrange(0, 256):
    sum_oi_numerator += input_v_count_list[ii] * (ii - ave_i)**2
  oi = sum_oi_numerator / sum_i
  input_std_deviation = math.sqrt(oi)

  #inputのtemplateに対する偏差値を求め補正結果を配列に格納
  correct_input_list = np.zeros(256)
  for iii in xrange(0, 256):
    deveation = template_std_deviation * (input_v_count_list[iii] - ave_i) / input_std_deviation + ave_t
    correct_input_list[iii] = deveation
  print 'correct_input_list\n' + str(correct_input_list)
  f_corre_in = open('correct_input_hist.csv', 'w')
  csvWriter = csv.writer(f_corre_in)
  csvWriter.writerow(correct_input_list)

  #補正したヒストグラムからinputを再生成(やらなくてもいい使わない。確認はする)
  #~~~
  #print 'correct_input_v_value' + str(correct_input_v_value)

  #作成した領域ラベルリストを用いてinputラベル画像生成
  label_input_img = np.zeros((input_x, input_y, 1), np.uint8)
  label_CsvFile = csv.reader(open('label_list.csv'),delimiter=',')
  label_CsvList = []
  for label in label_CsvFile:
    label_CsvList.append(label)
  for iy in xrange(input_y):
    for ix in xrange(input_x):
      label_input_img[ix,iy] = label_CsvList[correct_input_list[ix,iy]]
  cv2.imwrite('label_input_img.png', label_input_img)




#fastの16点がどのラベルと重なっているか。ラベルとラベルの接点は何個あるか。ラベル数:3,接点数:3の点をキーポイントとする。
def input_FeatureDetection(input_hsv, template_hsv):
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
input_hsv, template_hsv = rgb2Hsv(input_img, template)
v_count_list = template_splitRegion(template_hsv)
createInputLabelImage(input_hsv, v_count_list)
'''
while(True):
  cv2.imshow("input", input_hs)
  cv2.imshow("template", template_hs)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cv2.destroyAllWindows()
'''
