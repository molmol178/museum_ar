# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import csv
from collections import Counter

#RGBからHSVに変換
def rgb2Hsv(int template[[]]):

  cdef int template_hsv[[]]

  template_hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
  return (template_hsv)

#templateの領域をしきい値に応じて分割
def template_splitRegion(int template_hsv[[]]):

  cdef long template_y, template_x, template_HSV[], v_value[], separation_list[], sum_vj, sum_vl, sum_o1_numerator, sum_o2_numerator, o1, o2, o12, n
  cdef int v_count_list[[]], tmp_range, sum_k, sum_j, sum_l, a, b, label_list[], tx, ty
  cdef int lable_template_img[]


  #画像のV値を用いてヒストグラムを作成(特定のV値が何個あるか集計)
  template_y ,template_x = template_hsv.shape[:2]
  template_HSV = cv2.split(template_hsv)
  v_value = template_HSV[2]
  v_count_list = np.zeros(256)
  for y in xrange(template_y):
    for x in xrange(template_x):
      v_count_list[v_value[y,x]] += 1
  f = open('template_hist.csv', 'w')
  csvWriter = csv.writer(f)
  csvWriter.writerow(v_count_list)


  #0-255のそれぞれについて分離度を求めて、１次元配列に格納
  separation_list = np.zeros(256)
  tmp_range = 15
  for i in xrange(0 + tmp_range ,255 - tmp_range):
    sum_k = 0
    sum_j = 0
    sum_l = 0
    sum_vj = 0
    sum_vl = 0
    sum_o1_numerator = 0
    sum_o2_numerator = 0
    for k in xrange(i-tmp_range,i+tmp_range):
      if k != i:
        sum_k += v_count_list[k]
    for j in xrange(i-tmp_range,i - 1):
      sum_j += v_count_list[j]
      sum_vj += v_count_list[j] * j
    for l in xrange(i + 1,i+tmp_range):
      sum_l += v_count_list[l]
      sum_vl += v_count_list[l] * l
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
  print 'separation_list\n' + str(separation_list)
  f_separation = open('separation_list.csv', 'w')
  csvWriter = csv.writer(f_separation)
  csvWriter.writerow(separation_list)

  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  p1, = ax1.plot(v_count_list, label = 'histgram', color = 'g')
  p2, = ax2.plot(separation_list ,label = 'separability', color = 'b')

  ax1.set_ylabel("histgram")
  ax2.set_ylabel("separability")
  ax1.yaxis.label.set_color(p1.get_color())
  ax2.yaxis.label.set_color(p2.get_color())

  lines = [p1, p2]
  ax1.legend(lines, [l.get_label() for l in lines])
  plt.savefig('template_hists.png')
  plt.close('all')

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
       b += 1
       label_list[a] = b
  print 'label_list\n' + str(label_list)
  f_label = open('label_list.csv', 'w')
  csvWriter = csv.writer(f_label)
  csvWriter.writerow(label_list)

  #ラベル配列をルックアップテーブルとして用い、領域ごとに濃淡値を分けた濃淡画像として出力する。
  label_template_img = np.zeros((template_y, template_x), np.uint8)
  for ty in xrange(template_y):
    for tx in xrange(template_x):
      label_template_img[ty,tx] = label_list[v_value[ty,tx]]
  print 'v_value' + str(v_value)
  print 'label_template_img' + str(label_template_img)
  cv2.imwrite('label_template_img.tif', label_template_img)

  return (label_template_img)




#ラベルの面積がパッチより小さい部分は不確定として再ラベリング
def re_label(long label_template_img):

  cdef long dst_data, s1, s2, ai, bi, label, dst_D, tables
  cdef int next_label, area_template, x, y, index, table, i


  dst_data = np.zeros((len(label_template_img), len(label_template_img[0])), np.uint16)
  #dst_data = tiff.imread('label_template_img.tif')
  next_label = 1
  area_template = len(label_template_img) * len(label_template_img[0])
  table = []
  for x in xrange(0,area_template):
    table.append(x)

  for y in xrange(len(label_template_img)):
    for x in xrange(len(label_template_img[0])):

      '''
      if label_template_img[y,x] == 0:
        dst_data[y,x] = 0
        continue
      '''
      if x == 0 and y == 0:
        dst_data[y,x] = next_label
        next_label += 1
        continue
      if y == 0:
        if label_template_img[y,x] == label_template_img[y,x-1]:
          dst_data[y,x] = dst_data[y,x-1]
        else:
          dst_data[y,x] = next_label
          next_label += 1
        continue
      if x == 0:
        if label_template_img[y,x] == label_template_img[y-1,x]:
          dst_data[y,x] = dst_data[y-1,x]
        else:
          dst_data[y,x] = next_label
          next_label += 1
        continue

      s1 = label_template_img[y,x-1]
      s2 = label_template_img[y-1,x]

      if label_template_img[y,x] == s2 and label_template_img[y,x] == s1:
        ai = dst_data[y,x-1]
        bi = dst_data[y-1,x]

        if ai != bi:
          while(table[ai] != ai):
            ai = table[ai]
          while(table[bi] != bi):
            bi = table[bi]
          if ai != bi:
            table[bi] = ai
        if ai < bi:
          dst_data[y,x] = ai
        else:
          dst_data[y,x] = bi
        continue

      if label_template_img[y,x] == s2:
        dst_data[y,x] = dst_data[y-1,x]
        continue
      if label_template_img[y,x] == s1:
        dst_data[y,x] = dst_data[y,x-1]
        continue
      dst_data[y,x] = next_label
      next_label += 1

  #print 'next_label = '+ str(next_label)
  #print 'dst_data = \n' + str(dst_data)
  #cv2.imwrite('dst_data_pre.tif', dst_data)

  '''
  for i in xrange(0,255):
    print 'pre'
    print 'table = ' +str(table[i])
  '''


  #ラベル再割当
  for x in xrange(0,next_label):
    index = x
    while(table[index] != index):
      index = table[index]
      #print 'index = ' +str(index)
    table[x] = index
  for i in xrange(0,next_label):
    print 'i = ' + str(i) +' table = ' +str(table[i])

  label = np.zeros(next_label, np.uint8)
  index = 1

  for x in xrange(0,next_label):
    if table[x] == x:
      label[x] = index
      index += 1
  print 'label = ' +str(label)

  for y in xrange(len(label_template_img)):
    for x in xrange(len(label_template_img[0])):
      #print 'table[dst_data[y,x]] = ' + str(table[dst_data[y,x]])
      #print 'dst_data[y,x] = ' + str(dst_data[y,x])
      dst_D = dst_data[y,x]
      tables = table[dst_D]
      dst_data[y,x] = label[tables]
  #print 'table[dst_data[y,x]] = ' + str(table)

  print 'dst_data = \n' + str(dst_data)

  #OpenCVの関数だと255以上の画素値の画像は保存できない
  #cv2.imwrite('dst_data.tif', dst_data)
  #tiff.imsave('dst_data.tif', dst_data)
  #while(True):
  #  cv2.imshow('dst_data',dst_data)

  return dst_data


#ラベル数:3,接点数:3の点をキーポイントとする。
def featureDetection(long label_template_img):

  cdef int y_size, x_size, n, one_n, x, y, scan_y, scan_x, sum_ave_keypoint, one_d_count, one_d
  cdef long scanning_filter, sum_one_dimention_scanning, sum_xy, sum_boundary, scanning_center, one_dimention_scanning


  #3*3の四角のフィルターでinput全体を走査して３次元配列をつくる
  y_size, x_size = label_template_img.shape[:2]
  n = 16 #n=3 or 5 or 7 or 11 ...
  if n < 16:
    one_n = n + 2*(n-1) + n-2 # n=3の時8,n=5の時16,n=7の時24,n=11の時32
  else:
    one_n = 16
  scanning_filter = np.zeros((n,n),np.uint16)
  scan_y = len(scanning_filter)
  scan_x = len(scanning_filter[0])
  #8*nの配列、one_dimention_scanningをまとめるリスト
  sum_one_dimention_scanning = np.empty((0,one_n),np.uint16)
  #2*nの配列、one_dimention_scanningの時のxyをまとめるリスト
  sum_xy = np.empty((0,2),np.uint8)
  sum_boundary = np.empty((0,1),np.uint8)
  sum_ave_keypoint = np.empty((0,1),int)
  for y in xrange(0,y_size - scan_y,3):
    for x in xrange(0,x_size - scan_x,3):
      one_dimention_scanning = np.zeros(one_n, np.uint16)
      scanning_center = []
      for s_y in xrange(scan_y):
        for s_x in xrange(scan_x):
          scanning_filter[s_y, s_x] = label_template_img[s_y + y, s_x + x]
          #周の要素だけ取り出して１次元配列にする
          if n == 3:
            one_dimention_scanning = [scanning_filter[0,0],scanning_filter[0,1],scanning_filter[0,2],scanning_filter[1,2],scanning_filter[2,2],scanning_filter[2,1],scanning_filter[2,0],scanning_filter[1,0]]
            scanning_center = [scanning_filter[1,1]]
          if n == 5:
            one_dimention_scanning = [scanning_filter[0,0],scanning_filter[0,1],scanning_filter[0,2],scanning_filter[0,3],scanning_filter[0,4],scanning_filter[1,4],scanning_filter[2,4],scanning_filter[3,4],scanning_filter[4,4],scanning_filter[4,3],scanning_filter[4,2],scanning_filter[4,1],scanning_filter[4,0],scanning_filter[3,0],scanning_filter[2,0],scanning_filter[1,0]]
            scanning_center = [scanning_filter[2,2]]
          if n == 7:
            one_dimention_scanning = [scanning_filter[0,0],scanning_filter[0,1],scanning_filter[0,2],scanning_filter[0,3],scanning_filter[0,4],scanning_filter[0,5],scanning_filter[0,6],scanning_filter[1,6],scanning_filter[2,6],scanning_filter[3,6],scanning_filter[4,6],scanning_filter[5,6],scanning_filter[6,6],scanning_filter[6,5],scanning_filter[6,4],scanning_filter[6,3],scanning_filter[6,2],scanning_filter[6,1],scanning_filter[6,0],scanning_filter[5,0],scanning_filter[4,0],scanning_filter[3,0],scanning_filter[2,0],scanning_filter[1,0]]
            scanning_center = [scanning_filter[3,3]]
          if n == 11:
            one_dimention_scanning = [scanning_filter[0,0],scanning_filter[0,1],scanning_filter[0,2],scanning_filter[0,3],scanning_filter[0,4],scanning_filter[0,5],scanning_filter[0,6],scanning_filter[0,7],scanning_filter[0,8],scanning_filter[0,9],scanning_filter[0,10],scanning_filter[1,10],scanning_filter[2,10],scanning_filter[3,10],scanning_filter[4,10],scanning_filter[5,10],scanning_filter[6,10],scanning_filter[7,10],scanning_filter[8,10],scanning_filter[9,10],scanning_filter[10,10],scanning_filter[10,9],scanning_filter[10,8],scanning_filter[10,7],scanning_filter[10,6],scanning_filter[10,5],scanning_filter[10,4],scanning_filter[10,3],scanning_filter[10,2],scanning_filter[10,1],scanning_filter[10,0],scanning_filter[9,0],scanning_filter[8,0],scanning_filter[7,0],scanning_filter[6,0],scanning_filter[5,0],scanning_filter[4,0],scanning_filter[3,0],scanning_filter[2,0],scanning_filter[1,0]]
            scanning_center = [scanning_filter[5,5]]
          if n == 16:
            one_dimention_scanning = [scanning_filter[0,3],scanning_filter[0,4],scanning_filter[1,5],scanning_filter[2,6],scanning_filter[3,6],scanning_filter[4,6],scanning_filter[5,5],scanning_filter[6,4],scanning_filter[6,3],scanning_filter[6,2],scanning_filter[5,1],scanning_filter[4,0],scanning_filter[3,0],scanning_filter[2,0],scanning_filter[1,1],scanning_filter[0,2]]
            scanning_center = [scanning_filter[3,3]]
      one_d_count = 0
      for one_d in xrange(len(one_dimention_scanning) - 1):
        #要素の切れ目を数える
        if one_dimention_scanning[one_d] != one_dimention_scanning[one_d + 1]:
          one_d_count += 1
      #連続するラベルの数を数える
      counter =  Counter(one_dimention_scanning)
      word_list = []
      cnt_list = []
      for word, cnt in counter.most_common():
        word_list.append(word)
        cnt_list.append(cnt)
      #print 'word_list' + str(word_list)
      #print 'cnt_list' + str(cnt_list)
      #要素の切れ目が１の時
      if one_d_count == 1:
        element_check = []
        element_check = set(one_dimention_scanning)
        #要素数が2かどうかチェック、短い方のラベルが中心画素と同じ
        if len(element_check) == 2 and scanning_center == word_list[cnt_list.index(np.min(cnt_list))]:
          #短い方のラベルが120度以下の時
          if np.min(cnt_list) <= one_n / 3:
            #if np.min(cnt_list) >= 3:
              #print '切れ目１要素2 = ' + str(one_dimention_scanning)
              #print 'scanning_center = ' +str(scanning_center)
              ave_keypoint = math.ceil(np.sum(word_list) / len(word_list))
              sum_ave_keypoint = np.vstack((sum_ave_keypoint, ave_keypoint))
              sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
              sum_xy = np.vstack((sum_xy,[y + 3 ,x + 3]))
              sum_boundary = np.vstack((sum_boundary, [2]))
      #要素の切れ目が2の時
      elif one_d_count == 2:
        #要素数が３かどうかチェックする
        element_check = []
        element_check = set(one_dimention_scanning)
        if len(element_check) == 3:
          #連続するラベルの数が２以上の時
          #if np.min(cnt_list) >= 3:
            ave_keypoint = math.ceil(np.sum(word_list) / len(word_list))
            sum_ave_keypoint = np.vstack((sum_ave_keypoint, ave_keypoint))
            sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
            sum_xy = np.vstack((sum_xy,[y + 3,x + 3]))
            sum_boundary = np.vstack((sum_boundary, [3]))
            #print 'get   cnt_list = ' + str(cnt_list)
            #print 'one_dimention_scanning = ' + str(one_dimention_scanning)
        #要素の切れ目が2で最初と最後の要素がおなじ,短い方のラベルが中心画素と同じ
        if one_dimention_scanning[0] == one_dimention_scanning[-1] and scanning_center == word_list[cnt_list.index(np.min(cnt_list))]:
          #短い方のラベルが120度いかの時
          if np.min(cnt_list) <= one_n / 3:
            #if np.min(cnt_list) >= 3:
              #print '切れ目2要素2 = '+ str(one_dimention_scanning)
              #print 'scanning_center = ' +str(scanning_center)
              ave_keypoint = math.ceil(np.sum(word_list) / len(word_list))
              sum_ave_keypoint = np.vstack((sum_ave_keypoint, ave_keypoint))
              sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
              sum_xy = np.vstack((sum_xy,[y + 3,x + 3]))
              sum_boundary = np.vstack((sum_boundary, [2]))

      elif one_d_count == 3:
        if one_dimention_scanning[0] == one_dimention_scanning[-1]:
          #要素の切れ目が３の時は、最初と最後が同じなら要素数は３なので、キーポイントとする
          #連続するラベルの数が２以上の時
          #if np.min(cnt_list) >= 3:
            ave_keypoint = math.ceil(np.sum(word_list) / len(word_list))
            sum_ave_keypoint = np.vstack((sum_ave_keypoint, ave_keypoint))
            sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
            sum_xy = np.vstack((sum_xy,[y + 3,x + 3]))
            sum_boundary = np.vstack((sum_boundary, [3]))
            #print 'get  cnt_list = ' + str(cnt_list)
            #print 'one_dimention_scanning = ' + str(one_dimention_scanning)
            #print 'sum_ave_keypoint = ' + str(sum_ave_keypoint)
  return (sum_one_dimention_scanning, sum_xy, sum_boundary, sum_ave_keypoint)


#ラベル数:3,接点数:3の点をキーポイントとする。
def all_featureDetection(long label_template_img):

  cdef int y_size, x_size, n, one_n, x, y, scan_y, scan_x, sum_ave_keypoint, one_d_count, one_d
  cdef long scanning_filter, sum_one_dimention_scanning, sum_xy, sum_boundary, scanning_center, one_dimention_scanning


  #3*3の四角のフィルターでinput全体を走査して３次元配列をつくる
  y_size, x_size = label_template_img.shape[:2]
  n = 16 #n=3 or 5 or 7 or 11 ...
  if n < 16:
    one_n = n + 2*(n-1) + n-2 # n=3の時8,n=5の時16,n=7の時24,n=11の時32
  else:
    one_n = 16
  scanning_filter = np.zeros((n,n),np.uint16)
  scan_y = len(scanning_filter)
  scan_x = len(scanning_filter[0])
  #8*nの配列、one_dimention_scanningをまとめるリスト
  sum_one_dimention_scanning = np.empty((0,one_n),np.uint16)
  #2*nの配列、one_dimention_scanningの時のxyをまとめるリスト
  sum_xy = np.empty((0,2),np.uint8)
  sum_boundary = np.empty((0,1),np.uint8)
  sum_ave_keypoint = np.empty((0,1),int)
  for y in xrange(0,y_size - scan_y,3):
    for x in xrange(0,x_size - scan_x,3):
      one_dimention_scanning = np.zeros(one_n, np.uint16)
      scanning_center = []
      for s_y in xrange(scan_y):
        for s_x in xrange(scan_x):
          scanning_filter[s_y, s_x] = label_template_img[s_y + y, s_x + x]
          #周の要素だけ取り出して１次元配列にする
          if n == 3:
            one_dimention_scanning = [scanning_filter[0,0],scanning_filter[0,1],scanning_filter[0,2],scanning_filter[1,2],scanning_filter[2,2],scanning_filter[2,1],scanning_filter[2,0],scanning_filter[1,0]]
            scanning_center = [scanning_filter[1,1]]
          if n == 5:
            one_dimention_scanning = [scanning_filter[0,0],scanning_filter[0,1],scanning_filter[0,2],scanning_filter[0,3],scanning_filter[0,4],scanning_filter[1,4],scanning_filter[2,4],scanning_filter[3,4],scanning_filter[4,4],scanning_filter[4,3],scanning_filter[4,2],scanning_filter[4,1],scanning_filter[4,0],scanning_filter[3,0],scanning_filter[2,0],scanning_filter[1,0]]
            scanning_center = [scanning_filter[2,2]]
          if n == 7:
            one_dimention_scanning = [scanning_filter[0,0],scanning_filter[0,1],scanning_filter[0,2],scanning_filter[0,3],scanning_filter[0,4],scanning_filter[0,5],scanning_filter[0,6],scanning_filter[1,6],scanning_filter[2,6],scanning_filter[3,6],scanning_filter[4,6],scanning_filter[5,6],scanning_filter[6,6],scanning_filter[6,5],scanning_filter[6,4],scanning_filter[6,3],scanning_filter[6,2],scanning_filter[6,1],scanning_filter[6,0],scanning_filter[5,0],scanning_filter[4,0],scanning_filter[3,0],scanning_filter[2,0],scanning_filter[1,0]]
            scanning_center = [scanning_filter[3,3]]
          if n == 11:
            one_dimention_scanning = [scanning_filter[0,0],scanning_filter[0,1],scanning_filter[0,2],scanning_filter[0,3],scanning_filter[0,4],scanning_filter[0,5],scanning_filter[0,6],scanning_filter[0,7],scanning_filter[0,8],scanning_filter[0,9],scanning_filter[0,10],scanning_filter[1,10],scanning_filter[2,10],scanning_filter[3,10],scanning_filter[4,10],scanning_filter[5,10],scanning_filter[6,10],scanning_filter[7,10],scanning_filter[8,10],scanning_filter[9,10],scanning_filter[10,10],scanning_filter[10,9],scanning_filter[10,8],scanning_filter[10,7],scanning_filter[10,6],scanning_filter[10,5],scanning_filter[10,4],scanning_filter[10,3],scanning_filter[10,2],scanning_filter[10,1],scanning_filter[10,0],scanning_filter[9,0],scanning_filter[8,0],scanning_filter[7,0],scanning_filter[6,0],scanning_filter[5,0],scanning_filter[4,0],scanning_filter[3,0],scanning_filter[2,0],scanning_filter[1,0]]
            scanning_center = [scanning_filter[5,5]]
          if n == 16:
            one_dimention_scanning = [scanning_filter[0,3],scanning_filter[0,4],scanning_filter[1,5],scanning_filter[2,6],scanning_filter[3,6],scanning_filter[4,6],scanning_filter[5,5],scanning_filter[6,4],scanning_filter[6,3],scanning_filter[6,2],scanning_filter[5,1],scanning_filter[4,0],scanning_filter[3,0],scanning_filter[2,0],scanning_filter[1,1],scanning_filter[0,2]]
            scanning_center = [scanning_filter[3,3]]
      one_d_count = 0
      for one_d in xrange(len(one_dimention_scanning) - 1):
        #要素の切れ目を数える
        if one_dimention_scanning[one_d] != one_dimention_scanning[one_d + 1]:
          one_d_count += 1
      #連続するラベルの数を数える
      counter =  Counter(one_dimention_scanning)
      word_list = []
      cnt_list = []
      for word, cnt in counter.most_common():
        word_list.append(word)
        cnt_list.append(cnt)
      #要素の切れ目が１の時
      if one_d_count == 1:
        element_check = []
        element_check = set(one_dimention_scanning)
        #要素数が2かどうかチェック、短い方のラベルが中心画素と同じ
        if len(element_check) == 2:
              ave_keypoint = math.ceil(np.sum(word_list) / len(word_list))
              sum_ave_keypoint = np.vstack((sum_ave_keypoint, ave_keypoint))
              sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
              sum_xy = np.vstack((sum_xy,[y + 3 ,x + 3]))
              sum_boundary = np.vstack((sum_boundary, [2]))
      #要素の切れ目が2の時
      elif one_d_count == 2:
        #要素数が３かどうかチェックする
        element_check = []
        element_check = set(one_dimention_scanning)
        if len(element_check) == 3:
            ave_keypoint = math.ceil(np.sum(word_list) / len(word_list))
            sum_ave_keypoint = np.vstack((sum_ave_keypoint, ave_keypoint))
            sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
            sum_xy = np.vstack((sum_xy,[y + 3,x + 3]))
            sum_boundary = np.vstack((sum_boundary, [3]))
        #要素の切れ目が2で最初と最後の要素がおなじ,短い方のラベルが中心画素と同じ
        if one_dimention_scanning[0] == one_dimention_scanning[-1]:
              ave_keypoint = math.ceil(np.sum(word_list) / len(word_list))
              sum_ave_keypoint = np.vstack((sum_ave_keypoint, ave_keypoint))
              sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
              sum_xy = np.vstack((sum_xy,[y + 3,x + 3]))
              sum_boundary = np.vstack((sum_boundary, [2]))

      elif one_d_count == 3:
        if one_dimention_scanning[0] == one_dimention_scanning[-1]:
            ave_keypoint = math.ceil(np.sum(word_list) / len(word_list))
            sum_ave_keypoint = np.vstack((sum_ave_keypoint, ave_keypoint))
            sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
            sum_xy = np.vstack((sum_xy,[y + 3,x + 3]))
            sum_boundary = np.vstack((sum_boundary, [3]))
  return (sum_one_dimention_scanning, sum_xy, sum_boundary, sum_ave_keypoint)



#inputとtemplateのバイナリ化
'''
全てのラベルの内、パッチ内にどのラベルがあるかで0,1でバイナリ化する。
ラベルの数、順番の情報は使わない。（回転不変、拡大縮小不変のため）
'''
def featuredescription(int sum_scanning, int sum_xy, long dst_data):

  cdef long one_d_dst_data, one_d_dst, keypoint_binary
  cdef int s ,x, y

  #dst_dataを1次元配列にする
  one_d_dst_data = []
  for s in dst_data:
    one_d_dst_data.extend(s)
  #one_d_dst_dataの最大値分の連番リストを作る
  one_d_dst = np.arange(0,np.max(one_d_dst_data),1)
  #dst_dataの種類を数える
  element_check_label = []
  element_check_label = set(one_d_dst)
  print 'element check label = ' + str(element_check_label)
  #label_listの種類の長さ分の１次元リストを作る（keypoint_binary）
  keypoint_binary = np.zeros((len(sum_scanning),len(element_check_label)+1),np.uint16)
  #one_dimention_scanningの頭から走査してその値のkeypoint_binaryのインデックスに＋１する
  #print 'sum_scanning\n' + str(sum_scanning)
  #print 'len(sum_scanning[0])' + str(len(sum_scanning[0]))
  for y in xrange(len(sum_scanning)):
    for x in xrange(len(sum_scanning[0])):
      #すでに１が入っていたらone_dimention_scanningのカーソルを次に渡す
      if keypoint_binary[y,sum_scanning[y,x]] == 0:
        keypoint_binary[y,sum_scanning[y,x]] = 1
  #keypoint_binaryとsum_xyを二次元リストに保存する
  #print 'keypoint_binary\n' + str(keypoint_binary)
  f_template_key = open('template_keypoint_binary.csv', 'w')
  csvWriter = csv.writer(f_template_key, lineterminator='\n')
  csvWriter.writerows(keypoint_binary)

  f_template_xy = open('template_keypoint_xy.csv', 'w')
  csvWriter = csv.writer(f_template_xy, lineterminator='\n')
  csvWriter.writerows(sum_xy)


#main
def template_main():

  cdef long template[[]], template_hsv[[]], template_detection[[]],label_template_img[[]]
  cdef int one_dimention_scanning, sum_xy, sum_boundary[], f_boundary, xy[], change_xy, add_template, template_change_xy, top_left_template, bottomo_right_template, add_template, add_top_left_template, add_bottom_right_template, sum_ave_keypoint, f_ave_key


  template = cv2.imread('sample.png',1)
  template_hsv = cv2.imread('sample_gray.png',1)
  #dst_data = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

  template_detection = cv2.imread('template_detection.tif',1)

  #template_hsv = rgb2Hsv(template)
  label_template_img = template_splitRegion(template_hsv)
  #dst_data = re_label(label_template_img)
  sum_one_dimention_scanning, sum_xy, sum_boundary, sum_ave_keypoint = featureDetection(label_template_img)
  #sum_one_dimention_scanning, sum_xy, sum_boundary, sum_ave_keypoint = all_featureDetection(label_template_img)


  f_boundary = open('template_boundary.csv', 'w')
  csvWriter = csv.writer(f_boundary, lineterminator='\n')
  csvWriter.writerows(sum_boundary)

  f_ave_key = open('template_ave_key.csv', 'w')
  csvWriter = csv.writer(f_ave_key, lineterminator='\n')
  csvWriter.writerows(sum_ave_keypoint)

  for i in xrange(len(sum_xy)):
    xy = sum_xy[i]
    change_xy = [xy[1],xy[0]]
    add_template = tuple(change_xy)
    template_change_xy = [xy[1],xy[0]]

    top_left_template = [xy[1] - 3, xy[0] - 3]
    bottom_right_template = [xy[1] + 3, xy[0] + 3]
    add_template = tuple(template_change_xy)

    add_top_left_template = tuple(top_left_template)
    add_bottom_right_template = tuple(bottom_right_template)
    if sum_boundary[i] == 2:
      #blue [255,0,0]
      #cv2.circle(template,add_template,1,(255,0,0),-1)
      #template[xy[0],xy[1]] = [255,0,0]
      template_detection[xy[0],xy[1]] = [0,0,255]
      #cv2.circle(template,add_template,3,(255,0,0),-1)
      #cv2.rectangle(template,add_top_left_template,add_bottom_right_template,(255,0,0), 1)
      cv2.rectangle(template_detection,add_top_left_template,add_bottom_right_template,(0,0,255), 1)
    elif sum_boundary[i] == 3:
      #green [0,255,0]
      #cv2.circle(template,add_template,1,(0,255,0),-1)
      template_detection[xy[0],xy[1]] = [0,255,0]
      #template[xy[0],xy[1]] = [0,255,0]
      #cv2.circle(template,add_template,3,(0,255,0),-1)
      cv2.rectangle(template_detection,add_top_left_template,add_bottom_right_template,(0,255,0), 1)
      #cv2.rectangle(template,add_top_left_template,add_bottom_right_template,(0,255,0), 1)
  #cv2.imwrite('template_detection.tif', template)
  cv2.imwrite('template_detection.tif', template_detection)


  #featuredescription(sum_one_dimention_scanning, sum_xy, dst_data)
