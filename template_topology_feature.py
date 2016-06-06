# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import csv
from collections import Counter

input_img = cv2.imread('input_yaziri.png',1)
template = cv2.imread('template_yaziri.png',1)

#RGBからHSVに変換
def rgb2Hsv(input_img, template):
  input_hsv = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
  template_hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
  return (input_hsv, template_hsv)

#templateの領域をしきい値に応じて分割
def template_splitRegion(template_hsv):
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
  p1, = ax1.plot(v_count_list, label = 'template hist', color = 'g')
  p2, = ax2.plot(separation_list ,label = 'separation', color = 'b')

  ax1.set_ylabel("template hist")
  ax2.set_ylabel("separation")
  ax1.yaxis.label.set_color(p1.get_color())
  ax2.yaxis.label.set_color(p2.get_color())

  lines = [p1, p2]
  ax1.legend(lines, [l.get_label() for l in lines])
  plt.savefig('template_hists.png')

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

  re_label(label_template_img)
  return (label_template_img)




#ラベルの面積がパッチより小さい部分は不確定として再ラベリング
def re_label(label_template_img):
  #連番で上下からラベリングする（4近傍）
  label_scan = []
  re_label_scan = []
  re_label_template_img = np.zeros((len(label_template_img),len(label_template_img[0])),np.uint8)
  label_counter = 1
  for y in xrange(1, len(label_template_img) - 1):
    for x in xrange(1, len(label_template_img[0]) - 2):
      label_scan = [label_template_img[y,x],label_template_img[y,x + 1],label_template_img[y,x + 2],label_template_img[y + 1,x],label_template_img[y+ 1,x + 1]]
      if label_scan[4] != label_scan[0] and \
         label_scan[4] != label_scan[1] and \
         label_scan[4] != label_scan[2] and \
         label_scan[4] != label_scan[3] :
         #print 'new region!!! label_scan = ' + str(label_scan)
         re_label_template_img[y + 1, x + 1] = label_counter
         label_counter += 1
      if label_scan[4] == label_scan[0] or \
         label_scan[4] == label_scan[1] or \
         label_scan[4] == label_scan[2] or \
         label_scan[4] == label_scan[3] :
         #print 'old region!!! label_scan = ' + str(label_scan)
         re_label_template_img[y + 1, x + 1] = np.min(label_scan)
  cv2.imwrite('re_label_template_img.tif', re_label_template_img)

  #re_label画像全体をスキャンしてそれぞれのラベルのピクセル数をカウント
  one_d_re_label = []
  for s in re_label_template_img:
    one_d_re_label.extend(s)
  counter =  Counter(one_d_re_label)
  word_list = []
  cnt_list = []
  for word, cnt in counter.most_common():
    word_list.append(word)
    cnt_list.append(cnt)
  print 'word_list = ' + str(word_list)
  print 'cnt_list = ' + str(cnt_list)

  #パッチ（n*n）よりも小さい面積はラベルの座標を取得
  delete_list = np.zeros((0,2),np.uint8)
  for i in xrange(len(cnt_list)):
    if cnt_list[i] <= 50:
      for y in xrange(len(re_label_template_img)):
        for x in xrange(len(re_label_template_img[0])):
          if re_label_template_img[y,x] == word_list[i]:
            delete_list = np.vstack((delete_list, [y,x]))
  print 'delete_list = ' +str(delete_list)

  #元のラベル画像の座標を0にする（最後のdescriptionでラベル0の部分は処理しないようにスライスする）
  del_label = []
  deldel_label = []
  for de in xrange(0,len(delete_list)):
    #print 'delete list = ' + str(delete_list[de])
    del_label = delete_list[de]
    deldel_label == [del_label[1], del_label[0]]
    label_template_img[deldel_label] = 0
  cv2.imwrite('re_create_template_label_img.tif', label_template_img)

#ラベル数:3,接点数:3の点をキーポイントとする。
def featureDetection(label_template_img):
  #3*3の四角のフィルターでinput全体を走査して３次元配列をつくる
  y_size, x_size = label_template_img.shape[:2]
  n = 5 #n=3 or 5 or 7 or 11 ...
  one_n = n + 2*(n-1) + n-2 # n=3の時8,n=5の時16,n=7の時24,n=11の時32
  scanning_filter = np.zeros((n,n),np.uint8)
  scan_y = len(scanning_filter)
  scan_x = len(scanning_filter[0])
  #8*nの配列、one_dimention_scanningをまとめるリスト
  sum_one_dimention_scanning = np.empty((0,one_n),np.uint8)
  #2*nの配列、one_dimention_scanningの時のxyをまとめるリスト
  sum_xy = np.empty((0,2),np.uint8)
  for y in xrange(0,y_size - scan_y,n):
    for x in xrange(0,x_size - scan_x,n):
      one_dimention_scanning = np.zeros(one_n, np.uint8)
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
            scanning_center = [scanning_filter[4,4]]
          if n == 11:
            one_dimention_scanning = [scanning_filter[0,0],scanning_filter[0,1],scanning_filter[0,2],scanning_filter[0,3],scanning_filter[0,4],scanning_filter[0,5],scanning_filter[0,6],scanning_filter[0,7],scanning_filter[0,8],scanning_filter[0,9],scanning_filter[0,10],scanning_filter[1,10],scanning_filter[2,10],scanning_filter[3,10],scanning_filter[4,10],scanning_filter[5,10],scanning_filter[6,10],scanning_filter[7,10],scanning_filter[8,10],scanning_filter[9,10],scanning_filter[10,10],scanning_filter[10,9],scanning_filter[10,8],scanning_filter[10,7],scanning_filter[10,6],scanning_filter[10,5],scanning_filter[10,4],scanning_filter[10,3],scanning_filter[10,2],scanning_filter[10,1],scanning_filter[10,0],scanning_filter[9,0],scanning_filter[8,0],scanning_filter[7,0],scanning_filter[6,0],scanning_filter[5,0],scanning_filter[4,0],scanning_filter[3,0],scanning_filter[2,0],scanning_filter[1,0]]
            scanning_center = [scanning_filter[5,5]]
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
            if np.min(cnt_list) >= 3:
              print '切れ目１要素2 = ' + str(one_dimention_scanning)
              print 'scanning_center = ' +str(scanning_center)
              sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
              sum_xy = np.vstack((sum_xy,[y + n/2 ,x + n/2]))
      #要素の切れ目が2の時
      elif one_d_count == 2:
        #要素数が３かどうかチェックする
        element_check = []
        element_check = set(one_dimention_scanning)
        if len(element_check) == 3:
          #連続するラベルの数が２以上の時
          if np.min(cnt_list) >= 3:
            sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
            sum_xy = np.vstack((sum_xy,[y + n/2,x + n/2]))
            #print 'get   cnt_list = ' + str(cnt_list)
            #print 'one_dimention_scanning = ' + str(one_dimention_scanning)
        #要素の切れ目が2で最初と最後の要素がおなじ,短い方のラベルが中心画素と同じ
        if one_dimention_scanning[0] == one_dimention_scanning[-1] and scanning_center == word_list[cnt_list.index(np.min(cnt_list))]:
          #短い方のラベルが120度いかの時
          if np.min(cnt_list) <= one_n / 3:
            if np.min(cnt_list) >= 3:
              print '切れ目2要素2 = '+ str(one_dimention_scanning)
              print 'scanning_center = ' +str(scanning_center)
              sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
              sum_xy = np.vstack((sum_xy,[y + n/2,x + n/2]))

      elif one_d_count == 3:
        if one_dimention_scanning[0] == one_dimention_scanning[-1]:
          #要素の切れ目が３の時は、最初と最後が同じなら要素数は３なので、キーポイントとする
          #連続するラベルの数が２以上の時
          if np.min(cnt_list) >= 3:
            sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
            sum_xy = np.vstack((sum_xy,[y + n/2,x + n/2]))
            #print 'get  cnt_list = ' + str(cnt_list)
            #print 'one_dimention_scanning = ' + str(one_dimention_scanning)
  return (sum_one_dimention_scanning, sum_xy)

#inputとtemplateのバイナリ化
'''
全てのラベルの内、パッチ内にどのラベルがあるかで0,1でバイナリ化する。
ラベルの数、順番の情報は使わない。（回転不変、拡大縮小不変のため）
'''
def featureDescription(sum_scanning, sum_xy):
  #label_list読み込み
  num_label = []
  label_CsvFile_description = csv.reader(open('label_list.csv', 'rb'),delimiter=',')
  for label in label_CsvFile_description:
    num_label = map(float,label)
  print 'num_label' + str(num_label)
  #label_listの要素だけをリスト
  #label_listの種類を数える
  element_check_label = []
  element_check_label = set(num_label)
  print 'element check label = ' + str(element_check_label)
  #label_listの種類の長さ分の１次元リストを作る（keypoint_binary）
  keypoint_binary = np.zeros((len(sum_scanning),len(element_check_label)),np.uint8)
  #one_dimention_scanningの頭から走査してその値のkeypoint_binaryのインデックスに＋１する
  print 'sum_scanning\n' + str(sum_scanning)
  print 'len(sum_scanning[0])' + str(len(sum_scanning[0]))
  for y in xrange(len(sum_scanning)):
    for x in xrange(len(sum_scanning[0])):
      #すでに１が入っていたらone_dimention_scanningのカーソルを次に渡す
      if keypoint_binary[y,sum_scanning[y,x]] == 0:
        keypoint_binary[y,sum_scanning[y,x]] = 1
  #keypoint_binaryとsum_xyを二次元リストに保存する
  print 'keypoint_binary\n' + str(keypoint_binary)
  f_template_key = open('template_keypoint_binary.csv', 'w')
  csvWriter = csv.writer(f_template_key, lineterminator='\n')
  csvWriter.writerows(keypoint_binary)

  f_template_xy = open('template_keypoint_xy.csv', 'w')
  csvWriter = csv.writer(f_template_xy, lineterminator='\n')
  csvWriter.writerows(sum_xy)


#main
input_hsv, template_hsv = rgb2Hsv(input_img, template)
label_template_img = template_splitRegion(template_hsv)
sum_one_dimention_scanning, sum_xy = featureDetection(label_template_img)
featureDescription(sum_one_dimention_scanning, sum_xy)
