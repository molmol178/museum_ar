# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import csv
from collections import Counter

input_img = cv2.imread('test_2.png',1)
template = cv2.imread('test_2.png',1)

#RGBからHSVに変換
def rgb2Hsv(input_img, template):
  input_hsv = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
  template_hsv = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
  return (input_hsv, template_hsv)


#inputのV値をtemplateのV値に補正する
def createInputLabelImage(input_hsv):
  input_y ,input_x = input_hsv.shape[:2]
  input_HSV = cv2.split(input_hsv)
  input_v_value = input_HSV[2]

  #v_count_listの読み込み
  v_count_list = []
  read_template_v_count = csv.reader(open('template_hist.csv', 'rb'),delimiter=',')
  for v_count in read_template_v_count:
    v_count_list = map(float,v_count)

  #inputのヒストグラム作成
  input_v_count_list = np.zeros(256)
  for y in xrange(input_y):
    for x in xrange(input_x):
      input_v_count_list[input_v_value[y,x]] += 1
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
  print 'ave_t ' + str(ave_t)
  print 'template_std_deviation ' + str(template_std_deviation)

  #inputの標準偏差と全体の平均を求める
  for i in xrange(0, 256):
    sum_vi += input_v_count_list[i] * i
    sum_i += input_v_count_list[i]
  ave_i = sum_vi / sum_i
  for ii in xrange(0, 256):
    sum_oi_numerator += input_v_count_list[ii] * (ii - ave_i)**2
  oi = sum_oi_numerator / sum_i
  input_std_deviation = math.sqrt(oi)
  print 'ave_i ' + str(ave_i)
  print 'input_std_deviation ' + str(input_std_deviation)

  #inputのtemplateに対する偏差値を求め補正結果を配列に格納
  correct_input_list = np.zeros(256)
  for iii in xrange(0, 256):
    correct_input_list[iii] = template_std_deviation * ((input_v_count_list[iii] - ave_i) / input_std_deviation) + ave_t
  print 'correct_input_list\n' + str(correct_input_list)
  f_corre_in = open('correct_input_hist.csv', 'w')
  csvWriter = csv.writer(f_corre_in)
  csvWriter.writerow(correct_input_list)

  plt.plot(input_v_count_list, label = "input hist")
  plt.plot(correct_input_list, label = "correct input")
  plt.legend()
  plt.savefig('input_hist.png')


  #ヒストグラムは確認のために必要なだけで、処理には必要ない。偏差値の処理は画素１つずつ行ってラベル画像を作る。
  correct_input_v_value = np.zeros((input_y, input_x), np.uint8)
  print 'input_v_value\n' + str(input_v_value)
  for co_in_y in xrange(input_y):
    for co_in_x in xrange(input_x):
      correct_input_v_value[co_in_y, co_in_x] = template_std_deviation * (input_v_value[co_in_y, co_in_x] - ave_i) / input_std_deviation + ave_t
  print 'correct_input_v_value\n' + str(correct_input_v_value)
  cv2.imwrite('correct_input_v_value.png', correct_input_v_value)

  #作成した領域ラベルリスト(label_list.csvを読み込む)を用いてinputラベル画像生成
  label_input_img = np.zeros((input_y, input_x), np.uint8)
  label_CsvFile = csv.reader(open('label_list.csv', 'rb'),delimiter=',')
  for label in label_CsvFile:
    num_label = map(float,label)
  print 'num_label' +str(num_label)
  for iy in xrange(input_y):
    for ix in xrange(input_x):
      label_input_img[iy,ix] = num_label[correct_input_v_value[iy,ix]]
  cv2.imwrite('label_input_img.tif', label_input_img)
  return (label_input_img)
  featureDetection(label_input_img)

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
            print '切れ目１要素2 = ' + str(one_dimention_scanning)
            print 'scanning_center = ' +str(scanning_center)
            sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
            sum_xy = np.vstack((sum_xy,[y,x]))
      #要素の切れ目が2の時
      elif one_d_count == 2:
        #要素数が３かどうかチェックする
        element_check = []
        element_check = set(one_dimention_scanning)
        if len(element_check) == 3:
          #連続するラベルの数が２以上の時
          #if np.min(cnt_list) >= 2:
            sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
            sum_xy = np.vstack((sum_xy,[y,x]))
            #print 'get   cnt_list = ' + str(cnt_list)
            #print 'one_dimention_scanning = ' + str(one_dimention_scanning)
        #要素の切れ目が2で最初と最後の要素がおなじ,短い方のラベルが中心画素と同じ
        if one_dimention_scanning[0] == one_dimention_scanning[-1] and scanning_center == word_list[cnt_list.index(np.min(cnt_list))]:
          #短い方のラベルが120度以下の時
          if np.min(cnt_list) <= one_n / 3:
            print '切れ目2要素2 = '+ str(one_dimention_scanning)
            print 'scanning_center = ' +str(scanning_center)
            sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
            sum_xy = np.vstack((sum_xy,[y,x]))

      elif one_d_count == 3:
        if one_dimention_scanning[0] == one_dimention_scanning[-1]:
          #要素の切れ目が３の時は、最初と最後が同じなら要素数は３なので、キーポイントとする
          #連続するラベルの数が２以上の時
          #if np.min(cnt_list) >= 2:
            sum_one_dimention_scanning = np.vstack((sum_one_dimention_scanning,one_dimention_scanning))
            sum_xy = np.vstack((sum_xy,[y,x]))
            #print 'get  cnt_list = ' + str(cnt_list)
            #print 'one_dimention_scanning = ' + str(one_dimention_scanning)
  return (sum_one_dimention_scanning, sum_xy)

#inputとtemplateのバイナリ化
'''
全てのラベルの内、パッチ内にどのラベルがあるかで0,1でバイナリ化する。
ラベルの数、順番の情報は使わない。（回転不変、拡大縮小不変のため）
'''
def featureDescription(sum_scanning):
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
  #keypoint_binaryを二次元リストに保存する
  print 'keypoint_binary\n' + str(keypoint_binary)
  return (keypoint_binary)

#マッチング
'''
inputとtemplateのバイナリのハミング距離を取ってマッチング
'''
def featureMatching(input_keypoint_list, input_xy_list, input_img, template_img):
  #inputとtemplateのkeypoint_binary,xy_listを読み込む
  template_keypoint_list = []
  template_keypoint = csv.reader(open('template_keypoint_binary.csv', 'r'))
  for temp_key in template_keypoint:
    template_keypoint_list.append(map(int,temp_key))
  template_xy_list = []
  template_xy = csv.reader(open('template_keypoint_xy.csv', 'r'))
  for temp_xy in template_xy:
    template_xy_list.append(map(int,temp_xy))
  x_sum_img = len(input_img[0]) + len(template_img[0])
  sum_img = np.zeros((len(input_img), x_sum_img), np.uint8)
  sum_img = np.hstack((input_img, template_img))
  template_add_xsize = []
  #ハミング距離をとって0になったらマッチングとする
  for temp_y in xrange(0,len(template_keypoint_list)):
    for inp_y in xrange(0,len(input_keypoint_list)):
      calc_list = template_keypoint_list[temp_y] - input_keypoint_list[inp_y]
      #print 'template_xy' + str(template_xy_list[temp_y])
      #print 'input_xy' + str(input_xy_list[inp_y])
      if  all((x == 0 for x in calc_list)) == True:
        #マッチングした時のkeypoint_bineryと同じ場所のx,yの値を画像平面上にマッピングし、結果を出力
        #print 'get match'
        temp_xy = template_xy_list[temp_y]
        inp_xy = input_xy_list[inp_y]
        template_add_xsize = [temp_xy[1] + len(input_img[0]),temp_xy[0]]
        input_change_xy = [inp_xy[1],inp_xy[0]]
        add_input = tuple(input_change_xy)
        add_template = tuple(template_add_xsize)
        cv2.circle(sum_img,add_input,3,(0,0,0),-1)
        cv2.circle(sum_img,add_template,3,(0,0,255),-1)
        #cv2.line(sum_img,add_input,add_template,(255,255,0),1)
        #print 'add_input = ' +str(add_input) + ' add_template = ' +str(add_template)
  print 'quit calc'
  cv2.imwrite('sum_img.tif', sum_img)

#main
input_hsv, template_hsv = rgb2Hsv(input_img, template)
label_input_img = createInputLabelImage(input_hsv)
sum_one_dimention_scanning, sum_xy = featureDetection(label_input_img)
keypoint_binary = featureDescription(sum_one_dimention_scanning)
featureMatching(keypoint_binary, sum_xy, input_img, template)
