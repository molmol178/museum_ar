# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import csv
from collections import Counter

from template_topology_feature import re_label
from template_topology_feature import featureDetection
from template_topology_feature import all_featureDetection



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
  plt.close('all')

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

#inputとtemplateのバイナリ化
'''
全てのラベルの内、パッチ内にどのラベルがあるかで0,1でバイナリ化する。
ラベルの数、順番の情報は使わない。（回転不変、拡大縮小不変のため）
'''
def featureDescription(sum_scanning, dst_data):
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
  keypoint_binary = np.zeros((len(sum_scanning),len(element_check_label) + 1),np.uint16)
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
def featureMatching(input_keypoint_list, input_xy_list, input_img, template_img, input_boundary, input_ave_key):
  #inputとtemplateのkeypoint_binary,xy_listを読み込む
  template_keypoint_list = []
  template_keypoint = csv.reader(open('template_keypoint_binary.csv', 'r'))
  for temp_key in template_keypoint:
    template_keypoint_list.append(map(int,temp_key))
  print 'template_keypoint_list \n' + str(template_keypoint_list)
  print 'input_keypoint_list \n' + str(input_keypoint_list)



  template_xy_list = []
  template_xy = csv.reader(open('template_keypoint_xy.csv', 'r'))
  for temp_xy in template_xy:
    template_xy_list.append(map(int,temp_xy))
  #print 'template_xy_list \n' + str(template_xy_list)
  #print 'input_xy_list \n' + str(input_xy_list)



  template_boundary = []
  template_f = csv.reader(open('template_boundary.csv', 'r'))
  for temp_f in template_f:
    template_boundary.append(map(int,temp_f))
  #print 'template_boundary \n' + str(template_boundary)
  #print 'input_boundary \n' + str(input_boundary)

  template_ave_key = []
  template_ave = csv.reader(open('template_ave_key.csv', 'r'))
  for temp_ave in template_ave:
    template_ave_key.append(map(float,temp_ave))
  #print 'template_ave_key \n' + str(template_ave_key)
  #print 'input_ave_key \n' + str(input_ave_key)

  x_sum_img = len(input_img[0]) + len(template_img[0])
  sum_img = np.zeros((len(input_img), x_sum_img), np.uint8)
  sum_img = np.hstack((input_img, template_img))
  #template_add_xsize = []


  calc_list = np.zeros(len(template_keypoint_list[0]), np.uint8)

  #ハミング距離をとって0になったらマッチングとする
  for temp_y in xrange(0,len(template_keypoint_list)):
    for inp_y in xrange(0,len(input_keypoint_list)):

      temp_xy = template_xy_list[temp_y]
      inp_xy = input_xy_list[inp_y]

      #temp_bound = template_boundary[temp_y]
      #inp_bound = input_boundary[inp_y]


      template_add_xsize = [temp_xy[1] + len(input_img[0]),temp_xy[0]]
      input_change_xy = [inp_xy[1],inp_xy[0]]

      top_left_input = [inp_xy[1] - 3, inp_xy[0] - 3]
      bottom_right_input = [inp_xy[1] + 3, inp_xy[0] + 3]

      top_left_template = [temp_xy[1] + len(input_img[0]) - 3, temp_xy[0] - 3]
      bottom_right_template = [temp_xy[1] + len(input_img[0]) + 3, temp_xy[0] + 3]

      add_input = tuple(input_change_xy)
      add_template = tuple(template_add_xsize)

      add_top_left_input = tuple(top_left_input)
      add_bottom_right_input = tuple(bottom_right_input)

      add_top_left_template = tuple(top_left_template)
      add_bottom_right_template = tuple(bottom_right_template)

      calc_list = template_keypoint_list[temp_y] - input_keypoint_list[inp_y]
      calc_ave_key = template_ave_key[temp_y] - input_ave_key[inp_y]
      #print 'calc_list = ' + str(calc_list)
      #print 'calc_ave_key = '+ str(calc_ave_key)
      #print 'template_boundary' + str(template_boundary[temp_y])
      #print 'input_boundary' + str(input_boundary[inp_y])
      if all(x == 0 for x in calc_list) == True and calc_ave_key == 0:
        #マッチングした時のkeypoint_bineryと同じ場所のx,yの値を画像平面上にマッピングし、結果を出力
        print 'get match'
        #print 'calc_list = ' + str(calc_list)
        cv2.circle(sum_img,add_input,1,(0,0,255),-1)
        cv2.circle(sum_img,add_template,1,(0,0,255),-1)
        cv2.rectangle(sum_img,add_top_left_input,add_bottom_right_input,(0,0,255), 3)
        cv2.rectangle(sum_img,add_top_left_template,add_bottom_right_template,(0,0,255), 3)
        #cv2.line(sum_img,add_input,add_template,(0,0,255),1)
        #print 'add_input = ' +str(add_input) + ' add_template = ' +str(add_template)

      '''
      elif all(y == 0 for y in calc_list) == False:
        print 'not match'
        if inp_bound == 2 and temp_bound == 2:
          #print 'inp bound 2'
          cv2.circle(sum_img,add_input,1,(255,0,0),-1)
          cv2.rectangle(sum_img,add_top_left_input,add_bottom_right_input,(255,0,0), 1)
          cv2.circle(sum_img,add_template,1,(255,0,0),-1)
          cv2.rectangle(sum_img,add_top_left_template,add_bottom_right_template,(255,0,0), 1)

        elif inp_bound == 2 and temp_bound == 3:
          cv2.circle(sum_img,add_input,1,(255,0,0),-1)
          cv2.rectangle(sum_img,add_top_left_input,add_bottom_right_input,(255,0,0), 1)
          cv2.circle(sum_img,add_template,1,(0,255,0),-1)
          cv2.rectangle(sum_img,add_top_left_template,add_bottom_right_template,(0,255,0), 1)


        elif inp_bound == 3 and temp_bound ==2:
          #print 'inp bound 3'
          cv2.circle(sum_img,add_input,1,(0,255,0),-1)
          cv2.rectangle(sum_img,add_top_left_input,add_bottom_right_input,(0,255,0), 1)
          cv2.circle(sum_img,add_template,1,(255,0,0),-1)
          cv2.rectangle(sum_img,add_top_left_template,add_bottom_right_template,(255,0,0), 1)

        elif inp_bound ==3 and temp_bound == 3:
          cv2.circle(sum_img,add_input,1,(0,255,0),-1)
          cv2.rectangle(sum_img,add_top_left_input,add_bottom_right_input,(0,255,0), 1)
          cv2.circle(sum_img,add_template,1,(0,255,0),-1)
          cv2.rectangle(sum_img,add_top_left_template,add_bottom_right_template,(0,255,0), 1)
      '''
  print 'quit calc'
  cv2.imwrite('sum_img.tif', sum_img)

#main
if __name__ == '__main__':
  input_img = cv2.imread('sample_perspective.png',1)
  input_hsv = cv2.imread('sample_perspective_gray.png',1)
  #input_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
  template = cv2.imread('template_sika.png',1)
  #template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

  input_detection = cv2.imread('input_img.tif',1)

  #input_hsv, template_hsv = rgb2Hsv(input_img, template)
  label_input_img = createInputLabelImage(input_hsv)
  #dst_data = re_label(label_input_img)
  sum_one_dimention_scanning, sum_xy, sum_boundary, sum_ave_keypoint = featureDetection(label_input_img)
  #sum_one_dimention_scanning, sum_xy, sum_boundary, sum_ave_keypoint = all_featureDetection(label_input_img)


  for i in xrange(len(sum_xy)):
    xy = sum_xy[i]
    change_xy = [xy[1],xy[0]]
    add_input = tuple(change_xy)
    input_change_xy = [xy[1],xy[0]]

    top_left_input = [xy[1] - 3, xy[0] - 3]
    bottom_right_input = [xy[1] + 3, xy[0] + 3]
    add_input = tuple(input_change_xy)

    add_top_left_input = tuple(top_left_input)
    add_bottom_right_input = tuple(bottom_right_input)
    if sum_boundary[i] == 2:
      #blue
      #cv2.circle(input_img,add_input,1,(255,0,0),-1)
      #input_img[xy[0], xy[1]] = [255,0,0]
      input_detection[xy[0], xy[1]] = [0,0,255]
      #cv2.circle(input_img,add_input,3,(255,0,0),-1)
      #cv2.rectangle(input_img,add_top_left_input,add_bottom_right_input,(255,0,0), 1)
      cv2.rectangle(input_detection,add_top_left_input,add_bottom_right_input,(0,0,255), 1)
    elif sum_boundary[i] == 3:
      #green
      #cv2.circle(input_img,add_input,1,(0,255,0),-1)
      #input_img[xy[0], xy[1]] = [0,255,0]
      input_detection[xy[0], xy[1]] = [0,255,0]
      #cv2.circle(input_img,add_input,3,(0,255,0),-1)
      #cv2.rectangle(input_img,add_top_left_input,add_bottom_right_input,(0,255,0), 1)
      cv2.rectangle(input_detection,add_top_left_input,add_bottom_right_input,(0,255,0), 1)
  #cv2.imwrite('input_img.tif', input_img)
  cv2.imwrite('input_img.tif', input_detection)

  #keypoint_binary = featureDescription(sum_one_dimention_scanning, dst_data)
  #featureMatching(keypoint_binary, sum_xy, input_img, template, sum_boundary, sum_ave_keypoint)
