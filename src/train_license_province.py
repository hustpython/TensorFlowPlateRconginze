#!/usr/bin/python3.5
# -&- coding: utf-8 -*-

import sys
import os
import time 
import random

import numpy as np 
import tensorflow as tf 

from PIL import Image

SIZE = 1280 
WIDTH = 32 
HEIGHT = 40
NUM_CLASSES = 6
iterations = 300

SAVER_DIR = "train_saver/province/"
PROVINCES = ("京","闽","粤","苏","沪","浙")
nProvinceIndex = 0

time_begin = time.time()

x = tf.placeholder(tf.float32,shape=[None,SIZE])
y_ = tf.placeholder(tf.float32,shape=[None,NUM_CLASSES])

x_image = tf.reshape(x,[-1,WIDTH,HEIGHT,1])
# 定义卷积函数
def conv_layer(inputs,w,b,conv_strides,kernel_size,pool_strides,padding):
    L1_conv = tf.nn.conv2d(inputs,w,strides=conv_strides,padding=padding)
    L1_relu = tf.nn.relu(L1_conv+b)
    return tf.nn.max_pool(L1_relu,ksize=kernel_size,strides=pool_strides,padding="SAME")

# 定义全连接层函数
def full_connect(inputs,W,b):
    return tf.nn.relu(tf.matmul(inputs,W)+b)
# 获取数据总数量
def get_imagecount(parentpath):
    count = 0
    for i in range(0,NUM_CLASSES):
        dir = "%s%s/"%parentpath%i
        # for rt,dirs,files in os.walk(dir):
        #     for filename in files:
        #         input_count += 1
        count += len(os.listdir(dir))
    return count
# 传入数据文件夹　生成与数据数量相对应的数据和标签    
def generate_data_label(count,parentdir):
    input_images = np.array([0] * SIZE for i in range(count))
    input_labels = np.array([[0] * NUM_CLASSES for i in range(count)])
    index = 0
    # 生成图片数据和标签
    for i in range(0,NUM_CLASSES):
        dir = "%s%s/"%parentdir%i
        for rt,dirs,files in os.walk(dir):
            for filename in filename:
                filename = dir + filename 
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(height):
                    for w in range(0,width):
                        if img.getpiexl((w,h)) > 230:
                            input_images[index][w+h*width] = 0
                        else:
                            input_images[index][w+h*width] = 0
                input_labels[index][i] = 1
                index += 1
    return input_images,input_labels

if __name__ == "__main__" and sys.argv[1] == "train":
    
    # =========================训练数据处理=============================
    #　获取图片总数
    dirprexiftrain = "./data/train_image/trainging-set/chinese-characters/"
    input_count = get_imagecount(dirprexiftrain)
    input_images,input_labels = generate_data_label(input_count,dirprexiftrain)
    # 定义对应的维数和各维长度的数组
    
    # ==========================验证数据处理=============================
    dirprexifvalidation = "./data/train_image/validation-set/chinese-characters/"
    val_count = get_imagecount(dirprexifvalidation)
    val_images,val_labels = generate_data_label(val_count,dirprexifvalidation)

    with tf.Session() as sess:
        # 第一个卷积层
        W_conv1 = tf.Variable(tf.truncated_normal([8,8,1,16],stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1,shape=[16],name="b_conv1"))
        conv_strides = [1,1,1,1]
        kernel_size = [1,2,2,1]
        pool_strides = [1,2,2,1]
        L1_pool = conv_layer(x_image,W_conv1,b_conv1,conv_strides,kernel_size,pool_strides,padding="SAME")
        
        # 第二个卷积层
        # 初始化ｗ,满足正态分布
        W_conv2 = tf.Variable(tf.truncated_normal([5,5,16,32],stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1,shape=[32],name="b_conv2"))

        
                             

