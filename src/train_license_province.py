#!/usr/bin/python3.5
# -&- coding: utf-8 -*-

import sys
import os
import time 
import random

import numpy as np 
import tensorflow as tf 
from model1 import *
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

#x = tf.placeholder(tf.float32,shape=[None,SIZE])
y_ = tf.placeholder(tf.float32,shape=[None,NUM_CLASSES])

#x_image = tf.reshape(x,[-1,WIDTH,HEIGHT,1])

# 获取数据总数量
def get_imagecount(parentpath):
    count = 0
    for i in range(0,NUM_CLASSES):
        dir = "%s%s"%(parentpath,i)
        # for rt,dirs,files in os.walk(dir):
        #     for filename in files:
        #         input_count += 1
        count += len(os.listdir(dir))
    return count
# 传入数据文件夹　生成与数据数量相对应的数据和标签    
def generate_data_label(count,parentdir):
    input_images = np.array([[0] * SIZE for i in range(count)])
    input_labels = np.array([[0] * NUM_CLASSES for i in range(count)])
    index = 0
    # 生成图片数据和标签
    for i in range(0,NUM_CLASSES):
        dir = "%s%s/"%(parentdir,i)
        for rt,dirs,files in os.walk(dir):
            for filename in files:
                filename = dir + filename 
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(height):
                    for w in range(0,width):
                        if img.getpixel((w, h)) > 230:
                            input_images[index][w+h*width] = 0
                        else:
                            input_images[index][w+h*width] = 0
                input_labels[index][i] = 1
                index += 1
    return input_images,input_labels

if __name__ == "__main__" and sys.argv[1] == "train":
    
    # =========================训练数据处理=============================
    #　获取图片总数
    dirprexiftrain = "/home/mxq/Code/TensorFlowPlateRconginze/data/tf_car_license_dataset/train_images/training-set/chinese-characters/"
    input_count = get_imagecount(dirprexiftrain)
    input_images,input_labels = generate_data_label(input_count,dirprexiftrain)
    # 定义对应的维数和各维长度的数组
    
    # ==========================验证数据处理=============================
    dirprexifvalidation = "/home/mxq/Code/TensorFlowPlateRconginze/data/tf_car_license_dataset/train_images/validation-set/chinese-characters/"
    val_count = get_imagecount(dirprexifvalidation)
    val_images,val_labels = generate_data_label(val_count,dirprexifvalidation)

    with tf.Session() as sess:
        # 定义优化器和训练op
        # readout层
        W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1), name="W_fc2")
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b_fc2")
        
        # 定义优化器和训练op
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 初始化saver
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        time_elapsed = time.time() - time_begin
        print("读取图片文件耗费时间：%d秒" % time_elapsed)
        time_begin = time.time()

        print ("一共读取了 %s 个训练图像， %s 个标签" % (input_count, input_count))

        # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
        batch_size = 60
        iterations = iterations
        batches_count = int(input_count / batch_size)
        remainder = input_count % batch_size
        print ("训练数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count+1, batch_size, remainder))

        # 执行训练迭代
        for it in range(iterations):
            # 这里的关键是要把输入数组转为np.array
            for n in range(batches_count):
                train_step.run(feed_dict={x: input_images[n*batch_size:(n+1)*batch_size], y_: input_labels[n*batch_size:(n+1)*batch_size], keep_prob: 0.5})
            if remainder > 0:
                start_index = batches_count * batch_size
                train_step.run(feed_dict={x: input_images[start_index:input_count-1], y_: input_labels[start_index:input_count-1], keep_prob: 0.5})

            # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环
            iterate_accuracy = 0
            if it%5 == 0:
                iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
                print ('第 %d 次训练迭代: 准确率 %0.5f%%' % (it, iterate_accuracy*100))
                if iterate_accuracy >= 0.9999 and it >= 150:
                    break

        print ('完成训练!')
        time_elapsed = time.time() - time_begin
        print ("训练耗费时间：%d秒" % time_elapsed)
        time_begin = time.time()

        # 保存训练结果
        if not os.path.exists(SAVER_DIR):
            print ('不存在训练数据保存目录，现在创建保存目录')
            os.makedirs(SAVER_DIR)
        saver_path = saver.save(sess, "%smodel.ckpt"%(SAVER_DIR))
        
                             

