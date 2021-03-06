#!/usr/bin/python3.5
# -*- coding: utf-8 -*-  
 
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
iterations = 500
 
SAVER_DIR = "train-saver/province/"
 
PROVINCES = ("京","闽","粤","苏","沪","浙")
nProvinceIndex = 0
 
time_begin = time.time()
 
 
# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
# shape = [None,SIZE] 第一维是自动计算的
# 32 × 40 = 1280
x = tf.placeholder(tf.float32, shape=[None, SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

# shape = (?,32,40,1)
x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])

 
# 定义卷积函数
# 输入大小input 步长stride 核大小kernelsize paddingsize
# 卷积的输出大小 outsize = (input + padding × 2  - Kersize) / stride + 1
# 举例 输入图片大小 7 × 7,卷积核大小 3*3，步长1 ，padding 0
# 输出大小 (7 + 2*1 - 3) / 1 + 1 = 7 × 7

# relu 大于 0 返回 x,小于 0 返回 0
# max_pool ,返回kernelsize中最大的元素
# padding "SAME" 补0和原图尺寸一样 ; VALID 不继续补０
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')
 
# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)
 
 
if __name__ =='__main__' and sys.argv[1]=='train':
    # 第一次遍历图片目录是为了获取图片总数
    input_count = 0
    for i in range(0,NUM_CLASSES):
        dir = '/home/mxq/Codes/TensorFlowPlateRconginze/data/tf_car_license_dataset/train_images/training-set/chinese-characters/%s/' % i           # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                input_count += 1
 
    # 定义对应维数和各维长度的数组
    # inputsize : inputcount * 1280
    input_images = np.array([[0]*SIZE for i in range(input_count)])
    # inputlabelsize : inputcount * NUMCLASS(6)
    input_labels = np.array([[0]*NUM_CLASSES for i in range(input_count)])
 
    # 第二次遍历图片目录是为了生成图片数据和标签
    index = 0
    for i in range(0,NUM_CLASSES):
        dir = '/home/mxq/Codes/TensorFlowPlateRconginze/data/tf_car_license_dataset/train_images/training-set/chinese-characters/%s/' % i          # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                        if img.getpixel((w, h)) > 230:
                            input_images[index][w+h*width] = 0
                        else:
                            input_images[index][w+h*width] = 1
                input_labels[index][i] = 1
                index += 1
    # inputimages = [[1,0,(省略1276),0,1]，[0,1,(省略1276),0,1],...,[0,1,(省略1276),0,1]]
    # inputlabels = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],...,[0,0,0,0,0,1]]

    # 对验证数据集进行类似操作 =============================== 

    # 第一次遍历图片目录是为了获取图片总数
    val_count = 0
    for i in range(0,NUM_CLASSES):
        dir = '/home/mxq/Codes/TensorFlowPlateRconginze/data/tf_car_license_dataset/train_images/validation-set/chinese-characters/%s/' % i           # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                val_count += 1
 
    # 定义对应维数和各维长度的数组
    val_images = np.array([[0]*SIZE for i in range(val_count)])
    val_labels = np.array([[0]*NUM_CLASSES for i in range(val_count)])
 
    # 第二次遍历图片目录是为了生成图片数据和标签
    index = 0
    for i in range(0,NUM_CLASSES):
        dir = '/home/mxq/Codes/TensorFlowPlateRconginze/data/tf_car_license_dataset/train_images/validation-set/chinese-characters/%s/' % i          # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = dir + filename
                img = Image.open(filename)
                width = img.size[0]
                height = img.size[1]
                for h in range(0, height):
                    for w in range(0, width):
                        # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                        if img.getpixel((w, h)) > 230:
                            val_images[index][w+h*width] = 0
                        else:
                            val_images[index][w+h*width] = 1
                val_labels[index][i] = 1
                index += 1
    
    # 建立会话管理
    with tf.Session() as sess:
        # 第一个卷积层
        # 初始化W参数 truncated_normal 截尾正态分布 返回距离均值不超过2倍标准差的随机数
        # 卷积核大小　8×８　原通道１　卷积核个数　１６
        # 偏置　16*1
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="W_conv1")
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_conv1")
        conv_strides = [1, 1, 1, 1]
        # pool 层
        # [batch, height, width, channels]
        kernel_size = [1, 2, 2, 1]
        # [1, stride,stride, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')
        # 第一层参数　(8×８*1+1) * 16 = 1040
        # 输出大小:(32-2)/2+1 = 16　(40-2)/2+1 = 20 16×２0

        # 第二个卷积层
        # 卷积核大小　5*5 ，上一层通道数１６，核个数３２
        # 偏置层大小　３２＊１
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name="W_conv2")
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv2")
        conv_strides = [1, 1, 1, 1]
        # pool 层
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')
        # 输出大小:
        # 第二层参数　(５×５*１６+1) * ３２ = １２８３２
        # 输出大小　(16 - 1) /1 +1 =16 ; (20 - 1) /1 +1 =20 16×２０

        # 全连接层
        # １６×２０＊３２
        W_fc1 = tf.Variable(tf.truncated_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20*32])
        # flatten 1 * ? * ? * 512 
        # 输出　512 个节点
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)
        # 输出层　参数　512 ×　(１６＊３２＊20 + 1)
        # dropout　防止过拟合
        # 给一定的概率keep_prob 让某一部分节点"不起作用",不参与更新（<1)
        # run和predice的时候keep_prob　＝　１
        keep_prob = tf.placeholder(tf.float32)
 
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
 
        # readout层
        # 输出层１＊５１２　×　５１２　×　６　＝　６
        W_fc2 = tf.Variable(tf.truncated_normal([512, NUM_CLASSES], stddev=0.1), name="W_fc2")
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b_fc2")
        # 输出层参数:(512+1)*6=3078


        # 定义优化器和训练op
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        # tf.nn.softmax_entropy_with_logits
        # labels/logits:[batchsize,numclasses]
        # 第一步:softmax 求属于某一类的概率:
        # softmax(x)i = exp(xi)/(sum(exp(xj)))
        # 输出向量:[batchsize,numclasses]
        # 第二步:
        # softmax的输出向量与样本的实际标签作一个交叉熵
        # Ｈy'(y) = - sum(yi'log(yi))
        # 其中　ｙi'指实际标签中第i个值[0,0,1,0,0]
        # yi就是softmax的输出向量[y1,y2,y3，...]中第i个元素的值
        # 显然yi越准确,结果的值越小(别忘记了前面的负号),最后求一个平均
        
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
        # tf.argmax()返回最大数值的下标 
        # dimension=0 按列找 
        # dimension=1 按行找 
    
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #print(accuracy)
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
                start_index = batches_count * batch_size;
                train_step.run(feed_dict={x: input_images[start_index:input_count-1], y_: input_labels[start_index:input_count-1], keep_prob: 0.5})
 
            # 每完成五次迭代，判断准确度是否已达到100%，达到则退出迭代循环
            iterate_accuracy = 0
            if it%5 == 0:
                iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
                print ('第 %d 次训练迭代: 准确率 %0.5f%%' % (it, iterate_accuracy*100))
                if iterate_accuracy >= 0.9999 or it >= 150:
                    break;
 
        print ('完成训练!')
        time_elapsed = time.time() - time_begin
        print ("训练耗费时间：%d秒" % time_elapsed)
        time_begin = time.time()
 
        # 保存训练结果
        if not os.path.exists(SAVER_DIR):
            print ('不存在训练数据保存目录，现在创建保存目录')
            os.makedirs(SAVER_DIR)
        saver_path = saver.save(sess, "%smodel.ckpt"%(SAVER_DIR))
 
 
 
if __name__ =='__main__' and sys.argv[1]=='predict':
    saver = tf.train.import_meta_graph("%smodel.ckpt.meta"%(SAVER_DIR))
    with tf.Session() as sess:
        model_file=tf.train.latest_checkpoint(SAVER_DIR)
        saver.restore(sess, model_file)
 
        # 第一个卷积层
        W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
        b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')
 
        # 第二个卷积层
        W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
        b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')
 
 
        # 全连接层
        W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
        b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20*32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)
 
 
        # dropout　防止过拟合
        # 给一定的概率keep_prob 让某一部分节点"不起作用",不参与更新（<1)
        # run和predice的时候keep_prob　＝　１
        keep_prob = tf.placeholder(tf.float32)
 
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
 
        # readout层
        W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
        b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")
 
        # 定义优化器和训练op
        conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 
        for n in range(6):
            path = "/home/mxq/Codes/TensorflowPlateRecognize/data/tf_car_license_dataset/test_images/characterChinese/%s.bmp" % (n)
            img = Image.open(path)
            width = img.size[0]
            height = img.size[1]
 
            img_data = [[0]*SIZE for i in range(1)]
            for h in range(0, height):
                for w in range(0, width):
                    if img.getpixel((w, h)) < 190:
                        img_data[0][w+h*width] = 1
                    else:
                        img_data[0][w+h*width] = 0
            
            result = sess.run(conv, feed_dict = {x: np.array(img_data), keep_prob: 1.0})
            max1 = 0
            max2 = 0
            max3 = 0
            max1_index = 0
            max2_index = 0
            max3_index = 0
            for j in range(NUM_CLASSES):
                if result[0][j] > max1:
                    max1 = result[0][j]
                    max1_index = j
                    continue
                if (result[0][j]>max2) and (result[0][j]<=max1):
                    max2 = result[0][j]
                    max2_index = j
                    continue
                if (result[0][j]>max3) and (result[0][j]<=max2):
                    max3 = result[0][j]
                    max3_index = j
                    continue
            
            nProvinceIndex = max1_index
            print ("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (PROVINCES[max1_index],max1*100, PROVINCES[max2_index],max2*100, PROVINCES[max3_index],max3*100))
            print ("省份简称是: %s" % PROVINCES[nProvinceIndex])
