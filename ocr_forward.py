import tensorflow as tf
import os
import glob
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
import ocr_generated

conv1_filter=32
conv2_filter=64
conv3_filter=128
conv4_filter=256

def get_weight(shape, regularizer):
    #参数w初始化，并且对w进行正则化处理，防止模型过拟合
    w = tf.Variable(tf.truncated_normal((shape), stddev=0.1, dtype=tf.float32))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) 
    return w

def get_bias(shape): 
    #参数b初始化
    b = tf.Variable(tf.constant(0., shape=shape, dtype=tf.float32))  
    return b

def conv2d(x,w): 
    #卷积层函数tf.nn.conv2d
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, kernel_size):
    #池化层函数，在池化层采用最大池化，有效的提取特征
    return tf.nn.max_pool(x, ksize=kernel_size, strides=kernel_size, padding='VALID')

def forward(x, train, regularizer):
    #前向传播中共使用了四层神经网络
    #第一层卷积层和池化层实现
    conv1_w = get_weight([3, 3, 1, conv1_filter], regularizer)
    conv1_b = get_bias([conv1_filter])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b)) 
    pool1 = max_pool_2x2(relu1, [1,2,2,1])
    
    #通过keep_prob参数控制drop_out函数对神经元的筛选
    if train:
        keep_prob = 0.6     #防止过拟合
    else:
        keep_prob = 1.0
    #第二层卷积层和池化层实现
    conv2_w = get_weight([5, 5, conv1_filter, conv2_filter], regularizer)
    conv2_b = get_bias([conv2_filter])
    conv2 = conv2d(tf.nn.dropout(pool1, keep_prob), conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b)) 
    pool2 = max_pool_2x2(relu2, [1,2,1,1])
    #第三层卷积层和池化层
    conv3_w = get_weight([5, 5, conv2_filter, conv3_filter], regularizer)
    conv3_b = get_bias([conv3_filter])
    conv3 = conv2d(tf.nn.dropout(pool2, keep_prob), conv3_w)
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_b)) 
    pool3 = max_pool_2x2(relu3, [1,4,2,1])
    #第四层卷积层和池化层
    conv4_w = get_weight([5, 5, conv3_filter, conv4_filter], regularizer)
    conv4_b = get_bias([conv4_filter])
    conv4 = conv2d(tf.nn.dropout(pool3, keep_prob), conv4_w)
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_b)) 
    pool4 = max_pool_2x2(relu4, [1,7,1,1])
    
    rnn_inputs=tf.reshape(tf.nn.dropout(pool4,keep_prob),[-1,256,conv4_filter])
    
    num_hidden=512
    num_classes=len(ocr_generated.charactersNo)+1
    W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
    
    #前向传播、反向传播，利用双向LSTM长时记忆循环网络
    #seq_len = tf.placeholder(tf.int32, shape=[None])
    #labels=tf.sparse_placeholder(tf.int32, shape=[None,2])
    cell_fw = tf.nn.rnn_cell.LSTMCell(num_hidden>>1, state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.LSTMCell(num_hidden>>1, state_is_tuple=True)
    #outputs_fw_bw: (output_fw, output_bw) 是(output_fw, output_bw)的元组
    outputs_fw_bw, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, dtype=tf.float32)
    #tf.contat 连接前向和反向得到的结果，在指定维度上进行连接
    outputs1 = tf.concat(outputs_fw_bw, 2)
    
    shape = tf.shape(x)
    batch_s, max_timesteps = shape[0], shape[1]
    outputs = tf.reshape(outputs1, [-1, num_hidden])
    #全连接层实现
    logits0 = tf.matmul(tf.nn.dropout(outputs,keep_prob), W) + b
    logits1 = tf.reshape(logits0, [batch_s, -1, num_classes])
    logits = tf.transpose(logits1, (1, 0, 2))
    y = tf.cast(logits, tf.float32)
    return y
