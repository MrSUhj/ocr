import tensorflow as tf
import ocr_forward
import ocr_generated
import os
import glob
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter

REGULARIZER = 0.0001 
graphSize = (112,1024)
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "ocr_model"


def transform(im, flag=True):
    '''
    将传入的图片进行预处理：对图像进行图像缩放和数据增强
    Args:
        im :　传入的待处理的图片
    Return：
        graph : 返回经过预处理的图片
    #random.uniform(a, b)随机产生[a, b)之间的一个浮点数
    '''
    graph=np.zeros(graphSize[1]*graphSize[0]*1).reshape(graphSize[0],graphSize[1],1)
    deltaX=0
    deltaY=0
    ratio=1.464
    if flag:
        lowerRatio=max(1.269,im.size[1]*1.0/graphSize[0],im.size[0]*1.0/graphSize[1])
        upperRatio=max(lowerRatio,2.0)
        ratio=random.uniform(lowerRatio,upperRatio)
        deltaX=random.randint(0,int(graphSize[0]-im.size[1]/ratio))
        deltaY=random.randint(0,int(graphSize[1]-im.size[0]/ratio))
    else:
        ratio=max(1.464,im.size[1]*1.0/graphSize[0],im.size[0]*1.0/graphSize[1])
        deltaX=int(graphSize[0]-im.size[1]/ratio)>>1
        deltaY=int(graphSize[1]-im.size[0]/ratio)>>1
    height=int(im.size[1]/ratio)
    width=int(im.size[0]/ratio)
    data = im.resize((width,height),Image.ANTIALIAS).getdata()
    data = 1-np.asarray(data,dtype='float')/255.0
    data = data.reshape(height,width)
    graph[deltaX:deltaX+height,deltaY:deltaY+width,0]=data
    return graph

def create_sparse(Y,dtype=np.int32):
    '''
    对txt文本转化出来的数字序列Y作进一步的处理
    Args:
        Y
    Return:
        indices: 数组Y下标索引构成的新数组
        values: 下标索引对应的真实的数字码
        shape
    '''
    indices = []
    values = []
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            indices.append((i,j))
            values.append(Y[i][j])

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(Y), np.asarray(indices).max(0)[1] + 1], dtype=np.int64) #[64,180]

    return (indices, values, shape)

def backward():
    x = tf.placeholder(tf.float32, shape=[None, graphSize[0], graphSize[1],1])
    y = ocr_forward.forward(x, True, REGULARIZER)
    #y_: 表示真实标签数据
    #Y : 从文本中读取到的标签数据，训练时传给y_
    #y : 神经网络预测的标签

    global_step = tf.Variable(0, trainable=False)#全局步骤计数
    seq_len = tf.placeholder(tf.int32, shape=[None])
    y_ = tf.sparse_placeholder(tf.int32, shape=[None,2])

    Imgs, Y = ocr_generated.get_data()
    #损失函数使用的ctc_loss函数
    loss = tf.nn.ctc_loss(y_, y, seq_len)
    cost = tf.reduce_mean(loss)
    #优化函数使用的是Adam算法
    optimizer1 = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(cost, global_step=global_step)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, global_step=global_step)
    width1_decoded, width1_log_prob=tf.nn.ctc_beam_search_decoder(y, seq_len, merge_repeated=False,beam_width=1)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(y, seq_len, merge_repeated=False)
    width1_acc = tf.reduce_mean(tf.edit_distance(tf.cast(width1_decoded[0], tf.int32), y_))
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y_))
    nBatchArray=np.arange(Y.shape[0])
    epoch=100
    batchSize=32
    saver=tf.train.Saver(max_to_keep=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)
    bestDevErr=100.0
    with sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        #saver.restore(sess, "model/model.ckpt")
        #print(outputs.get_shape())
        for ep in range(epoch):
            np.random.shuffle(nBatchArray)
            for i in range(0, Y.shape[0], batchSize):
                batch_output = create_sparse(Y[nBatchArray[i:i+batchSize]])
                X=[None]*min(Y.shape[0]-i,batchSize)
                for j in range(len(X)):
                    X[j]=transform(Imgs[nBatchArray[i+j]])
    
                feed_dict={x:X,seq_len :np.ones(min(Y.shape[0]-i,batchSize)) * 256, y_:batch_output}
                if ep<50:
                    sess.run(optimizer1, feed_dict=feed_dict)
                else:
                    sess.run(optimizer2, feed_dict=feed_dict)
                print(ep,i,"loss:",tf.reduce_mean(loss.eval(feed_dict=feed_dict)).eval(),"err:",tf.reduce_mean(width1_acc.eval(feed_dict=feed_dict)).eval())
            #saver.save(sess, "model/model.ckpt")
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

def main():
    backward()

if __name__ == '__main__':
    main()
