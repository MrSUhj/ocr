import os
import glob
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
import ocr_forward
import tensorflow as tf


REGULARIZER = 0.0001 
graphSize = (112,1024)

def transform(im,flag=True):
    '''
    对image做预处理，将其形状强制转化成(112, 1024, 1)的ndarray对象并返回
    Args:
        im = Image Object
    Return:
        graph = Ndarray Object
    '''
    graph=np.zeros(graphSize[1]*graphSize[0]*1).reshape(graphSize[0],graphSize[1],1)
    deltaX=0
    deltaY=0
    ratio=1.464
    if flag:
        lowerRatio=max(1.269,im.size[1]*1.0/graphSize[0],im.size[0]*1.0/graphSize[1])
        upperRatio=max(lowerRatio,1.659)
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

def countMargin(v,minSum,direction=True):
    '''
    Args:
        v = list
        minSum = Int
    Return:
        v中比minSum小的项数
    '''
    if direction:
        for i in range(len(v)):
            if v[i]>minSum:
                return i
        return len(v)
    for i in range(len(v)-1,-1,-1):
        if v[i]>minSum:
            return len(v)-i-1
    return len(v)

def splitLine(seg,dataSum,h,maxHeight):
    i=0
    while i<len(seg)-1:
        if seg[i+1]-seg[i]<maxHeight:
            i+=1
            continue
        x=countMargin(dataSum[seg[i]:],3,True)
        y=countMargin(dataSum[:seg[i+1]],3,False)
        if seg[i+1]-seg[i]-x-y<maxHeight:
            i+=1
            continue
        idx=dataSum[seg[i]+x+h:seg[i+1]-h-y].argmin()+h
        if 0.33<=idx/(seg[i+1]-seg[i]-x-y)<=0.67:
            seg.insert(i+1,dataSum[seg[i]+x+h:seg[i+1]-y-h].argmin()+seg[i]+x+h)
        else:
            i+=1

def getLine(im,data,upperbound=8,lowerbound=25,threshold=30,h=40,minHeight=35,maxHeight=120,beginX=20,endX=-20,beginY=125,endY=1100,merged=True):
    '''
    
    '''
    dataSum=data[:,beginX:endX].sum(1)   #dataSum是一个一维向量
    lastPosition=beginY
    seg=[]
    flag=True
    cnt=0
    for i in range(beginY,endY):
        if dataSum[i]<=lowerbound:
            flag=True
            if dataSum[i]<=upperbound:
                cnt=0
                continue
        if flag:
            cnt+=1
            if cnt>=threshold:
                lineNo=np.argmin(dataSum[lastPosition:i])+lastPosition if threshold<=i-beginY else beginY
                if not merged or len(seg)==0 or lineNo-seg[-1]-countMargin(dataSum[seg[-1]:],5,True)-countMargin(dataSum[:lineNo],5,False)>minHeight:
                    seg.append(lineNo)
                else:
                    avg1=dataSum[max(0,seg[-1]-1):seg[-1]+2]
                    avg1=avg1.sum()/avg1.shape[0]
                    avg2=dataSum[max(0,lineNo-1):lineNo+2]
                    avg2=avg2.sum()/avg2.shape[0]
                    if avg1>avg2:
                        seg[-1]=lineNo
                lastPosition=i
                flag=False
    lineNo=np.argmin(dataSum[lastPosition:]>10)+lastPosition if threshold<i else beginY
    if not merged or len(seg)==0 or lineNo-seg[-1]-countMargin(dataSum[seg[-1]:],10,True)-countMargin(dataSum[:lineNo],10,False)>minHeight:
        seg.append(lineNo)
    else:
        avg1=dataSum[max(0,seg[-1]-1):seg[-1]+2]
        avg1=avg1.sum()/avg1.shape[0]
        avg2=dataSum[max(0,lineNo-1):lineNo+2]
        avg2=avg2.sum()/avg2.shape[0]
        if avg1>avg2:
            seg[-1]=lineNo
    splitLine(seg,dataSum,h,maxHeight)
    results=[]
    for i in range(0,len(seg)-1):
        results.append(im.crop((0,seg[i]+countMargin(dataSum[seg[i]:],0),im.size[0],seg[i+1]-countMargin(dataSum[:seg[i+1]],0,False))))
    return results

def calEditDistance(text1,text2):
    dp=np.asarray([0]*(len(text1)+1)*(len(text2)+1)).reshape(len(text1)+1,len(text2)+1)
    dp[0]=np.arange(len(text2)+1)
    dp[:,0]=np.arange(len(text1)+1)
    for i in range(1,len(text1)+1):
        for j in range(1,len(text2)+1):
            if text1[i-1]==text2[j-1]:
                dp[i,j]=dp[i-1,j-1]
            else:
                dp[i,j]=min(dp[i,j-1],dp[i-1,j],dp[i-1,j-1])+1
    return dp[-1,-1]

def test():
    x = tf.placeholder(tf.float32, shape=[None, graphSize[0], graphSize[1], 1])
    y = ocr_forward.forward(x, False, REGULARIZER)
    
    seq_len = tf.placeholder(tf.int32, shape=[None])
    labels=tf.sparse_placeholder(tf.int32, shape=[None,2])
    
    loss = tf.nn.ctc_loss(labels, y, seq_len)
    cost = tf.reduce_mean(loss)
    width1_decoded, width1_log_prob=tf.nn.ctc_beam_search_decoder(y, seq_len, merge_repeated=False,beam_width=1)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(y, seq_len, merge_repeated=False)
    width1_acc = tf.reduce_mean(tf.edit_distance(tf.cast(width1_decoded[0], tf.int32), labels))
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))
    saver=tf.train.Saver(max_to_keep=1)
    result=0
    imgFiles=glob.glob(os.path.join("test_img","*"))
    imgFiles.sort()
    txtFiles=glob.glob(os.path.join("test_txt","*"))
    txtFiles.sort()
    for i in range(len(imgFiles)):
        goldLines=[]
        fin=open(txtFiles[i])
        lines=fin.readlines()
        fin.close()
        for j in range(len(lines)):
            goldLines.append(lines[j])
        im = Image.open(imgFiles[i])
        width, height = im.size
        im = im.convert("L")
        data = im.getdata()
        data = 1-np.asarray(data,dtype='float')/255.0
        data = data.reshape(height,width)
        #getLine()将图片切割成一行一行的词条
        Imgs = getLine(im,data)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess=tf.Session(config=config)
        with sess:
            saver.restore(sess,"model/model.ckpt")
            X=[None]*len(Imgs)
            for j in range(len(Imgs)):
                X[j]=transform(Imgs[j],False)
            feed_dict={inputs:X,seq_len :np.ones(len(X)) * 256}
            predict = decoded[0].eval(feed_dict=feed_dict)
            j=0
            predict_text=""
            gold_text="".join(goldLines)
            for k in range(predict.dense_shape[0]):
                while j<len(predict.indices) and predict.indices[j][0]==k:
                    predict_text+=characters[predict.values[j]]
                    j+=1
                predict_text+="\n"
            predict_text=predict_text.rstrip("\n")
            print("predict_text:")
            print(predict_text)
            fout=open("predict%s%s.txt"%(os.sep,txtFiles[i][txtFiles[i].find(os.sep)+1:txtFiles[i].rfind('.')]),'w')
            fout.write(predict_text)
            fout.close()
            print("gold_text:")
            print(gold_text)
            cer=calEditDistance(predict_text,gold_text)*1.0/len(gold_text)
            print("预测正确率： ", end='')
            print(cer)
            print()
            result+=cer
    print("test composition err:",result*1.0/len(imgFiles))

def main():
    test()

if __name__ == '__main__':
    main()
