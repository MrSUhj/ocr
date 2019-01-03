import os
import glob
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter

#记录一个问题： tf.placeholder 报错InvalidArgumentError: You must feed a value for placeholder tensor 'inputs/x_input'
#chr函数： 将数字转化成字符
#ord函数： 将字符转化成数字
#characterNo字典：a-z, A-Z, 0-10, " .,?\'-:;!/\"<>&(+" 为key分别对应值是0-25,26-51,52-61,62...
#characters列表： 存储的是cahracterNo字典的key
#建立characterNo字典的意思是： 为了将之后手写体对应的txt文件中的句子转化成 数字编码便于存储和运算求距离
charactersNo={}
characters=[]
length=[]

for i in range(26):
    charactersNo[chr(ord('a')+i)]=i
    characters.append(chr(ord('a')+i))
for i in range(26):
    charactersNo[chr(ord('A')+i)]=i+26
    characters.append(chr(ord('A')+i))
for i in range(10):
    charactersNo[chr(ord('0')+i)]=i+52
    characters.append(chr(ord('0')+i))
punctuations=" .,?\'-:;!/\"<>&(+"
for p in punctuations:
    charactersNo[p]=len(charactersNo)
    characters.append(p)


def get_data():
    #读取了train_img和train_txt文件夹下的所有文件的读取路径
    #下面代码的作用是： 
    #Imgs:列表结构 存储的是手写的英文图片
    #Y: 数组结构 存储的是图片对应的txt文件中句子，只不过存储的是字符转码后的数字
    #length: 数组结构 存储的是图片对应的txt文件中句子含有字符的数量
    imgFiles=glob.glob(os.path.join("train_img", "*"))
    imgFiles.sort()
    txtFiles=glob.glob(os.path.join("train_txt", "*"))
    txtFiles.sort()
    Imgs=[]
    Y=[]
    length=[]
    for i in range(len(imgFiles)):
        fin=open(txtFiles[i])
        line=fin.readlines()
        line=line[0]
        fin.close()
        y=np.asarray([0]*(len(line)))
        succ=True
        for j in range(len(line)):
            if line[j] not in charactersNo:
                succ=False
                break
            y[j]=charactersNo[line[j]]
        if not succ:
            continue
        Y.append(y)
        length.append(len(line))
        im = Image.open(imgFiles[i])
        width,height = im.size#1499,1386
        im = im.convert("L")
        Imgs.append(im)
    
    
    #np.asarray()函数 和 np.array()函数： 将list等结构转化成数组
    #区别是np.asarray()函数不是copy对象，而np.array()函数是copy对象
    print("train:",len(Imgs),len(Y))
    Y = np.asarray(Y)
    length = np.asarray(length)
    return Imgs, Y
