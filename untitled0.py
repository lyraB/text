# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 07:56:27 2018

@author: yfh
"""

import numpy as np
import mxnet as mx
import logging
import pickle

def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding='latin1')
    return np.array(dict['data']).reshape(10000,3072),np.array(dict['labels']).reshape(10000)


def to4d(img):
    return img.reshape(img.shape[0],3,32,32).astype(np.float32)/255


def fit(batch_num,model,val_iter,batch_size):
    (train_img, train_lbl) = unpickle('E:/learning/cifar-10-batches-py/data_batch_'+str(batch_num))
    train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)#构建训练数据迭代器，且其中shuffle表示采用可拖动的方式，意味着可以将在早期已经训练过的数据在后面再次训练
    model.fit(
        X=train_iter,
        eval_data=val_iter,
        batch_end_callback=mx.callback.Speedometer(batch_size,200)
    )


(val_img, val_lbl) = unpickle('E:/learning/cifar-10-batches-py/test_batch')

batch_size = 100#定义每次处理的数据量为100
val_iter = mx.io.NDArrayIter(to4d(val_img),val_lbl,batch_size)#测试数据迭代器

data = mx.sym.Variable('data')
cv1 = mx.sym.Convolution(data=data,name='cv1',num_filter=32,kernel=(3,3))
act1 = mx.sym.Activation(data=cv1,name='relu1',act_type='relu')
poing1 = mx.sym.Pooling(data=act1,name='poing1',kernel=(2,2),pool_type='max')
do1 = mx.sym.Dropout(data=poing1,name='do1',p=0.25)
cv2 = mx.sym.Convolution(data=do1,name='cv2',num_filter=32,kernel=(3,3))
act2 = mx.sym.Activation(data=cv2,name='relu2',act_type='relu')
poing2 = mx.sym.Pooling(data=act2,name='poing2',kernel=(2,2),pool_type='avg')
do2 = mx.sym.Dropout(data=poing2,name='do2',p=0.25)
cv3 = mx.sym.Convolution(data=do2,name='cv3',num_filter=64,kernel=(3,3))
act3 = mx.sym.Activation(data=cv3,name='relu3',act_type='relu')
poing3 = mx.sym.Pooling(data=act3,name='poing3',kernel=(2,2),pool_type='avg')
do3 = mx.sym.Dropout(data=poing3,name='do3',p=0.25)
data = mx.sym.Flatten(data=do3)
fc1 = mx.sym.FullyConnected(data=data,name='fc1',num_hidden=64)
act4 = mx.sym.Activation(data=fc1,name='relu4',act_type='relu')
do4 = mx.sym.Dropout(data=act4,name='do4',p=0.25)
fc2 = mx.sym.FullyConnected(data=do4,name='fc2',num_hidden=10)
mlp = mx.sym.SoftmaxOutput(data=fc2,name='softmax')

logging.getLogger().setLevel(logging.DEBUG)

model = mx.model.FeedForward(
    ctx=mx.cpu(),
    symbol=mlp,
    num_epoch=10,
    learning_rate=0.1
)
for batch_num in range(1,6):
    fit(batch_num, model, val_iter, batch_size)