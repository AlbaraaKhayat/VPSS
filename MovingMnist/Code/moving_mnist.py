import numpy as np
import tensorflow as tf
import threading
import time
import scipy.misc as misc
import sys

data_array = np.zeros((100, 30, 64, 64, 1),dtype=np.float32)

class MyThread(threading.Thread):
    def __init__(self, ):

        super(MyThread, self).__init__()  
        self.data = Xall
        self.N = length of possible start
        self.seq_len = dms
        self.num_digits = num_digits
        self.image_size = image_size
        self.digit_size = 28
        self.channels = 1 
        self.data_slice = data_slice


    def run(self):
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,image_size, image_size, self.channels),dtype=np.float32)
        idx = np.random.randint(self.N)
        digit = self.data[idx]
        data_array[self.data_slice, 0:self.seq_len] = x

class mnist(object):
    def __init__(self,train, batch_size = 1, seq_len = 0, image_size = 160):
        self.train = train
        self.seq_len = seq_len
        self.image_size = image_size
        self.batch_size = batch_size
        self.channels = 1 

    def getbatch(self,sources=None):
        # data_array = np.zeros((100, 15, 64, 64, 1),dtype=np.float32)
        th_pool = []
        for i in range(self.batch_size):
            th = MyThread(self.split, self.seq_len, self.image_size, i)
            th.start()
            th_pool.append(th)
        for i in range(self.batch_size):
            th_pool[i].join()
        return data_array[0:self.batch_size, 0:self.seq_len]
    
