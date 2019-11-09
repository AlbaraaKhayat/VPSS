import numpy as np
import threading
import os

data_array = np.zeros((10, 30, 160, 160, 1),dtype=np.float32)

class MyThread(threading.Thread):
    def __init__(self, starts=None, paths=None, seq_len = 30, image_size = 160, eid = None, batch_n = -1):

        super(MyThread, self).__init__()  
        self.seq_len = seq_len
        self.image_size = image_size
        self.channels = 1 
        self.batch_n = batch_n
        self.eid = eid
        self.starts, self.paths = starts, paths
        
    def run(self):
        x = np.zeros((self.seq_len, self.image_size, self.image_size, self.channels),dtype=np.float32)   
        
        digit = self.data[self.eid+self.batch_n]
        data_array[self.batch_n, 0:self.seq_len] = x

class mnist(object):
    def __init__(self, train, batch_size = 1, seq_len = 10, image_size = 160):
        self.train = train
        self.seq_len = seq_len
        self.image_size = image_size
        self.batch_size = batch_size
        self.channels = 1 
        self.starts, self.paths = getsources(seq_len = 20, split='train',
                                             DATA_DIR=None, 
                                             batch_size=1,
                                             latency = 321, 
                                             shuffle=True, 
                                             skip=1,
                                             sequence_start_mode='all', 
                                             N_seq=None)
        
        
       
        
    def getbatch(self, eid=None):
        # data_array = np.zeros((10, 20, 160, 160, 1),dtype=np.float32)
        th_pool = []
        for i in range(self.batch_size):
            th = MyThread(starts = self.starts,
                          paths = self.paths, 
                          seq_len = self.seq_len, 
                          image_size = self.image_size, 
                          eid = eid, batch_n = i)
            th.start()
            th_pool.append(th)
        for i in range(self.batch_size):
            th_pool[i].join()
        return data_array[0:self.batch_size, 0:self.seq_len]
    

    
    
    def getsources(self, seq_len = 20,split='train',
                 DATA_DIR=None, 
                 batch_size=1,
                 latency = 321, 
                 shuffle=True, 
                 skip=1,
                 sequence_start_mode='all', 
                 N_seq=None):

        self.nt = seq_len
        self.dms = 1
        self.latency = latency # Allowed time between frames
        self.DATA_DIR = DATA_DIR
        self.split = split
        self.skip = skip 
        self.sequence_start_mode = sequence_start_mode
        self.sources = np.load(os.path.join(self.DATA_DIR, 'meta', 'abha_train_source_list.npz')) 
        self.paths = sorted(self.sources['paths'])
        self.cnt = np.copy(self.paths)
        self.sources = sorted(self.sources['days'])

        for i in range(len(self.paths)):
            self.cnt[i] = int(self.paths[i][-10:-8])*3600 + int(self.paths[i][-8:-6]) * 60 + int(self.paths[i][-6:-4])

        if self.split == 'train':
            k = self.seq_len
        else: k = 1

        if self.sequence_start_mode == 'all':  
            idx = 0
            possible_starts = []
            while idx < (len(self.paths) - self.nt *self.skip * self.dms + 1):
                if self.sources[idx] == self.sources[idx + self.nt *self.skip * self.dms - 1]:
                    if (int(self.cnt[idx + self.nt *self.skip * self.dms - 1]) - int(self.cnt[idx])) < (self.nt *self.skip * self.dms - 1) * self.latency:
                        possible_starts.append(idx)
                        idx += 1
                    else:
                        idx += 1
                else: idx += 1    

        elif self.sequence_start_mode == 'unique':  
            idx = 0
            possible_starts = []
            while idx < (len(self.paths) - self.nt *self.skip * self.dms + 1):
                if self.sources[idx] == self.sources[idx + self.nt *self.skip * self.dms - 1]:
                    if (int(self.cnt[idx + self.nt *self.skip * self.dms - 1]) - int(self.cnt[idx])) < (self.nt *self.skip * self.dms - 1) * self.latency:
                        possible_starts.append(idx)
                        idx += k * self.dms *self.skip 
                    else:
                        idx += 1
                else: idx += 1

        if self.split == 'train':
            self.possible_starts = possible_starts
        if self.split == 'spec':
            self.possible_starts = list(range(61326,61613))     #20140329
            self.possible_starts += list(range(157981,158268))  #20160413
            self.possible_starts += list(range(158268,158555))  #20160414
            self.possible_starts += list(range(198103,198374))  #20170214
            self.possible_starts += list(range(198630,198896))  #20170217
        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  
            self.possible_starts = self.possible_starts[:N_seq]  
        print(split+': ',len(self.possible_starts))
        return self.possible_starts, self.paths

