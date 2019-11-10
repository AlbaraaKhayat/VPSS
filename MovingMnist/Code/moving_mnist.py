import numpy as np
import threading
import os

data_array = np.zeros((10, 30, 160, 160, 1),dtype=np.float32)

class MyThread(threading.Thread):
    def __init__(self, starts=None, 
                 paths=None, 
                 seq_len = 30, 
                 image_size = 160, 
                 eid = None, 
                 batch_n = -1
                 batch_size = 2,
                 DATA_DIR=None):

        super(MyThread, self).__init__()  
        self.seq_len = int(seq_len)
        self.image_size = int(image_size)
        self.channels = 1 
        self.batch_n = int(batch_n)
        self.eid = int(eid)
        self.starts, self.paths = starts, paths
        self.DATA_DIR = DATA_DIR
        self.m  = (167 - self.image_size)/2
        self.mn = (167 - self.image_size)/2
        
    def run(self):
        x = np.zeros((self.seq_len, 
                      self.image_size, 
                      self.image_size, 
                      self.channels), dtype=np.float32)   
        idx = self.starts[self.eid + self.batch_n]
        buff2 = self.load_data(idx)[:, int(self.m ) + 1:int(-self.m ),
                                       int(self.mn) + 1:int(-self.mn),:]
        buff2 /= 65
        data_array[self.batch_n, 0:self.seq_len] = buff2


    def load_data(self,idx):
        images = []
        for z in range(idx, idx + self.seq_len):
            path = os.path.join(self.DATA_DIR, str(self.paths[z][-34:].decode('UTF-8'))) 
            im = np.load(path, allow_pickle = True)
            im = im['a']
            im = self.pooling(im,(4,4))
            im = np.expand_dims(im, axis=2)
            images.append(im)
        return images


    def pooling(self,mat,ksize):
        m, n = mat.shape[:2]
        ky,kx = ksize
        _ceil = lambda x,y: int(np.ceil(x/float(y)))
        ny=m//ky
        nx=n//kx
        mat_pad = mat[:ny*ky, :nx*kx, ...]
        new_shape = (ny,ky,nx,kx) + mat.shape[2:]
        return np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))


class mnist(object):
    def __init__(self, split='train',
                 batch_size = 1, 
                 seq_len = 10, 
                 image_size = 160, 
                 DATA_DIR = None):
        
        self.split = split
        self.seq_len = seq_len
        self.image_size = image_size
        self.batch_size = batch_size
        self.channels = 1 
        self.DATA_DIR = DATA_DIR
        self.shuffle = True
        self.N_seq = None
        self.dms = 1
        self.latency = 321 # Allowed time between frames
        self.sequence_start_mode = 'all'
        self.starts, self.paths = self.getsources()
  
    
    def getbatch(self, eid=None):
        th_pool = []
        for i in range(self.batch_size):
            th = MyThread(starts = self.starts,
                          paths = self.paths, 
                          seq_len = self.seq_len, 
                          image_size = self.image_size, 
                          eid = eid, batch_n = i,
                          DATA_DIR = self.DATA_DIR)
            th.start()
            th_pool.append(th)
        for i in range(self.batch_size):
            th_pool[i].join()
        # data_array = np.zeros((10, 20, 160, 160, 1),dtype=np.float32)
        return data_array[0:self.batch_size, 0:self.seq_len]
    
   
    def getsources(self):
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
            while idx < (len(self.paths) - self.seq_len *self.skip * self.dms + 1):
                if self.sources[idx] == self.sources[idx + self.seq_len *self.skip * self.dms - 1]:
                    if (int(self.cnt[idx + self.seq_len *self.skip * self.dms - 1]) - int(self.cnt[idx])) < (self.seq_len *self.skip * self.dms - 1) * self.latency:
                        possible_starts.append(idx)
                        idx += 1
                    else:
                        idx += 1
                else: idx += 1    

        elif self.sequence_start_mode == 'unique':  
            idx = 0
            possible_starts = []
            while idx < (len(self.paths) - self.seq_len *self.skip * self.dms + 1):
                if self.sources[idx] == self.sources[idx + self.seq_len *self.skip * self.dms - 1]:
                    if (int(self.cnt[idx + self.seq_len *self.skip * self.dms - 1]) - int(self.cnt[idx])) < (self.seq_len *self.skip * self.dms - 1) * self.latency:
                        possible_starts.append(idx)
                        idx += k * self.dms *self.skip 
                    else:
                        idx += 1
                else: idx += 1

        self.possible_starts = possible_starts
        if self.split == 'train':          
            for i in range(61326,61613):
                try:
                    self.possible_starts.remove(i)
                except ValueError:
                    pass  
                        
        if self.split == 'valid':
            self.possible_starts = self.possible_starts[int(5 * len(self.possible_starts) / 6):]
            
        if self.split == 'spec':
            self.possible_starts  = list(range(61326,  61613,  self.seq_len))  #20140329
            self.possible_starts += list(range(157981, 158268, self.seq_len))  #20160413
            self.possible_starts += list(range(158268, 158555, self.seq_len))  #20160414
            self.possible_starts += list(range(198103, 198374, self.seq_len))  #20170214
            self.possible_starts += list(range(198630, 198896, self.seq_len))  #20170217
            
        if self.shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)

        if self.N_seq is not None and len(self.possible_starts) > self.N_seq:  
            self.possible_starts = self.possible_starts[:self.N_seq]  

        print(split+': ',len(self.possible_starts))
        return self.possible_starts, self.paths

