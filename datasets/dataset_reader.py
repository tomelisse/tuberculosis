import h5py
import numpy as np

class HdfDataset():
    def __init__(self, savepath):
        ''' dataset initialization '''
        with h5py.File(savepath) as f:
            self.n_examples = f['training/images'].shape[0]
            self.h_input    = f['training/images'].shape[1]
            self.w_input    = f['training/images'].shape[2]
            self.d_input    = f['training/images'].shape[3]
            self.n_classes  = f['training/labels'].shape[1]
            # self.n_classes  = len(np.unique(f['training/labels']))
            print 'Accesing dataset with {} examples and {} classes'.format(self.n_examples, self.n_classes)

        self.where_in_epoch   = 0
        # self.epochs_started   = 1 
        self.epochs_completed = 0
        self.savepath         = savepath
        self.indices          = np.arange(self.n_examples)

    def next_batch(self, batch_size):
        ''' return a batch of training examples of a given batch_size '''
        start = self.where_in_epoch
        self.where_in_epoch += batch_size
        stop  = self.where_in_epoch
        # examples' permutation between epochs should be added
        if stop >= self.n_examples:
            # the start of a new epoch
            self.epochs_completed+= 1
            self.where_in_epoch = 0
            # shuffle
            np.random.shuffle(self.indices)
        indices = np.sort(self.indices[start:stop]).tolist()
        with h5py.File(self.savepath) as f:
            # h5py doesn't support int arrays for fancy indexing
            # images = f['training/images'][self.indices][start:stop]
            # labels = f['training/labels'][self.indices][start:stop]
            images = f['training/images'][indices]
            labels = f['training/labels'][indices]
        return images, labels

