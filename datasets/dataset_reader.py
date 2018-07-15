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
            self.n_classes  = len(np.unique(f['training/labels']))
            print 'Accesing dataset with {} examples and {} classes'.format(self.n_examples, self.n_classes)

        self.where_in_epoch   = 0
        self.epochs_completed = 0
        self.savepath         = savepath

    def next_batch(self, batch_size):
        ''' return a batch of training examples of a given batch_size '''
        if self.where_in_epoch + batch_size > self.n_examples:
            # the start of a new epoch
            self.epochs_completed += 1
            print 'epochs completed: {}'.format(self.epochs_completed)
            self.where_in_epoch = 0
        start = self.where_in_epoch
        self.where_in_epoch += batch_size
        stop = self.where_in_epoch
        # examples' permutation between epochs should be added
        with h5py.File(self.savepath) as f:
            images = f['training/images'][start:stop]
            labels = f['training/labels'][start:stop]
        return images, labels

