from datasets import data_preparator as dp
from datasets import dataset_reader as dr
from trainers import classifier as tc

if __name__ == '__main__':
    datapath = 'datasets/tuberculosis.storage'
    netpath  = 'saved/cnn.ckpt'

    # dataset generation - to be run just once
    dp.make_dataset(datapath)

    # read the data
    dataset = dr.HdfDataset(datapath)

    # define the classifier and its parameters
    hparams = [0.1, 1, 1]
    iparams = [dataset.n_examples, dataset.h_input, dataset.w_input, dataset.d_input, dataset.n_classes]
    aparams = [4, 4, 3, # first conv
               4, 4,    # first pool
               2, 2, 3, # second conv
               2, 2]    # second pool
    classifier = tc.CNN(netpath, hparams, iparams, aparams)

    # train it!
    # classifier.train(dataset)
