# from  sklearn.preprocessing import LabelBinarizer
from glob import glob
import numpy as np
import h5py
import cv2

def read_paths(how_many):
    query = 'data/MontgomerySet/CXR_png/*png'
    paths = glob(query)
    return paths[:how_many]

def labels_to_vectors(labels):
    new_labels = []
    for label in labels:
        new_label = np.zeros(2)
        new_label[label] = 1
        new_labels.append(new_label)
    return new_labels

def read_data():
    images = []
    labels = []
    paths = read_paths(100)
    for path in paths:
        image = cv2.imread(path)
        if image.shape[0] == 4020:
            image = image.transpose((1,0,2))
        small = cv2.resize(image,(72, 64))
        images.append(small)
        labels.append([int('1.png' in path)])
    labels = labels_to_vectors(labels)
    # is one-hot encoding necessary in binary problems?
    # lb = LabelBinarizer()
    # lb.fit(labels)
    # vectors = lb.transform(labels)
    # vectors = lb.fit_transform(labels)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def make_dataset(savepath):
    images, labels = read_data()
    # maybe it would make sense to permute the examples
    n_examples = images.shape[0]
    # savepath = 'datasets/tuberculosis.storage'
    with h5py.File(savepath, 'w') as f:
        training = f.create_group('training')
        training.create_dataset('images', data = images[:n_examples/2])
        training.create_dataset('labels', data = labels[:n_examples/2])

        testing = f.create_group('testing')
        testing.create_dataset('images', data = images[n_examples/2:])
        testing.create_dataset('labels', data = labels[n_examples/2:])

    with h5py.File(savepath) as f:
        print f['training/images'].shape
        print f['training/labels'].shape
        print f['testing/images'].shape
        print f['testing/labels'].shape

