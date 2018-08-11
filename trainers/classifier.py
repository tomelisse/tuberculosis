import tensorflow as tf
from matplotlib import pyplot as plt

class CNN(object):
    def __init__(self, netname, hparams, iparams, aparams):
        ''' initialize the classifier '''
        self.netname = netname
        self.savedir = 'saved/'
        self.netpath  = 'saved/' + netname + '.ckpt'

        # hyperparameters
        self.learning_rate = hparams[0]
        self.n_epochs      = hparams[1]
        self.batch_size    = hparams[2]

        # input parameters
        self.n_examples = iparams[0]
        self.h_input    = iparams[1]
        self.w_input    = iparams[2]
        self.d_input    = iparams[3]
        self.n_classes  = iparams[4]

        # architecture params
        self.h_filter1  = aparams[0]
        self.w_filter1  = aparams[1]
        self.n_filters1 = aparams[2]
        self.h_pool1    = aparams[3]
        self.w_pool1    = aparams[4]

        self.h_filter2  = aparams[5]
        self.w_filter2  = aparams[6]
        self.n_filters2 = aparams[7]
        self.h_pool2    = aparams[8]
        self.w_pool2    = aparams[9]

        self.n_fc       = aparams[10]

        # plotting containers
        self.losses     = []
        self.accuracies = []

        # define the graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.make_architecture()
            self.define_operations()

    @staticmethod
    def convolutional_layer(input, W, b, ksize):
        strides = [1, 1, 1, 1]
        convoluted = tf.nn.conv2d(input, W, 
                               strides = strides, 
                               padding = 'SAME')
        biased = tf.nn.bias_add(convoluted, b)
        relued = tf.nn.relu(biased)
        pooled = tf.nn.max_pool(relued, ksize, 
                             strides = ksize, 
                             padding = 'SAME')
        # print 'pooled:', pooled.get_shape().as_list()
        return pooled

    @staticmethod
    def fully_connected_layer(input, W, b, sigmoid = True):
        matmuled = tf.matmul(input, W)
        biased   = tf.nn.bias_add(matmuled, b)
        if sigmoid:
            biased   = tf.nn.sigmoid(biased)
        # sigmoid would be better perhaps
        # if relued:
        #     biased  = tf.nn.relu(biased)
        # when to apply dropout?
        # output   = tf.nn.dropout(relued, self.dropout)
        return biased 

    def make_architecture(self):
        ''' defines tf variables and placeholders '''
        input_shape = (None, self.h_input, self.w_input, self.d_input)
        self.input  = tf.placeholder(tf.float32, shape = input_shape)
        # None shape?
        labels_shape = (None, self.n_classes)
        self.labels = tf.placeholder(tf.float32, shape = labels_shape)
        print 'labels: ', self.labels.get_shape().as_list()

        # the 1st conv layer
        w1_init = tf.random_normal([self.h_filter1, 
                                    self.w_filter1, 
                                    self.d_input, 
                                    self.n_filters1])
        W1 = tf.Variable(w1_init)
        b1_init = tf.random_normal([self.n_filters1])
        b1 = tf.Variable(b1_init)
        ksize1 = [1, self.h_pool1, self.w_pool1, 1] 
        conv1 = CNN.convolutional_layer(self.input, W1, b1, ksize1) 
        print 'conv1:', conv1.get_shape().as_list()
        
        # the 2nd conv layer
        w2_init = tf.random_normal([self.h_filter2, 
                                    self.w_filter2, 
                                    self.n_filters1/ksize1[3], 
                                    self.n_filters2])
        W2 = tf.Variable(w2_init)
        b2_init = tf.random_normal([self.n_filters2])
        b2 = tf.Variable(b2_init)
        ksize2 = [1, self.h_pool2, self.w_pool2, 1] 
        conv2 = CNN.convolutional_layer(conv1, W2, b2, ksize2) 
        print 'conv2:', conv2.get_shape().as_list()

        # the 1st fully-connected_layer
        shape     = conv2.get_shape().as_list()
        new_shape = [-1, shape[1]*shape[2]*shape[3]] 
        conv2_flattened = tf.reshape(conv2, new_shape)
        wfc_init = tf.random_normal([new_shape[1], self.n_fc])
        Wfc = tf.Variable(wfc_init)
        bfc_init = tf.random_normal([self.n_fc])
        bfc = tf.Variable(bfc_init)
        fc = CNN.fully_connected_layer(conv2_flattened, Wfc, bfc, True)
        print 'fc:', fc.get_shape().as_list()

        # the last fully-connected layer
        # if one-hot encoded change 1 -> self.n_classes
        wlast_init = tf.random_normal([self.n_fc, self.n_classes])
        Wlast = tf.Variable(wlast_init)
        # if one-hot encoded change 1 -> self.n_classes
        blast_init = tf.random_normal([self.n_classes])
        blast = tf.Variable(blast_init)
        predictions = CNN.fully_connected_layer(fc, Wlast, blast, False)
        self.predictions = tf.nn.softmax(predictions)
        print 'pred:', self.predictions.get_shape().as_list()

    def define_operations(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels, 
                                                       logits = self.predictions)
        self.loss = tf.reduce_mean(loss) 
        correct = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.labels, 1))
        correct = tf.cast(correct, tf.float32)
        self.accuracy = tf.reduce_mean(correct)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.training_step = optimizer.minimize(self.loss)

    def make_plots(self):
        fig, ax = plt.subplots()
        ax.plot(self.losses)
        ax.set_title('Losses')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        fig.savefig(self.savedir + '/losses.png')

        fig, ax = plt.subplots()
        ax.plot(self.accuracies)
        ax.set_title('Accuracy')
        ax.set_xlabel('Step')
        ax.set_ylabel('Acc')
        fig.savefig(self.savedir + '/accuracies.png')

    # the only function parameter should be self
    def draw_ROC_curves(self, predictions, labels):
        ''' ROC curves '''
        # plotting containers
        fall_outs = []
        recalls   = []
        # let's concentrate of the 1st of the 2 classes
        predictions = [pred[0] for pred in predictions]
        labels = [lab[0] for lab in labels.tolist()]

        thresholds = map((0.1).__mul__, range(1,10))
        # thresholds = [0.8, 0.9]
        all_pos = sum(labels)
        all_neg = len(labels) - all_pos
        
        for th in thresholds:
            true_pos = 0
            false_pos = 0
            for pred, lab in zip(predictions, labels):
                if pred > th:
                    if lab == 1:
                        true_pos += 1
                    else:
                        false_pos += 1
            fall_outs.append(false_pos/all_neg)
            recalls.append(true_pos/all_pos)

        print fall_outs
        print recalls
        fig, ax = plt.subplots()
        ax.plot(fall_outs, recalls)
        ax.set_title('ROC curve')
        ax.set_xlabel('fall_out')
        ax.set_ylabel('recall')
        ax.set_xlim(xmin = 0, xmax = 1)
        ax.set_ylim(ymin = 0, ymax = 1)
        plt.show()
        fig.savefig(self.savedir + '/ROC.png')


    def test(self, dataset):
        ''' testing '''
        with tf.Session(graph = self.graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path = self.savedir + self.netname + '.ckpt')
            images, labels = dataset.test_batch()
            # filling for the placeholders
            feed_dict = {self.input : images, self.labels : labels}
            # operations to be run and variables to be evaluated
            fetches = [self.loss, self.accuracy, self.predictions]
            # run!
            loss, acc, preds = sess.run(fetches = fetches, feed_dict = feed_dict)
            print 'testing loss {} and accuracy {}'.format(loss, acc)
            self.draw_ROC_curves(preds, labels)

    def train(self, dataset):
        ''' training '''
        with tf.Session(graph = self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            while dataset.epochs_completed < self.n_epochs:
                images, labels = dataset.next_batch(self.batch_size)
                # filling for the placeholders
                feed_dict = {self.input : images, self.labels : labels}
                # operations to be run and variables to be evaluated
                fetches = [self.training_step, self.loss, self.accuracy]
                # run!
                _, loss, acc = sess.run(fetches = fetches, feed_dict = feed_dict)
                if dataset.where_in_epoch == 0:
                    print 'loss {} acc {}'.format(loss, acc)
                    print 'epoch {} completed'.format(dataset.epochs_completed)
                self.losses.append(loss)
                self.accuracies.append(acc)
            self.make_plots()
            saver = tf.train.Saver()
            saver.save(sess, save_path = self.savedir + self.netname + '.ckpt')
            print 'saved'
