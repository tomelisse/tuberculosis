import tensorflow as tf

class CNN(object):
    def __init__(self, savepath, hparams, iparams, aparams):
        ''' initialize the classifier '''
        self.savepath = savepath

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
        convoluted = tf.conv2d(input, W, 
                               strides = strides, 
                               padding = 'SAME')
        biased = tf.nn.bias_add(convoluted, b)
        relued = tf.nn.relu(biased)
        pooled = tf.max_pool(relued, ksize, 
                             strides = strides, 
                             padding = 'SAME')
        print 'pooled:', pooled.get_shape().as_list()
        return pooled

    @staticmethod
    def fully_connected_layer(input, W, b):
        matmuled = tf.matmul(input, W)
        biased   = tf.nn.bias_add(matmuled, b)
        # sigmoid would be better perhaps
        relued   = tf.nn.relu(biased)
        output   = tf.nn.dropout(relued, self.dropout)
        return output

    def make_architecture(self):
        ''' defines tf variables and placeholders '''
        input_shape = (None, self.h_input, self.w_input, self.d_input)
        self.input  = tf.placeholder(tf.float32, shape = input_shape)
        # None shape?
        self.labels = tf.placeholder(tf.float32)

        # the 1st conv layer
        w1_init = tf.random_normal([self.h_filter1, 
                                    self.w_filter1, 
                                    self.d_input, 
                                    self.n_filters1])
        W1 = tf.Variable(w1_init)
        b1_init = tf.random_normal([self.n_filters1])
        b1 = tf.Variable(b1_init)
        ksize1 = [1, self.h_pool1, self.w_pool1, 1] 
        conv1 = convolutional_layer(self.input, W1, b1, ksize1) 
        
        # the 2nd conv layer
        w2_init = tf.random_normal([self.h_filter2, 
                                    self.w_filter2, 
                                    self.n_filters1/ksize1[3], 
                                    self.n_filters2])
        W2 = tf.Variable(w2_init)
        b2_init = tf.random_normal([self.n_filters2])
        b2 = tf.Variable(b2_init)
        ksize2 = [1, self.h_pool2, self.w_pool2, 1] 
        conv2 = convolutional_layer(conv1, W2, b2, ksize2) 

        # the fully-connected_layer
        shape     = conv2.get_shape().as_list()
        new_shape = [shape[0], shape[1]*shape[2]*shape[3]] 
        conv2_flattened = tf.reshape(conv2, new_shape)
        wfc_init = tf.random_normal([new_shape[1], self.n_classes])
        Wfc = tf.Variable(wfc_init)
        bfc_init = tf.random_normal(self.n_classes)
        bfc = tf.Variable(bfc_init)
        # maybe we should use 2 fcs
        self.predictions = fully_connected_layer(conv2_flattened, Wfc, bfc)

    def define_operations(self):
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.labels, 
                                                       logits = self.predictions)
        correct = tf.equal(self.predictions, self.labels)
        self.accuracy = tf.reduce_mean(correct)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.training_step = optimizer.minimize(self.loss)

    def train(self, dataset):
        ''' training '''
        with tf.Session(graph = self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            while dataset.epochs_completed < self.n_epochs:
                images, labels = dataset.next_batch(self.batch_size)
                # filling for the placeholders
                feed_dict = {self.input = images, self.labels = labels}
                # operations to be run and variables to be evaluated
                fetches = [self.training_step, self.loss, self.accuracy]
                # run!
                _, loss, acc = sess.run(fetches = fetches, feed_dict = feed_dict)
                print epoch, loss, batch
                self.losses.append(loss)
                self.accuracies.append(acc)
            saver = tf.train.Saver()
            saver.save(sess, savepath = savepath)
            print 'saved'
