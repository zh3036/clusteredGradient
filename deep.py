from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


mnist = input_data.read_data_sets("/home/zihanz/cluterML/clusteredGradient/inputdata", one_hot=True)


class clusterdTrain(object):
    """docstring for train"""

    def __init__(self, images=None, lables = None):
        super(clusterdTrain, self).__init__()
        if images is None or lables is None:
            final_ind = np.load("final_ind_cluster.npy")
            self.images = np.load("minstImage.npy")[final_ind]
            self.lables = np.load("minstLabels.npy")[final_ind]
        else:
            self.images = images
            self.lables = lables
        self.length = len(self.images)
        self.curPos = 0
        self.epoch = 0
    def next_batch(self, size):
        self.curPos += size
        if self.curPos > self.length:
            self.epoch += 1
            self.curPos = size 
        return [self.images[self.curPos - size:self.curPos], \
                self.lables[self.curPos - size:self.curPos]]




def Grad(Train,R=6000,B=20):
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    #tf.global_variables_initializer().run()
    tf.initialize_all_variables()
    # Train
    loss= 10
    acc = 0
    ploss = False
    ploss = True
    loss_list_normal=[]
    acc_list = []
    for i in range(R):
        batch_xs, batch_ys = Train.next_batch(B)


        if ploss:
            _, loss, acc = sess.run([train_step,cross_entropy,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
        else:
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 1 ==0:
    #         print (i, loss)
            loss_list_normal.append(loss)
            acc_list.append(acc)
    # Test trained model

    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
    return loss_list_normal,acc_list



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')



def deepGrad(Train,R=6000,B=20):
    x = tf.placeholder(tf.float32, [None, 784])
    W_conv1 = weight_variable([5, 5, 1, 4])
    b_conv1 = bias_variable([4])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    
    W_conv2 = weight_variable([5, 5, 4, 8])
    b_conv2 = bias_variable([8])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 8, 128])
    b_fc1 = bias_variable([128])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*8])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([128, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    
    
    
    
    

#     W = tf.Variable(tf.zeros([784, 10]))
#     b = tf.Variable(tf.zeros([10]))
#     y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #sess = tf.InteractiveSession()
    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    #tf.initialize_all_variables().run(session=sess)
    #tf.initialize_local_variables().run(session=sess)
    
    loss = 10 
    acc = 0
    ploss = True
#     ploss = False
    loss_list=[]
    acc_list = []
    for i in range(R):
        batch = Train.next_batch(B)
        if i%1 == 0:
            print (i)
            test_accuracy = accuracy.eval(session=sess, feed_dict={
                x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            train_loss = cross_entropy.eval(session=sess,feed_dict={x:batch[0],y_:batch[1],keep_prob: 1.0})
            acc_list.append(test_accuracy)
            loss_list.append(train_loss)
            
        train_step.run(session=sess,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(session=sess,feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))    
    return loss_list,acc_list







#images = np.load("minstImages.npy")
images = mnist.train.images
labels = mnist.train.labels
#lables = np.load("minstLabels.npy")
final_ind = np.load("final_ind_cluster.npy")
final_labels  = labels[final_ind]
final_images = images[final_ind]
clusteredData = clusterdTrain(final_images,final_labels)


R=1500
B=30
loss_list_normal,acc_list_normal = deepGrad(mnist.train,R,B)
loss_list_clus,acc_list_clus = deepGrad(clusteredData,R,B)

np.save("loss_list_normal_R_"+str(R)+"_B_"+str(B),loss_list_normal)

np.save("acc_list_normal_R_"+str(R)+"_B_"+str(B),acc_list_normal)

np.save("loss_list_clus_R_"+str(R)+"_B_"+str(B),loss_list_clus)

np.save("loss_list_clus_R_"+str(R)+"_B_"+str(B),loss_list_clus)


