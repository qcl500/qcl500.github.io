# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:44:36 2018

@author: zhou
"""

import tensorflow as tf
import cv2
import numpy as np
import os
import time
from utils import  merge, init_setting, pre_setting, read_data , imsave, imread

class srcnn(object):
    
    #Hyperparameter
    def __init__(self, 
                 sess, 
                 image_size=33,
                 label_size=21,
                 epoch=2000, 
                 batch_size=64, 
                 scale=3):
        
        self.sess=sess
        self.image_size=image_size
        self.label_size=label_size
        self.epoch=epoch
        self.batch_size=batch_size
        self.scale=scale
        self.learning_rate=0.001
        
        self.build_model()
        
        self.checkpoint_dir='backup'
        self.load(self.checkpoint_dir)
        
        
        
    def build_model(self):
        #only build in the Y channel
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, 1], name='labels')
        
        #init w and b
        self.weights = {
          'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
          'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
          'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
        }
        self.biases = {
          'b1': tf.Variable(tf.zeros([64]), name='b1'),
          'b2': tf.Variable(tf.zeros([32]), name='b2'),
          'b3': tf.Variable(tf.zeros([1]), name='b3')
        }
        
        self.conv3=self.model_body()
        
        #loss
        self.loss=tf.reduce_mean(tf.square(self.labels-self.conv3))
        
        self.optimizer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()
        
            
    def model_body(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
        return conv3
    
    def train(self):
        
        init_setting()
        data_path='./dataset/train.h5'
        train_data, train_label = read_data(data_path)
        counter=0
        start_time=time.time()
    
        print('Training...')
        
        for epx in range(self.epoch):
            batch_id=len(train_data)//self.batch_size
            for idx in range(batch_id):
                batch_images = train_data[idx*self.batch_size : (idx+1)*self.batch_size]
                batch_labels = train_label[idx*self.batch_size : (idx+1)*self.batch_size]
                
                counter +=1
                
                _, err = self.sess.run([self.optimizer, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                          % ((epx+1), counter, time.time()-start_time, err))

                if counter % 500 == 0:
                    self.save(self.checkpoint_dir, counter)    
    
    def predict(self,test_image):
        
        arrdata, arrCrCb, nx, ny=pre_setting(test_image)

        result = self.sess.run(self.conv3, feed_dict={self.images: arrdata})

        result = merge(result, arrCrCb, [nx, ny])

        
        return result


    
        
    def save(self, checkpoint_dir, step):

        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False 
        

if __name__=='__main__':


    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        
        model = srcnn(sess)

#        model.train()
#        print('finished the train')

#        
        image=imread('timg.jpg')

        new_image = model.predict(image)

    
    
    
    
    
    
    
    
    
    
    
    
    
    