import tensorflow as tf
from functools import partial
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import functools
from queues import *
from generator import *       
from utils_multistep_lr import *
import math
import numpy as np
class BWNet(Model):
    def conv2d(self,inputs,filters=16,kernel_size=3,strides=1,padding='SAME',activation=None,
                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                       kernel_regularizer=layers.l2_regularizer(2e-4),
                       use_bias=False,name="conv"):
      return tf.layers.conv2d(inputs,filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format="channels_last",activation=activation,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer,
                       use_bias=use_bias,name=name)
    def _build_model(self, inputs):
        if self.data_format == 'NCHW': 
            reduction_axis = [2,3]
            concat_axis = 1
            _inputs = tf.cast(tf.transpose(inputs, [0, 3, 1, 2]), tf.float32)
        else:          
            reduction_axis = [1,2]
            concat_axis = 3
            _inputs = tf.cast(inputs, tf.float32)
        self.inputImage = _inputs
        with arg_scope([layers.batch_norm],
                       decay=0.9, center=True, scale=True, 
                       updates_collections=None, is_training=self.is_training,
                       fused=True, data_format=self.data_format),\
            arg_scope([layers.avg_pool2d],
                       kernel_size=[3,3], stride=[2,2], padding='SAME',
                       data_format=self.data_format),\
            arg_scope([layers.max_pool2d],
                       kernel_size=[3,3], stride=[2,2], padding='SAME',
                       data_format=self.data_format):
          arr = np.loadtxt('kernel_0.8_20.txt',dtype=np.float32)
          arr = np.reshape(arr,[1,20,8,8])
          arr = np.transpose(arr,(2,3,0,1))
          print(arr)
          with tf.variable_scope('DCT_preprocess'):
              W_DCT = tf.get_variable('W', initializer=arr, \
                          dtype=tf.float32, \
                          regularizer=None,trainable=False)
              conv = tf.nn.conv2d(_inputs,W_DCT, [1,1,1,1], 'SAME',data_format=self.data_format)
              actv = tf.clip_by_value(conv,-8,8)
          with tf.variable_scope('Layer2'): 
              conv1=self.conv2d(actv,filters=10,name="conv1",kernel_size=1)
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=20,strides=2,name="conv2")
              actv2=tf.nn.relu(layers.batch_norm(conv2))
              conv3=self.conv2d(actv2,filters=20,name="conv3",kernel_size=1)
              actv3=layers.batch_norm(conv3)
              conv_sc=self.conv2d(actv,filters=20,strides=2,name="conv_sc")
              actv_sc=layers.batch_norm(conv_sc)
              res=tf.add_n([actv3,actv_sc])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer3'):  
              conv1=self.conv2d(res,filters=20,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=20,name="conv2")
              actv2=layers.batch_norm(conv2)
              res=tf.add_n([res,actv2])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer4'): 
              conv1=self.conv2d(res,filters=20,name="conv1",kernel_size=1)
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=40,strides=2,name="conv2")
              actv2=tf.nn.relu(layers.batch_norm(conv2))
              conv3=self.conv2d(actv2,filters=40,name="conv3",kernel_size=1)
              actv3=layers.batch_norm(conv3)
              conv_sc=self.conv2d(res,filters=40,strides=2,name="conv_sc")
              actv_sc=layers.batch_norm(conv_sc)
              res=tf.add_n([actv3,actv_sc])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer5'): 
              conv1=self.conv2d(res,filters=40,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=40,name="conv2")
              actv2=layers.batch_norm(conv2)
              res=tf.add_n([res,actv2])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer6'): 
              conv1=self.conv2d(res,filters=40,name="conv1",kernel_size=1)
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=80,strides=2,name="conv2")
              actv2=tf.nn.relu(layers.batch_norm(conv2))
              conv3=self.conv2d(actv2,filters=80,name="conv3",kernel_size=1)
              actv3=layers.batch_norm(conv3)
              conv_sc=self.conv2d(res,filters=80,strides=2,name="conv_sc")
              actv_sc=layers.batch_norm(conv_sc)
              res=tf.add_n([actv3,actv_sc])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer7'): 
              conv1=self.conv2d(res,filters=80,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=80,name="conv2")
              actv2=layers.batch_norm(conv2)
              res=tf.add_n([res,actv2])
              res=tf.nn.relu(res)  
          with tf.variable_scope('Layer8'): 
              conv1=self.conv2d(res,filters=80,name="conv1",kernel_size=1)
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=160,strides=2,name="conv2")
              actv2=tf.nn.relu(layers.batch_norm(conv2))
              conv3=self.conv2d(actv2,filters=160,name="conv3",kernel_size=1)
              actv3=layers.batch_norm(conv3)
              conv_sc=self.conv2d(res,filters=160,strides=2,name="conv_sc")
              actv_sc=layers.batch_norm(conv_sc)
              res=tf.add_n([actv3,actv_sc])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer9'): 
              conv1=self.conv2d(res,filters=160,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=160,name="conv2")
              actv2=layers.batch_norm(conv2)
              res=tf.add_n([res,actv2])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer10'): 
              conv1=self.conv2d(res,filters=160,name="conv1",kernel_size=1)
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=320,strides=2,name="conv2")
              actv2=tf.nn.relu(layers.batch_norm(conv2))
              conv3=self.conv2d(actv2,filters=320,name="conv3",kernel_size=1)
              actv3=layers.batch_norm(conv3)
              conv_sc=self.conv2d(res,filters=320,strides=2,name="conv_sc")
              actv_sc=layers.batch_norm(conv_sc)
              res=tf.add_n([actv3,actv_sc])
              res=tf.nn.relu(res)
          with tf.variable_scope('Layer11'): 
              conv1=self.conv2d(res,filters=320,name="conv1")
              actv1=tf.nn.relu(layers.batch_norm(conv1))
              conv2=self.conv2d(actv1,filters=320,name="conv2")
              actv2=layers.batch_norm(conv2)
              res=tf.add_n([res,actv2])
              # avgp = tf.reduce_mean(res, reduction_axis,  keep_dims=True )
              # ip = layers.fully_connected(layers.flatten(avgp),num_outputs=2,activation_fn=None, normalizer_fn=None,
                # weights_initializer=layers.xavier_initializer(uniform=False),
                # biases_initializer=None, scope='ip')
          with tf.variable_scope('Layer12'):
              conv1=self.conv2d(res,filters=32,name="conv1")
              actv1=tf.nn.relu(conv1)
              conv2=self.conv2d(actv1,filters=2,name="conv2")
              soft_max = tf.nn.softmax(conv2,dim=3,name="feature_pre")
              x=tf.reduce_mean(soft_max, axis=[1,2],keep_dims=True)
              logits=layers.flatten(x)
              print(logits.get_shape())
        self.outputs = logits
