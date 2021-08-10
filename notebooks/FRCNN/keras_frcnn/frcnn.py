# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:09:54 2021

@author: Blaine Perry
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from . import resnet as nn
from . import train_helpers
from . import config
from keras_frcnn.tfrecord_parser import batch_processor



class FRCNN(keras.Model):
    def __init__(self, rpn, frcnn, C, **kwargs):
        super(FRCNN, self).__init__(**kwargs)
        self.rpn = rpn
        self.frcnn = frcnn
        self.C = C
        (feature_map_width, feature_map_height) = nn.get_feature_map_size(rpn)
        self.feature_map_width = feature_map_width
        self.feature_map_height = feature_map_height
        #self.loss_tracker = keras.metrics.Mean(name="loss")
    
    def call(self, X, training=False):
        
        return self.frcnn(X, training=training)
    
    def compute_loss(self, frcnn_pred, frcnn_targets):
        
        num_anchors = self.frcnn.output_shape[0][3]
        num_classes = self.frcnn.output_shape[2][2]
        
        lambda_rpn_regr = 1.0
        lambda_rpn_class = 1.0
        
        lambda_cls_regr = 1.0
        lambda_cls_class = 1.0
        
        epsilon = 1e-4
        
       	def rpn_loss_regr(y_true, y_pred):
       
       		x = y_true[:, :, :, 4 * num_anchors:] - y_pred
       		x_abs = K.abs(x)
       		x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
       
       		return lambda_rpn_regr * K.sum(
       			y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])
        
       	def rpn_loss_cls(y_true, y_pred):
               return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])

        def class_loss_regr(y_true, y_pred):
            def class_loss_regr_fixed_num(y_true, y_pred):
                #subtract 1 here to take out the background class
                num_classes_cls_regr = num_classes - 1
                
                x = y_true[:, :, 4*num_classes_cls_regr:] - y_pred
                x_abs = K.abs(x)
                x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
                return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes_cls_regr] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes_cls_regr])
            return class_loss_regr_fixed_num(y_true, y_pred)

        
        def class_loss_cls(y_true, y_pred):
            return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
        
        rpn_loss_cls = rpn_loss_cls(frcnn_targets[0], frcnn_pred[0])
        rpn_loss_regr = rpn_loss_regr(frcnn_targets[1], frcnn_pred[1])
        class_loss_cls = class_loss_cls(frcnn_targets[2], frcnn_pred[2])
        class_loss_regr = class_loss_regr(frcnn_targets[3], frcnn_pred[3])
        
        

        return [rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr]


    def train_step(self, batch):
        #if running in graph mode, we will through the training step with a placeholder to build the graph
        #this currently does not work
        if batch['image'].shape[0] is None:
            #roi_input_shape = [None, self.frcnn.input_shape[1][1], self.frcnn.input_shape[1][2]]
            
            Y_rpn_cls_shape = [None, self.frcnn.output_shape[0][1], self.frcnn.output_shape[0][2], 2 * self.frcnn.output_shape[0][3]]       
            Y_rpn_reg_shape = [None, self.frcnn.output_shape[1][1], self.frcnn.output_shape[1][2], 2 * self.frcnn.output_shape[1][3]] 
            Y_cnn_cls_shape = [None, self.frcnn.output_shape[2][1], self.frcnn.output_shape[2][2]] 
            Y_cnn_reg_shape = [None, self.frcnn.output_shape[3][1], 2 * self.frcnn.output_shape[3][2]] 
            
            
            #X = [batch['image'], tf.compat.v1.placeholder(shape = roi_input_shape, dtype='float32')]
            X = [batch['image'], batch['roi_input']]
            

            Y = [tf.compat.v1.placeholder(shape = Y_rpn_cls_shape, dtype='float32'),
                 tf.compat.v1.placeholder(shape = Y_rpn_reg_shape, dtype='float32'),
                 tf.compat.v1.placeholder(shape = Y_cnn_cls_shape, dtype='float32'),
                 tf.compat.v1.placeholder(shape = Y_cnn_reg_shape, dtype='float32')]
        else:
            X = batch_processor(batch, self.C)
            C = self.C
            #initialize new X and img_data 
            X_temp = np.zeros(shape=(len(X),C.im_size, C.im_size, 3),dtype='uint8')
            img_data_temp = []
            for i in range(len(X)):
                img_data_temp.append(X[i])
                X_temp[i] = X[i]['rawimage']
            
            X = X_temp
            img_data = img_data_temp
            
            P_rpn = self.rpn(X, training = False)

            X, Y, pos_samples, discard = train_helpers.second_stage_helper(X, P_rpn, img_data, C)

            
        with tf.GradientTape() as tape:
            # Forward pass
            frcnn_pred = self(X, training=True)
            loss = self.compute_loss(frcnn_pred, Y)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        #self.loss_tracker.update_state(loss)
        return {"rpn_loss_cls": loss[0].numpy(), "rpn_loss_regr": loss[1].numpy(), "class_loss_cls": loss[2].numpy(), "class_loss_regr": loss[3].numpy()}

    def test_step(self, batch):
        X = batch_processor(batch, self.C)
        C = self.C
        #initialize new X and img_data 
        X_temp = np.zeros(shape=(len(X),C.im_size, C.im_size, 3),dtype='uint8')
        img_data_temp = []
        for i in range(len(X)):
            img_data_temp.append(X[i])
            X_temp[i] = X[i]['rawimage']
            
        X = X_temp
        img_data = img_data_temp
            
        P_rpn = self.rpn(X, training = False)

        X, Y, pos_samples, discard = train_helpers.second_stage_helper(X, P_rpn, img_data)
        
        frcnn_pred = self(X, training=True)
        loss = self.compute_loss(frcnn_pred, Y)
        return {"rpn_loss_cls": loss[0].numpy(), "rpn_loss_regr": loss[1].numpy(), "class_loss_cls": loss[2].numpy(), "class_loss_regr": loss[3].numpy()}
