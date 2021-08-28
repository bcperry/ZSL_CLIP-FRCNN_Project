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
        self.accuracy = tf.keras.metrics.CategoricalAccuracy()
        #self.loss_tracker = keras.metrics.Mean(name="loss")
    
    def call(self, X, training=False):
        
        return self.frcnn(X, training=training)
    
    def compute_loss(self, frcnn_pred, frcnn_targets):
        
        lambda_rpn_regr = 1.0
        lambda_rpn_class = 1.0
        
        lambda_cls_regr = 1.0
        lambda_cls_class = 1.0
        
        epsilon = 1e-4
        num_anchors = self.rpn.outputs[0].shape[3]
        num_classes = int(self.frcnn.outputs[3].shape[2]/4)
        

        def rpn_loss_regr(y_true, y_pred):
            x = y_true[:, :, :, 4 * num_anchors:] - y_pred
            x_abs = K.abs(x)
            x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

            return lambda_rpn_regr * K.sum((y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))), axis=[1, 2, 3]) / K.sum((epsilon + y_true[:, :, :, :4 * num_anchors]), axis=[1, 2, 3])

        def rpn_loss_cls(y_true, y_pred):
            return lambda_rpn_class * K.sum((y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])), axis=[1, 2, 3]) / K.sum((epsilon + y_true[:, :, :, :num_anchors]), axis=[1, 2, 3])

        def class_loss_regr(y_true, y_pred):

            def class_loss_regr_fixed_num(y_true, y_pred):
                x = y_true[:, :, 4*num_classes:] - y_pred
                x_abs = K.abs(x)
                x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
                return lambda_cls_regr * K.sum((y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))), axis=[1, 2]) / K.sum((epsilon + y_true[:, :, :4*num_classes]), axis=[1, 2])
            
            return class_loss_regr_fixed_num(y_true, y_pred)

        
        def class_loss_cls(y_true, y_pred):
            #return lambda_cls_class * K.mean(categorical_crossentropy(y_true, y_pred), axis=[1])
            return lambda_cls_class * categorical_crossentropy(y_true, y_pred)
        
        rpn_loss_cls = rpn_loss_cls(frcnn_targets[0], frcnn_pred[0])
        rpn_loss_regr = rpn_loss_regr(frcnn_targets[1], frcnn_pred[1])
        class_loss_cls = class_loss_cls(frcnn_targets[2], frcnn_pred[2])
        class_loss_regr = class_loss_regr(frcnn_targets[3], frcnn_pred[3])
        self.accuracy.update_state(frcnn_targets[2], frcnn_pred[2].numpy())
  
        
        return [rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr]
    
    def train_step(self, batch):

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

        X, Y, pos_samples, discard, _ = train_helpers.second_stage_helper(X, P_rpn, img_data, C)

        with tf.GradientTape() as tape:
            # Forward pass
            frcnn_pred = self(X, training=True)
            loss = self.compute_loss(frcnn_pred, Y)
            
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        total_loss = np.average((loss[0].numpy() + loss[1].numpy() + np.average(loss[2].numpy(), axis=1) + loss[3].numpy()))
        return {"rpn_loss_cls": loss[0].numpy(), "rpn_loss_regr": loss[1].numpy(), "class_loss_cls": loss[2].numpy(), "class_loss_regr": loss[3].numpy(), "total_loss": total_loss, 'classifier_accuracy': self.accuracy.result().numpy()}
    def test_step(self, batch):
        self.accuracy.reset_state()
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

        X, Y, pos_samples, discard, _ = train_helpers.second_stage_helper(X, P_rpn, img_data, C)
        
        frcnn_pred = self(X, training=True)
        loss = self.compute_loss(frcnn_pred, Y)
        total_loss = np.average((loss[0].numpy() + loss[1].numpy() + np.average(loss[2].numpy(), axis=1) + loss[3].numpy()))
        return {"rpn_loss_cls": loss[0].numpy(), "rpn_loss_regr": loss[1].numpy(), "class_loss_cls": loss[2].numpy(), "class_loss_regr": loss[3].numpy(), "total_loss": total_loss, 'classifier_accuracy': self.accuracy.result().numpy()}
