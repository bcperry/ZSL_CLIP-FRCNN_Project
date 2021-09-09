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
from keras_frcnn.debug_helper import show_train_img


class FRCNN(keras.Model):
    def __init__(self, rpn, frcnn, C, **kwargs):
        super(FRCNN, self).__init__(**kwargs)
        self.rpn = rpn
        self.frcnn = frcnn
        self.C = C
        (feature_map_width, feature_map_height) = nn.get_feature_map_size(rpn)
        self.feature_map_width = feature_map_width
        self.feature_map_height = feature_map_height
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name="classifier_accuracy")
        self.prev_batch = None
        self.total_loss = keras.metrics.Mean(name="total_loss")
        self.rpn_cls_loss = keras.metrics.Mean(name="rpn_cls_loss")
        self.rpn_reg_loss = keras.metrics.Mean(name="rpn_reg_loss")
        self.class_cls_loss = keras.metrics.Mean(name="class_cls_loss")
        self.class_reg_loss = keras.metrics.Mean(name="class_reg_loss")



    def call(self, X, training=False):
        pred_0 = []
        pred_1 = []
        pred_2 = []
        pred_3 = []
        
        #in order to keep the images and ROIs together, we have to split the batch and run it individually before re-combining the answers
        for im in range(X[0].shape[0]):
            preds = self.frcnn([X[0][im:im+1], X[1][im:im+1]], training=training)
            pred_0.append(preds[0][0])
            pred_1.append(preds[1][0])
            pred_2.append(preds[2][0])
            pred_3.append(preds[3][0])
                 
        pred_0 = tf.stack(pred_0, axis=0)
        pred_1 = tf.stack(pred_1, axis=0)
        pred_2 = tf.stack(pred_2, axis=0)
        pred_3 = tf.stack(pred_3, axis=0)
        return [pred_0, pred_1, pred_2, pred_3]

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
                x = y_true[:, :, 4*(num_classes):] - y_pred
                x_abs = K.abs(x)
                x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
                return lambda_cls_regr * K.sum((y_true[:, :, :4*(num_classes)] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))), axis=[1, 2]) / K.sum((epsilon + y_true[:, :, :4*(num_classes)]), axis=[1, 2])
            
            return class_loss_regr_fixed_num(y_true, y_pred)


        def class_loss_cls(y_true, y_pred):
            return lambda_cls_class * K.mean(categorical_crossentropy(y_true, y_pred), axis=[1])
            #return lambda_cls_class * categorical_crossentropy(y_true, y_pred)
        
        rpn_loss_cls = rpn_loss_cls(frcnn_targets[0], frcnn_pred[0])
        rpn_loss_regr = rpn_loss_regr(frcnn_targets[1], frcnn_pred[1])
        class_loss_cls = class_loss_cls(frcnn_targets[2], frcnn_pred[2])
        class_loss_regr = class_loss_regr(frcnn_targets[3], frcnn_pred[3])
        self.accuracy.update_state(frcnn_targets[2], frcnn_pred[2])
  
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
        
        X, Y, pos_samples, discard, _, batch_order = train_helpers.second_stage_helper(X, P_rpn, img_data, C)
        
        if X is None:
            #revert and train on the previous batch
            #TODO: this is a hack, fix it later it can fail on the first batch.
            #print("The RPN failed to propose useable regions in this batch, reverting to training on last good training batch")
            X, Y, pos_samples = self.prev_batch
        
        #for im in range(len(img_data)):
        #    show_train_img(img_data[batch_order.index(im)], np.expand_dims(X[0][im], axis = 0), [np.expand_dims(Y[0][im], axis = 0), np.expand_dims(Y[1][im], axis = 0)], C, pos_samples, X[0][im], True)
        
        with tf.GradientTape() as tape:
            # Forward pass
            frcnn_pred = self(X, training=True)
            loss = self.compute_loss(frcnn_pred, Y)
            
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.prev_batch = [X, Y, pos_samples]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        total_loss = np.average((loss[0].numpy() + loss[1].numpy() + loss[2].numpy() + loss[3].numpy()))
        self.rpn_cls_loss.update_state(loss[0].numpy()[0])
        self.rpn_reg_loss.update_state(loss[1].numpy()[0])
        self.class_cls_loss.update_state(loss[2].numpy()[0])
        self.class_reg_loss.update_state(loss[3].numpy()[0])
        self.total_loss.update_state(total_loss)


        return {"rpn_cls_loss": self.rpn_cls_loss.result(), "rpn_reg_loss": self.rpn_reg_loss.result(), 
                "class_cls_loss": self.class_cls_loss.result(), "class_reg_loss": self.class_reg_loss.result(), 
                "total_loss": self.total_loss.result(), 
                'classifier_accuracy': self.accuracy.result()}
        #initialize new X and img_data 
        X_temp = np.zeros(shape=(len(X),C.im_size, C.im_size, 3),dtype='uint8')
        img_data_temp = []
        for i in range(len(X)):
            img_data_temp.append(X[i])
            X_temp[i] = X[i]['rawimage']
            

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

        X, Y, pos_samples, discard, _, batch_order = train_helpers.second_stage_helper(X, P_rpn, img_data, C)
        
        frcnn_pred = self(X, training=False)
        loss  = self.compute_loss(frcnn_pred, Y)
        total_loss = np.average((loss[0].numpy() + loss[1].numpy() + loss[2].numpy() + loss[3].numpy()))
        self.rpn_cls_loss.update_state(loss[0].numpy()[0])
        self.rpn_reg_loss.update_state(loss[1].numpy()[0])
        self.class_cls_loss.update_state(loss[2].numpy()[0])
        self.class_reg_loss.update_state(loss[3].numpy()[0])
        self.total_loss.update_state(total_loss)


        return {"rpn_cls_loss": self.rpn_cls_loss.result(), "rpn_reg_loss": self.rpn_reg_loss.result(), 
                "class_cls_loss": self.class_cls_loss.result(), "class_reg_loss": self.class_reg_loss.result(), 
                "total_loss": self.total_loss.result(), 
                'classifier_accuracy': self.accuracy.result()}