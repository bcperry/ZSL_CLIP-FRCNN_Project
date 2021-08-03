# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:09:54 2021

@author: Blaine Perry
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_frcnn.roi_helpers as roi_helpers
import random
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy




from tensorflow.python.eager import def_function

from tensorflow.python.ops import math_ops


import tensorflow.python.ops.numpy_ops.np_config as npc
npc.enable_numpy_behavior()


class FRCNN(keras.Model):
    def __init__(self, rpn, frcnn, C, class_mapping, num_classes, **kwargs):
        super(FRCNN, self).__init__(**kwargs)
        self.rpn = rpn
        self.frcnn = frcnn
        self.class_mapping = class_mapping
        self.C = C
        self.num_classes = num_classes

    
    def call(self, X, training=False):

        return self.frcnn(X)
    
    def compute_loss(self, frcnn_pred, frcnn_targets):
        
        num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        
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
            #num_classes - 1 since we dont count the background class
       		x = y_true[:, :, 4*self.num_classes-1:] - y_pred
       		x_abs = K.abs(x)
       		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
       		return lambda_cls_regr * K.sum(y_true[:, :, :4*self.num_classes-1] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*self.num_classes-1])
        
        def class_loss_cls(y_true, y_pred):
        	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
        
        rpn_loss_cls = rpn_loss_cls()
        rpn_loss_regr = rpn_loss_regr()
        class_loss_cls = class_loss_cls()
        class_loss_regr = class_loss_regr()
        
        

        return [rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr]


    def train_step(self, data):
        
        if print(data['image'].shape[0]) is None:
            data_2 = [data['image'], tf.compat.v1.placeholder(shape = [None, 200, 4], dtype='uint8')]
        
        else:
            rpn_pred = self.rpn(data['image'])
            
            #R input and output are in feature space
            R = roi_helpers.rpn_to_roi(rpn_pred[0], rpn_pred[1], self.C, use_regr=True, overlap_thresh=0.1, max_boxes=900)
            
            class_id = tf.sparse.to_dense(data['image/object/class/label']).numpy()
            
            image = data['image'].numpy()
            (w, h) = image.shape[:2]
            
            x1 = (tf.sparse.to_dense(data['image/object/bbox/xmin']).numpy() * w).round().astype(int)
            x2 = (tf.sparse.to_dense(data['image/object/bbox/xmax']).numpy() * w).round().astype(int)
            y1 = (tf.sparse.to_dense(data['image/object/bbox/ymin']).numpy() * h).round().astype(int)
            y2 = (tf.sparse.to_dense(data['image/object/bbox/ymax']).numpy() * h).round().astype(int)
            
            #drop 0s due to the sparse to dense
            zeroes_map = class_id != 0
            class_id = class_id[zeroes_map]
            x1 = x1[zeroes_map]
            x2 = x2[zeroes_map]
            y1 = y1[zeroes_map]
            y2 = y2[zeroes_map]
                
            try:
                assert len(class_id) == len(x1) == len(x2) == len(y1) == len(y2)
            except Exception as e:
                print(f'Exception: {e}')
                
            
            
            img_data = {}
            bboxes = {}
            width = data['image'].shape[1]
            height = data['image'].shape[2]
            for i in range(len(class_name)):
                img_data['bboxes'].append({'class': class_name[i], 'x1': x1[i], 'x2': x2[i], 'y1': y1[i], 'y2': y2[i]})
            
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, self.C, self.class_mapping)
            
            if X2 is None:
                return None
    
    
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)
            
            neg_samples = list(neg_samples[0])
            pos_samples = list(pos_samples[0])
            
    
            if self.C.num_rois > 1:
                selected_pos_samples = []
                selected_neg_samples= []
                
                if len(pos_samples) < self.C.num_rois//2:
                    selected_pos_samples = pos_samples
                else:
                    selected_pos_samples = np.random.choice(pos_samples, self.C.num_rois//2, replace=False)
                #if there are no negative samples, use only positive samples
                if len(neg_samples) == 0:
                    selected_pos_samples = np.random.choice(pos_samples, self.C.num_rois, replace=True).tolist()
                else:
                #add negative samples to fill out the total necessary
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, self.C.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, self.C.num_rois - len(selected_pos_samples), replace=True).tolist()
    
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples
                selected_neg_samples = neg_samples
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)
                    
            #cast Y1 and Y2 to float32 in case there are no positive samples in the selection
            Y1 = Y1.astype('float32')
            Y2 = Y2.astype('float32')
            
            frcnn_targets = [Y[0], Y[1], Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                
        with tf.GradientTape() as tape:
            # Forward pass
            frcnn_pred = self(data_2, training=True)
            loss = self.compute_loss(frcnn_pred, frcnn_targets)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        caption_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
