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




class Dual_FRCNN(keras.Model):
    #def __init__(self, rpn, frcnn, text_encoder, C, **kwargs):
    def __init__(self, rpn, frcnn, text_encoder, C, **kwargs):
        super(Dual_FRCNN, self).__init__(**kwargs)
        self.rpn = rpn
        self.frcnn = frcnn
        self.text_encoder = text_encoder
        self.C = C
        (feature_map_width, feature_map_height) = nn.get_feature_map_size(rpn)
        self.feature_map_width = feature_map_width
        self.feature_map_height = feature_map_height
        self.temperature = C.temperature
        self.prev_batch = None
        #self.loss_tracker = keras.metrics.Mean(name="loss")
    
    def call(self, X, training=False):
        # Place each encoder on a separate GPU (if available).
        # TF will fallback on available devices if there are fewer than 2 GPUs.
        text_inputs = X[1]
        text_embedding = []
        with tf.device("/gpu:1"):
            # Get the embeddings for the captions.
            for im in range(len(text_inputs)):
                text_embedding.append(self.text_encoder(text_inputs[im], training=training))
                 
        text_embedding = tf.stack(text_embedding, axis=0)
        with tf.device("/gpu:0"):
            # Get the embeddings for the images.
            image_embedding = self.frcnn(X[0], training=training)

        return image_embedding, text_embedding
    
    def compute_loss(self, frcnn_pred, text_embedding, frcnn_targets):

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
            return lambda_cls_class * K.mean(categorical_crossentropy(y_true, y_pred), axis=[1])

        
        def dual_loss_cls(text_embedding, image_embeddings):
             # logits[i][j] is the dot_similarity(caption_i, image_j).
            logits = (
                tf.matmul(text_embedding, image_embeddings, transpose_b=True)
                / self.temperature
            )
            # images_similarity[i][j] is the dot_similarity(image_i, image_j).
            images_similarity = tf.matmul(
                image_embeddings, image_embeddings, transpose_b=True
            )
            # captions_similarity[i][j] is the dot_similarity(caption_i, caption_j).
            captions_similarity = tf.matmul(
                text_embedding, text_embedding, transpose_b=True
            )
            # targets[i][j] = avarage dot_similarity(caption_i, caption_j) and dot_similarity(image_i, image_j).
            targets = keras.activations.softmax(
                (captions_similarity + images_similarity) / (2 * self.temperature)
            )

            # Compute the loss for the captions using crossentropy
            captions_loss = keras.losses.categorical_crossentropy(
                 y_true=targets, y_pred=logits, from_logits=True
             )
            # Compute the loss for the images using crossentropy
            images_loss = keras.losses.categorical_crossentropy(
                y_true=tf.transpose(targets, perm=[0,2,1]), y_pred=tf.transpose(logits, perm=[0,2,1]), from_logits=True
            )

            # Return the mean of the loss over the batch.
            return lambda_cls_class * (captions_loss + images_loss) / 2

        rpn_loss_cls = rpn_loss_cls(frcnn_targets[0], frcnn_pred[0])
        rpn_loss_regr = rpn_loss_regr(frcnn_targets[1], frcnn_pred[1])
        dual_loss_cls = dual_loss_cls(text_embedding, frcnn_pred[2])
        class_loss_regr = class_loss_regr(frcnn_targets[3], frcnn_pred[3])
        
        

        return [rpn_loss_cls, rpn_loss_regr, dual_loss_cls, class_loss_regr]


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

        X, Y, pos_samples, discard, text_batch = train_helpers.second_stage_helper(X, P_rpn, img_data, C)
        

        #TODO: TEST
        bert_embeddings = np.zeros(shape=(text_batch.shape[0], text_batch.shape[1], 512), dtype=float)
        for i,im in enumerate(text_batch):
            for j,text in enumerate(im):        
                bert_embeddings[i][j] = train_helpers.bert_embed(text, C)
        text_batch = bert_embeddings
        #TODO: TEST
        
        if X is None:
            #revert and train on the previous batch
            #TODO: this is a hack, fix it later it can fail on the first batch.
            print("The RPN failed to propose useable regions in this batch, reverting to training on last good training batch")
            X, text_batch, Y = self.prev_batch

        with tf.GradientTape() as tape:
            # Forward pass
            frcnn_pred, text_embedding = self([X, text_batch], training=True)
            loss = self.compute_loss(frcnn_pred, text_embedding, Y)
        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        #self.loss_tracker.update_state(loss)
        total_loss = np.average((loss[0].numpy() + loss[1].numpy() + np.average(loss[2].numpy()) + loss[3].numpy()) / 4)
        self.prev_batch = [X, text_batch, Y]
        return {"rpn_loss_cls": loss[0].numpy(), "rpn_loss_regr": loss[1].numpy(), "embedding_loss": loss[2].numpy(), "class_loss_regr": loss[3].numpy(), "total_loss": total_loss}

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

        X, Y, pos_samples, discard, text_batch = train_helpers.second_stage_helper(X, P_rpn, img_data, C)
        
        frcnn_pred, text_embedding = self([X, text_batch], training=True)
        loss = self.compute_loss(frcnn_pred, text_embedding, Y)
        total_loss = np.average((loss[0].numpy() + loss[1].numpy() + np.average(loss[2].numpy()) + loss[3].numpy()) / 4)
        return {"rpn_loss_cls": loss[0].numpy(), "rpn_loss_regr": loss[1].numpy(), "embedding_loss": loss[2].numpy(), "class_loss_regr": loss[3].numpy(), "total_loss": total_loss}
