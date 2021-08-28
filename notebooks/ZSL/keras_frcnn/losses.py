from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
import keras

from keras_frcnn import config

import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

C = config.Config()



def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
        return lambda_rpn_regr * K.sum((
       			y_true * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))), axis=[1, 2, 3]) / K.sum((epsilon + y_true), axis=[1, 2, 3])
    return rpn_loss_regr_fixed_num

def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_rpn_class * K.sum((y_true * K.binary_crossentropy(y_pred, y_true)), axis=[1, 2, 3]) / K.sum((epsilon + y_true), axis=[1, 2, 3])
    return rpn_loss_cls_fixed_num

def class_loss_regr(y_true, y_pred):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum((y_true * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))), axis=[1, 2]) / K.sum((epsilon + y_true), axis=[1, 2])
    return class_loss_regr_fixed_num(y_true, y_pred)


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true, y_pred), axis=[1])

def dual_loss_cls(text_embedding, image_embeddings):
     # logits[i][j] is the dot_similarity(caption_i, image_j).
    logits = (
        tf.matmul(text_embedding, image_embeddings, transpose_b=True)
        / C.temperature
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
        (captions_similarity + images_similarity) / (2 * C.temperature)
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