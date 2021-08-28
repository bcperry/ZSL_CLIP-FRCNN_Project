

from __future__ import division
import re
import argparse
import os


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import callbacks as callbacks
from tensorflow.keras.callbacks import Callback

from keras_frcnn import config
from keras_frcnn import losses as losses
from keras_frcnn import resnet as nn
from keras_frcnn import train_helpers
from keras_frcnn.dual_frcnn import Dual_FRCNN
from keras_frcnn.frcnn import FRCNN
from keras_frcnn import CLIP


from keras_frcnn.tfrecord_parser import get_data
from tensorflow.keras import layers

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.model_path = r"C:\Data_drive\Github\GEOINT_Zero-shot_Object_identification\notebooks\ZSL\outputs\model\ZSL_FRCNN_fine_tune_epoch-total_loss-2.23.hdf5"
C.train_path = r"C:\Data_drive\Data"
model_type = 'ZSL'


model_path_regex = re.match("^(.+)(\.hdf5)$", C.model_path)
if model_path_regex.group(2) != '.hdf5':
    print('Output weights must have .hdf5 filetype')
    exit(1)


train_dataset, total_train_records = get_data('train', C)
val_dataset, total_val_records = get_data('test', C)

C.class_mapping = train_helpers.get_class_map(C)

C.class_text = train_helpers.get_class_text(C)

#find the largest class id and add 1 for the background class
num_ids = C.class_mapping[max(C.class_mapping, key=C.class_mapping.get)] + 1

print(f'Num train samples {total_train_records}')
print(f'Num val samples {total_val_records}')

#since we are using square images, we know the image size we will be scaling to
input_shape_img = (C.im_size, C.im_size, 3)
roi_input = Input(shape=(C.num_rois, 4))

print('Building models.')
# define the base network (resnet here)
base_model = ResNet50(input_shape = input_shape_img, weights='imagenet', include_top=False)

#freeze the feature extractor
base_model.trainable = False

shared_layers = Model(inputs=base_model.inputs, outputs=base_model.get_layer('conv4_block6_out').output)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers.output, num_anchors)
model_rpn = Model(shared_layers.input, rpn[:2])
optimizer = Adam()

if model_type == 'ZSL':
    classifier_ZSL = nn.classifier_ZSL(shared_layers.output, roi_input, C.num_rois, C.num_projection_layers, C.projection_dims, C.dropout_rate, nb_classes=num_ids, trainable=True)
    # this is a model that holds both the RPN and the classifier, used to train the model end to end
    model_all = Model([shared_layers.input, roi_input], rpn[:2] + classifier_ZSL)
    #build the text encoder
    text_encoder = CLIP.create_text_encoder(C)
else:
    classifier = nn.classifier(shared_layers.output, roi_input, C.num_rois, nb_classes=num_ids, trainable=True)
    # this is a model that holds both the RPN and the classifier, used to train the model end to end
    model_all = Model([shared_layers.input, roi_input], rpn[:2] + classifier)

if model_type == 'ZSL':
    Dual_FRCNN = Dual_FRCNN(model_rpn, model_all, text_encoder, C)
    Dual_FRCNN.compile(optimizer= optimizer, run_eagerly = True)
    
print('Models sucessfully built.')

try:
    if (C.model_path == None):
        print('Loaded imagenet weights to the vision backbone.')
    else:
        print('loading weights from {C.model_path}')
        Dual_FRCNN.load_weights(C.model_path, by_name=True)

except:
    print('Could not load pretrained model weights.')