from __future__ import division
import re
import argparse
import os
import glob
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import callbacks as callbacks
from tensorflow.keras.callbacks import Callback

from keras_frcnn import config
from keras_frcnn import resnet as nn
from keras_frcnn import train_helpers
from keras_frcnn.dual_frcnn import Dual_FRCNN
from keras_frcnn.frcnn import FRCNN
from keras_frcnn import CLIP

from keras_frcnn.tfrecord_parser import get_data


import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

#TODO:delete
#run on 1070 only
os.environ["CUDA_VISIBLE_DEVICES"]="1"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='test_tfrecords_dir', default=None, help='data folder containing tfrecord files and label file')
parser.add_argument('--input-weight-path', type=str, dest='input_weight_path', default=None, help='file containing pre-trained model weights')
parser.add_argument('--num-epochs', type=int, dest='num_epochs', default=50, help='number of epochs for training')
parser.add_argument('--model-type', type=str, dest='model_type', default='ZSL', help='ZSL or FRCNN')
parser.add_argument('--azure', type=str, dest='azure', default=True, help='is the model running on Azure')

args = parser.parse_args()

num_epochs = args.num_epochs
data_dir = args.test_tfrecords_dir
model_type = args.model_type
azure = args.azure == "True"
input_weight_path = args.input_weight_path 


if azure:
    from azureml.core import Run
    # start an Azure ML run
    run = Run.get_context()
    class LogRunMetrics(Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            run.log('Total Loss', log['total_loss'])
else:
    class LogRunMetrics(Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            print('epoch complete')

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.model_type = model_type

if model_type == 'ZSL':
    #for the ZSL to work properly, we can only have 1 roi per image, and must have a batch size greater than 1
    C.num_rois = 1
    if C.batch_size<=4:
        C.batch_size = 5

if data_dir is not None:
    C.data_path = data_dir

if num_epochs is not None:
    C.num_epochs = num_epochs

if input_weight_path is not None:
    C.input_weight_path = input_weight_path


model_path_regex = re.match("^(.+)(\.hdf5)$", C.model_path)
if model_path_regex.group(2) != '.hdf5':
    print('Output weights must have .hdf5 filetype')
    exit(1)


train_dataset, total_train_records = get_data(C.train_path, C)
val_dataset, total_val_records = get_data(C.val_path, C)

C.class_mapping = train_helpers.get_class_map(C, None)

#this gets the class text associated with the class names
if C.text_dict_pickle is not None:
    C.class_text = train_helpers.get_class_text(C)
    C.class_text = list(C.class_text.values())
else:
    C.class_text = train_helpers.get_class_map(C, r'pascal_class_text.txt')
    C.class_text = list(C.class_text.keys())

#this will renumber from 0 - x if the class keys are out of order
for i,item in enumerate(list(C.class_mapping.keys())):
    C.class_conv[C.class_mapping[item]] = i
    C.class_mapping[item]=i
    
num_ids = len(C.training_classes) + 1

print(f'Num train samples {total_train_records}')
print(f'Num val samples {total_val_records}')

#since we are using square images, we know the image size we will be scaling to
input_shape_img = (None, None, 3)
roi_input = Input(shape=(C.num_rois, 4))

optimizer = Adam(learning_rate=1e-5)
#optimizer = Adam(clipnorm=1.0)

#set a very small learning rate for the final pass
full_optimizer = Adam(learning_rate=1e-7, clipnorm=1.0)

print('Building models.')
# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
strategy = tf.distribute.get_strategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.

    # define the base network (resnet here)
    base_model = ResNet50(input_shape = input_shape_img, weights='imagenet', include_top=False)

    #freeze the feature extractor
    base_model.trainable = False

    shared_layers = Model(inputs=base_model.inputs, outputs=base_model.get_layer('conv4_block6_out').output)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers.output, num_anchors)
    model_rpn = Model(shared_layers.input, rpn[:2])

    if model_type == 'ZSL':
        classifier_ZSL = nn.classifier_ZSL(shared_layers.output, roi_input, C.num_rois, C.num_projection_layers, C.projection_dims, C.dropout_rate, nb_classes=num_ids, trainable=True)
        # this is a model that holds both the RPN and the classifier, used to train the model end to end
        model_all = Model([shared_layers.input, roi_input], rpn[:2] + classifier_ZSL)
        #build the text encoder
        text_base_embedding, text_encoder = CLIP.create_text_encoder(C)
        C.bert_embeddings = text_base_embedding.predict(tf.convert_to_tensor(C.class_text))


    else:
        classifier = nn.classifier(shared_layers.output, roi_input, C.num_rois, nb_classes=num_ids, trainable=True)
        # this is a model that holds both the RPN and the classifier, used to train the model end to end
        model_all = Model([shared_layers.input, roi_input], rpn[:2] + classifier)

if model_type == 'ZSL':
    with strategy.scope():
        model = Dual_FRCNN(model_rpn, model_all, text_encoder, C)
else:
    with strategy.scope():
        model = FRCNN(model_rpn, model_all, C)

print('Models sucessfully built.')
start_epoch = 0
#check if model has already started training by checking for a model in the outputs folder
if os.path.isdir('./outputs/model/') and C.input_weight_path is None:

    fine_tune = None
    epoch_weights = None
    files = glob.glob('./outputs/model/*')
    
    for file in files:
        fine_tune = re.search('fine_tune_epoch', file)
        if fine_tune is not None:
            fine_tune_weights = file
        else:
            epoch = int(re.search('epoch(.+?)-', file).group(1))
            
            if epoch >= start_epoch:
                epoch_weights = file
                start_epoch = epoch
    if fine_tune is not None:
        start_epoch = 0
        C.input_weight_path = fine_tune_weights
    else:
        C.input_weight_path = epoch_weights

#load weights to the model
try:
    if (C.input_weight_path == None):
        print('Loaded imagenet weights to the vision backbone.')
    else:
        model.built = True
        if model_type == 'FRCNN':
            model.load_weights(C.input_weight_path, by_name=True)
            print(f'loading FRCNN weights from {C.input_weight_path}')
        else:
            if (C.input_weight_path == None):
                print('Loaded imagenet weights to the vision backbone.')
                print('Loaded pretrained BERT weights to the text encoder.')
            else:
                print(f'loading dual encoder weights from {C.input_weight_path}')
                
                prev_wt = model_all.get_layer('res5c_branch2c').weights[1].numpy()[0]
                
                model.load_weights(C.input_weight_path, by_name=True, skip_mismatch=True)
                
                post_wt = model_all.get_layer('res5c_branch2c').weights[1].numpy()[0]

                #make the rpn un-trainable
                #model_rpn.trainable = False
                
                #there is an issue where, due to the way the model is built, when starting training of the ZSL from a pre-trained FRCNN, we cant load some weights, the below code fixes that
                if prev_wt == post_wt:
                    print('weights failed to load properly')
                    print('loading weights manually')
                    import h5py
                    f = h5py.File(C.input_weight_path, 'r')
                    
                    #load the vision side weights
                    for layer in model_all.layers:
                        layer_name = layer.name
                        value_list = []
                        if layer_name in list(f['model_2']):
                            for layer_component in range(len(layer.weights)):
                                layer_info = layer.weights[layer_component].name
                                _, sub_name = layer_info.split('/')

                                saved_shape = f['model_2'][layer_name][sub_name].shape
                                
                                saved_values = np.zeros(saved_shape, dtype=float)
                                
                                f['model_2'][layer_name][sub_name].read_direct(saved_values)
                                value_list.append(saved_values)

                            if not np.array_equal(value_list[0], layer.weights[0]):
                                print('loading weights to: ' + layer.name)
                                layer.set_weights(value_list)
                    #load the text encoder weights
                    for layer in text_encoder.layers:
                        layer_name = layer.name
                        value_list = []
                        if layer_name in list(f['text_encoder']):
                            for layer_component in range(len(layer.weights)):
                                layer_info = layer.weights[layer_component].name
                                _, sub_name = layer_info.split('/')

                                saved_shape = f['text_encoder'][layer_name][sub_name].shape
                                
                                saved_values = np.zeros(saved_shape, dtype=float)
                                
                                f['text_encoder'][layer_name][sub_name].read_direct(saved_values)
                                value_list.append(saved_values)

                            if not np.array_equal(value_list[0], layer.weights[0]):
                                print('loading weights to: ' + layer.name)
                                layer.set_weights(value_list)
                            
except:
    print('Could not load pretrained model weights.')


print('Starting training')

#Azure note:
    # create a ./outputs/model folder in the compute target
    # files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)
if model_type == 'ZSL':
    checkpoint_path = './outputs/model/ZSL_FRCNN_epoch{epoch:02d}-total_loss-{total_loss:.2f}.hdf5'
else:
    checkpoint_path = './outputs/model/FRCNN_epoch{epoch:02d}-total_loss-{total_loss:.2f}.hdf5'

#set up callbacks
checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Create a learning rate scheduler callback.
reduce_lr = callbacks.ReduceLROnPlateau(monitor="total_loss", factor=0.2, patience=3)

# Create an early stopping callback.
early_stopping = callbacks.EarlyStopping(monitor="total_loss", patience=15, restore_best_weights=True)

#this will reduce the time between evaluation by shortening the epoch lenth to less than the full training dataset size
steps_per_epoch = int(total_train_records / C.batch_size)
validation_steps = int(total_val_records / C.batch_size)

#this will reduce the time between evaluation by shortening the epoch lenth to less than the full training dataset size

#while steps_per_epoch >= 2000:
#   steps_per_epoch = int(steps_per_epoch / 2)
#this will reduce the amount of validataion data used to generate validation losses
while validation_steps >= 100:
    validation_steps = int(validation_steps / 2)
 
model.compile(optimizer= optimizer, run_eagerly = True)
    
hist = model.fit(x=train_dataset, epochs=C.num_epochs, steps_per_epoch = steps_per_epoch, validation_steps = validation_steps, initial_epoch = start_epoch, verbose='auto', validation_data=val_dataset, callbacks=[reduce_lr, checkpoint, early_stopping, LogRunMetrics()])
     

print('Primary training complete, starting fine tuning for 1 epoch.')
#set all layers of the FRCNN model to trainable, however we dont set the BERT model layers trainable
model_all.trainable = True
'''

if model_type == 'ZSL':
    checkpoint_path = './outputs/model/ZSL_FRCNN_fine_tune_epoch-total_loss-{total_loss:.2f}.hdf5'
else:
    checkpoint_path = './outputs/model/FRCNN_fine_tune_epoch-total_loss-{total_loss:.2f}.hdf5'

#set up callbacks
checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


#recompile the model
with strategy.scope():
    model.compile(optimizer= full_optimizer, run_eagerly=True)
model.fit(x=train_dataset, epochs=1, verbose='auto', steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_data=val_dataset, callbacks=[reduce_lr, early_stopping, checkpoint, LogRunMetrics()])
'''