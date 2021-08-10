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
from keras_frcnn.frcnn import FRCNN

from keras_frcnn.tfrecord_parser import get_data

from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='test_tfrecords_dir', default=None, help='data folder containing tfrecord files and label file')
parser.add_argument('--num-epochs', type=int, dest='num_epochs', default=50, help='number of epochs for training')

args = parser.parse_args()

num_epochs = args.num_epochs
data_dir = args.test_tfrecords_dir

# pass the settings from the command line, and persist them in the config object
C = config.Config()


if data_dir is not None:
    C.train_path = data_dir

if num_epochs is not None:
    C.num_epochs = num_epochs


model_path_regex = re.match("^(.+)(\.hdf5)$", C.model_path)
if model_path_regex.group(2) != '.hdf5':
    print('Output weights must have .hdf5 filetype')
    exit(1)


train_dataset, total_train_records = get_data('train', C)
val_dataset, total_val_records = get_data('test', C)

C.class_mapping = train_helpers.get_class_map(C)

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

classifier = nn.classifier(shared_layers.output, roi_input, C.num_rois, nb_classes=num_ids, trainable=True)

model_rpn = Model(shared_layers.input, rpn[:2])

# this is a model that holds both the RPN and the classifier, used to train the model end to end
model_all = Model([shared_layers.input, roi_input], rpn[:2] + classifier)

print('Models sucessfully built.')

try:
    if (C.input_weight_path == None):
        print('Loaded imagenet weights to backbone.')
    else:
        print('loading weights from {C.input_weight_path}')
        model_all.load_weights(C.input_weight_path, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam()

model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_all.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), losses.class_loss_cls, losses.class_loss_regr(num_ids-1)], metrics={f'dense_class_{num_ids}': 'accuracy'})

print('Starting training')

vis = True

# start an Azure ML run

run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['val_loss'])

#Azure note:
    # create a ./outputs/model folder in the compute target
    # files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

checkpoint_path = './outputs/model/frcnn_epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'


#set up callbacks
checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Create a learning rate scheduler callback.
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_class_loss_cls", factor=0.2, patience=3)

# Create an early stopping callback.
early_stopping = callbacks.EarlyStopping(monitor="val_class_loss_cls", patience=5, restore_best_weights=True)


FRCNN = FRCNN(model_rpn, model_all, C)
FRCNN.compile(optimizer= optimizer, run_eagerly=True)
FRCNN.fit(x=train_dataset, epochs=C.num_epochs, verbose='auto', validation_data=val_dataset, callbacks=[reduce_lr, early_stopping, checkpoint, LogRunMetrics()])


print('Primary training complete, starting fine tuning for 1 epoch.')
#set all layers of the model to trainable
FRCNN.trainable = True
#set a very small learning rate
optimizer = Adam(learning_rate=1e-5)

#recompile the model
FRCNN.compile(optimizer= Adam(learning_rate=1e-5), run_eagerly=True)
FRCNN.fit(x=train_dataset, epochs=1, verbose='auto', validation_data=val_dataset, callbacks=[reduce_lr, early_stopping, checkpoint, LogRunMetrics()])
