from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
import re
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras_frcnn import data_generators
import config
from keras_frcnn import losses as losses
from keras.utils import generic_utils
from tensorflow.keras.applications.resnet50 import ResNet50

from keras_frcnn.tfrecord_parser import get_data
from keras_frcnn.train_helpers import second_stage_helper
from keras_frcnn import resnet as nn
import argparse
#from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='test_tfrecords_dir', default=None, help='data folder containing tfrecord files and label file')
parser.add_argument('--num-epochs', type=int, dest='num_epochs', default=50, help='number of epochs for training')


args = parser.parse_args()

num_epochs = args.num_epochs
data_dir = args.test_tfrecords_dir


sys.setrecursionlimit(40000)

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
   

train_imgs, classes_count, class_mapping, train_dataset = get_data(C.train_path, 'train')
val_imgs, _, _, val_dataset = get_data(C.train_path, 'test')


if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print(f'Num classes (including bg) = {len(classes_count)}')


random.shuffle(train_imgs)

num_imgs = len(train_imgs)

print(f'Num train samples {len(train_imgs)}')
print(f'Num val samples {len(val_imgs)}')

#since we are using square images, we know the image size we will be scaling to
input_shape_img = (C.im_size, C.im_size, 3)

img_input = Input(shape=input_shape_img)
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

classifier = nn.classifier(shared_layers.output, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(shared_layers.input, rpn[:2])
model_classifier = Model([shared_layers.input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([shared_layers.input, roi_input], rpn[:2] + classifier)

print('Models sucessfully built.')
#find the feature map dimensions
(feature_map_width, feature_map_height) = nn.get_feature_map_size(model_rpn)

#Create the data generators
data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, feature_map_width, feature_map_height, mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, feature_map_width, feature_map_height, mode='val')


try:
    if (C.input_weight_path == None):
        print('Loaded imagenet weights to backbone.')
    else:
        print('loading weights from {C.input_weight_path}')
        model_rpn.load_weights(C.input_weight_path, by_name=True)
        model_classifier.load_weights(C.input_weight_path, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam()

model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_all.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={f'dense_class_{len(classes_count)}': 'accuracy'})


epoch_length = len(train_imgs)
num_epochs = int(C.num_epochs)
iter_num = 0

training_losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

# start an Azure ML run
#run = Run.get_context()

for epoch_num in range(num_epochs):

    progbar = generic_utils.Progbar(epoch_length)
    print(f'Epoch {epoch_num + 1}/{num_epochs}')

    while True:
        try:

            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print(f'Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes} for {epoch_length} previous iterations')
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            
            #X is in the resized image space, Y is in the feature space
            X, Y, img_data = next(data_gen_train)
            
            X2, Y2, pos_samples = second_stage_helper(X, model_rpn, img_data, C)
                

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))


            loss_all = model_all.train_on_batch([X, X2], [Y[0], Y[1], Y2[0], Y2[1]], return_dict=True)


            training_losses[iter_num, 0] = loss_all['rpn_out_class_loss']
            training_losses[iter_num, 1] = loss_all['rpn_out_regress_loss']
            
            training_losses[iter_num, 2] = loss_all['dense_class_{}_loss'.format(len(classes_count))]
            training_losses[iter_num, 3] = loss_all['dense_regress_{}_loss'.format(len(classes_count))]
            training_losses[iter_num, 4] = loss_all['dense_class_{}_accuracy'.format(len(classes_count))]

            progbar.update(iter_num+1, [('rpn_cls', training_losses[iter_num, 0]), ('rpn_regr', training_losses[iter_num, 1]),
                                      ('detector_cls', training_losses[iter_num, 2]), ('detector_regr', training_losses[iter_num, 3])])

            iter_num += 1

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(training_losses[:, 0])
                loss_rpn_regr = np.mean(training_losses[:, 1])
                loss_class_cls = np.mean(training_losses[:, 2])
                loss_class_regr = np.mean(training_losses[:, 3])
                class_acc = np.mean(training_losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []
                

                if C.verbose:
                    print(f'Mean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_bboxes}')
                    print(f'Classifier accuracy for bounding boxes from RPN: {class_acc}')
                    print(f'Loss RPN classifier: {loss_rpn_cls}')
                    print(f'Loss RPN regression: {loss_rpn_regr}')
                    print(f'Loss Detector classifier: {loss_class_cls}')
                    print(f'Loss Detector regression: {loss_class_regr}')
                    print(f'Elapsed time: {time.time() - start_time}')

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                
                #log the total loss for azure
                #run.log('Loss', curr_loss)

                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print(f'Total loss decreased from {best_loss} to {curr_loss}, saving weights')
                    best_loss = curr_loss
                
                # create a ./outputs/model folder in the compute target
                # files saved in the "./outputs" folder are automatically uploaded into run history
                os.makedirs('./outputs/model', exist_ok=True)


                # save model weights
                print("Training completed. Saving model...")
                model_all.save_weights('./outputs/model/' + model_path_regex.group(1) + "_" + '{:04d}'.format(epoch_num) + model_path_regex.group(2))
                print("model saved in ./outputs/model folder")
                

                break
        
        except Exception as e:
            print(f'Exception: {e}')
            continue
        



print('Primary training complete, starting fine tuning for 1 epoch.')
#set all layers of the model to trainable
model_all.trainable = True
#set a very small learning rate
optimizer = Adam(learning_rate=1e-5)

#recompile the model
model_all.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={f'dense_class_{len(classes_count)}': 'accuracy'})

progbar = generic_utils.Progbar(epoch_length)
print('Fine Tune Epoch')

while True:
    try:

        if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
            mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
            rpn_accuracy_rpn_monitor = []
            print(f'Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes} for {epoch_length} previous iterations')
            if mean_overlapping_bboxes == 0:
                print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
        
        #X is in the resized image space, Y is in the feature space
        X, Y, img_data = next(data_gen_train)
        
        X2, Y2, pos_samples = second_stage_helper(X, model_rpn, img_data, C)
            

        if X2 is None:
            rpn_accuracy_rpn_monitor.append(0)
            rpn_accuracy_for_epoch.append(0)
            continue

        rpn_accuracy_rpn_monitor.append(len(pos_samples))
        rpn_accuracy_for_epoch.append((len(pos_samples)))

        loss_all = model_all.train_on_batch([X, X2], [Y[0], Y[1], Y2[0], Y2[1]], return_dict=True)


        training_losses[iter_num, 0] = loss_all['rpn_out_class_loss']
        training_losses[iter_num, 1] = loss_all['rpn_out_regress_loss']
        
        training_losses[iter_num, 2] = loss_all['dense_class_{}_loss'.format(len(classes_count))]
        training_losses[iter_num, 3] = loss_all['dense_regress_{}_loss'.format(len(classes_count))]
        training_losses[iter_num, 4] = loss_all['dense_class_{}_accuracy'.format(len(classes_count))]

        progbar.update(iter_num+1, [('rpn_cls', training_losses[iter_num, 0]), ('rpn_regr', training_losses[iter_num, 1]),
                                    ('detector_cls', training_losses[iter_num, 2]), ('detector_regr', training_losses[iter_num, 3])])

        iter_num += 1

        if iter_num == epoch_length:
            loss_rpn_cls = np.mean(training_losses[:, 0])
            loss_rpn_regr = np.mean(training_losses[:, 1])
            loss_class_cls = np.mean(training_losses[:, 2])
            loss_class_regr = np.mean(training_losses[:, 3])
            class_acc = np.mean(training_losses[:, 4])

            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
            rpn_accuracy_for_epoch = []
            

            if C.verbose:
                print(f'Mean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_bboxes}')
                print(f'Classifier accuracy for bounding boxes from RPN: {class_acc}')
                print(f'Loss RPN classifier: {loss_rpn_cls}')
                print(f'Loss RPN regression: {loss_rpn_regr}')
                print(f'Loss Detector classifier: {loss_class_cls}')
                print(f'Loss Detector regression: {loss_class_regr}')
                print(f'Elapsed time: {time.time() - start_time}')

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            
            #log the total loss for azure
            #run.log('Loss', curr_loss)

            iter_num = 0
            start_time = time.time()

            if curr_loss < best_loss:
                if C.verbose:
                    print(f'Total loss decreased from {best_loss} to {curr_loss}, saving weights')
                best_loss = curr_loss
            
            # create a ./outputs/model folder in the compute target
            # files saved in the "./outputs" folder are automatically uploaded into run history
            os.makedirs('./outputs/model', exist_ok=True)


            # save model weights
            print("Training completed. Saving model...")
            model_all.save_weights('./outputs/model/' + model_path_regex.group(1) + "_" + '{:04d}'.format(num_epochs + 1) + model_path_regex.group(2))
            print("model saved in ./outputs/model folder")
            

            break
    
    except Exception as e:
        print(f'Exception: {e}')
        continue

