from __future__ import division
import argparse
import os
import time
import cv2

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50


from keras_frcnn import config
from keras_frcnn import resnet as nn
from keras_frcnn import train_helpers
from keras_frcnn import CLIP

from keras_frcnn.dual_frcnn import Dual_FRCNN
from keras_frcnn.frcnn import FRCNN
import keras_frcnn.roi_helpers as roi_helpers

import logging


def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	ratio = (img_min_side/width, img_min_side/height)
	img = cv2.resize(img, (C.im_size, C.im_size), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	#img = img.astype(np.float32)
	'''
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
    '''
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
	width_ratio, height_ratio = ratio
	real_x1 = int(round(x1 // width_ratio))
	real_y1 = int(round(y1 // height_ratio))
	real_x2 = int(round(x2 // width_ratio))
	real_y2 = int(round(y2 // height_ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


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
parser.add_argument('--path', type=str, dest='path', default=None, help='data folder containing tfrecord files and label file')
parser.add_argument('--input-weight-path', type=str, dest='input_weight_path', default=None, help='file containing pre-trained model weights')
parser.add_argument('--num-rois', type=int, dest='num_rois', default=5, help='number of regions of interest to process')
parser.add_argument('--model-type', type=str, dest='model_type', default='ZSL', help='ZSL or FRCNN')


args = parser.parse_args()

model_type = args.model_type
input_weight_path = args.input_weight_path 
img_path = args.path
num_rois = args.num_rois

# pass the settings from the command line, and persist them in the config object
C = config.Config()


if input_weight_path is not None:
    C.input_weight_path = input_weight_path
else:
    print('must provide a weight file')

C.num_rois = num_rois

C.class_mapping = train_helpers.get_class_map(C, None)
class_mapping_inv = {v: k for k, v in C.class_mapping.items()}
class_to_color = {C.class_mapping[v]: np.random.randint(0, 255, 3) for v in C.class_mapping}

#this gets the class text associated with the class names
if C.text_dict_pickle is not None:
    C.class_text = train_helpers.get_class_text(C)
    C.class_text = list(C.class_text.values())
else:
    C.class_text = train_helpers.get_class_map(C, r'pascal_class_text.txt')
    C.class_text = list(C.class_text.keys())

#find the largest class id and add 1 for the background class
num_ids = len(C.training_classes)

#since we are using square images, we know the image size we will be scaling to
input_shape_img = (None, None, 3)
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

print('Models sucessfully built.')
#check if model has already started training by checking for a model in the outputs folder
        
if model_type == 'ZSL':

    model = Dual_FRCNN(model_rpn, model_all, text_encoder, C)
    model.built = True
else:

    model = FRCNN(model_rpn, model_all, C)
    model.built = True
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
                prev_wt = model_all.get_layer('time_distributed_5').weights[1].numpy()[0]
                prev_wt2 = model_all.get_layer('res5c_branch2a').weights[1].numpy()[0]
                model.load_weights(C.input_weight_path, by_name=True, skip_mismatch=True)
                post_wt = model_all.get_layer('time_distributed_5').weights[1].numpy()[0]
                post_wt2 = model_all.get_layer('res5c_branch2a').weights[1].numpy()[0]
                if prev_wt2 == post_wt2:
                    print('weights failed to load properly')
                    import sys
                    sys.exit()
except:
    print('Could not load pretrained model weights.')
    exit()
all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True

if model_type == 'ZSL':
    print(f"Generating embeddings for {len(C.class_text)} labels...")
    text_embeddings = text_encoder.predict(C.bert_embeddings)
    print(f"Text embeddings shape: {text_embeddings.shape}.")
    

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    st = time.time()
    filepath = os.path.join(img_path,img_name)
    #reads in BGR
    img = cv2.imread(filepath) 

    X, ratio = format_img(img, C)

    X = np.transpose(X, (0, 2, 3, 1))
    
	# get the feature maps and output from the RPN
    P_rpn = model_rpn.predict(X)

    R = roi_helpers.rpn_to_roi(C, P_rpn[0], P_rpn[1], use_regr=True, overlap_thresh=0.8, max_boxes=300)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
			#pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        P_all = model_all.predict([X, ROIs])
        
        
        if model_type == 'ZSL':
        
            dot_similarity = tf.matmul(P_all[2][0], 100 * text_embeddings, transpose_b=True)
            true_probs = tf.keras.activations.softmax(dot_similarity, axis=1)
            results = []
            probabilities = []
            results_topk = []
            probs_topk = []
            
            for roi in range(true_probs.shape[0]):
                results.append(tf.math.top_k(dot_similarity[roi], 1).indices.numpy())
                probabilities.append(tf.math.top_k(true_probs[roi], 1).values.numpy())
                results_topk.append(tf.math.top_k(dot_similarity[roi], 5).indices.numpy())
                probs_topk.append(tf.math.top_k(true_probs[roi], 5).values.numpy())
            results = np.array(results)
            probabilities = np.array(probs)
            results_topk = np.array(results_topk)
            probs_topk = np.array(probs_topk)
            
            #replace the image embedding output with the class probabilities calculated above
            P_all[2] = np.expand_dims(true_probs, axis = 0)
            
        P_cls = P_all[2]
        P_regr = P_all[3]

        for ii in range(P_cls.shape[1]):
            
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == 0:
                continue

            cls_name = class_mapping_inv[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.1)

        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[C.class_mapping[key]][0]), int(class_to_color[C.class_mapping[key]][1]), int(class_to_color[C.class_mapping[key]][2])),2)

            textLabel = f'{key}: {int(100*new_probs[jk])}'
            all_dets.append((key,100*new_probs[jk]))

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            
    print(f'Elapsed time = {(time.time() - st)}')
    print(all_dets)
    cv2.imshow('image window', img)
    # add wait key. window waits until user presses a key
    cv2.waitKey(0)
    # and finally destroy/close all open windows
    cv2.destroyAllWindows()
        
    
