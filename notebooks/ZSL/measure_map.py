import cv2
import numpy as np
import os
import time

import argparse
from keras_frcnn import config
import keras_frcnn.resnet as nn

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from sklearn.metrics import average_precision_score
from keras_frcnn import train_helpers
from tensorflow.keras.applications.resnet50 import ResNet50
from keras_frcnn.tfrecord_parser import batch_processor


import tensorflow as tf



from keras_frcnn import CLIP

from keras_frcnn.dual_frcnn import Dual_FRCNN
from keras_frcnn.frcnn import FRCNN

from keras_frcnn.tfrecord_parser import get_data

#TODO:delete
#run on 1070 only
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_map(pred, gt, f):
	T = {}
	P = {}
	iou_result = 0
	fx, fy = f

	for bbox in gt:
		bbox['bbox_matched'] = False

	pred_probs = np.array([s['prob'] for s in pred])
	#print(pred)
	#print(pred_probs)
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	for box_idx in box_idx_sorted_by_prob:
		pred_box = pred[box_idx]
		pred_class = pred_box['class']
		pred_x1 = pred_box['x1']
		pred_x2 = pred_box['x2']
		pred_y1 = pred_box['y1']
		pred_y2 = pred_box['y2']
		pred_prob = pred_box['prob']
		if pred_class not in P:
			P[pred_class] = []
			T[pred_class] = []
		P[pred_class].append(pred_prob)
		found_match = False

		for gt_box in gt:
			gt_class = gt_box['class']
			gt_x1 = gt_box['x1']/fx
			gt_x2 = gt_box['x2']/fx
			gt_y1 = gt_box['y1']/fy
			gt_y2 = gt_box['y2']/fy
			gt_seen = gt_box['bbox_matched']
			if gt_class != pred_class:
				continue
			if gt_seen:
				continue
			iou = 0
			iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
			iou_result += iou
			#print('IoU = ' + str(iou))
			if iou >= 0.5:
				found_match = True
				gt_box['bbox_matched'] = True
				break
			else:
				continue

		T[pred_class].append(int(found_match))
	for gt_box in gt:
		if not gt_box['bbox_matched']: # and not gt_box['difficult']:
			if gt_box['class'] not in P:
				P[gt_box['class']] = []
				T[gt_box['class']] = []

			T[gt_box['class']].append(1)
			P[gt_box['class']].append(0)

	#import pdb
	#pdb.set_trace()
	return T, P, iou_result

def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	fx = width/float(new_width)
	fy = height/float(new_height)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	img = img[:, :, (2, 1, 0)]
	'''
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
    '''
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)

	return img, fx, fy


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, dest='path', default=None, help='data folder containing tfrecord files and label file')
parser.add_argument('--input-weight-path', type=str, dest='input_weight_path', default=None, help='file containing pre-trained model weights')
parser.add_argument('--model-type', type=str, dest='model_type', default='ZSL', help='ZSL or FRCNN')
parser.add_argument('--num-rois', type=int, dest='num_rois', default=4, help='number of ROIs to search in the image')

args = parser.parse_args()

model_type = args.model_type
input_weight_path = args.input_weight_path 
img_path = args.path

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.num_rois = args.num_rois

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False




# pass the settings from the command line, and persist them in the config object
C = config.Config()

val_dataset, total_val_records = get_data(C.val_path, C)

if input_weight_path is not None:
    C.input_weight_path = input_weight_path
else:
    print('must provide a weight file')


C.class_mapping = train_helpers.get_class_map(C)
class_mapping_inv = {v: k for k, v in C.class_mapping.items()}
class_to_color = {C.class_mapping[v]: np.random.randint(0, 255, 3) for v in C.class_mapping}

if C.text_dict_pickle is not None:
    C.class_text = train_helpers.get_class_text(C)
else:
    C.class_text = [f"This is a picture of a {key}" for key in C.class_mapping.keys()]

#find the largest class id and add 1 for the background class
num_ids = len(C.training_classes) + 1

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
                model.load_weights(C.input_weight_path, by_name=True)
                print(f'loading dual encoder weights from {C.input_weight_path}')
except:
    print('Could not load pretrained model weights.')



#test_imgs = [s for s in all_imgs if s['imageset'] == 'test']
begin = time.time()
T = {}
P = {}
iou_result = 0
idx = 0
for img_data in val_dataset:
    print('{}/{}'.format(idx + 1,total_val_records))
    st = time.time()
    
    X = batch_processor(img_data, C)

    #initialize new X and img_data 
    X_temp = np.zeros(shape=(len(X),C.im_size, C.im_size, 3),dtype='uint8')
    img_data_temp = []
    for i in range(len(X)):
        img_data_temp.append(X[i])
        img_data_temp.append(X[i])
        X_temp[i] = X[i]['rawimage']
    X = X_temp
    img_data = img_data_temp
        
    X, fx, fy = format_img(X[0], C)


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

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
			# pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        P_all = model_all.predict([X, ROIs])
        P_cls = P_all[2]
        P_regr = P_all[3]

        for ii in range(P_cls.shape[1]):

            if np.argmax(P_cls[0, ii, :]) == 0:
                continue

            cls_name = class_mapping_inv[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
            all_dets.append(det)


    print('Elapsed time = {}'.format(time.time() - st))
    t, p, iou = get_map(all_dets, img_data[0]['bboxes'], (fx, fy))
    iou_result += iou
    for key in t.keys():
        if key not in T:
            T[key] = []
            P[key] = []
        T[key].extend(t[key])
        P[key].extend(p[key])
    all_aps = []
    for key in T.keys():
        ap = average_precision_score(T[key], P[key])
        all_aps.append(ap)
        print('{} AP: {}'.format(key, ap))

    print('mAP = {}'.format(np.nanmean(np.array(all_aps))))
    idx += 1
    if idx == total_val_records:
        print('Completely Elapsed time = {}'.format(time.time() - begin))
        print('IoU@0.50 = ' + str(iou_result/total_val_records))
        exit()
