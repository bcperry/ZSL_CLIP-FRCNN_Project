# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 08:59:10 2021

@author: blaine perry
"""

import keras_frcnn.roi_helpers as roi_helpers
import numpy as np
import random
from . import config
from . import data_generators
import glob


def get_class_map(C):
    
    label_file = glob.glob(C.train_path + '/*labels.txt')
    print(label_file)
    class_mapping = {}
    file = open(label_file[0], "r")
    
    #create a class label dictionary
    for line in file:
        key, value = line.split(':')
        class_mapping[value.strip()] = int(key)

    if 'bg' not in class_mapping:
        #add the background class
        if class_mapping[min(class_mapping, key=class_mapping.get)] != 0:
            class_mapping['bg'] = 0
        else:
            print('a class with id 0 exists')
            exit()
    
    return class_mapping

def second_stage_helper(X, P_rpn, img_data, C):
    
    class_mapping = get_class_map(C)

    #find the largest class id
    num_ids = class_mapping[max(class_mapping, key=class_mapping.get)]
    
    discard = []
    batch_pos_samples = 0
    
    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios
    num_anchors = len(anchor_sizes) * len(anchor_ratios)


    #P_rpn = model_rpn.predict_on_batch(X)
    #test = model_all.predict_on_batch([X, batch['roi_input']])
    
    Y1_rpn_batch = np.zeros(shape=(len(X), P_rpn[0].shape[1], P_rpn[0].shape[2], num_anchors * 2), dtype='float32')
    Y2_rpn_batch = np.zeros(shape=(len(X), P_rpn[0].shape[1], P_rpn[0].shape[2], num_anchors * 8), dtype='float32')
    
    X2_batch = np.zeros(shape=(len(X), C.num_rois, 4), dtype='float32')
    Y1_batch = np.zeros(shape=(len(X), C.num_rois, num_ids+1), dtype='float32')
    Y2_batch = np.zeros(shape=(len(X), C.num_rois, (num_ids)*8), dtype='float32')

    for im in range(X.shape[0]):
        #get the rpn targets
        _, rpn_targets, _ = data_generators.calc_targets(X[im], img_data[im], P_rpn[0].shape[1], P_rpn[0].shape[2])
        
        
        #R input and output are in feature space
        R = roi_helpers.rpn_to_roi(P_rpn[0][im:im+1], P_rpn[1][im:im+1], use_regr=True, overlap_thresh=0.1, max_boxes=900)
    
        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data[im])

        if X2 is None:
            discard.append(im)
            continue
        
        neg_samples = np.where(Y1[0, :, -1] == 1)
        pos_samples = np.where(Y1[0, :, -1] == 0)
        
        neg_samples = list(neg_samples[0])
        pos_samples = list(pos_samples[0])
        batch_pos_samples += len(pos_samples)
        
        if C.num_rois > 1:
            selected_pos_samples = []
            selected_neg_samples= []
            
            if len(pos_samples) < C.num_rois//2:
                selected_pos_samples = pos_samples
            else:
                selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False)
            #if there are no negative samples, use only positive samples
            if len(neg_samples) == 0:
                selected_pos_samples = np.random.choice(pos_samples, C.num_rois, replace=True).tolist()
            else:
            #add negative samples to fill out the total necessary
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
        
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
        
        X2 = X2[:, sel_samples, :]
        Y1 = Y1[:, sel_samples, :]
        Y2 = Y2[:, sel_samples, :]
        
        Y1_rpn_batch[im] = rpn_targets[0]
        Y2_rpn_batch[im] = rpn_targets[1]
        
        X2_batch[im] = X2[0]
        Y1_batch[im] = Y1[0]
        Y2_batch[im] = Y2[0]
        
    #if any of the images in the batch fail to predict regions, drop them from the batch
    if len(discard) > 0:
        X = np.delete(X, discard, 0)
        X2_batch = np.delete(X2_batch, discard, 0)
        Y1_rpn_batch = np.delete(Y1_rpn_batch, discard, 0)
        Y2_rpn_batch = np.delete(Y2_rpn_batch, discard, 0)
        Y1_batch = np.delete(Y1_batch, discard, 0)
        Y2_batch = np.delete(Y2_batch, discard, 0)
    
    return [X, X2_batch], [Y1_rpn_batch, Y2_rpn_batch, Y1_batch, Y2_batch], batch_pos_samples, discard