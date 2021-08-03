# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 08:59:10 2021

@author: blaine perry
"""

import keras_frcnn.roi_helpers as roi_helpers
import numpy as np
import random

def second_stage_helper(X, model_rpn, img_data, C):

    P_rpn = model_rpn.predict_on_batch(X)
                
    #R input and output are in feature space
    R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr=True, overlap_thresh=0.1, max_boxes=900)
    
    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
    X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, C.class_mapping)
    
    if X2 is None:
        return None, None, None, None
    
    neg_samples = np.where(Y1[0, :, -1] == 1)
    pos_samples = np.where(Y1[0, :, -1] == 0)
    
    neg_samples = list(neg_samples[0])
    pos_samples = list(pos_samples[0])
    
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
    
    return X2, [Y1, Y2], pos_samples