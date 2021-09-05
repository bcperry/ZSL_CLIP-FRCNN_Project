# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 08:59:10 2021

@author: blaine perry
"""

import keras_frcnn.roi_helpers as roi_helpers
import numpy as np
import random
from . import data_generators
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

def bert_embed(text, C):
    index =  C.class_text.index(text)
    embed = C.bert_embeddings[index]
    return embed

def get_class_map(C, filename):
    
    if filename is not None:
        class_text = filename
    else:
        class_text = C.class_text
    
    label_file = glob.glob(C.data_path + class_text)
    class_mapping = {}
    file = open(label_file[0], "r")
    
    if filename is None:
        class_mapping['bg'] = 0
    
    #create a class label dictionary
    for line in file:
        key, value = line.split(':')
        if int(key) not in C.training_classes:
            continue
        class_mapping[value.strip()] = int(key)
    
    return class_mapping

def get_class_text(C):

    pickle_file = glob.glob(C.data_path + C.text_dict_pickle)
    class_text = pd.read_pickle(pickle_file[0])
    class_text_mapping = {}
    
    for key in C.class_mapping:
        text_key = C.class_mapping[key]
        #We need to accound for the background class
        if text_key == 0:
            class_text_mapping[text_key] = 'A background with features: None'
            continue
        #these keys exist in the data but are not in the descriptions
        if text_key == 75 or text_key == 82 :
            class_text_mapping[text_key] = 'A unknown with features: '
            continue
        text_list = class_text[text_key]
        desc = "A " + key + " with features: "
        for i in text_list:
            desc = desc + i + ', '
        class_text_mapping[text_key] = desc
        
    return class_text_mapping


def get_data_parallel(inputs):
    C = inputs[0]
    X = inputs[1]
    img_data = inputs[2]
    P_rpn = inputs[3]
    
    X, rpn_targets, _ = data_generators.calc_targets(C, X, img_data, P_rpn[0].shape[1], P_rpn[0].shape[2])
    
    
    #R input and output are in feature space
    R = roi_helpers.rpn_to_roi(C, P_rpn[0], P_rpn[1], use_regr=True, overlap_thresh=0.7, max_boxes=300)

    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
    X2, Y1, Y2, IouS, class_text = roi_helpers.calc_iou(C, R, img_data)

    if X2 is None:
        return None
    
    neg_samples = np.where(Y1[0, :, 0] == 1)
    pos_samples = np.where(Y1[0, :, 0] == 0)
    
    neg_samples = list(neg_samples[0])
    pos_samples = list(pos_samples[0])

    
    if C.num_rois > 1:
        selected_pos_samples = []
        selected_neg_samples= []
        
        if len(pos_samples) < int(C.num_rois//2):
            selected_pos_samples = pos_samples
        else:
            selected_pos_samples = np.random.choice(pos_samples, int(C.num_rois//2), replace=False).tolist()
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
        #shuffle the list so that the positives are not always the first two
        random.shuffle(sel_samples)
    else:
        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
        selected_pos_samples = pos_samples
        selected_neg_samples = neg_samples
        if np.random.randint(0, 2):
            sel_samples = random.choice(neg_samples)
        else:
            
            if len(selected_pos_samples) == 0:
                sel_samples = random.choice(neg_samples)
            else:
                sel_samples = random.choice(pos_samples)
            
    #cast Y1 and Y2 to float32 in case there are no positive samples in the selection
    Y1 = Y1.astype('float32')
    Y2 = Y2.astype('float32')
    
    
    
    X2 = X2[:, sel_samples, :]
    Y1 = Y1[:, sel_samples, :]
    Y2 = Y2[:, sel_samples, :]
    selected_sample_texts = class_text[:, sel_samples]
    if C.num_rois == 1:
        selected_sample_texts = np.expand_dims(selected_sample_texts, axis = 0)
        X2 = np.expand_dims(X2, axis = 0)
        Y1 = np.expand_dims(Y1, axis = 0)
        Y2 = np.expand_dims(Y2, axis = 0)
    
    Y1_rpn_batch = rpn_targets[0][0]
    Y2_rpn_batch = rpn_targets[1][0]
    
    X2_batch = X2[0]
    Y1_batch = Y1[0]
    Y2_batch = Y2[0]
    text_batch = selected_sample_texts[0]

    return [X, X2_batch], [Y1_rpn_batch, Y2_rpn_batch, Y1_batch, Y2_batch], pos_samples, text_batch


def parallelize(C, X, img_data, P_rpn):
    """
    Create a thread pool and calculate data for the batch
    """
    X_batch = []
    Y1_rpn_batch = []
    Y2_rpn_batch = []
    X2_batch = []
    Y1_batch = []
    Y2_batch = []
    text_batch = []
    
    batch_pos_samples = 0
    bad_images = 0
    
    #for debugging*************************************************************************************************************************************
    '''
    for im in range(X.shape[0]):
        get_data_parallel([C, X[im], img_data[im], [P_rpn[0][im:im+1], P_rpn[1][im:im+1]]])
'''

    with ThreadPoolExecutor(max_workers=C.batch_size) as executor:
        futures = [executor.submit(get_data_parallel, [C, X[im], img_data[im], [P_rpn[0][im:im+1], P_rpn[1][im:im+1]]]) for im in range(X.shape[0])]
    for future in as_completed(futures):
        
        #if there are no good bounding boxes, ignore the image
        if future.result() == None:
            bad_images += 1
            continue
        
        X = future.result()[0][0][0]
        X2 = future.result()[0][1]
        Y1_rpn = future.result()[1][0]
        Y2_rpn = future.result()[1][1]
        Y1 =  future.result()[1][2]
        Y2 = future.result()[1][3]
        pos_samples = future.result()[2]
        text = future.result()[3]
        
        X_batch.append(X)
        Y1_rpn_batch.append(Y1_rpn)
        Y2_rpn_batch.append(Y2_rpn)
        X2_batch.append(X2)
        Y1_batch.append(Y1)
        Y2_batch.append(Y2)
        text_batch.append(text)
        batch_pos_samples = batch_pos_samples + len(pos_samples)
        
    if bad_images == X.shape[0]:
        #print('no valid images were found')
        return(None, None, None, None)
    
    #if we ignored any images, replace them with the last good image.  otherwise, the model will not update properly
    for i in range(bad_images):
        X_batch.append(X)
        Y1_rpn_batch.append(Y1_rpn)
        Y2_rpn_batch.append(Y2_rpn)
        X2_batch.append(X2)
        Y1_batch.append(Y1)
        Y2_batch.append(Y2)
        text_batch.append(text)
        batch_pos_samples = batch_pos_samples + len(pos_samples)
        
    X_batch = np.array(X_batch)
    Y1_rpn_batch = np.array(Y1_rpn_batch)
    Y2_rpn_batch = np.array(Y2_rpn_batch)
    X2_batch = np.array(X2_batch)
    Y1_batch = np.array(Y1_batch)
    Y2_batch = np.array(Y2_batch)
    text_batch = np.array(text_batch)
    
    return [X_batch, X2_batch], [Y1_rpn_batch, Y2_rpn_batch, Y1_batch, Y2_batch], batch_pos_samples, text_batch    
        
def second_stage_helper(X, P_rpn, img_data, C):
    
    X, Y, pos_samples, text_batch = parallelize(C, X, img_data, P_rpn)
    
    if X is None:
        return(None, None, None, None, None)
    
    if X[0].shape[0] != C.batch_size:
        discard = C.batch_size - X[0].shape[0]
    else:
        discard = 0

    return X, Y, pos_samples, discard, text_batch
    