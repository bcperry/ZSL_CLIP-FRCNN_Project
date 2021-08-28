import numpy
class Config:

    def __init__(self):
        
        #common user defined information
        
        #path to the training data
        self.data_path = r'C:\\Data_drive\\Data\\'
        self.train_path = r"pascal_train.record"
        self.val_path = r"pascal_test.record"
        self.class_text = r'pascal_class_labels.txt'
        
        #number of epochs to train
        self.num_epochs = 25
        
        # number of ROIs at once
        self.num_rois = 4
        
        self.output_weight_path = r'C:\Data_drive\workspace\FRCNN\model.hdf5'
        
        #path to the input weights. If trainining for the first time use None to load imagenet weights and/ or standard BERT weights
        self.input_weight_path = None#r"C:\Data_drive\Github\GEOINT_Zero-shot_Object_identification\notebooks\ZSL\outputs\FRCNN model\FRCNN_epoch23-total_loss-0.40.hdf5"
        
        self.text_dict_pickle = None#r"xview_attribute_dict_text.pickle"
        
        self.batch_size = 40
        
        #CLIP projection settings
        
        self.num_projection_layers=1
        
        self.projection_dims=256
        
        self.dropout_rate=0.1
        
        self.temperature=0.05
        
        
        
        self.verbose = True
  
        self.network = 'resnet50'
  
        # setting for data augmentation
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False
  
        # anchor box information
        self.anchor_box_scales = [128, 256, 512]#[5, 10, 20, 50, 100]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
  
        # size to resize the smallest side of the image
        self.im_size = 413
  
        # image channel-wise mean to subtract
        self.img_channel_mean = [103.939, 116.779, 123.68] #[R, G, B]
        self.img_scaling_factor = 1.0
  
  
  
        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 16
  
        self.balanced_classes = False
  
        # scaling the stdev
        self.std_scaling = 4.0#1.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]#[1.0, 1.0, 1.0, 1.0]
  
        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7
  
        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5
  
        # placeholder for the class mapping, automatically generated by the parser
        self.class_mapping = None
  
        self.model_path = 'model_frcnn.resnet.hdf5'
        
        self.training_classes = list(range(1,21))#[11, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 47, 49, 50, 53, 59, 60, 61, 62, 63, 64, 65, 66, 91]
        weight_list = list(numpy.ones(len(self.training_classes)))
        weight_list[0] = .1
        weight_list = weight_list * self.batch_size
        self.cce_weight = weight_list
