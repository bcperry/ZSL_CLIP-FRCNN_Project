import tensorflow as tf
import glob
import numpy


def batch_processor(batch, C):
    
    imgs = {}
    #read in the class labels
    class_dict = C.class_dict

        
    for im in range(batch['image'].numpy().shape[0]):

        class_id = tf.sparse.to_dense(batch['image/object/class/label']).numpy()[im]
        
        image = batch['image'].numpy()[im]
        (w, h) = image.shape[:2]
        
        x1 = (tf.sparse.to_dense(batch['image/object/bbox/xmin']).numpy()[im] * w).round().astype(int)
        x2 = (tf.sparse.to_dense(batch['image/object/bbox/xmax']).numpy()[im] * w).round().astype(int)
        y1 = (tf.sparse.to_dense(batch['image/object/bbox/ymin']).numpy()[im] * h).round().astype(int)
        y2 = (tf.sparse.to_dense(batch['image/object/bbox/ymax']).numpy()[im] * h).round().astype(int)
        
        #drop 0s due to the sparse to dense
        zeroes_map = class_id != 0
        class_id = class_id[zeroes_map]
        x1 = x1[zeroes_map]
        x2 = x2[zeroes_map]
        y1 = y1[zeroes_map]
        y2 = y2[zeroes_map]
        
        try:
            assert len(class_id) == len(x1) == len(x2) == len(y1) == len(y2)
        except Exception as e:
            print(f'Exception: {e}')
            
        class_name = []
        
        for cls in range(len(class_id)):
            class_name.append(class_dict[class_id[cls]])
    
        for i in range(len(class_name)):
            #Convert from class id numbers to text
            for cls in range(len(class_id)):
                class_name.append(class_dict[class_id[cls]])
                                   
            if im not in imgs:
                imgs[im] = {}
                imgs[im]['image_number'] = im
                (rows,cols) = image.shape[:2]
                
                imgs[im]['width'] = cols
                imgs[im]['height'] = rows
                imgs[im]['bboxes'] = []
                imgs[im]['rawimage'] = image
            
            imgs[im]['bboxes'].append({'class': class_name[i], 'x1': x1[i], 'x2': x2[i], 'y1': y1[i], 'y2': y2[i]})

    return imgs


def get_data(path, C):
    
    label_file = glob.glob(C.data_path + C.class_text)
    
    record_file = glob.glob(C.data_path + path)

    #read in the class labels
    class_dict = {}
    file = open(label_file[0], "r")

    #create a class label dictionary
    for line in file:
        key, value = line.split(':')
        class_dict[int(key)] = value.strip()
    C.class_dict = class_dict   
    #for debugging within tf.function

    #import pdb
    #tf.data.experimental.enable_debug_mode()

    
    with open(record_file[0],'r') as f:

        print('Parsing annotation files')


        #tfrecord reader helper functions

        feature_description = {
                    'image/height': tf.io.FixedLenFeature([], tf.int64),
                    'image/width': tf.io.FixedLenFeature([], tf.int64),
                    'image/encoded': tf.io.FixedLenFeature([], tf.string),
                    'image/format': tf.io.FixedLenFeature([], tf.string),
                    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            }

        def read_example(example):
            #pdb.Pdb(nosigint=True).set_trace()
            features = tf.io.parse_single_example(example, feature_description)
            raw_image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
            features["image"] = tf.cast(tf.image.resize(raw_image, size=(C.im_size, C.im_size)), tf.uint8)
            
            features.pop("image/encoded")
            features.pop("image/format")
            features.pop("image/height")
            features.pop("image/width")
            return features

        def get_dataset(file_pattern, batch_size):

            return (
                tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
                .map(
                    read_example,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    deterministic=False,
                )
                .shuffle(batch_size * 10)
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                .batch(batch_size)
                .repeat()
            )

        
        TFdataset = get_dataset(record_file, C.batch_size)
        
        total_records = sum(1 for _ in tf.data.TFRecordDataset(record_file[0]))
        print("total records in the TFrecord file is : " + str(total_records))
        
        return TFdataset, total_records
