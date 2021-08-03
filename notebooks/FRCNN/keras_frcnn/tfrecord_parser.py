import tensorflow as tf
import glob


def get_data(input_path, data_type):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}
    
    label_file = glob.glob(input_path + '/*labels.txt')
    
    record_file = glob.glob(input_path + "/*" + data_type + "*.record")
  

    #read in the class labels
    class_dict = {}
    file = open(label_file[0], "r")

    #create a class label dictionary
    for line in file:
        key, value = line.split(':')
        class_dict[int(key)] = value.strip()
        
    #for debugging within tf.function
    '''
    import pdb
    tf.data.experimental.enable_debug_mode()
    '''
    
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
            features["image"] = tf.cast(tf.image.resize(raw_image, size=(500, 500)), tf.uint8)

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
            )

        
        
        batch_size = 5
        TFdataset = get_dataset(input_path + "/*" + data_type + "*.record", batch_size)

        record_file = glob.glob(input_path + "/*" + data_type + "*.record")
        
        total_records = sum(1 for _ in tf.data.TFRecordDataset(record_file[0]))
        print("total records in the TFrecord file is : " + str(total_records))
        
        #find the number of classes in the tfrecord file
        filename = -1
        for example in TFdataset:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
            #loop through each image in the batch
            for im in range(example['image'].numpy().shape[0]):
                
                filename  = filename + 1
                
                class_id = tf.sparse.to_dense(example['image/object/class/label']).numpy()[im]
                
                image = example['image'].numpy()[im]
                (w, h) = image.shape[:2]
                
                x1 = (tf.sparse.to_dense(example['image/object/bbox/xmin']).numpy()[im] * w).round().astype(int)
                x2 = (tf.sparse.to_dense(example['image/object/bbox/xmax']).numpy()[im] * w).round().astype(int)
                y1 = (tf.sparse.to_dense(example['image/object/bbox/ymin']).numpy()[im] * h).round().astype(int)
                y2 = (tf.sparse.to_dense(example['image/object/bbox/ymax']).numpy()[im] * h).round().astype(int)
                
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
                
                #Convert from class id numbers to text
                for cls in range(len(class_id)):
                    class_name.append(class_dict[class_id[cls]])
                                       
                for i in range(len(class_name)):
    
                    if class_name[i] not in classes_count:
                        classes_count[class_name[i]] = 1
                    else:
                        classes_count[class_name[i]] += 1
    
                    if class_name[i] not in class_mapping:
                        if class_name[i] == 'bg' and found_bg == False:
                            print('Found class name with special name bg. Will be treated as a background region.')
                            found_bg = True
                        class_mapping[class_name[i]] = len(class_mapping)
    
                    if filename not in all_imgs:
                        all_imgs[filename] = {}
    
                        (rows,cols) = image.shape[:2]
                        all_imgs[filename]['filepath'] = filename
                        all_imgs[filename]['width'] = cols
                        all_imgs[filename]['height'] = rows
                        all_imgs[filename]['bboxes'] = []
                        all_imgs[filename]['imageset'] = data_type
                        all_imgs[filename]['rawimage'] = image
                    
                    #ignore bounding boxes which are on the images edges
                    if x1[i] == 0 or x2[i] == 1 or y1[i] == 0 or y2[i] == 1:
                        continue
                    
                    all_imgs[filename]['bboxes'].append({'class': class_name[i], 'x1': x1[i], 'x2': x2[i], 'y1': y1[i], 'y2': y2[i]})
    
            all_data = []
            for key in all_imgs:
                all_data.append(all_imgs[key])
    
            # make sure the bg class is last in the list
            if found_bg:
                if class_mapping['bg'] != len(class_mapping) - 1:
                    key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                    val_to_switch = class_mapping['bg']
                    class_mapping['bg'] = len(class_mapping) - 1
                    class_mapping[key_to_switch] = val_to_switch
            
        return all_data, classes_count, class_mapping, TFdataset


