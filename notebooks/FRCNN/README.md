# keras-frcnn
Keras implementation of Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.
cloned from https://github.com/kbardool/keras-frcnn


USAGE:
- `train_frcnn.py` can be used to train a model. 

- Running `train_frcnn.py` will write weights to disk to an hdf5 file, as well as all the setting of the training run to a `pickle` file. These
settings can then be loaded by `test_frcnn.py` for any testing.

- test_frcnn.py can be used to perform inference, given pretrained weights and a config file. Specify a path to the folder containing
images:
    `python test_frcnn.py -p /path/to/test_data/`


NOTES:
- config.py contains all settings for the train or test run. The default settings match those in the original Faster-RCNN
paper. 

