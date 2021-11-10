# keras-frcnn-Zero-shot
Keras implementation of Faster R-CNN with CLIP

This package can be used to train a dual text/image imbedded Faster-RCNN model

USAGE:
- `train_model.py` can be used to train a the model  The steps should be as follows:
- first train a pure Faster-RCNN by calling:
`python train_model.py --data-folder /path/to/folder/containing/tfrecords --input-weight-path /path/to/*.hdf5 --num-epochs X --model-type FRCNN --azure False`

- After training the pure Faster-RCNN, train the dual FRCNN by calling
`python train_model.py --data-folder /path/to/folder/containing/tfrecords --input-weight-path /path/to/previously/traied/*.hdf5 --num-epochs X --model-type ZSL --azure False`

- Running `train_ZSL.py` will write weights to an hdf5 file after each epoch with the format `MODEL_TYPE_epochXX-total_loss-XX.XX.hdf5` in both instances 

- `test_model.py` can be used to run inference on the model with `python test_model.py --path /path/to/folder/containing/images --input-weight-path /path/to/previously/traied/*.hdf5 --num-rois X --model-type ZSL`

NOTES:
- config.py contains all settings for the train or test run. The default settings match those in the original Faster-RCNN paper. The anchor box sizes are [128, 256, 512] and the ratios are [1:1, 1:2, 2:1].
