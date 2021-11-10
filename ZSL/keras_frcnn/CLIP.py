
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from keras.callbacks import Callback
from tensorflow.keras.layers import Input


"""
## Implement the projection head
The projection head is used to transform the image and the text embeddings to
the same embedding space with the same dimensionality.
"""

def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = layers.Dense(projection_dims)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([projected_embeddings, x])
        projected_embeddings = layers.LayerNormalization()(x)
    return projected_embeddings


"""
## Implement the vision encoder
In this example, we use Resnet50V2
from [Keras Applications](https://keras.io/api/applications/) as the base for the
vision encoder.
"""

def create_vision_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained resnet model to be used as the base encoder.
    ResNet = keras.applications.ResNet50V2(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # Set the trainability of the base encoder.
    for layer in ResNet.layers:
        layer.trainable = trainable
    # Receive the images as inputs.
    inputs = layers.Input(shape=(299, 299, 3), name="image_input")
    # Preprocess the input image.
    ResNet_input = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    # Generate the embeddings for the images using the resnet model.
    embeddings = ResNet(ResNet_input)
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return keras.Model(inputs, embeddings, name="vision_encoder")


"""
## Implement the text encoder
We use [BERT](https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1)
from [TensorFlow Hub](https://tfhub.dev) as the text encoder
"""


def create_text_encoder(C, trainable=False):
    # Load the BERT preprocessing module.
    preprocess = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        name="text_preprocessing",
    )
    # Load the pre-trained BERT model to be used as the base encoder.
    bert = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
        "bert",
    )
    # Set the trainability of the base encoder.
    bert.trainable = trainable
    # Receive the text as inputs.
    inputs = layers.Input(shape=(), dtype=tf.string, name="text_input")
    # Preprocess the text.
    bert_inputs = preprocess(inputs)
    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(bert_inputs)["pooled_output"]
    
    proj_inputs = Input(shape=embeddings.shape[1], name='bert_projections')
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        proj_inputs, C.num_projection_layers, C.projection_dims, C.dropout_rate
    )
    # Create the text encoder model.
    return keras.Model(inputs, embeddings, name="bert_encoder"), keras.Model(proj_inputs, outputs, name="text_encoder")