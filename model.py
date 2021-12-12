import tensorflow as tf
from tensorflow.keras import layers

from main import CLASS_NUMBER, IMAGE_SHAPE



def model_v1():
    inputs = layers.Input(shape=IMAGE_SHAPE)
    image = layers.Lambda(lambda img: img/255)(inputs)

    ### conv1
    image = layers.Convolution2D(filters=16,
                            kernel_size=[3,3],
                            padding='valid',
                            activation=tf.nn.relu)(image)

    image = layers.BatchNormalization(momentum=0.01)(image)
    image = layers.MaxPooling2D(pool_size=[2,2],
                            strides=[2,2],
                            padding='same')(image)

    ### conv2
    image = layers.Convolution2D(filters=16,
                            kernel_size=[3,3],
                            padding='valid',
                            activation=tf.nn.relu)(image)
    image = layers.BatchNormalization(momentum=0.01)(image)
    image = layers.MaxPooling2D(pool_size=[2,2],
                            strides=[2,2],
                            padding='same')(image)
    ### conv3
    image = layers.Convolution2D(filters=32,
                            kernel_size=[5,5],
                            padding='valid',
                            activation=tf.nn.relu)(image)
    image = layers.BatchNormalization(momentum=0.01)(image)
    image = layers.MaxPooling2D(pool_size=[2,2],
                            strides=[2,2],
                            padding='same')(image)
    ### conv4
    image = layers.Convolution2D(filters=32,
                            kernel_size=[5,5],
                            padding='valid',
                            activation=tf.nn.relu)(image)
    image = layers.BatchNormalization(momentum=0.01)(image)
    image = layers.MaxPooling2D(pool_size=[2,2],
                            strides=[2,2],
                            padding='same')(image)
    ### conv5
    image = layers.Convolution2D(filters=32,
                            kernel_size=[5,5],
                            padding='valid',
                            activation=tf.nn.relu)(image)
    image = layers.BatchNormalization(momentum=0.01)(image)
    image = layers.MaxPooling2D(pool_size=[2,2],
                            strides=[2,2],
                            padding='same')(image)
    ### flatten
    image = layers.Flatten()(image)

    image = layers.Dense(units=100, 
                    activation=tf.nn.relu)(image)
    
    outputs = layers.Dense(units=CLASS_NUMBER, activation=tf.nn.relu)(image)

    return inputs, outputs