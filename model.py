import tensorflow as tf
from tensorflow.keras import layers

from main import CLASS_NUMBER, IMAGE_SHAPE


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

def model_v1():

    ## inputs and normalize
    inputs = layers.Input(shape=IMAGE_SHAPE)
    image = layers.Lambda(lambda img: img/255)(inputs)

    ## data augmentation
    image = data_augmentation(image)

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


def model_v2():
    ## inputs and normalize
    inputs = layers.Input(shape=IMAGE_SHAPE)
    image = layers.Lambda(lambda img: img/255)(inputs)

    # conv1
    image = layers.Conv2D(32, (3, 3), 
                        activation='relu', 
                        kernel_initializer='he_uniform',
                        padding='same')(image)
    image = layers.Conv2D(32, (3, 3), 
                        activation='relu', 
                        kernel_initializer='he_uniform',
                        padding='same')(image)
    image = layers.MaxPooling2D((2, 2))(image)

    # conv2
    image = layers.Conv2D(64, (3, 3), 
                        activation='relu', 
                        kernel_initializer='he_uniform',
                        padding='same')(image)
    image = layers.Conv2D(64, (3, 3), 
                        activation='relu', 
                        kernel_initializer='he_uniform',
                        padding='same')(image)
    image = layers.MaxPooling2D((2, 2))(image)

    # flatten
    image = layers.Flatten()(image)

    # fully connected
    image = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(image)
    outputs = layers.Dense(CLASS_NUMBER, activation='softmax')(image)

    return inputs, outputs


def model_mobilenetv2():
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False
    
    inputs = layers.Input(shape=IMAGE_SHAPE)
    image = layers.Lambda(lambda img: img/255)(inputs)

    ## data augmentation
    image = data_augmentation(image) 

    ## preprocessinig
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    ## using MobileNetV2
    image = base_model(image, training=False)

    image = tf.keras.layers.GlobalAveragePooling2D()(image)
    image = tf.keras.layers.Dropout(0.2)(image)
    outputs = tf.keras.layers.Dense(CLASS_NUMBER)(image)

    return inputs, outputs