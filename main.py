import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import kaggle

import os


CLASS_NAMES = ['Azadirachta Indica (Neem)',
              'Carissa Carandas (Karanda)',
              'Ficus Religiosa (Peepal Tree)']

CLASS_NUMBER = 3

IMAGE_SIZE = (150, 200)
IMAGE_SHAPE = (150, 200, 3)

SHUFFLE_SIZE = 10000
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2


def download_directory():
    src_directory = './dataset/'

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('aminizahra/leaf-edge', path='.', unzip=True)




def get_dataset():
    def image_parser(filename, label):
        image_str = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_str, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image = tf.image.resize(image, IMAGE_SIZE)

        return image, label

    filenames = []
    labels = []
    df_size = 0

    for label_idx in range(len(CLASS_NAMES)):
        path = f'./dataset/{CLASS_NAMES[label_idx]}'
        directory_finames = [os.path.join(path, image_file) for image_file in os.listdir(path)]

        filenames = filenames + directory_finames
        labels = labels + [tf.one_hot(label_idx, CLASS_NUMBER) for i in range(0, len(directory_finames))]

        print(label_idx, ':', CLASS_NAMES[label_idx], '->', tf.one_hot(label_idx, CLASS_NUMBER))

        df_size += len(directory_finames)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(image_parser)

    dataset = dataset.shuffle(SHUFFLE_SIZE)

    train_size = int(TRAIN_SPLIT * df_size)

    train = dataset.take(train_size)
    test = dataset.skip(train_size)


    train = train.batch(BATCH_SIZE)
    test = test.batch(BATCH_SIZE)

    train = train.prefetch(buffer_size=tf.data.AUTOTUNE)
    test = test.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train, test

#def show_example(test, model):


def train_and_evaluate_model(train, test):
    from model import model_v2 as create_model

    inputs, outputs = create_model()
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        train,
        epochs=25,
        verbose=2,
        validation_data=test
    )

    max_val_accuracy = max(history.history["val_accuracy"])
    print("Max validation accuracy is : {:.4f}".format(max_val_accuracy))

    score = model.evaluate(test, verbose=0)

    print('Test loss      : {:5.4f}'.format(score[0]))
    print('Test accuracy  : {:5.4f}'.format(score[1]))


def main():
    train,test = get_dataset()

    """
    plt.figure(figsize=(10, 10))

    for images, labels in train.take(1):
      for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        for j in range(len(CLASS_NAMES)):
            if tf.equal(labels[i], tf.one_hot(j, CLASS_NUMBER)).numpy().all():
                plt.title(CLASS_NAMES[j])
        plt.axis("off")

    plt.show()
    """
    
    train_and_evaluate_model(train, test)
    

if __name__ == '__main__':
    main()
