import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from PIL import Image
from sklearn.model_selection import train_test_split

from pathlib import Path
from typing import Tuple
import shutil

SEED = 1534

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.1, seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomZoom((0.0, 0.3), seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, seed=SEED),
    ]
)


class CnnRes:
    def __init__(self):
        self.train_directory = './Raw-Data/state-farm-distracted-driver-detection/imgs/train'
        self.input_image_shape = (224, 224, 3)  # Height, Wdith, Color channels

    def run(self):
        train_ds = self.get_data('training')
        valid_ds = self.get_data('validation')

        images_batch, labels_batch = list(train_ds.take(1)).pop()
        self.plot_images(images_batch, labels_batch)

        self.plot_images(data_augmentation(images_batch), labels_batch)

    def get_data(self, subset):
        return tf.keras.preprocessing.image_dataset_from_directory(
            self.train_directory,
            labels="inferred",  # Inferred from directory name
            label_mode="int",
            class_names=None,
            color_mode="rgb",
            batch_size=40,  # Since we have such small amount of data, we can use the full dataset as batch
            image_size=(224, 224),  # This is the default image input size of the model we wll be using
            shuffle=True,
            seed=SEED,
            validation_split=0.2,
            subset=subset,
            interpolation="bilinear",
            follow_links=False, )

    @staticmethod
    def plot_images(images, labels, num_columns=8):
        num_images = len(labels)
        num_rows = num_images // num_columns + 1
        plt.figure(figsize=(3 * num_columns, 3 * num_rows))
        for i, (image, label) in enumerate(zip(images, labels)):
            ax = plt.subplot(num_rows, num_columns, i + 1)
            plt.imshow(image / 255)
            plt.title(int(label))
            plt.axis("off")
        plt.show()

    def load_base_model(self):
        # Load the base model
        base_model = tf.keras.applications.EfficientNetB0(
            # Load an EfficientNet neural network architecture (implementation included in Keras)
            weights="imagenet",  # Load weights for the neural network pre-trained on ImageNet.
            input_shape=self.input_image_shape,  # Expected size of the input images
            include_top=False,  # Do not include the top layer (the classifier) which only works for ImagenNet classes
        )

        # Freeze the base_model so that it doesn't change the feature extraction weights
        base_model.trainable = False
        return base_model

    def create_model(self, base_model):
        # Define the input to be expected for the model
        inputs = tf.keras.Input(shape=self.input_image_shape)

        # First, we apply random data augmentation to the inputs
        x = data_augmentation(inputs)

        # Pre-trained weights often require that input be normalized from (0, 255) to a range (-1., +1.)
        # This is done in a specific way by the preprocessing function included with the model
        preprocessing = tf.keras.applications.efficientnet.preprocess_input
        x = preprocessing(x)
        # The base model contains batchnorm layers. We want to keep them in inference mode
        # when we unfreeze the base model for fine-tuning, so we make sure that the
        # base_model is running in inference mode here.
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Regularize feature extraction output with dropout
        x = tf.keras.layers.Dropout(0.2)(x)

        # Add our classifier layer to single output
        outputs = tf.keras.layers.Dense(1)(x)

        # Finally, the model is defined by joining the inputs to the outputs
        model = tf.keras.Model(inputs, outputs)

        # We can inspect the resulting layers of the model to make sure everything is correct
        # Note the difference between the total number of parameters and the number of trainable parameters
        model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy()], )

    def fine_tuning(self, base_model, model, train_ds):
        base_model.trainable = True
        model.summary()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Low learning rate
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
        )
        epochs = 10
        fine_tune_hist = model.fit(train_ds, epochs=epochs, validation_data=valid_ds)
        model.save("models/efficientnet_gvd.h5")
        model = tf.keras.models.load_model('models/efficientnet_gvd.h5')


if __name__ == '__main__':
    cn = CnnRes()
    cn.run()
